"""Restart recovery: the edges a restart takes and the report it produces (DESIGN.md §7).

A restart moves jobs backwards along edges the forward lifecycle never takes, so those edges live
here rather than in :data:`~image_processor.ledger.schema.TRANSITIONS`. The rules:

* ``INFERENCING -> READY`` — the executor cell died mid-job. The attempt count is kept, so the
  retry budget still shrinks and a poison input eventually exhausts (DESIGN.md §6.2).
* ``CLAIMED -> READY`` and ``WAITING_MODEL -> READY`` — the job was admitted but no work started.
* ``RETRY_WAIT -> READY`` — only when the backoff timer has already elapsed.
* ``RESULT_COMMITTED``/``PUBLISH_PENDING``/``PUBLISH_EXHAUSTED`` — untouched. The result is durable
  and its outbox rows are still eligible; the publisher picks them up.
* ``CLEANUP_PENDING`` — untouched and listed in the report, because only the filesystem can say
  what happened; :class:`~image_processor.completion.actions.Completer.reconcile` decides.

The report also carries the sidecar bindings of every committed-but-unpublished job so the caller
can run the DESIGN.md §7 filesystem reconciliation (an orphan sidecar is verified and adopted or
removed; a committed record whose sidecar is absent returns to re-inference through
:meth:`~image_processor.ledger.ledger.Ledger.requeue_for_reinference`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from image_processor.types import JobState

#: Edges restart recovery may take. Deliberately separate from the forward lifecycle table.
RECOVERY_EDGES = frozenset(
    {
        (JobState.INFERENCING, JobState.READY),
        (JobState.CLAIMED, JobState.READY),
        (JobState.WAITING_MODEL, JobState.READY),
        (JobState.RETRY_WAIT, JobState.READY),
        (JobState.RESULT_COMMITTED, JobState.READY),
        (JobState.PUBLISH_PENDING, JobState.READY),
    }
)

#: States recovery restarts unconditionally, mapped to the state they restart at.
RESTART_STATES = {
    JobState.INFERENCING: JobState.READY,
    JobState.CLAIMED: JobState.READY,
    JobState.WAITING_MODEL: JobState.READY,
}


@dataclass(frozen=True)
class SidecarRecord:
    """The durable sidecar binding of a job whose result is committed but not yet cleaned up."""

    inference_id: str
    state: JobState
    sidecar_path: str
    sidecar_sha256: str


@dataclass(frozen=True)
class RecoveryReport:
    """What one :meth:`~image_processor.ledger.ledger.Ledger.recover` pass did and found."""

    transitions: dict = field(default_factory=dict)
    cleanup_pending: tuple = ()
    publish_pending: tuple = ()
    sidecars: tuple = ()

    @property
    def total(self) -> int:
        """The number of jobs recovery moved."""
        return sum(self.transitions.values())

    def count(self, current: JobState, new: JobState) -> int:
        """Return how many jobs moved along ``current -> new`` in this pass.

        Args:
            current: The state the jobs were found in.
            new: The state they were moved to.

        Returns:
            The count, or ``0`` when no job took that edge.
        """
        return self.transitions.get(edge_key(current, new), 0)


def edge_key(current: JobState, new: JobState) -> str:
    """Render an edge as the ``"FROM->TO"`` key used in :attr:`RecoveryReport.transitions`."""
    return f"{current.value}->{new.value}"


@dataclass(frozen=True)
class RecoveryMove:
    """One planned restart transition."""

    inference_id: str
    current: JobState
    new: JobState


def plan_recovery(rows: Iterable, now_ms: int) -> list:
    """Decide which jobs a restart moves, without touching the database.

    Args:
        rows: ``(inference_id, state, next_attempt_at_ms)`` triples for every non-terminal job.
        now_ms: The current wall clock in milliseconds; a ``RETRY_WAIT`` job whose
            ``next_attempt_at_ms`` is at or before this is due and restarts at ``READY``.

    Returns:
        The :class:`RecoveryMove` list, in the order the rows arrived.
    """
    moves = []
    for inference_id, raw_state, next_attempt_at_ms in rows:
        state = JobState(raw_state)
        target: Optional[JobState] = RESTART_STATES.get(state)
        if target is None and state is JobState.RETRY_WAIT:
            if next_attempt_at_ms is None or next_attempt_at_ms <= now_ms:
                target = JobState.READY
        if target is not None:
            moves.append(RecoveryMove(inference_id, state, target))
    return moves


def build_report(
    moves: Iterable,
    cleanup_pending: Iterable,
    publish_pending: Iterable,
    sidecars: Iterable,
) -> RecoveryReport:
    """Assemble a :class:`RecoveryReport` from the applied moves and the observed backlog.

    Args:
        moves: The :class:`RecoveryMove` values that were applied.
        cleanup_pending: :class:`~image_processor.types.CleanupIntent` values awaiting
            reconciliation.
        publish_pending: Inference ids whose outbox is still eligible.
        sidecars: :class:`SidecarRecord` values for committed-but-uncleaned jobs.

    Returns:
        The immutable report.
    """
    transitions: dict = {}
    for move in moves:
        key = edge_key(move.current, move.new)
        transitions[key] = transitions.get(key, 0) + 1
    intents: tuple = tuple(cleanup_pending)
    return RecoveryReport(
        transitions=transitions,
        cleanup_pending=intents,
        publish_pending=tuple(publish_pending),
        sidecars=tuple(sidecars),
    )
