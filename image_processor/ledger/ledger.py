"""The SQLite job ledger, outbox, cleanup intents, and recovery (LLD §5, DESIGN.md §7).

Durability model, following the same shape as file-replicator's ``state.rs``:

* **WAL** so a committed transaction survives a process crash and readers never block the writer.
  ``synchronous`` is configurable and defaults to ``FULL``, the regulated profile in DESIGN.md §7.
* **One writer.** Every mutation is a closure queued to a single writer thread that owns the only
  write connection and wraps the closure in ``BEGIN IMMEDIATE`` / ``COMMIT``. Callers block on a
  :class:`~concurrent.futures.Future`, so a caller sees the write only after it is durable and a
  failed closure rolls back whole. Reads use a second connection and never take the writer's turn.
* **Write-ahead.** The state that authorizes a side effect is committed before the side effect
  runs: ``commit_result`` stores the result, the sidecar digest, and every outbox row in one
  transaction before an outbox row becomes eligible, and a cleanup intent is committed before the
  first file mutation.

Outbox eligibility is derived from job state, not from a flag: :meth:`Ledger.pending_outbox`
returns rows only for jobs in ``PUBLISH_PENDING``. A row committed alongside ``RESULT_COMMITTED``
is durable but invisible to the publisher until the same transaction advances the job.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from image_processor.ledger import schema as _schema
from image_processor.ledger.recovery import (
    RECOVERY_EDGES,
    RecoveryReport,
    SidecarRecord,
    build_report,
    plan_recovery,
)
from image_processor.types import (
    CleanupIntent,
    Job,
    JobState,
    TERMINAL_STATES,
)

log = logging.getLogger(__name__)

#: ``synchronous`` values SQLite accepts. Checked before the value reaches a PRAGMA statement.
SYNCHRONOUS_MODES = frozenset({"OFF", "NORMAL", "FULL", "EXTRA"})

#: Default reservation budget: the total bytes of outbox and evidence capacity admission may hold.
DEFAULT_RESERVE_BUDGET_BYTES = 256 * 1024 * 1024

#: States after which a job's admission reservation is released.
_RELEASING_STATES = frozenset(TERMINAL_STATES)

#: Position of ``created_at_ms`` in a :data:`~image_processor.ledger.schema.JOB_SELECT` row.
_CREATED_AT_IDX = _schema.JOB_COLUMNS.index("created_at_ms")


class LedgerError(Exception):
    """Base class for every ledger failure."""


class LedgerConflict(LedgerError):
    """A compare-and-set transition lost: the row is missing or not in the expected state."""


class IllegalTransition(LedgerError):
    """The requested edge is not in the DESIGN.md §7 state diagram."""


class LedgerClosed(LedgerError):
    """The ledger was closed and can no longer accept writes."""


@dataclass(frozen=True)
class OutboxRow:
    """One prepared message awaiting confirmed publication.

    ``encoded_bytes`` are the exact bytes ``AppFacade.prepare()`` froze; the publisher retries the
    same bytes rather than re-encoding. ``gating`` marks the rows that must be confirmed before
    cleanup may run, which is the ``app/inference/result`` message (DESIGN.md §12.1, D-IP-6).
    """

    id: Optional[int]
    inference_id: str
    topic: str
    encoded_bytes: bytes
    attempts: int = 0
    gating: bool = True
    last_error: Optional[str] = None


def _now_ms() -> int:
    """Return the current wall clock in milliseconds."""
    return int(time.time() * 1000)


class Ledger:
    """Durable job state for one component process.

    Args:
        path: The state database file. Parent directories are created.
        synchronous: The SQLite ``synchronous`` PRAGMA. ``FULL`` (the default) is the regulated
            profile; ``NORMAL`` trades a fsync per commit for throughput on a development host.
        busy_timeout_ms: How long a connection waits for a lock before raising.
        reserve_budget_bytes: The admission byte budget. :meth:`admit` refuses a job whose
            reservation would push the outstanding total past it.
        clock: Returns the current wall clock in milliseconds. Injected by tests.
    """

    def __init__(
        self,
        path: Path,
        synchronous: str = "FULL",
        busy_timeout_ms: int = 5000,
        reserve_budget_bytes: int = DEFAULT_RESERVE_BUDGET_BYTES,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        mode = str(synchronous).upper()
        if mode not in SYNCHRONOUS_MODES:
            raise ValueError(f"unsupported synchronous mode: {synchronous!r}")
        self.path = Path(path)
        self.synchronous = mode
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.reserve_budget_bytes = int(reserve_budget_bytes)
        self._clock = clock
        self._closed = False
        self._read_lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()

        if self.path.parent and str(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._write_conn = self._connect()
        _schema.apply_schema(self._write_conn)
        self._write_conn.commit()
        self._read_conn = self._connect()

        self._writer = threading.Thread(
            target=self._writer_loop, name=f"ledger-writer-{self.path.name}", daemon=True
        )
        self._writer.start()

    # -- connection plumbing ---------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with the durability PRAGMAs this ledger runs under."""
        conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA synchronous={self.synchronous}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    def _writer_loop(self) -> None:
        """Run queued write closures in arrival order, each in its own immediate transaction."""
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            fn, future = item
            try:
                self._write_conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(self._write_conn)
                except BaseException:
                    self._write_conn.execute("ROLLBACK")
                    raise
                self._write_conn.execute("COMMIT")
            except BaseException as exc:  # noqa: BLE001 - reported to the calling thread
                if self._write_conn.in_transaction:
                    self._write_conn.execute("ROLLBACK")
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._queue.task_done()

    def _write(self, fn: Callable):
        """Queue ``fn`` for the writer thread and block until its transaction commits.

        Args:
            fn: A callable taking the write connection and returning the caller's result.

        Returns:
            Whatever ``fn`` returned.

        Raises:
            LedgerClosed: The ledger is closed.
        """
        if self._closed:
            raise LedgerClosed(f"ledger {self.path} is closed")
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((fn, future))
        return future.result()

    def _read(self, sql: str, params: tuple = ()) -> list:
        """Run a query on the read connection and return every row."""
        with self._read_lock:
            return self._read_conn.execute(sql, params).fetchall()

    def _read_one(self, sql: str, params: tuple = ()):
        """Run a query on the read connection and return the first row or ``None``."""
        with self._read_lock:
            return self._read_conn.execute(sql, params).fetchone()

    def close(self) -> None:
        """Drain the writer queue, stop the writer thread, and close both connections."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._writer.join(timeout=30)
        with self._read_lock:
            self._read_conn.close()
        self._write_conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- jobs ------------------------------------------------------------------------------

    def admit(self, job: Job, reserve_bytes: int) -> bool:
        """Admit a job and reserve its outbox and evidence capacity (DESIGN.md §7).

        Admission is idempotent on ``inference_id``: re-discovering the same input under the same
        model finds the row already there and changes nothing, which is what makes a lost
        filesystem hint harmless. The reservation is held until the job reaches a terminal state,
        so a finished job is never stranded by a full outbox.

        Args:
            job: The job to admit. Its state must be ``DISCOVERED`` or ``READY``.
            reserve_bytes: Capacity to hold for this job's maximum configured result.

        Returns:
            ``True`` when the job was newly admitted, ``False`` when it already exists or the
            reservation does not fit the budget.

        Raises:
            IllegalTransition: ``job.state`` is not an admission state.
            ValueError: ``reserve_bytes`` is negative.
        """
        if job.state not in _schema.INITIAL_STATES:
            raise IllegalTransition(f"cannot admit a job in {job.state.value}")
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes must not be negative")

        def _txn(conn) -> bool:
            existing = conn.execute(
                "SELECT 1 FROM jobs WHERE inference_id = ?", (job.inference_id,)
            ).fetchone()
            if existing:
                return False
            held = conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM reservations").fetchone()[0]
            if held + reserve_bytes > self.reserve_budget_bytes:
                log.warning(
                    "admission refused for %s: reserving %d over budget %d (held %d)",
                    job.inference_id,
                    reserve_bytes,
                    self.reserve_budget_bytes,
                    held,
                )
                return False
            now = self._clock()
            conn.execute(
                "INSERT INTO jobs (inference_id, route_id, state, source_json, model_json, "
                "transform_version, attempts, next_attempt_at_ms, staged_path, config_generation, "
                "created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.inference_id,
                    job.route_id,
                    job.state.value,
                    _schema.encode_source(job.source),
                    _schema.encode_model(job.model),
                    job.transform_version,
                    job.attempts,
                    job.next_attempt_at_ms,
                    job.staged_path,
                    job.config_generation,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO reservations (inference_id, bytes) VALUES (?, ?)",
                (job.inference_id, reserve_bytes),
            )
            return True

        return self._write(_txn)

    def reserved_bytes(self) -> int:
        """Return the total capacity currently held by admitted, non-terminal jobs."""
        return self._read_one("SELECT COALESCE(SUM(bytes), 0) FROM reservations")[0]

    def get(self, inference_id: str) -> Optional[Job]:
        """Return the job, or ``None`` when it was never admitted.

        Args:
            inference_id: The job identity.

        Returns:
            The :class:`~image_processor.types.Job`, or ``None``.
        """
        row = self._read_one(
            _schema.JOB_SELECT + " WHERE inference_id = ?", (inference_id,)
        )
        return _schema.row_to_job(row) if row else None

    def last_error(self, inference_id: str) -> Optional[str]:
        """Return the last recorded error for a job, or ``None``.

        Args:
            inference_id: The job identity.

        Returns:
            The stored error string, or ``None`` when the job is clean or absent.
        """
        row = self._read_one("SELECT last_error FROM jobs WHERE inference_id = ?", (inference_id,))
        return row[0] if row else None

    def _set_state(self, conn, inference_id: str, expected, new: JobState, **fields) -> Job:
        """Compare-and-set one job's state inside an open transaction.

        Args:
            conn: The write connection, already in a transaction.
            expected: The state the job must currently be in, or ``None`` to skip the check.
            new: The state to move to.
            **fields: Columns from :data:`~image_processor.ledger.schema.MUTABLE_JOB_FIELDS`.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The row is missing or is not in ``expected``.
            IllegalTransition: The edge is not legal.
            ValueError: A field outside the mutable set was passed.
        """
        unknown = set(fields) - _schema.MUTABLE_JOB_FIELDS
        if unknown:
            raise ValueError(f"not settable on a transition: {sorted(unknown)}")
        row = conn.execute(
            "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
        ).fetchone()
        if row is None:
            raise LedgerConflict(f"no such job: {inference_id}")
        current = JobState(row[0])
        if expected is not None and current is not expected:
            raise LedgerConflict(
                f"{inference_id} is {current.value}, expected {expected.value}"
            )
        if current is not new and not _schema.is_legal(current, new):
            raise IllegalTransition(f"{current.value} -> {new.value} is not a lifecycle edge")
        assignments = ["state = ?", "updated_at_ms = ?"]
        params: list = [new.value, self._clock()]
        for key in sorted(fields):
            assignments.append(f"{key} = ?")
            params.append(fields[key])
        params.append(inference_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE inference_id = ?", tuple(params)
        )
        if new in _RELEASING_STATES:
            conn.execute("DELETE FROM reservations WHERE inference_id = ?", (inference_id,))
        return _schema.row_to_job(
            conn.execute(
                _schema.JOB_SELECT + " WHERE inference_id = ?", (inference_id,)
            ).fetchone()
        )

    def transition(
        self, inference_id: str, expected: JobState, new: JobState, **fields
    ) -> Job:
        """Move a job along a lifecycle edge, compare-and-set on ``expected``.

        Args:
            inference_id: The job identity.
            expected: The state the job must currently be in.
            new: The state to move to; must be a DESIGN.md §7 edge from ``expected``.
            **fields: Optional column updates (``attempts``, ``next_attempt_at_ms``,
                ``staged_path``, ``config_generation``, ``last_error``).

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is missing or in a different state.
            IllegalTransition: The edge is not in the state diagram.
        """
        return self._write(
            lambda conn: self._set_state(conn, inference_id, expected, new, **fields)
        )

    def claimable(self, route_id: Optional[str], limit: int) -> list:
        """Return ``READY`` jobs whose retry timer has elapsed, oldest first.

        Args:
            route_id: Restrict to one route, or ``None`` for every route.
            limit: The maximum number of jobs to return.

        Returns:
            A list of :class:`~image_processor.types.Job`.
        """
        now = self._clock()
        sql = (
            _schema.JOB_SELECT
            + " WHERE state = ? AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?)"
        )
        params: list = [JobState.READY.value, now]
        if route_id is not None:
            sql += " AND route_id = ?"
            params.append(route_id)
        sql += " ORDER BY created_at_ms, inference_id LIMIT ?"
        params.append(int(limit))
        return [_schema.row_to_job(r) for r in self._read(sql, tuple(params))]

    def by_state(
        self,
        states: Iterable,
        route_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> tuple:
        """Page through jobs in the given states, oldest first.

        Args:
            states: The :class:`~image_processor.types.JobState` values to include.
            route_id: Restrict to one route, or ``None`` for every route.
            cursor: The ``next_cursor`` from a previous call, or ``None`` to start.
            limit: The page size.

        Returns:
            ``(jobs, next_cursor)``; ``next_cursor`` is ``None`` on the last page.

        Raises:
            ValueError: The cursor is malformed.
        """
        wanted = [s.value for s in states]
        if not wanted:
            return [], None
        sql = _schema.JOB_SELECT + " WHERE state IN (%s)" % ", ".join("?" * len(wanted))
        params: list = list(wanted)
        if route_id is not None:
            sql += " AND route_id = ?"
            params.append(route_id)
        if cursor:
            created, _, ident = cursor.partition("|")
            if not ident:
                raise ValueError(f"malformed cursor: {cursor!r}")
            try:
                created_ms = int(created)
            except ValueError as exc:
                raise ValueError(f"malformed cursor: {cursor!r}") from exc
            sql += " AND (created_at_ms > ? OR (created_at_ms = ? AND inference_id > ?))"
            params.extend([created_ms, created_ms, ident])
        sql += " ORDER BY created_at_ms, inference_id LIMIT ?"
        params.append(int(limit) + 1)
        rows = self._read(sql, tuple(params))
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1]
            next_cursor = f"{last[_CREATED_AT_IDX]}|{last[0]}"
        return [_schema.row_to_job(r) for r in rows], next_cursor

    # -- result and outbox -----------------------------------------------------------------

    def _insert_outbox_row(self, conn, inference_id: str, row: OutboxRow) -> int:
        """Insert one outbox row inside an open transaction and return its id."""
        cursor = conn.execute(
            "INSERT INTO outbox (inference_id, topic, payload, gating, attempts, last_error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                inference_id,
                row.topic,
                row.encoded_bytes,
                1 if row.gating else 0,
                row.attempts,
                row.last_error,
            ),
        )
        return cursor.lastrowid

    def commit_result(
        self,
        inference_id: str,
        result_json: bytes,
        sidecar: Optional[tuple],
        outbox: list,
    ) -> None:
        """Commit the result, the sidecar binding, and every outbox row in one transaction.

        This is the ordered-durability step of DESIGN.md §7. The sidecar is already written and
        atomically installed by the caller; this call records its digest, the result bytes and
        their digest, and the prepared messages, then advances the job to ``RESULT_COMMITTED`` and,
        when there is at least one outbox row, on to ``PUBLISH_PENDING`` — which is what makes the
        outbox eligible. Either all of that is durable or none of it is.

        Args:
            inference_id: The job identity; it must be in ``INFERENCING``.
            result_json: The exact result body bytes to retain.
            sidecar: ``(path, sha256)`` of the installed evidence sidecar, or ``None``.
            outbox: The prepared messages, in publication order.

        Raises:
            LedgerConflict: The job is missing or not in ``INFERENCING``.
        """
        digest = hashlib.sha256(result_json).hexdigest()
        rows = list(outbox)

        def _txn(conn) -> None:
            state_row = conn.execute(
                "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
            ).fetchone()
            if state_row is None:
                raise LedgerConflict(f"no such job: {inference_id}")
            current = JobState(state_row[0])
            if current is not JobState.INFERENCING:
                raise LedgerConflict(
                    f"{inference_id} is {current.value}, expected {JobState.INFERENCING.value}"
                )
            conn.execute(
                "UPDATE jobs SET result_json = ?, result_sha256 = ?, sidecar_path = ?, "
                "sidecar_sha256 = ?, updated_at_ms = ? WHERE inference_id = ?",
                (
                    result_json,
                    digest,
                    sidecar[0] if sidecar else None,
                    sidecar[1] if sidecar else None,
                    self._clock(),
                    inference_id,
                ),
            )
            for row in rows:
                self._insert_outbox_row(conn, inference_id, row)
            self._set_state(conn, inference_id, JobState.INFERENCING, JobState.RESULT_COMMITTED)
            if rows:
                self._set_state(
                    conn, inference_id, JobState.RESULT_COMMITTED, JobState.PUBLISH_PENDING
                )

        self._write(_txn)

    def result_bytes(self, inference_id: str) -> Optional[bytes]:
        """Return the committed result body bytes, or ``None`` when nothing is committed."""
        row = self._read_one(
            "SELECT result_json FROM jobs WHERE inference_id = ?", (inference_id,)
        )
        return row[0] if row and row[0] is not None else None

    def outbox_for(self, inference_id: str) -> list:
        """Return every outbox row of one job, oldest first, published or not.

        Args:
            inference_id: The job identity.

        Returns:
            A list of :class:`OutboxRow`.
        """
        rows = self._read(
            "SELECT id, inference_id, topic, payload, attempts, gating, last_error "
            "FROM outbox WHERE inference_id = ? ORDER BY id",
            (inference_id,),
        )
        return [_row_to_outbox(r) for r in rows]

    def pending_outbox(self, limit: int) -> list:
        """Return eligible unpublished outbox rows in insertion order.

        A row is eligible only while its job is in ``PUBLISH_PENDING``: rows committed alongside
        ``RESULT_COMMITTED`` are durable but invisible, and rows of a job that has already reached
        ``PUBLISHED`` or ``PUBLISH_EXHAUSTED`` are not re-offered.

        Args:
            limit: The maximum number of rows to return.

        Returns:
            A list of :class:`OutboxRow` ordered by id.
        """
        rows = self._read(
            "SELECT o.id, o.inference_id, o.topic, o.payload, o.attempts, o.gating, o.last_error "
            "FROM outbox o JOIN jobs j ON j.inference_id = o.inference_id "
            "WHERE o.published_at_ms IS NULL AND j.state = ? ORDER BY o.id LIMIT ?",
            (JobState.PUBLISH_PENDING.value, int(limit)),
        )
        return [_row_to_outbox(r) for r in rows]

    def mark_published(self, outbox_id: int) -> None:
        """Record transport confirmation for one row, advancing the job when its gating set is done.

        Confirmed publish is positive transport acceptance (DESIGN.md §7). When every gating row
        of the job carries a confirmation, the job moves ``PUBLISH_PENDING -> PUBLISHED``, which is
        what unlocks cleanup.

        Args:
            outbox_id: The row id from :meth:`pending_outbox`.

        Raises:
            LedgerConflict: No such outbox row.
        """

        def _txn(conn) -> None:
            row = conn.execute(
                "SELECT inference_id FROM outbox WHERE id = ?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise LedgerConflict(f"no such outbox row: {outbox_id}")
            inference_id = row[0]
            conn.execute(
                "UPDATE outbox SET published_at_ms = ?, last_error = NULL WHERE id = ?",
                (self._clock(), outbox_id),
            )
            outstanding = conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE inference_id = ? AND gating = 1 "
                "AND published_at_ms IS NULL",
                (inference_id,),
            ).fetchone()[0]
            if outstanding:
                return
            state = JobState(
                conn.execute(
                    "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
                ).fetchone()[0]
            )
            if state is JobState.PUBLISH_PENDING:
                self._set_state(conn, inference_id, JobState.PUBLISH_PENDING, JobState.PUBLISHED)

        self._write(_txn)

    def mark_publish_attempt(self, outbox_id: int, error: str) -> None:
        """Count a failed publication attempt and record why it failed.

        Args:
            outbox_id: The row id from :meth:`pending_outbox`.
            error: A bounded description of the failure.

        Raises:
            LedgerConflict: No such outbox row.
        """

        def _txn(conn) -> None:
            cursor = conn.execute(
                "UPDATE outbox SET attempts = attempts + 1, last_error = ? "
                "WHERE id = ? AND published_at_ms IS NULL",
                (error, outbox_id),
            )
            if cursor.rowcount == 0:
                raise LedgerConflict(f"no unpublished outbox row: {outbox_id}")

        self._write(_txn)

    def exhaust_publish(self, inference_id: str) -> Job:
        """Give up on publication for now: ``PUBLISH_PENDING -> PUBLISH_EXHAUSTED``.

        The rows stay durable. An operator resumes with :meth:`retry_publication`.

        Args:
            inference_id: The job identity.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is not in ``PUBLISH_PENDING``.
        """
        return self.transition(
            inference_id, JobState.PUBLISH_PENDING, JobState.PUBLISH_EXHAUSTED
        )

    def retry_publication(self, inference_id: str) -> Job:
        """Resume publication: ``PUBLISH_EXHAUSTED -> PUBLISH_PENDING`` (DESIGN.md §13).

        The per-row attempt counters and errors of the still-unpublished rows are cleared, because
        an operator retry restarts the policy budget rather than continuing an exhausted one.

        Args:
            inference_id: The job identity.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is not in ``PUBLISH_EXHAUSTED``.
        """

        def _txn(conn) -> Job:
            job = self._set_state(
                conn, inference_id, JobState.PUBLISH_EXHAUSTED, JobState.PUBLISH_PENDING
            )
            conn.execute(
                "UPDATE outbox SET attempts = 0, last_error = NULL "
                "WHERE inference_id = ? AND published_at_ms IS NULL",
                (inference_id,),
            )
            return job

        return self._write(_txn)

    # -- cleanup ---------------------------------------------------------------------------

    def record_cleanup_intent(self, intent: CleanupIntent) -> Job:
        """Persist a cleanup intent before any file mutation runs (DESIGN.md §7).

        The intent records the action, the deterministic target, the source digest, and the bundle
        members, which is what lets recovery decide from observed filesystem state alone. A job on
        the publish ladder moves ``PUBLISHED -> CLEANUP_PENDING`` (or ``CLEANUP_FAILED ->
        CLEANUP_PENDING`` on a retry). A job whose lifecycle ends at a direct terminal edge —
        ``INPUT_INVALID -> QUARANTINED`` and ``PROCESSING_EXHAUSTED -> RETAINED_FAILED`` — keeps
        its state; the intent still lands first, and :meth:`complete_cleanup` takes that edge.

        Args:
            intent: The write-ahead record of the mutation about to be attempted.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is missing.
            IllegalTransition: The job is in a state cleanup never runs from.
        """

        def _txn(conn) -> Job:
            row = conn.execute(
                "SELECT state FROM jobs WHERE inference_id = ?", (intent.inference_id,)
            ).fetchone()
            if row is None:
                raise LedgerConflict(f"no such job: {intent.inference_id}")
            current = JobState(row[0])
            conn.execute(
                "INSERT INTO cleanup_intents (inference_id, action, source_path, source_sha256, "
                "target_path, members_json, observed, created_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?) "
                "ON CONFLICT (inference_id) DO UPDATE SET action = excluded.action, "
                "source_path = excluded.source_path, source_sha256 = excluded.source_sha256, "
                "target_path = excluded.target_path, members_json = excluded.members_json, "
                "observed = NULL, created_at_ms = excluded.created_at_ms",
                (
                    intent.inference_id,
                    intent.action.value,
                    intent.source_path,
                    intent.source_sha256,
                    intent.target_path,
                    json.dumps(list(intent.members)),
                    self._clock(),
                ),
            )
            if current in (JobState.PUBLISHED, JobState.CLEANUP_FAILED):
                return self._set_state(
                    conn, intent.inference_id, current, JobState.CLEANUP_PENDING
                )
            if current in (
                JobState.INPUT_INVALID,
                JobState.PROCESSING_EXHAUSTED,
                JobState.CLEANUP_PENDING,
            ):
                return _schema.row_to_job(
                    conn.execute(
                        _schema.JOB_SELECT + " WHERE inference_id = ?", (intent.inference_id,)
                    ).fetchone()
                )
            raise IllegalTransition(f"cleanup does not run from {current.value}")

        return self._write(_txn)

    def cleanup_intent(self, inference_id: str) -> Optional[CleanupIntent]:
        """Return the stored cleanup intent for a job, or ``None``."""
        row = self._read_one(
            "SELECT inference_id, action, source_path, source_sha256, target_path, members_json "
            "FROM cleanup_intents WHERE inference_id = ?",
            (inference_id,),
        )
        return _schema.row_to_intent(row) if row else None

    def cleanup_observed(self, inference_id: str) -> Optional[str]:
        """Return the observed outcome recorded against a cleanup intent, or ``None``."""
        row = self._read_one(
            "SELECT observed FROM cleanup_intents WHERE inference_id = ?", (inference_id,)
        )
        return row[0] if row else None

    def complete_cleanup(self, inference_id: str, observed: str) -> Job:
        """Record a successful cleanup and take the job's completing edge.

        The edge depends on where the job entered cleanup: ``CLEANUP_PENDING -> COMPLETED`` on the
        publish ladder, ``INPUT_INVALID -> QUARANTINED`` for a rejected input, and
        ``PROCESSING_EXHAUSTED -> RETAINED_FAILED`` for an exhausted one.

        Args:
            inference_id: The job identity.
            observed: What the filesystem showed, retained against the intent as evidence.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is missing or has no open intent.
            IllegalTransition: The job is not in a state cleanup completes from.
        """

        def _txn(conn) -> Job:
            row = conn.execute(
                "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
            ).fetchone()
            if row is None:
                raise LedgerConflict(f"no such job: {inference_id}")
            current = JobState(row[0])
            target = _COMPLETING_EDGES.get(current)
            if target is None:
                raise IllegalTransition(f"cleanup does not complete from {current.value}")
            conn.execute(
                "UPDATE cleanup_intents SET observed = ? WHERE inference_id = ?",
                (observed, inference_id),
            )
            return self._set_state(conn, inference_id, current, target)

        return self._write(_txn)

    def fail_cleanup(self, inference_id: str, error: str) -> Job:
        """Record a cleanup failure. Cleanup failure is never success (DESIGN.md §7).

        A job on the publish ladder moves ``CLEANUP_PENDING -> CLEANUP_FAILED``. A job on a direct
        terminal edge keeps its non-terminal state with the error attached, so it stays visible to
        :meth:`pending_cleanup` and is retried rather than being recorded as done.

        Args:
            inference_id: The job identity.
            error: A bounded description of the failure.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is missing.
        """

        def _txn(conn) -> Job:
            row = conn.execute(
                "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
            ).fetchone()
            if row is None:
                raise LedgerConflict(f"no such job: {inference_id}")
            current = JobState(row[0])
            conn.execute(
                "UPDATE cleanup_intents SET observed = NULL WHERE inference_id = ?",
                (inference_id,),
            )
            target = (
                JobState.CLEANUP_FAILED if current is JobState.CLEANUP_PENDING else current
            )
            return self._set_state(conn, inference_id, current, target, last_error=error)

        return self._write(_txn)

    def pending_cleanup(self, limit: int) -> list:
        """Return cleanup intents that have not been observed to succeed, oldest first.

        Args:
            limit: The maximum number of intents to return.

        Returns:
            A list of :class:`~image_processor.types.CleanupIntent`.
        """
        rows = self._read(
            "SELECT inference_id, action, source_path, source_sha256, target_path, members_json "
            "FROM cleanup_intents WHERE observed IS NULL ORDER BY created_at_ms, inference_id "
            "LIMIT ?",
            (int(limit),),
        )
        return [_schema.row_to_intent(r) for r in rows]

    # -- model generations -----------------------------------------------------------------

    def set_route_generation(
        self, route_id: str, desired: str, active: Optional[str]
    ) -> None:
        """Persist a route's desired and active model generation (DESIGN.md §9).

        Args:
            route_id: The route.
            desired: The generation configuration asks for.
            active: The generation currently serving, or ``None`` while the route is staging.
        """

        def _txn(conn) -> None:
            conn.execute(
                "INSERT INTO route_generations (route_id, desired, active, updated_at_ms) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (route_id) DO UPDATE SET "
                "desired = excluded.desired, active = excluded.active, "
                "updated_at_ms = excluded.updated_at_ms",
                (route_id, desired, active, self._clock()),
            )

        self._write(_txn)

    def route_generation(self, route_id: str) -> tuple:
        """Return ``(desired, active)`` for a route; ``(None, None)`` when nothing is recorded."""
        row = self._read_one(
            "SELECT desired, active FROM route_generations WHERE route_id = ?", (route_id,)
        )
        return (row[0], row[1]) if row else (None, None)

    # -- key/value -------------------------------------------------------------------------

    def kv_get(self, key: str) -> Optional[str]:
        """Return a small durable value, or ``None``.

        Used for reconciliation watermarks such as the camera ``sb/capture-status`` cursor.

        Args:
            key: The value's key.

        Returns:
            The stored string, or ``None``.
        """
        row = self._read_one("SELECT value FROM kv WHERE key = ?", (key,))
        return row[0] if row else None

    def kv_set(self, key: str, value: Optional[str]) -> None:
        """Store a small durable value.

        Args:
            key: The value's key.
            value: The string to store; ``None`` clears it to SQL NULL.
        """

        def _txn(conn) -> None:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

        self._write(_txn)

    # -- recovery --------------------------------------------------------------------------

    def requeue_for_reinference(self, inference_id: str, reason: str) -> Job:
        """Return a committed job to re-inference, dropping its result and outbox rows.

        DESIGN.md §7: a committed record whose evidence sidecar is absent returns to
        re-inference. The result bytes, the result digest, the sidecar binding, and every
        unpublished outbox row go with it, so the retry prepares a fresh message rather than
        publishing a result whose evidence no longer exists.

        Args:
            inference_id: The job identity.
            reason: Why the record was invalidated; retained as the job's last error.

        Returns:
            The job as it now stands.

        Raises:
            LedgerConflict: The job is missing.
            IllegalTransition: The job holds no invalidatable committed result.
        """

        def _txn(conn) -> Job:
            row = conn.execute(
                "SELECT state FROM jobs WHERE inference_id = ?", (inference_id,)
            ).fetchone()
            if row is None:
                raise LedgerConflict(f"no such job: {inference_id}")
            current = JobState(row[0])
            if (current, JobState.READY) not in RECOVERY_EDGES:
                raise IllegalTransition(f"{current.value} holds no committed result to invalidate")
            conn.execute("DELETE FROM outbox WHERE inference_id = ?", (inference_id,))
            conn.execute(
                "UPDATE jobs SET result_json = NULL, result_sha256 = NULL, sidecar_path = NULL, "
                "sidecar_sha256 = NULL WHERE inference_id = ?",
                (inference_id,),
            )
            return self._force_state(conn, inference_id, current, JobState.READY, last_error=reason)

        return self._write(_txn)

    def _force_state(self, conn, inference_id: str, current: JobState, new: JobState, **fields) -> Job:
        """Take a recovery edge, which the forward lifecycle table does not contain.

        Args:
            conn: The write connection, already in a transaction.
            inference_id: The job identity.
            current: The state the job is in.
            new: The state to restart at.
            **fields: Columns from :data:`~image_processor.ledger.schema.MUTABLE_JOB_FIELDS`.

        Returns:
            The job as it now stands.

        Raises:
            IllegalTransition: The edge is not a declared recovery edge.
        """
        if (current, new) not in RECOVERY_EDGES:
            raise IllegalTransition(f"{current.value} -> {new.value} is not a recovery edge")
        unknown = set(fields) - _schema.MUTABLE_JOB_FIELDS
        if unknown:
            raise ValueError(f"not settable on a transition: {sorted(unknown)}")
        assignments = ["state = ?", "updated_at_ms = ?"]
        params: list = [new.value, self._clock()]
        for key in sorted(fields):
            assignments.append(f"{key} = ?")
            params.append(fields[key])
        params.append(inference_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE inference_id = ?", tuple(params)
        )
        return _schema.row_to_job(
            conn.execute(
                _schema.JOB_SELECT + " WHERE inference_id = ?", (inference_id,)
            ).fetchone()
        )

    def recover(self) -> RecoveryReport:
        """Reconcile the ledger after a restart and report what moved (DESIGN.md §7).

        Jobs that were mid-flight restart at ``READY`` with their attempt count intact; a
        ``RETRY_WAIT`` job whose backoff has elapsed becomes claimable again; committed results and
        their outbox rows are left alone for the publisher; and open cleanup intents are returned
        so the caller can reconcile them against observed filesystem state.

        Returns:
            The :class:`~image_processor.ledger.recovery.RecoveryReport` for this pass.
        """
        now = self._clock()

        def _txn(conn) -> RecoveryReport:
            rows = conn.execute(
                "SELECT inference_id, state, next_attempt_at_ms FROM jobs "
                "WHERE state IN (?, ?, ?, ?) ORDER BY created_at_ms, inference_id",
                (
                    JobState.INFERENCING.value,
                    JobState.CLAIMED.value,
                    JobState.WAITING_MODEL.value,
                    JobState.RETRY_WAIT.value,
                ),
            ).fetchall()
            moves = plan_recovery(rows, now)
            for move in moves:
                self._force_state(conn, move.inference_id, move.current, move.new)
            pending_cleanup = [
                _schema.row_to_intent(r)
                for r in conn.execute(
                    "SELECT inference_id, action, source_path, source_sha256, target_path, "
                    "members_json FROM cleanup_intents WHERE observed IS NULL "
                    "ORDER BY created_at_ms, inference_id"
                ).fetchall()
            ]
            publish_pending = [
                r[0]
                for r in conn.execute(
                    "SELECT inference_id FROM jobs WHERE state = ? "
                    "ORDER BY created_at_ms, inference_id",
                    (JobState.PUBLISH_PENDING.value,),
                ).fetchall()
            ]
            sidecars = [
                SidecarRecord(r[0], JobState(r[1]), r[2], r[3])
                for r in conn.execute(
                    "SELECT inference_id, state, sidecar_path, sidecar_sha256 FROM jobs "
                    "WHERE sidecar_path IS NOT NULL AND state IN (?, ?, ?) "
                    "ORDER BY created_at_ms, inference_id",
                    (
                        JobState.RESULT_COMMITTED.value,
                        JobState.PUBLISH_PENDING.value,
                        JobState.PUBLISH_EXHAUSTED.value,
                    ),
                ).fetchall()
            ]
            return build_report(moves, pending_cleanup, publish_pending, sidecars)

        report = self._write(_txn)
        log.info(
            "ledger recovery moved %d job(s): %s", report.total, report.transitions or "nothing"
        )
        return report


#: The completing edge each cleanup-bearing state takes when its mutation succeeds.
_COMPLETING_EDGES = {
    JobState.CLEANUP_PENDING: JobState.COMPLETED,
    JobState.INPUT_INVALID: JobState.QUARANTINED,
    JobState.PROCESSING_EXHAUSTED: JobState.RETAINED_FAILED,
}


def _row_to_outbox(row) -> OutboxRow:
    """Build an :class:`OutboxRow` from an ``(id, inference_id, topic, payload, attempts, gating,
    last_error)`` row."""
    return OutboxRow(
        id=row[0],
        inference_id=row[1],
        topic=row[2],
        encoded_bytes=bytes(row[3]),
        attempts=row[4],
        gating=bool(row[5]),
        last_error=row[6],
    )
