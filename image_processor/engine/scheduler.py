"""Model-aware scheduling over the executor cells (DESIGN.md §10.3, §7, LLD §6).

The scheduler is the parent's decision-making half: which model is resident, which job runs next,
and what a failure means for the durable record. It holds work in one lane per model generation,
because that is the unit residency is keyed by -- routes bound to the same digest share one
session, and a lane is what a load is amortized over.

One pass of :meth:`Scheduler.run_once` is deterministic and does the whole cycle:

1. Retry timers that have come due move ``RETRY_WAIT -> READY`` and rejoin their lane.
2. Lanes are ranked: work whose model is already resident on the cell first, then by queue age
   weighted by route priority, so a cold lane cannot starve behind a hot one.
3. The chosen lane's model is made resident, which may mean admitting it against the device budget
   and evicting the cheapest sessions to make room. A cold digest is loaded once per pass no
   matter how many lanes want it, and at most ``loadConcurrencyPerGpu`` loads run per device.
4. One job per cell is dispatched, which is what "one in-flight inference per cell" means.
5. Every reply is applied in cell order: the ledger edge, the retry arithmetic, and the callback
   the app registered.

The ledger is the authority for what happened, not this object. The scheduler owns
``READY -> CLAIMED -> WAITING_MODEL -> INFERENCING`` and the failure edges out of those, and stops
there: a successful result stays ``INFERENCING`` until the app commits it with the sidecar and the
outbox rows in one transaction (DESIGN.md §7). Nothing here drops an accepted job. When a model
cannot be made resident, the job waits in its lane and the pass reports the deferral; back-pressure
is latency and a degraded route, never a discarded image.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from image_processor.engine.cell import CellDead, CellError, CellTimeout
from image_processor.engine.protocol import (
    CONTAMINATING,
    PERMANENT,
    TRANSIENT,
    Infer,
    LoadFailed,
    LoadModel,
    Loaded,
    Stats,
    Unload,
)
from image_processor.engine.residency import ResidencyStats, setting
from image_processor.engine.supervisor import SupervisorError
from image_processor.ledger.ledger import IllegalTransition, LedgerConflict
from image_processor.types import InferenceResult, Job, JobState, Timings

logger = logging.getLogger(__name__)

#: Attempts one job gets before ``PROCESSING_EXHAUSTED``.
DEFAULT_MAX_ATTEMPTS = 5

#: First retry delay, in seconds. Doubles per attempt.
DEFAULT_RETRY_BACKOFF_SECS = 2.0

#: Ceiling on the retry delay, in seconds.
DEFAULT_MAX_RETRY_BACKOFF_SECS = 300.0

#: How long a dispatched inference may take before the cell is considered wedged.
DEFAULT_INFER_TIMEOUT_S = 300.0

#: How long a model load may take before the cell is considered wedged.
DEFAULT_LOAD_TIMEOUT_S = 600.0

#: How long an unload or a stats read may take.
DEFAULT_CONTROL_TIMEOUT_S = 60.0

#: Fraction of a session's measured footprint an unload must return before the cell is trusted to
#: keep serving. Below it, DESIGN.md §10.4 recycles the cell.
DEFAULT_RECLAIM_RATIO = 0.5

#: How long :meth:`Scheduler.run_forever` waits between passes that dispatched nothing.
DEFAULT_POLL_INTERVAL_S = 0.05


def _now_ms() -> int:
    """Return the current wall clock in milliseconds."""
    return int(time.time() * 1000)


@dataclass
class Queued:
    """One job waiting in a lane.

    Attributes:
        job: The durable job, as the ledger last returned it.
        priority: The route priority, from configuration.
        first_seen_ms: When the scheduler first saw this job. Retries keep it, so a job that has
            been round the retry loop keeps the age that protects it from starvation.
        queued_at_ms: When it last entered a lane. The queue time stamped on the result.
    """

    job: Job
    priority: int = 100
    first_seen_ms: int = 0
    queued_at_ms: int = 0


@dataclass
class Lane:
    """The queue for one model generation.

    Attributes:
        digest: The bundle digest every job in this lane is pinned to.
        jobs: The queued jobs, oldest first.
        burst_remaining: Jobs left in the drain burst a load was justified by. While it is above
            zero the session is leased, which is what stops a freshly loaded model from being
            evicted before it has done the work it was loaded for.
        load_failures: Consecutive transient load failures, which set the load backoff.
        blocked_until_ms: When this lane may try to load again after a transient failure.
        blocked_reason: Why the lane is permanently blocked, or ``None``.
    """

    digest: str
    jobs: deque = field(default_factory=deque)
    burst_remaining: int = 0
    load_failures: int = 0
    blocked_until_ms: int = 0
    blocked_reason: Optional[str] = None

    @property
    def priority(self) -> int:
        """The highest route priority among the queued jobs."""
        return max((entry.priority for entry in self.jobs), default=0)

    def oldest_ms(self, now_ms: int) -> int:
        """Return the age of the oldest queued job in milliseconds.

        Args:
            now_ms: The current wall clock in milliseconds.

        Returns:
            The age, or ``0`` when the lane is empty.
        """
        if not self.jobs:
            return 0
        return max(now_ms - self.jobs[0].first_seen_ms, 0)


class Scheduler:
    """Chooses what runs next, keeps models resident, and records what happened.

    Args:
        ledger: The durable job ledger. Every state change goes through it.
        supervisor: The executor supervisor. Cells come from it, and recycles go to it.
        cache: The content-addressed bundle cache, or anything with the same ``get(digest)``
            returning a :class:`~image_processor.types.CachedBundle`.
        policy: The :class:`~image_processor.engine.residency.ResidencyPolicy`.
        cfg: The ``scheduler`` configuration block (DESIGN.md §11): ``maxBatchLatencyMs``,
            ``hotTtlSecs``, ``minResidencySecs``, plus the retry budget ``maxAttempts``,
            ``retryBackoffSecs``, and ``maxRetryBackoffSecs``. Any object or mapping carrying those
            names, in either spelling.
        on_result: Called with ``(job, result)`` for a successful inference and for a terminal
            failure. It is the app's result pipeline (WP6).
        route_priorities: Route id to priority, from ``instances[].priority``. Used when
            :meth:`submit` is not given one.
        clock: Returns the current wall clock in milliseconds. Injected by tests.
        rng: The jitter source. Injected by tests to make backoff reproducible.
        max_attempts: Overrides the configured retry budget.
        retry_backoff_secs: Overrides the configured first retry delay.
        max_retry_backoff_secs: Overrides the configured retry delay ceiling.
        load_concurrency_per_gpu: Overrides ``runtime.loadConcurrencyPerGpu``, which the
            supervisor read.
        infer_timeout_s: Per-inference deadline.
        load_timeout_s: Per-load deadline.
        control_timeout_s: Deadline for an unload or a stats read.
        reclaim_ratio: How much of a session's footprint an unload must return before the cell is
            trusted to keep serving.
        poll_interval_s: How long :meth:`run_forever` waits after a pass that dispatched nothing.
    """

    def __init__(
        self,
        ledger,
        supervisor,
        cache,
        policy,
        cfg=None,
        on_result=None,
        route_priorities=None,
        clock=_now_ms,
        rng=None,
        max_attempts=None,
        retry_backoff_secs=None,
        max_retry_backoff_secs=None,
        load_concurrency_per_gpu=None,
        infer_timeout_s: float = DEFAULT_INFER_TIMEOUT_S,
        load_timeout_s: float = DEFAULT_LOAD_TIMEOUT_S,
        control_timeout_s: float = DEFAULT_CONTROL_TIMEOUT_S,
        reclaim_ratio: float = DEFAULT_RECLAIM_RATIO,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        """Build the scheduler over its collaborators."""
        self.ledger = ledger
        self.supervisor = supervisor
        self.cache = cache
        self.policy = policy
        self.cfg = cfg
        self._on_result = on_result
        self.route_priorities = dict(route_priorities or {})
        self._clock = clock
        self._rng = rng or random.Random(0)

        self.max_attempts = int(
            max_attempts
            if max_attempts is not None
            else setting(cfg, "maxAttempts", default=DEFAULT_MAX_ATTEMPTS)
        )
        self.retry_backoff_secs = float(
            retry_backoff_secs
            if retry_backoff_secs is not None
            else setting(cfg, "retryBackoffSecs", default=DEFAULT_RETRY_BACKOFF_SECS)
        )
        self.max_retry_backoff_secs = float(
            max_retry_backoff_secs
            if max_retry_backoff_secs is not None
            else setting(cfg, "maxRetryBackoffSecs", default=DEFAULT_MAX_RETRY_BACKOFF_SECS)
        )
        self.max_batch_latency_ms = float(setting(cfg, "maxBatchLatencyMs", default=20) or 0)
        self.hot_ttl_secs = float(
            setting(cfg, "hotTtlSecs", default=getattr(policy, "hot_ttl_secs", 120.0))
        )
        self.load_concurrency_per_gpu = int(
            load_concurrency_per_gpu
            if load_concurrency_per_gpu is not None
            else getattr(supervisor, "load_concurrency_per_gpu", 1)
        )
        self.infer_timeout_s = float(infer_timeout_s)
        self.load_timeout_s = float(load_timeout_s)
        self.control_timeout_s = float(control_timeout_s)
        self.reclaim_ratio = float(reclaim_ratio)
        self.poll_interval_s = float(poll_interval_s)

        self._lock = threading.RLock()
        self._inbox = deque()
        self._inbox_lock = threading.Lock()
        self._lanes = {}
        self._resident = {}
        self._retry = []
        self._paused = False
        self._stop = threading.Event()
        self._thread = None
        self.counters = {
            "dispatched": 0,
            "succeeded": 0,
            "retried": 0,
            "exhausted": 0,
            "blocked": 0,
            "loads": 0,
            "evictions": 0,
            "deferred": 0,
            "recycles": 0,
            "callbackFailures": 0,
        }

    # -- queue management ------------------------------------------------------------------

    def submit(self, job: Job, priority=None) -> None:
        """Take one admitted job into its lane.

        The job joins its lane at the start of the next pass. Submission itself takes only the
        inbox lock, so a source thread is never held up by a model load in flight.

        Args:
            job: The job, in ``READY``, ``CLAIMED``, ``WAITING_MODEL``, or ``RETRY_WAIT``. The
                advanced states are what recovery hands back after a restart.
            priority: The route priority, or ``None`` to read it from ``route_priorities``.

        Raises:
            ValueError: The job is in a state the scheduler cannot act on. It is refused loudly
                rather than dropped quietly.
        """
        schedulable = {
            JobState.READY,
            JobState.CLAIMED,
            JobState.WAITING_MODEL,
            JobState.RETRY_WAIT,
        }
        if job.state not in schedulable:
            raise ValueError(f"a {job.state.value} job cannot be scheduled: {job.inference_id}")
        now = self._clock()
        entry = Queued(
            job=job,
            priority=int(
                priority if priority is not None else self.route_priorities.get(job.route_id, 100)
            ),
            first_seen_ms=now,
            queued_at_ms=now,
        )
        with self._inbox_lock:
            self._inbox.append(entry)

    def _drain_inbox(self, now_ms: int) -> int:
        """Move submitted jobs into their lanes at the start of a pass.

        Submission lands in an inbox rather than straight in a lane so that a source thread never
        waits behind a pass: a cold model load takes seconds, and discovery must not stall for it.

        Args:
            now_ms: The current wall clock in milliseconds.

        Returns:
            The number of jobs taken in.
        """
        with self._inbox_lock:
            if not self._inbox:
                return 0
            entries = list(self._inbox)
            self._inbox.clear()
        for entry in entries:
            if entry.job.state is JobState.RETRY_WAIT:
                self._retry.append((int(entry.job.next_attempt_at_ms or now_ms), entry))
            else:
                self._lane(entry.job.model.digest).jobs.append(entry)
        return len(entries)

    def _lane(self, digest: str) -> Lane:
        """Return the lane for a digest, creating it on first use.

        Args:
            digest: The bundle digest.

        Returns:
            The lane.
        """
        lane = self._lanes.get(digest)
        if lane is None:
            lane = Lane(digest=digest)
            self._lanes[digest] = lane
        return lane

    def _requeue(self, lane: Lane, entry: Queued, front: bool = True) -> None:
        """Put a job back in its lane without changing its attempt.

        Args:
            lane: The lane it came from.
            entry: The queued job.
            front: Whether it goes back at the head, which is what a recycled cell owes the job it
                was running.
        """
        if front:
            lane.jobs.appendleft(entry)
        else:
            lane.jobs.append(entry)

    def pause(self) -> None:
        """Stop claiming new work. Jobs already in flight finish (DESIGN.md §13)."""
        with self._lock:
            self._paused = True
        logger.info("scheduler paused")

    def resume(self) -> None:
        """Start claiming work again."""
        with self._lock:
            self._paused = False
        logger.info("scheduler resumed")

    @property
    def paused(self) -> bool:
        """Whether the scheduler is currently refusing to claim new work."""
        return self._paused

    def queued(self) -> int:
        """Return how many jobs are waiting across the inbox, the lanes, and the retry timers."""
        with self._lock:
            waiting = sum(len(lane.jobs) for lane in self._lanes.values()) + len(self._retry)
        with self._inbox_lock:
            return waiting + len(self._inbox)

    # -- durable edges ---------------------------------------------------------------------

    def _transition(self, entry: Queued, expected: JobState, new: JobState, **fields):
        """Move one job along a ledger edge and keep the queued copy current.

        Args:
            entry: The queued job.
            expected: The state the job must be in.
            new: The state to move to.
            **fields: Column updates the ledger accepts.

        Returns:
            The updated job, or ``None`` when the ledger refused the edge. A refusal means
            something else already moved the job -- an operator command, recovery -- so the
            scheduler drops its claim on it rather than fighting for it.
        """
        try:
            job = self.ledger.transition(entry.job.inference_id, expected, new, **fields)
        except (LedgerConflict, IllegalTransition) as exc:
            logger.warning(
                "job %s did not move %s -> %s: %s",
                entry.job.inference_id,
                expected.value,
                new.value,
                exc,
            )
            return None
        entry.job = job
        return job

    def _claim(self, entry: Queued) -> bool:
        """Advance a job to ``WAITING_MODEL``, through whatever edges it still owes.

        A job that is already ``INFERENCING`` is one a recycled cell handed back, and it needs no
        edge at all: durably it never stopped being inferred, which is exactly what lets it run
        again at the same attempt.

        Args:
            entry: The queued job.

        Returns:
            ``True`` when the job may be given a session, ``False`` when the ledger refused.
        """
        state = entry.job.state
        if state is JobState.READY:
            if self._transition(entry, JobState.READY, JobState.CLAIMED) is None:
                return False
            state = JobState.CLAIMED
        if state is JobState.CLAIMED:
            if self._transition(entry, JobState.CLAIMED, JobState.WAITING_MODEL) is None:
                return False
            state = JobState.WAITING_MODEL
        return state in (JobState.WAITING_MODEL, JobState.INFERENCING)

    def _backoff_ms(self, attempts: int) -> int:
        """Return the retry delay for an attempt, with jitter.

        The delay doubles per attempt up to the configured ceiling, and half of it is jittered, so
        a burst of jobs that failed together does not come back together. The jittered half never
        collapses the delay to zero, which would turn a retry storm into a spin.

        Args:
            attempts: The attempt number that just failed.

        Returns:
            The delay in milliseconds.
        """
        delay = self.retry_backoff_secs * (2 ** max(int(attempts) - 1, 0))
        delay = min(delay, self.max_retry_backoff_secs)
        half = delay / 2.0
        return int((half + self._rng.uniform(0.0, half)) * 1000.0)

    def _promote_retries(self, now_ms: int) -> int:
        """Move retry timers that have come due back to ``READY``.

        Args:
            now_ms: The current wall clock in milliseconds.

        Returns:
            The number of jobs returned to their lanes.
        """
        if not self._retry:
            return 0
        due = [item for item in self._retry if item[0] <= now_ms]
        if not due:
            return 0
        self._retry = [item for item in self._retry if item[0] > now_ms]
        promoted = 0
        for _, entry in due:
            if self._transition(entry, JobState.RETRY_WAIT, JobState.READY) is None:
                continue
            entry.queued_at_ms = now_ms
            self._lane(entry.job.model.digest).jobs.append(entry)
            promoted += 1
        return promoted

    def _retry_later(self, entry: Queued, error: str, attempts=None, delay_ms=None) -> None:
        """Send one job to ``RETRY_WAIT`` with its backoff.

        Args:
            entry: The queued job.
            error: The bounded error to record.
            attempts: The attempt count to store, or ``None`` to keep the job's own.
            delay_ms: The delay to wait, or ``None`` to derive it from the attempt count. A model
                load that failed transiently passes the lane's own backoff, because a load failure
                is a property of the model generation rather than of this job's attempt.
        """
        count = entry.job.attempts if attempts is None else int(attempts)
        wait = self._backoff_ms(count) if delay_ms is None else int(delay_ms)
        due = self._clock() + wait
        moved = self._transition(
            entry,
            entry.job.state,
            JobState.RETRY_WAIT,
            attempts=count,
            next_attempt_at_ms=due,
            last_error=error,
        )
        if moved is None:
            return
        self._retry.append((due, entry))
        self.counters["retried"] += 1

    def _exhaust(self, entry: Queued, error: str, attempts=None):
        """Send one job to ``PROCESSING_EXHAUSTED``.

        Args:
            entry: The queued job.
            error: The bounded error to record.
            attempts: The attempt count to store, or ``None`` to keep the job's own.

        Returns:
            The updated job, or ``None`` when the ledger refused the edge.
        """
        count = entry.job.attempts if attempts is None else int(attempts)
        job = self._transition(
            entry,
            entry.job.state,
            JobState.PROCESSING_EXHAUSTED,
            attempts=count,
            last_error=error,
        )
        if job is not None:
            self.counters["exhausted"] += 1
        return job

    def _block(self, entry: Queued, error: str):
        """Give one job the terminal verdict its state allows.

        A job waiting on a model that cannot load is ``BLOCKED_CONFIGURATION``. A job a recycled
        cell handed back is durably still ``INFERENCING``, and the diagram gives that state no
        edge to ``BLOCKED_CONFIGURATION``, so it ends at ``PROCESSING_EXHAUSTED`` instead -- the
        same terminal meaning by the edge DESIGN.md §7 actually declares.

        Args:
            entry: The queued job.
            error: The bounded error to record.

        Returns:
            The updated job, or ``None`` when the ledger refused the edge.
        """
        if entry.job.state is JobState.INFERENCING:
            return self._exhaust(entry, error)
        job = self._transition(
            entry, JobState.WAITING_MODEL, JobState.BLOCKED_CONFIGURATION, last_error=error
        )
        if job is not None:
            self.counters["blocked"] += 1
        return job

    def _notify(self, job: Job, result) -> None:
        """Hand one answer to the app.

        Args:
            job: The job as the ledger last returned it.
            result: The :class:`~image_processor.types.InferenceResult`.
        """
        try:
            self.on_result(job, result)
        except Exception as exc:
            self.counters["callbackFailures"] += 1
            logger.exception(
                "the result callback failed for %s, leaving the job for recovery: %s",
                job.inference_id,
                exc,
            )

    def on_result(self, job: Job, result) -> None:
        """Deliver one answer to the app's result pipeline (LLD §6).

        It is called for a successful inference, where the job is still ``INFERENCING`` and the
        app commits the result, the sidecar, and the outbox rows in one transaction, and for a
        terminal failure, where the job is already ``PROCESSING_EXHAUSTED`` or
        ``BLOCKED_CONFIGURATION`` and the app publishes the failed result and runs the failure
        completion. A retry is not an answer, so it is not delivered.

        Args:
            job: The job as the ledger last returned it.
            result: The :class:`~image_processor.types.InferenceResult`.
        """
        if self._on_result is not None:
            self._on_result(job, result)

    # -- cells and residency ---------------------------------------------------------------

    def _resident_for(self, cell) -> dict:
        """Return the residency record for one cell, creating it on first use.

        Args:
            cell: The cell handle.

        Returns:
            Digest to :class:`~image_processor.engine.residency.ResidencyStats`.
        """
        return self._resident.setdefault(cell.cell_id, {})

    def _device_resident_mib(self, cell) -> int:
        """Return what this component holds on the device this cell runs on.

        With more than one cell per GPU the budget is a property of the device, not of the cell,
        so every cell on that device counts toward it.

        Args:
            cell: The cell handle.

        Returns:
            The total measured footprint in MiB.
        """
        total = 0
        for other in self.supervisor.cells():
            if other.device != cell.device:
                continue
            for stats in self._resident.get(other.cell_id, {}).values():
                total += stats.footprint_mib
        return total

    def _leases(self, cell) -> set:
        """Return the digests that may not be evicted from one cell.

        A lease is held by a freshly loaded model that has not yet drained the burst it was loaded
        for (DESIGN.md §10.2, §10.3). Work in flight holds one too, and gets it from the shape of a
        pass rather than from this set: a cell evicts only while it is choosing its own next job,
        and it has at most one job in flight, so the session running that job is never a candidate.

        Queued work does not lease a session by itself. It makes it expensive instead, through the
        retained value, which is what lets a starving lane eventually take the memory it needs
        rather than waiting behind a busy model forever.

        Args:
            cell: The cell handle.

        Returns:
            The leased digests.
        """
        return {digest for digest, lane in self._lanes.items() if lane.burst_remaining > 0}

    def _refresh_stats(self, cell, now_ms: int) -> dict:
        """Bring one cell's residency records up to date before they are priced.

        Args:
            cell: The cell handle.
            now_ms: The current wall clock in milliseconds.

        Returns:
            The refreshed records.
        """
        records = self._resident_for(cell)
        for digest, stats in records.items():
            lane = self._lanes.get(digest)
            stats.queued_jobs = len(lane.jobs) if lane else 0
            stats.priority = lane.priority if lane and lane.jobs else 0
            stats.leased = bool(lane and lane.burst_remaining > 0)
            stats.load_ms = self.policy.measured_load_ms(digest) or stats.load_ms
        return records

    def _recycle(self, cell, reason: str):
        """Ask the supervisor to recycle one cell and forget what it held.

        Args:
            cell: The cell handle.
            reason: Why.

        Returns:
            The request that was in flight, or ``None``.
        """
        self._resident.pop(cell.cell_id, None)
        self.counters["recycles"] += 1
        try:
            return self.supervisor.recycle(cell, reason)
        except SupervisorError as exc:
            logger.error("cell %s could not be recycled: %s", cell.cell_id, exc.message)
            return None

    def _cell_call(self, cell, message, timeout_s: float):
        """Send one control request to a cell, recycling the cell when it cannot answer.

        Args:
            cell: The cell handle.
            message: The request.
            timeout_s: The deadline in seconds.

        Returns:
            The reply, or ``None`` when the cell died or missed its deadline. The caller leaves
            the work queued; the next pass finds a fresh cell.
        """
        try:
            return cell.call(message, timeout_s)
        except (CellDead, CellTimeout) as exc:
            self._recycle(cell, f"a control request failed: {exc}")
            return None
        except CellError as exc:
            logger.warning("cell %s refused a control request: %s", cell.cell_id, exc)
            return None

    def _memory(self, cell, state) -> bool:
        """Read one cell's device memory once per pass, and reconcile what it holds.

        The reading is what admission is judged against, and the resident list that comes with it
        is the cell's own truth: a cell that restarted between passes holds nothing, and the
        scheduler drops the records it was keeping for it rather than dispatching work to a
        session that is gone.

        Args:
            cell: The cell handle.
            state: The pass state.

        Returns:
            ``True`` when the reading is available, ``False`` when the cell could not answer and
            has been recycled.
        """
        if cell.cell_id in state.free:
            return True
        reply = self._cell_call(cell, Stats(), self.control_timeout_s)
        if reply is None:
            return False
        state.free[cell.cell_id] = int(reply.device_free_mib or 0)
        state.total[cell.cell_id] = int(reply.device_total_mib or 0)
        records = self._resident_for(cell)
        for digest in list(records):
            if digest not in (reply.resident or ()):
                logger.info(
                    "cell %s no longer holds %s; forgetting it", cell.cell_id, digest
                )
                records.pop(digest, None)
        return True

    def _unload(self, cell, digest: str, state) -> bool:
        """Evict one session and account for what came back (DESIGN.md §10.4).

        Args:
            cell: The cell holding it.
            digest: The bundle digest to evict.
            state: The pass state.

        Returns:
            ``True`` when the cell is still usable afterwards.
        """
        records = self._resident_for(cell)
        stats = records.get(digest)
        reply = self._cell_call(cell, Unload(digest), self.control_timeout_s)
        if reply is None:
            return False
        records.pop(digest, None)
        self.counters["evictions"] += 1
        state.free[cell.cell_id] = state.free.get(cell.cell_id, 0) + int(reply.freed_mib or 0)
        expected = int(reply.expected_mib or (stats.footprint_mib if stats else 0))
        if (
            expected > 0
            and state.total.get(cell.cell_id, 0) > 0
            and reply.freed_mib < expected * self.reclaim_ratio
        ):
            self._recycle(
                cell,
                f"unloading {digest} returned {reply.freed_mib} MiB of {expected} MiB",
            )
            return False
        return True

    # WP6 -- the `evict-model` verb (DESIGN.md §13) releases an idle session on operator
    # request. Residency is this object's business, so the refusal rule that protects a draining
    # burst lives here rather than in the command handler.
    def evict(self, digest: str) -> dict:
        """Release one resident model generation, refusing a leased one.

        Args:
            digest: The bundle digest to evict.

        Returns:
            ``{"evicted": bool, "digest": str, "cells": [...], "reason": str}``. A generation
            still leased by a draining burst is refused rather than taken away from it.
        """
        with self._lock:
            state = _Pass(now_ms=self._clock())
            holders = []
            for cell in self.supervisor.cells():
                if digest not in self._resident.get(cell.cell_id, {}):
                    continue
                if digest in self._leases(cell):
                    return {
                        "evicted": False,
                        "digest": digest,
                        "cells": [],
                        "reason": "the generation is leased by work that has not drained",
                    }
                holders.append(cell)
            if not holders:
                return {
                    "evicted": False,
                    "digest": digest,
                    "cells": [],
                    "reason": "the generation is not resident",
                }
            released = []
            for cell in holders:
                # Read the device before unloading, so the reclaim check that decides whether the
                # cell is still usable has the same reading a scheduling pass would give it.
                self._memory(cell, state)
                if self._unload(cell, digest, state):
                    released.append(cell.cell_id)
            return {
                "evicted": bool(released),
                "digest": digest,
                "cells": sorted(released),
                "reason": "" if released else "the cell did not release it",
            }

    def reset_lane(self, digest: str) -> None:
        """Clear a lane's blocks so a re-staged model generation is tried again.

        The ``preload-model`` and ``reload-model-catalog`` commands use it: a digest whose load
        failed permanently is not retried on its own, because the same bundle on the same machine
        fails the same way, but an operator who has fixed the machine or re-staged the bundle can
        say so.

        Args:
            digest: The bundle digest.
        """
        with self._lock:
            lane = self._lanes.get(digest)
            if lane is None:
                return
            lane.blocked_reason = None
            lane.blocked_until_ms = 0
            lane.load_failures = 0

    def _failed_result(self, entry: Queued, code: str, error_class: str, error: str):
        """Build the answer for a job that failed before a session ever ran it.

        Args:
            entry: The queued job.
            code: Stable SCREAMING_SNAKE code.
            error_class: One of the protocol error classes.
            error: The bounded detail.

        Returns:
            A ``FAILED`` :class:`~image_processor.types.InferenceResult`, so the app's result
            pipeline handles a model that never loaded exactly as it handles one that ran and
            failed.
        """
        return InferenceResult(
            inference_id=entry.job.inference_id,
            status="FAILED",
            normalized=None,
            decision=None,
            providers=[],
            gpu_device=None,
            gpu_class=None,
            timings=Timings(
                queue_ms=float(max(self._clock() - entry.queued_at_ms, 0)),
                model_load_ms=0.0,
                preprocess_ms=0.0,
                inference_ms=0.0,
                postprocess_ms=0.0,
                total_ms=float(max(self._clock() - entry.queued_at_ms, 0)),
            ),
            memory_high_water_mib=None,
            error=f"{code}: {error}",
            error_class=error_class,
        )

    def _block_lane(self, lane: Lane, code: str, error: str) -> None:
        """Give every job pinned to one digest the same permanent verdict.

        Args:
            lane: The lane whose model cannot load.
            code: Stable SCREAMING_SNAKE code.
            error: The bounded detail.
        """
        lane.blocked_reason = f"{code}: {error}"
        while lane.jobs:
            entry = lane.jobs.popleft()
            if not self._claim(entry):
                continue
            job = self._block(entry, lane.blocked_reason)
            if job is not None:
                self._notify(job, self._failed_result(entry, code, PERMANENT, error))

    def _ensure_resident(self, cell, lane: Lane, entry: Queued, state) -> str:
        """Make one lane's model resident on one cell (DESIGN.md §10.2).

        Any job this method finishes with -- blocked, exhausted, or sent to a retry timer -- is
        removed from the lane before it returns, so the caller only ever pops a job it dispatched.

        Args:
            cell: The cell that would run the work.
            lane: The lane whose model is needed.
            entry: The head job of that lane.
            state: The pass state.

        Returns:
            ``resident`` when a session is ready, ``deferred`` when the load must wait, ``failed``
            when this job has been given a durable verdict, ``lost`` when the cell went away.
        """
        records = self._resident_for(cell)
        if lane.digest in records:
            return "resident"

        if lane.blocked_reason is not None:
            lane.jobs.popleft()
            job = self._block(entry, lane.blocked_reason)
            if job is not None:
                self._notify(
                    job, self._failed_result(entry, "MODEL_BLOCKED", PERMANENT, lane.blocked_reason)
                )
            return "failed"

        if lane.blocked_until_ms > state.now_ms:
            return "deferred"

        device_key = cell.device or "cpu"
        if state.loads_left.get(device_key, 0) <= 0:
            return "deferred"

        try:
            bundle = self.cache.get(lane.digest)
        except Exception as exc:
            self._block_lane(lane, "BUNDLE_INVALID", str(exc))
            return "failed"
        if bundle is None:
            lane.jobs.popleft()
            lane.load_failures += 1
            delay = self._backoff_ms(lane.load_failures)
            lane.blocked_until_ms = state.now_ms + delay
            self._retry_later(
                entry,
                f"MODEL_NOT_STAGED: {lane.digest} is not in the bundle cache",
                delay_ms=delay,
            )
            return "failed"

        if not self._memory(cell, state):
            return "lost"

        estimate = int(getattr(bundle.manifest, "estimated_device_mib", 0) or 0)
        admission = self._admit(cell, lane, estimate, state)
        if not admission and admission.reason == "OVER_BUDGET":
            self._block_lane(
                lane,
                "MODEL_OVER_BUDGET",
                f"{lane.digest} needs {admission.required_mib} MiB, over the device budget",
            )
            return "failed"
        if not admission:
            if not self._evict_for(cell, lane, admission, state):
                return "lost"
            admission = self._admit(cell, lane, estimate, state)
        if not admission:
            self.counters["deferred"] += 1
            logger.info(
                "deferring %s on cell %s: %d MiB short",
                lane.digest,
                cell.cell_id,
                admission.shortfall_mib,
            )
            return "deferred"

        return self._load(cell, lane, entry, bundle, admission, state)

    def _admit(self, cell, lane: Lane, estimate_mib: int, state):
        """Ask the residency policy whether one model fits on one cell right now.

        Args:
            cell: The cell handle.
            lane: The lane whose model is being admitted.
            estimate_mib: The manifest's ``estimatedDeviceMib``.
            state: The pass state.

        Returns:
            The :class:`~image_processor.engine.residency.Admission`.
        """
        return self.policy.admit(
            lane.digest,
            estimate_mib,
            state.free.get(cell.cell_id, 0),
            total_mib=state.total.get(cell.cell_id, 0),
            resident_mib=self._device_resident_mib(cell),
        )

    def _evict_for(self, cell, lane: Lane, admission, state) -> bool:
        """Free room for one model by evicting the cheapest sessions.

        Args:
            cell: The cell holding the sessions.
            lane: The lane whose model needs the room.
            admission: The refusal that named the shortfall.
            state: The pass state.

        Returns:
            ``True`` when the cell is still usable afterwards.
        """
        records = self._refresh_stats(cell, state.now_ms)
        victims = self.policy.victims(
            admission.shortfall_mib, records, self._leases(cell)
        )
        for victim in victims:
            logger.info(
                "evicting %s from cell %s to make room for %s", victim, cell.cell_id, lane.digest
            )
            if not self._unload(cell, victim, state):
                return False
        return True

    def _load(self, cell, lane: Lane, entry: Queued, bundle, admission, state) -> str:
        """Load one model generation into one cell.

        Args:
            cell: The cell to load into.
            lane: The lane whose model it is.
            entry: The head job, which owns the durable verdict if the load fails.
            bundle: The cached bundle.
            admission: The admission that allowed the load, whose ``required_mib`` bounds the
                provider arena.
            state: The pass state.

        Returns:
            ``resident``, ``deferred``, ``failed``, or ``lost``, as :meth:`_ensure_resident`
            defines them.
        """
        manifest = bundle.manifest
        request = LoadModel(
            digest=lane.digest,
            bundle_root=str(bundle.root),
            providers=tuple(getattr(self.supervisor, "providers", ()) or cell.providers),
            provider_policy=manifest.provider_policy,
            providers_permitted=tuple(manifest.providers_permitted or ()),
            warmup=True,
            required_provider=getattr(self.supervisor, "required_provider", None),
            allow_cpu_only=bool(getattr(self.supervisor, "allow_cpu_only", False)),
            gpu_mem_limit_mib=admission.required_mib if cell.device is not None else None,
        )
        state.loads_left[cell.device or "cpu"] = state.loads_left.get(cell.device or "cpu", 1) - 1
        reply = self._cell_call(cell, request, self.load_timeout_s)
        if reply is None:
            return "lost"
        if isinstance(reply, LoadFailed):
            return self._load_failed(cell, lane, entry, reply, state)
        if not isinstance(reply, Loaded):
            lane.jobs.popleft()
            self._retry_later(entry, f"UNEXPECTED_REPLY: {type(reply).__name__} answered a load")
            return "failed"

        self.policy.record_load(lane.digest, reply.device_mib, reply.load_ms)
        state.free[cell.cell_id] = max(
            state.free.get(cell.cell_id, 0) - int(reply.device_mib or 0), 0
        )
        self._resident_for(cell)[lane.digest] = ResidencyStats(
            digest=lane.digest,
            size_mib=int(reply.device_mib or 0),
            estimate_mib=int(getattr(manifest, "estimated_device_mib", 0) or 0),
            measured_load_peak_mib=int(reply.device_mib or 0) or None,
            load_ms=float(reply.load_ms or 0.0),
            queued_jobs=len(lane.jobs),
            priority=lane.priority,
            last_used_ms=state.now_ms,
            loaded_at_ms=state.now_ms,
        )
        lane.burst_remaining = len(lane.jobs)
        lane.load_failures = 0
        lane.blocked_until_ms = 0
        self.counters["loads"] += 1
        logger.info(
            "cell %s made %s resident in %.0f ms for a burst of %d",
            cell.cell_id,
            lane.digest,
            reply.load_ms,
            lane.burst_remaining,
        )
        return "resident"

    def _load_failed(self, cell, lane: Lane, entry: Queued, reply: LoadFailed, state) -> str:
        """Apply the ``WAITING_MODEL`` failure edges (DESIGN.md §7).

        A permanent model failure blocks the whole lane, because every job in it is pinned to the
        same digest and would fail identically. A transient one backs the lane off and returns
        this job to its retry timer without spending an inference attempt: the load failed, the
        job did not.

        Args:
            cell: The cell that tried to load.
            lane: The lane whose model failed.
            entry: The head job.
            reply: The failure.
            state: The pass state.

        Returns:
            ``failed``, ``deferred``, or ``lost``.
        """
        if reply.memory_pressure:
            self.policy.record_memory_pressure(lane.digest)
        detail = f"{reply.code}: {reply.error}"

        if reply.error_class == CONTAMINATING:
            self._recycle(cell, f"a load poisoned the cell: {detail}")
            return "lost"

        if reply.error_class == PERMANENT:
            self._block_lane(lane, reply.code, reply.error)
            return "failed"

        lane.jobs.popleft()
        lane.load_failures += 1
        delay = self._backoff_ms(lane.load_failures)
        lane.blocked_until_ms = state.now_ms + delay
        self._retry_later(entry, detail, delay_ms=delay)
        logger.warning(
            "cell %s could not load %s (%s); the lane waits %d ms",
            cell.cell_id,
            lane.digest,
            detail,
            delay,
        )
        return "failed"

    def _ranked_lanes(self, cell, now_ms: int) -> list:
        """Rank the lanes with work for one cell (DESIGN.md §10.3).

        Work whose model is already resident on this cell goes first, because serving it costs no
        load. A cold lane that has waited longer than ``hotTtlSecs`` joins that first tier rather
        than staying behind it: preferring resident work is what makes a load pay for itself, but
        on its own it would let one busy model starve every other route indefinitely, and
        DESIGN.md §10.3 requires weighted age and priority to prevent exactly that. Inside a tier
        the oldest queue weighted by route priority wins, and the digest breaks the tie so a pass
        is reproducible.

        Args:
            cell: The cell that would run the work.
            now_ms: The current wall clock in milliseconds.

        Returns:
            The lanes, best first.
        """
        records = self._resident_for(cell)
        starving_ms = self.hot_ttl_secs * 1000.0
        lanes = [lane for lane in self._lanes.values() if lane.jobs]
        return sorted(
            lanes,
            key=lambda lane: (
                0
                if (lane.digest in records or lane.oldest_ms(now_ms) >= starving_ms)
                else 1,
                -(lane.oldest_ms(now_ms) * max(lane.priority, 1)),
                lane.digest,
            ),
        )

    def _batch(self, lane: Lane) -> list:
        """Choose the jobs one dispatch carries.

        Phase 1 dispatches one job (DESIGN.md §10.3, LLD §6). The micro-batching seam is here: a
        manifest that declares a dynamic batch axis may take several same-shape jobs from the head
        of the lane, bounded by ``maxBatchSize`` and ``maxBatchLatencyMs``, and the cell's
        ``Infer`` message and the family preprocessing grow a batch dimension to match. Nothing
        else in the pass needs to change.

        Args:
            lane: The lane being dispatched from.

        Returns:
            The jobs for this dispatch.
        """
        return [lane.jobs[0]] if lane.jobs else []

    # -- one pass --------------------------------------------------------------------------

    def run_once(self) -> int:
        """Run one scheduling pass.

        Returns:
            The number of jobs dispatched. A pass that dispatches nothing has still done its
            bookkeeping: retry timers, evictions, and durable verdicts all happen here.
        """
        with self._lock:
            state = _Pass(now_ms=self._clock())
            self._drain_inbox(state.now_ms)
            self._promote_retries(state.now_ms)
            if self._paused:
                return 0
            cells = [cell for cell in self.supervisor.cells() if cell.is_alive()]
            if not cells:
                return 0
            for cell in cells:
                state.loads_left.setdefault(cell.device or "cpu", self.load_concurrency_per_gpu)

            pending = []
            for cell in cells:
                dispatched = self._dispatch_one(cell, state)
                if dispatched is not None:
                    pending.append(dispatched)
        for cell, lane, entry in pending:
            self._collect(cell, lane, entry)
        return len(pending)

    def _dispatch_one(self, cell, state):
        """Give one cell one job, if any lane can be served.

        Args:
            cell: The cell handle.
            state: The pass state.

        Returns:
            ``(cell, lane, entry)`` when a job was sent, otherwise ``None``.
        """
        if not any(lane.jobs for lane in self._lanes.values()):
            return None
        if not self._memory(cell, state):
            return None
        for lane in self._ranked_lanes(cell, state.now_ms):
            entry = lane.jobs[0]
            if not self._claim(entry):
                lane.jobs.popleft()
                continue
            outcome = self._ensure_resident(cell, lane, entry, state)
            if outcome == "lost":
                return None
            if outcome in ("failed", "deferred"):
                continue
            if entry.job.state is JobState.WAITING_MODEL:
                if self._transition(entry, JobState.WAITING_MODEL, JobState.INFERENCING) is None:
                    lane.jobs.popleft()
                    continue
            lane.jobs.popleft()
            request = Infer(
                inference_id=entry.job.inference_id,
                staged_path=str(entry.job.staged_path or ""),
                sha256=entry.job.source.sha256,
                digest=lane.digest,
                transform_version=entry.job.transform_version,
                queue_ms=float(max(state.now_ms - entry.queued_at_ms, 0)),
            )
            try:
                cell.send(request)
            except (CellDead, CellTimeout, CellError) as exc:
                self._recycle(cell, f"a dispatch failed: {exc}")
                self._requeue(lane, entry)
                return None
            stats = self._resident_for(cell).get(lane.digest)
            if stats is not None:
                stats.last_used_ms = state.now_ms
                stats.hits += 1
            if lane.burst_remaining > 0:
                lane.burst_remaining -= 1
            self.counters["dispatched"] += 1
            return (cell, lane, entry)
        return None

    def _collect(self, cell, lane: Lane, entry: Queued) -> None:
        """Wait for one dispatched job's answer and apply it (DESIGN.md §7).

        The wait happens without the pass lock held, so a submission or a status read is never
        stuck behind an inference; the answer is applied with the lock held.

        Args:
            cell: The cell that is running it.
            lane: The lane it came from.
            entry: The queued job.
        """
        try:
            result = cell.receive(self.infer_timeout_s)
        except (CellDead, CellTimeout) as exc:
            with self._lock:
                self._recycle(cell, f"a dispatched job was lost: {exc}")
                self._transition(
                    entry, JobState.INFERENCING, JobState.INFERENCING, last_error=str(exc)
                )
                self._requeue(lane, entry)
            return
        with self._lock:
            self._apply(cell, lane, entry, result)

    def _apply(self, cell, lane: Lane, entry: Queued, result) -> None:
        """Record one answer against the ledger and hand it on (DESIGN.md §7).

        Args:
            cell: The cell that produced it.
            lane: The lane the job came from.
            entry: The queued job.
            result: The reply the cell sent.
        """
        if not isinstance(result, InferenceResult):
            self._transition(
                entry, JobState.INFERENCING, JobState.INFERENCING, last_error="UNEXPECTED_REPLY"
            )
            self._requeue(lane, entry)
            logger.error(
                "cell %s answered %s with a %s", cell.cell_id, entry.job.inference_id,
                type(result).__name__,
            )
            return

        if result.status == "SUCCEEDED":
            self.counters["succeeded"] += 1
            self._notify(entry.job, result)
            return

        error_class = result.error_class or TRANSIENT
        detail = result.error or "the cell reported no detail"

        if error_class == CONTAMINATING:
            self._recycle(cell, f"a job poisoned the cell: {detail}")
            self._transition(
                entry, JobState.INFERENCING, JobState.INFERENCING, last_error=detail
            )
            self._requeue(lane, entry)
            return

        if error_class == PERMANENT:
            job = self._exhaust(entry, detail)
            if job is not None:
                self._notify(job, result)
            return

        attempts = int(entry.job.attempts) + 1
        if attempts >= self.max_attempts:
            job = self._exhaust(entry, detail, attempts=attempts)
            if job is not None:
                self._notify(job, result)
            return
        self._retry_later(entry, detail, attempts=attempts)

    # -- running ---------------------------------------------------------------------------

    def run_forever(self, stop=None) -> None:
        """Run passes until the stop event is set.

        Args:
            stop: The event that ends the loop, or ``None`` to use the scheduler's own, which
                :meth:`stop` sets.
        """
        event = stop or self._stop
        event.clear()
        logger.info("scheduler loop started")
        while not event.is_set():
            try:
                dispatched = self.run_once()
            except Exception as exc:
                logger.exception("a scheduling pass failed: %s", exc)
                dispatched = 0
            if not dispatched:
                event.wait(self.poll_interval_s)
        logger.info("scheduler loop stopped")

    def start(self) -> "Scheduler":
        """Run :meth:`run_forever` on a daemon thread.

        Returns:
            This scheduler.
        """
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever, name="image-processor-scheduler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 30.0) -> None:
        """Stop the loop and wait for the thread to finish.

        Args:
            timeout_s: How long to wait for the thread.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)

    def status(self) -> dict:
        """Summarize the scheduler for ``get-queue`` and ``get-models`` (DESIGN.md §13).

        Returns:
            The lanes with their queue depth, age, and priority; what each cell holds and what is
            leased; the executor recycle count; and the pass counters.
        """
        with self._lock:
            now = self._clock()
            lanes = [
                {
                    "digest": lane.digest,
                    "queued": len(lane.jobs),
                    "oldestMs": lane.oldest_ms(now),
                    "priority": lane.priority,
                    "burstRemaining": lane.burst_remaining,
                    "blockedUntilMs": lane.blocked_until_ms,
                    "blockedReason": lane.blocked_reason,
                }
                for lane in sorted(self._lanes.values(), key=lambda entry: entry.digest)
            ]
            cells = []
            for cell in self.supervisor.cells():
                records = self._resident.get(cell.cell_id, {})
                cells.append(
                    {
                        "cellId": cell.cell_id,
                        "device": cell.device,
                        "alive": cell.is_alive(),
                        "resident": sorted(records),
                        "residentMib": {
                            digest: stats.footprint_mib for digest, stats in records.items()
                        },
                        "leased": sorted(self._leases(cell)),
                    }
                )
            return {
                "paused": self._paused,
                "queued": sum(len(lane.jobs) for lane in self._lanes.values())
                + len(self._inbox),
                "retryWaiting": len(self._retry),
                "lanes": lanes,
                "cells": cells,
                "recycleCount": getattr(self.supervisor, "recycle_count", 0),
                "counters": dict(self.counters),
            }


@dataclass
class _Pass:
    """What one scheduling pass carries between its steps.

    Attributes:
        now_ms: The wall clock the whole pass is judged against, so a pass is one instant.
        free: Device memory free per cell, read once and then tracked across loads and evictions.
        total: Device memory installed per cell.
        loads_left: Loads still allowed per device this pass (``loadConcurrencyPerGpu``).
    """

    now_ms: int
    free: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    loads_left: dict = field(default_factory=dict)


__all__ = [
    "DEFAULT_CONTROL_TIMEOUT_S",
    "DEFAULT_INFER_TIMEOUT_S",
    "DEFAULT_LOAD_TIMEOUT_S",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_RETRY_BACKOFF_SECS",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_RECLAIM_RATIO",
    "DEFAULT_RETRY_BACKOFF_SECS",
    "Lane",
    "Queued",
    "Scheduler",
]
