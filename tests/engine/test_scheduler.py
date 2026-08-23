"""Scheduling decisions and the durable edges they produce (DESIGN.md §10.3, §7, LLD §6).

The cells are fakes and the ledger is real, which is the split that matters: what a pass decides is
asserted against scripted replies, and what it records is asserted against the SQLite state machine
that refuses an illegal edge. Every clock is manual, so a backoff is a number rather than a wait.
"""

import threading

import pytest

from image_processor.engine.residency import ResidencyPolicy
from image_processor.engine.scheduler import Lane, Queued, Scheduler
from image_processor.engine.protocol import (
    CONTAMINATING,
    CUDA_PROVIDER,
    PERMANENT,
    TRANSIENT,
    Infer,
    LoadFailed,
    LoadModel,
    Stats,
    Unload,
)
from image_processor.ledger import Ledger
from image_processor.types import JobState
from tests.engine.executor_support import (
    DIGEST_A,
    DIGEST_B,
    FakeCache,
    FakeCell,
    FakeSupervisor,
    ManualClock,
    admit_ready,
    build_job,
    failed_result,
    ok_result,
)


class Harness:
    """A scheduler over fake cells, a fake cache, and a real ledger.

    Attributes:
        scheduler: The subject.
        supervisor: The fake supervisor.
        cells: Its fake cells.
        cache: The fake bundle cache.
        clock: The manual millisecond clock.
        results: ``(job, result)`` for every call the scheduler made to the app.
    """

    def __init__(self, ledger, cells=None, digests=(DIGEST_A, DIGEST_B), **kwargs) -> None:
        """Build the harness."""
        self.ledger = ledger
        self.clock = ManualClock()
        self.cells = list(cells or [FakeCell()])
        self.supervisor = FakeSupervisor(self.cells)
        self.cache = FakeCache()
        for digest in digests:
            self.cache.add(digest)
        self.results = []
        self.policy = ResidencyPolicy(
            resident_memory_budget_percent=80,
            reserve_mib=0,
            min_residency_secs=0,
            hot_ttl_secs=120,
            clock=self.clock,
        )
        fields = {
            "on_result": lambda job, result: self.results.append((job, result)),
            "clock": self.clock,
            "rng": _FixedJitter(),
            "max_attempts": 3,
            "retry_backoff_secs": 4.0,
            "max_retry_backoff_secs": 60.0,
        }
        fields.update(kwargs)
        self.scheduler = Scheduler(
            ledger, self.supervisor, self.cache, self.policy, **fields
        )

    def sent(self, cell_index: int = 0, kind=None) -> list:
        """Return what one cell was asked to do.

        Args:
            cell_index: Which cell.
            kind: Restrict to one message type, or ``None`` for all.

        Returns:
            The requests, in order.
        """
        calls = self.cells[cell_index].calls
        return [call for call in calls if kind is None or isinstance(call, kind)]

    def state(self, inference_id: str) -> JobState:
        """Return one job's durable state.

        Args:
            inference_id: The job identity.

        Returns:
            The state the ledger holds.
        """
        return self.ledger.get(inference_id).state


class _FixedJitter:
    """A jitter source that always returns the midpoint, so backoff is exact."""

    def uniform(self, low: float, high: float) -> float:
        """Return the midpoint of the range.

        Args:
            low: The lower bound.
            high: The upper bound.

        Returns:
            ``high``, so a delay is the full jittered value.
        """
        return high


@pytest.fixture()
def ledger(tmp_path):
    """A real ledger in a temporary directory."""
    store = Ledger(tmp_path / "state.db", synchronous="NORMAL")
    try:
        yield store
    finally:
        store.close()


def test_one_pass_walks_a_job_from_ready_to_inferencing_and_hands_over_the_result(ledger):
    harness = Harness(ledger)
    job = admit_ready(ledger, "job-1")
    harness.scheduler.submit(job)

    assert harness.scheduler.run_once() == 1

    loads = harness.sent(kind=LoadModel)
    assert len(loads) == 1
    assert loads[0].digest == DIGEST_A
    assert loads[0].providers == (CUDA_PROVIDER,)
    assert loads[0].gpu_mem_limit_mib > 0

    infers = harness.sent(kind=Infer)
    assert len(infers) == 1
    assert infers[0].inference_id == "job-1"
    assert infers[0].staged_path == job.staged_path
    assert infers[0].sha256 == job.source.sha256
    assert infers[0].transform_version == "t1"

    assert harness.state("job-1") is JobState.INFERENCING
    assert [entry[0].inference_id for entry in harness.results] == ["job-1"]
    assert harness.results[0][1].status == "SUCCEEDED"
    assert harness.scheduler.counters["succeeded"] == 1


def test_a_pass_with_nothing_queued_dispatches_nothing(ledger):
    harness = Harness(ledger)
    assert harness.scheduler.run_once() == 0
    assert harness.sent() == []


def test_a_cold_digest_is_loaded_once_however_many_jobs_want_it(ledger):
    harness = Harness(ledger)
    for index in range(3):
        harness.scheduler.submit(admit_ready(ledger, f"job-{index}"))

    for _ in range(3):
        harness.scheduler.run_once()

    assert len(harness.sent(kind=LoadModel)) == 1
    assert len(harness.sent(kind=Infer)) == 3
    assert harness.scheduler.counters["loads"] == 1
    assert harness.scheduler.queued() == 0


def test_a_resident_lane_is_served_before_an_older_cold_one(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "old-cold", digest=DIGEST_B))
    harness.clock.advance(60_000)
    harness.scheduler.submit(admit_ready(ledger, "new-hot", digest=DIGEST_A))

    harness.scheduler.run_once()
    assert harness.sent(kind=Infer)[0].digest == DIGEST_B

    harness.scheduler.submit(admit_ready(ledger, "second-hot", digest=DIGEST_B))
    harness.scheduler.run_once()
    assert harness.sent(kind=Infer)[1].digest == DIGEST_B
    assert harness.state("new-hot") is JobState.READY


def test_among_cold_lanes_the_oldest_weighted_by_priority_goes_first(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "low", digest=DIGEST_A), priority=1)
    harness.clock.advance(1_000)
    harness.scheduler.submit(admit_ready(ledger, "high", digest=DIGEST_B), priority=1000)
    harness.clock.advance(1_000)

    harness.scheduler.run_once()
    assert harness.sent(kind=Infer)[0].inference_id == "high"


def test_a_paused_scheduler_claims_nothing_and_resumes_where_it_left_off(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    harness.scheduler.pause()

    assert harness.scheduler.paused is True
    assert harness.scheduler.run_once() == 0
    assert harness.state("job-1") is JobState.READY
    assert harness.scheduler.queued() == 1

    harness.scheduler.resume()
    assert harness.scheduler.run_once() == 1
    assert harness.state("job-1") is JobState.INFERENCING


def test_a_transient_failure_waits_with_backoff_and_comes_back_at_the_next_attempt(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_infer = lambda message: failed_result(message.inference_id, TRANSIENT)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    job = ledger.get("job-1")
    assert job.state is JobState.RETRY_WAIT
    assert job.attempts == 1
    assert job.next_attempt_at_ms == harness.clock.now + 4_000
    assert harness.results == []
    assert harness.scheduler.counters["retried"] == 1

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-1") is JobState.RETRY_WAIT

    harness.clock.advance(4_000)
    harness.scheduler.run_once()
    assert ledger.get("job-1").attempts == 2
    assert ledger.get("job-1").next_attempt_at_ms == harness.clock.now + 8_000


def test_the_retry_budget_ends_in_processing_exhausted_and_the_app_is_told(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_infer = lambda message: failed_result(message.inference_id, TRANSIENT)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    for _ in range(3):
        harness.scheduler.run_once()
        harness.clock.advance(120_000)

    job = ledger.get("job-1")
    assert job.state is JobState.PROCESSING_EXHAUSTED
    assert job.attempts == 3
    assert [entry[1].status for entry in harness.results] == ["FAILED"]
    assert harness.scheduler.counters["exhausted"] == 1


def test_a_permanent_failure_is_not_retried_at_all(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_infer = lambda message: failed_result(message.inference_id, PERMANENT)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    assert harness.state("job-1") is JobState.PROCESSING_EXHAUSTED
    assert ledger.get("job-1").attempts == 0
    assert harness.results[0][1].error_class == PERMANENT
    assert harness.scheduler.counters["retried"] == 0


def test_a_contaminating_failure_recycles_the_cell_and_requeues_at_the_same_attempt(ledger):
    harness = Harness(ledger)
    answers = [failed_result("job-1", CONTAMINATING), None]
    harness.cells[0].on_infer = lambda message: answers.pop(0) if answers else None
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    assert harness.supervisor.recycle_count == 1
    assert harness.state("job-1") is JobState.INFERENCING
    assert ledger.get("job-1").attempts == 0
    assert ledger.last_error("job-1")
    assert harness.results == []
    assert harness.scheduler.queued() == 1

    harness.scheduler.run_once()
    assert harness.results[0][1].status == "SUCCEEDED"
    assert ledger.get("job-1").attempts == 0
    assert len(harness.sent(kind=LoadModel)) == 2


def test_a_cell_that_dies_after_dispatch_returns_the_job_to_its_lane(ledger):
    harness = Harness(ledger)

    def die_after_taking(message):
        harness.cells[0].alive = False
        return ok_result(message.inference_id)

    harness.cells[0].on_infer = die_after_taking
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    assert harness.supervisor.recycle_count == 1
    assert harness.state("job-1") is JobState.INFERENCING
    assert harness.scheduler.queued() == 1
    assert harness.results == []


def test_a_reply_that_is_not_a_result_is_not_taken_as_an_answer(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_infer = lambda message: LoadFailed(digest=DIGEST_A, error="wrong reply")
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    assert harness.results == []
    assert harness.state("job-1") is JobState.INFERENCING
    assert harness.scheduler.queued() == 1


def test_a_model_that_cannot_load_blocks_every_job_pinned_to_it(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_load = lambda message: LoadFailed(
        digest=message.digest, error="the head is unreadable",
        error_class=PERMANENT, code="FAMILY_REFUSED",
    )
    for index in range(2):
        harness.scheduler.submit(admit_ready(ledger, f"job-{index}"))

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-0") is JobState.BLOCKED_CONFIGURATION
    assert harness.state("job-1") is JobState.BLOCKED_CONFIGURATION
    assert [entry[1].error_class for entry in harness.results] == [PERMANENT, PERMANENT]
    assert "FAMILY_REFUSED" in harness.results[0][1].error
    assert harness.scheduler.counters["blocked"] == 2

    harness.scheduler.submit(admit_ready(ledger, "job-late"))
    harness.scheduler.run_once()
    assert harness.state("job-late") is JobState.BLOCKED_CONFIGURATION
    assert len(harness.sent(kind=LoadModel)) == 1


def test_a_blocked_lane_is_tried_again_once_an_operator_says_so(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_load = lambda message: LoadFailed(
        digest=message.digest, error="nope", error_class=PERMANENT, code="MODEL_INVALID"
    )
    harness.scheduler.submit(admit_ready(ledger, "job-0"))
    harness.scheduler.run_once()

    harness.cells[0].on_load = None
    harness.scheduler.reset_lane(DIGEST_A)
    harness.scheduler.reset_lane("sha256:not-a-lane")
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    assert harness.scheduler.run_once() == 1
    assert harness.state("job-1") is JobState.INFERENCING


def test_a_transient_load_failure_backs_the_lane_off_without_spending_an_attempt(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_load = lambda message: LoadFailed(
        digest=message.digest, error="CUDA failure 2: out of memory",
        error_class=TRANSIENT, code="CUDA_OOM", memory_pressure=True,
    )
    harness.scheduler.submit(admit_ready(ledger, "job-0"))
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    assert harness.scheduler.run_once() == 0
    job = ledger.get("job-0")
    assert job.state is JobState.RETRY_WAIT
    assert job.attempts == 0
    assert harness.state("job-1") is JobState.READY
    assert harness.policy.required_mib(DIGEST_A, 1024) > harness.policy.activation_peak_factor * 1024

    assert harness.scheduler.run_once() == 0
    assert len(harness.sent(kind=LoadModel)) == 1

    harness.clock.advance(10_000)
    harness.cells[0].on_load = None
    assert harness.scheduler.run_once() == 1


def test_a_load_that_poisons_the_cell_recycles_it_and_keeps_the_job(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_load = lambda message: LoadFailed(
        digest=message.digest, error="an illegal memory access was encountered",
        error_class=CONTAMINATING, code="CUDA_ILLEGAL_ADDRESS",
    )
    harness.scheduler.submit(admit_ready(ledger, "job-0"))

    assert harness.scheduler.run_once() == 0
    assert harness.supervisor.recycle_count == 1
    assert harness.state("job-0") is JobState.WAITING_MODEL
    assert harness.scheduler.queued() == 1


def test_a_bundle_that_is_not_staged_yet_waits_rather_than_failing(ledger):
    harness = Harness(ledger, digests=())
    harness.scheduler.submit(admit_ready(ledger, "job-0"))

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-0") is JobState.RETRY_WAIT
    assert ledger.get("job-0").attempts == 0
    assert harness.sent(kind=LoadModel) == []

    harness.cache.add(DIGEST_A)
    harness.clock.advance(10_000)
    assert harness.scheduler.run_once() == 1


def test_a_cached_bundle_that_no_longer_verifies_blocks_the_lane(ledger):
    harness = Harness(ledger)
    harness.cache.raises[DIGEST_A] = RuntimeError("DIGEST_MISMATCH: the cached bundle changed")
    harness.scheduler.submit(admit_ready(ledger, "job-0"))

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-0") is JobState.BLOCKED_CONFIGURATION
    assert harness.results[0][1].status == "FAILED"


def test_a_cold_model_evicts_the_cheapest_session_to_make_room(ledger):
    cell = FakeCell(total_mib=4096, free_mib=4096, load_mib=3000)
    harness = Harness(ledger, cells=[cell])
    harness.scheduler.submit(admit_ready(ledger, "first", digest=DIGEST_A))
    harness.scheduler.run_once()
    assert cell.resident == {DIGEST_A: 3000}

    harness.clock.advance(240_000)
    harness.scheduler.submit(admit_ready(ledger, "second", digest=DIGEST_B))
    harness.scheduler.run_once()

    unloads = harness.sent(kind=Unload)
    assert [message.digest for message in unloads] == [DIGEST_A]
    assert cell.resident == {DIGEST_B: 3000}
    assert harness.state("second") is JobState.INFERENCING
    assert harness.scheduler.counters["evictions"] == 1


def test_a_session_with_work_still_queued_for_it_is_not_evicted_mid_burst(ledger):
    cell = FakeCell(total_mib=4096, free_mib=4096, load_mib=3000)
    harness = Harness(ledger, cells=[cell])
    for index in range(3):
        harness.scheduler.submit(admit_ready(ledger, f"hot-{index}", digest=DIGEST_A), priority=1)
    harness.scheduler.run_once()
    assert cell.resident == {DIGEST_A: 3000}

    harness.scheduler.submit(admit_ready(ledger, "cold", digest=DIGEST_B), priority=1000)
    harness.clock.advance(240_000)
    assert harness.scheduler.run_once() == 1

    assert harness.sent(kind=Unload) == []
    assert cell.resident == {DIGEST_A: 3000}
    assert harness.state("cold") is JobState.WAITING_MODEL
    assert harness.state("hot-1") is JobState.INFERENCING
    assert harness.scheduler.counters["deferred"] >= 1

    for _ in range(2):
        harness.scheduler.run_once()
    harness.clock.advance(240_000)
    harness.scheduler.run_once()
    assert [message.digest for message in harness.sent(kind=Unload)] == [DIGEST_A]
    assert harness.state("cold") is JobState.INFERENCING


def test_a_model_larger_than_the_whole_budget_is_a_configuration_failure(ledger):
    cell = FakeCell(total_mib=2048, free_mib=2048)
    harness = Harness(ledger, cells=[cell])
    harness.cache.add(DIGEST_A, estimated_device_mib=4096)
    harness.scheduler.submit(admit_ready(ledger, "job-0"))

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-0") is JobState.BLOCKED_CONFIGURATION
    assert "MODEL_OVER_BUDGET" in harness.results[0][1].error


def test_an_unload_that_does_not_give_the_memory_back_recycles_the_cell(ledger):
    from image_processor.engine.protocol import Unloaded

    cell = FakeCell(total_mib=4096, free_mib=4096, load_mib=3000)
    harness = Harness(ledger, cells=[cell])
    harness.scheduler.submit(admit_ready(ledger, "first", digest=DIGEST_A))
    harness.scheduler.run_once()

    cell.on_unload = lambda message: Unloaded(
        digest=message.digest, freed_mib=0, was_resident=True, expected_mib=3000
    )
    harness.clock.advance(240_000)
    harness.scheduler.submit(admit_ready(ledger, "second", digest=DIGEST_B))
    harness.scheduler.run_once()

    assert harness.supervisor.recycle_count == 1
    assert harness.state("second") is JobState.WAITING_MODEL


def test_one_load_per_device_per_pass(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "a-job", digest=DIGEST_A))
    harness.scheduler.submit(admit_ready(ledger, "b-job", digest=DIGEST_B))

    harness.scheduler.run_once()
    assert len(harness.sent(kind=LoadModel)) == 1
    harness.scheduler.run_once()
    assert len(harness.sent(kind=LoadModel)) == 2


def test_two_cells_each_take_one_job_in_the_same_pass(ledger):
    cells = [FakeCell(cell_id="gpu0-0", device="0"), FakeCell(cell_id="gpu1-0", device="1")]
    harness = Harness(ledger, cells=cells)
    harness.scheduler.submit(admit_ready(ledger, "job-a", digest=DIGEST_A))
    harness.scheduler.submit(admit_ready(ledger, "job-b", digest=DIGEST_B))

    assert harness.scheduler.run_once() == 2
    assert len(harness.sent(0, Infer)) == 1
    assert len(harness.sent(1, Infer)) == 1
    assert harness.state("job-a") is JobState.INFERENCING
    assert harness.state("job-b") is JobState.INFERENCING


def test_a_pass_with_no_live_cell_does_nothing(ledger):
    harness = Harness(ledger)
    harness.cells[0].die()
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    assert harness.scheduler.run_once() == 0
    assert harness.state("job-1") is JobState.READY


def test_a_job_in_a_state_the_scheduler_cannot_act_on_is_refused_loudly(ledger):
    harness = Harness(ledger)
    with pytest.raises(ValueError):
        harness.scheduler.submit(build_job("job-x", state=JobState.COMPLETED))


def test_a_job_submitted_on_a_retry_timer_waits_for_it(ledger):
    harness = Harness(ledger)
    admit_ready(ledger, "job-1")
    ledger.transition("job-1", JobState.READY, JobState.CLAIMED)
    ledger.transition("job-1", JobState.CLAIMED, JobState.WAITING_MODEL)
    ledger.transition(
        "job-1", JobState.WAITING_MODEL, JobState.RETRY_WAIT,
        next_attempt_at_ms=harness.clock.now + 5_000,
    )
    harness.scheduler.submit(ledger.get("job-1"))

    assert harness.scheduler.run_once() == 0
    assert harness.scheduler.queued() == 1
    harness.clock.advance(5_000)
    assert harness.scheduler.run_once() == 1


def test_a_job_the_ledger_has_moved_on_from_leaves_the_lane(ledger):
    harness = Harness(ledger)
    job = admit_ready(ledger, "job-1")
    harness.scheduler.submit(job)
    ledger.transition("job-1", JobState.READY, JobState.CLAIMED)
    ledger.transition("job-1", JobState.CLAIMED, JobState.WAITING_MODEL)

    assert harness.scheduler.run_once() == 0
    assert harness.scheduler.queued() == 0


def test_a_result_callback_that_raises_leaves_the_job_for_recovery(ledger):
    harness = Harness(ledger)

    def explode(job, result):
        raise RuntimeError("the app is broken")

    harness.scheduler._on_result = explode
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    harness.scheduler.run_once()

    assert harness.state("job-1") is JobState.INFERENCING
    assert harness.scheduler.counters["callbackFailures"] == 1


def test_a_scheduler_without_a_callback_still_runs(ledger):
    harness = Harness(ledger, on_result=None)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    assert harness.scheduler.run_once() == 1


def test_the_load_budget_per_device_is_configurable(ledger):
    harness = Harness(ledger, load_concurrency_per_gpu=2)
    harness.scheduler.submit(admit_ready(ledger, "a-job", digest=DIGEST_A))
    harness.scheduler.submit(admit_ready(ledger, "b-job", digest=DIGEST_B))
    harness.scheduler.run_once()
    assert len(harness.sent(kind=LoadModel)) == 1


def test_the_backoff_doubles_and_is_capped(ledger):
    harness = Harness(ledger, max_retry_backoff_secs=10.0)
    assert harness.scheduler._backoff_ms(1) == 4_000
    assert harness.scheduler._backoff_ms(2) == 8_000
    assert harness.scheduler._backoff_ms(3) == 10_000
    assert harness.scheduler._backoff_ms(0) == 4_000


def test_a_cell_that_cannot_report_its_memory_is_recycled_and_the_pass_moves_on(ledger):
    harness = Harness(ledger)
    from image_processor.engine.cell import CellDead

    def refuse(message):
        raise CellDead("the child exited")

    original = harness.cells[0]._answer

    def answering(message):
        if isinstance(message, Stats):
            refuse(message)
        return original(message)

    harness.cells[0]._answer = answering
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    assert harness.scheduler.run_once() == 0
    assert harness.supervisor.recycle_count == 1
    assert harness.state("job-1") is JobState.READY


def test_a_supervisor_that_refuses_to_recycle_does_not_break_the_pass(ledger):
    harness = Harness(ledger)
    harness.supervisor.refuse_recycle = "the budget is spent"
    harness.cells[0].on_infer = lambda message: failed_result(message.inference_id, CONTAMINATING)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    harness.scheduler.run_once()
    assert harness.scheduler.counters["recycles"] == 1
    assert harness.scheduler.queued() == 1


def test_the_status_summary_answers_get_queue_and_get_models(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "job-1", digest=DIGEST_A))
    harness.scheduler.submit(admit_ready(ledger, "job-2", digest=DIGEST_A))
    harness.scheduler.submit(admit_ready(ledger, "job-3", digest=DIGEST_B))
    harness.scheduler.run_once()

    status = harness.scheduler.status()
    assert status["paused"] is False
    assert status["queued"] == 2
    assert status["retryWaiting"] == 0
    lanes = {lane["digest"]: lane for lane in status["lanes"]}
    assert lanes[DIGEST_A]["queued"] == 1
    assert lanes[DIGEST_A]["burstRemaining"] == 1
    assert lanes[DIGEST_B]["queued"] == 1
    assert lanes[DIGEST_B]["blockedReason"] is None
    cell = status["cells"][0]
    assert cell["cellId"] == "gpu0-0"
    assert cell["resident"] == [DIGEST_A]
    assert cell["leased"] == [DIGEST_A]
    assert cell["residentMib"][DIGEST_A] > 0
    assert status["recycleCount"] == 0
    assert status["counters"]["dispatched"] == 1


def test_the_loop_runs_passes_until_it_is_stopped(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    harness.scheduler.start()
    harness.scheduler.start()
    try:
        for _ in range(100):
            if harness.results:
                break
            threading.Event().wait(0.02)
    finally:
        harness.scheduler.stop(10.0)
    assert [entry[0].inference_id for entry in harness.results] == ["job-1"]
    assert harness.state("job-1") is JobState.INFERENCING


def test_a_pass_that_raises_does_not_end_the_loop(ledger):
    harness = Harness(ledger)
    stop = threading.Event()
    passes = []

    def explode():
        passes.append(1)
        if len(passes) >= 3:
            stop.set()
        raise RuntimeError("a pass failed")

    harness.scheduler.run_once = explode
    harness.scheduler.run_forever(stop)
    assert len(passes) >= 3


def test_stopping_a_loop_that_never_started_is_harmless(ledger):
    Harness(ledger).scheduler.stop(1.0)


def test_the_batching_seam_hands_over_one_job_in_phase_one(ledger):
    harness = Harness(ledger)
    lane = Lane(digest=DIGEST_A)
    assert harness.scheduler._batch(lane) == []
    entry = Queued(job=build_job("job-1"), priority=100, first_seen_ms=0, queued_at_ms=0)
    lane.jobs.append(entry)
    lane.jobs.append(Queued(job=build_job("job-2"), priority=100))
    assert harness.scheduler._batch(lane) == [entry]
    assert lane.priority == 100
    assert lane.oldest_ms(1_000) == 1_000
    assert Lane(digest=DIGEST_A).oldest_ms(1_000) == 0


def test_a_restarted_cell_is_no_longer_credited_with_what_it_held(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))
    harness.scheduler.run_once()
    assert harness.scheduler.status()["cells"][0]["resident"] == [DIGEST_A]

    harness.cells[0].resident.clear()
    harness.scheduler.submit(admit_ready(ledger, "job-2"))
    harness.scheduler.run_once()
    assert len(harness.sent(kind=LoadModel)) == 2


def test_a_load_answered_with_the_wrong_message_is_retried(ledger):
    harness = Harness(ledger)
    harness.cells[0].on_load = lambda message: "not a reply"
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    assert harness.scheduler.run_once() == 0
    assert harness.state("job-1") is JobState.RETRY_WAIT
    assert "UNEXPECTED_REPLY" in (ledger.last_error("job-1") or "")


def test_a_cell_that_is_already_busy_is_left_alone_for_this_pass(ledger):
    harness = Harness(ledger)
    from image_processor.engine.cell import CellError

    def busy(message):
        raise CellError("already has a request in flight")

    harness.cells[0]._answer = busy
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    assert harness.scheduler.run_once() == 0
    assert harness.supervisor.recycle_count == 0
    assert harness.state("job-1") is JobState.READY


def test_a_submission_is_counted_before_the_pass_that_lanes_it(ledger):
    harness = Harness(ledger)
    harness.scheduler.submit(admit_ready(ledger, "job-1"))

    assert harness.scheduler.queued() == 1
    assert harness.scheduler.status()["queued"] == 1
    assert harness.scheduler._lanes == {}

    harness.scheduler.run_once()
    assert list(harness.scheduler._lanes) == [DIGEST_A]
    assert harness.scheduler.queued() == 0
