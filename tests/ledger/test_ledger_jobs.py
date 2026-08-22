"""Admission, the reservation budget, and the compare-and-set transition gate."""

import itertools
import threading

import pytest
from ledger_support import admitted, build_job, drive, path_to

from image_processor.ledger import (
    TRANSITIONS,
    IllegalTransition,
    Ledger,
    LedgerClosed,
    LedgerConflict,
)
from image_processor.types import JobState

#: Edges the diagram does not contain, one per interesting confusion.
ILLEGAL_EDGES = [
    (JobState.DISCOVERED, JobState.CLAIMED),
    (JobState.READY, JobState.INFERENCING),
    (JobState.READY, JobState.COMPLETED),
    (JobState.CLAIMED, JobState.INFERENCING),
    (JobState.WAITING_MODEL, JobState.RESULT_COMMITTED),
    (JobState.INFERENCING, JobState.PUBLISH_PENDING),
    (JobState.INFERENCING, JobState.PUBLISHED),
    (JobState.RESULT_COMMITTED, JobState.PUBLISHED),
    (JobState.RESULT_COMMITTED, JobState.COMPLETED),
    (JobState.PUBLISH_PENDING, JobState.CLEANUP_PENDING),
    (JobState.PUBLISH_PENDING, JobState.COMPLETED),
    (JobState.PUBLISHED, JobState.COMPLETED),
    (JobState.CLEANUP_FAILED, JobState.COMPLETED),
    (JobState.CLEANUP_PENDING, JobState.PUBLISHED),
    (JobState.INPUT_INVALID, JobState.READY),
    (JobState.QUARANTINED, JobState.READY),
    (JobState.COMPLETED, JobState.CLEANUP_PENDING),
    (JobState.RETAINED_FAILED, JobState.READY),
    (JobState.BLOCKED_CONFIGURATION, JobState.READY),
    (JobState.PROCESSING_EXHAUSTED, JobState.READY),
]


def test_admit_inserts_and_reserves(ledger):
    job = build_job()
    assert ledger.admit(job, 4096) is True
    assert ledger.get(job.inference_id) == job
    assert ledger.reserved_bytes() == 4096


def test_admit_is_idempotent_on_inference_id(ledger):
    job = build_job()
    assert ledger.admit(job, 4096) is True
    again = build_job(relative_path="somewhere/else.jpg")
    assert ledger.admit(again, 4096) is False
    assert ledger.get(job.inference_id).source.relative_path == job.source.relative_path
    assert ledger.reserved_bytes() == 4096


def test_admit_refuses_when_the_budget_is_exhausted(db_path, clock):
    store = Ledger(db_path, synchronous="NORMAL", reserve_budget_bytes=1000, clock=clock)
    try:
        assert store.admit(build_job(inference_id="a"), 600) is True
        assert store.admit(build_job(inference_id="b"), 600) is False
        assert store.get("b") is None
        assert store.reserved_bytes() == 600
        assert store.admit(build_job(inference_id="c"), 400) is True
        assert store.reserved_bytes() == 1000
    finally:
        store.close()


def test_reservation_is_released_at_a_terminal_state(db_path, clock):
    store = Ledger(db_path, synchronous="NORMAL", reserve_budget_bytes=1000, clock=clock)
    try:
        assert store.admit(build_job(inference_id="a"), 900) is True
        assert store.admit(build_job(inference_id="b"), 900) is False
        drive(store, "a", JobState.COMPLETED)
        assert store.reserved_bytes() == 0
        assert store.admit(build_job(inference_id="b"), 900) is True
    finally:
        store.close()


def test_admit_refuses_a_non_initial_state(ledger):
    with pytest.raises(IllegalTransition):
        ledger.admit(build_job(state=JobState.INFERENCING), 1)


def test_admit_refuses_a_negative_reservation(ledger):
    with pytest.raises(ValueError):
        ledger.admit(build_job(), -1)


def test_admit_at_ready_is_allowed(ledger):
    job = build_job(state=JobState.READY)
    assert ledger.admit(job, 1) is True
    assert ledger.get(job.inference_id).state is JobState.READY


def test_get_returns_none_for_an_unknown_job(ledger):
    assert ledger.get("nope") is None
    assert ledger.last_error("nope") is None


@pytest.mark.parametrize(
    "current,new", sorted(TRANSITIONS, key=lambda e: (e[0].value, e[1].value))
)
def test_every_legal_edge_is_accepted(ledger, current, new):
    job = admitted(ledger, current, inference_id=f"{current.value}-{new.value}")
    assert job.state is current
    assert ledger.transition(job.inference_id, current, new).state is new
    assert ledger.get(job.inference_id).state is new


@pytest.mark.parametrize("current,new", ILLEGAL_EDGES)
def test_illegal_edges_are_refused(ledger, current, new):
    assert (current, new) not in TRANSITIONS
    job = admitted(ledger, current, inference_id=f"{current.value}-x-{new.value}")
    with pytest.raises(IllegalTransition):
        ledger.transition(job.inference_id, current, new)
    assert ledger.get(job.inference_id).state is current


def test_transition_is_compare_and_set(ledger):
    job = admitted(ledger, JobState.READY)
    with pytest.raises(LedgerConflict):
        ledger.transition(job.inference_id, JobState.CLAIMED, JobState.WAITING_MODEL)
    assert ledger.get(job.inference_id).state is JobState.READY


def test_transition_on_a_missing_job_conflicts(ledger):
    with pytest.raises(LedgerConflict):
        ledger.transition("nope", JobState.READY, JobState.CLAIMED)


def test_transition_updates_declared_fields_only(ledger):
    job = admitted(ledger, JobState.INFERENCING)
    moved = ledger.transition(
        job.inference_id,
        JobState.INFERENCING,
        JobState.RETRY_WAIT,
        attempts=3,
        next_attempt_at_ms=42,
        last_error="cuda oom",
        staged_path="/staging/a.jpg",
    )
    assert (moved.attempts, moved.next_attempt_at_ms, moved.staged_path) == (
        3,
        42,
        "/staging/a.jpg",
    )
    assert ledger.last_error(job.inference_id) == "cuda oom"
    with pytest.raises(ValueError):
        ledger.transition(job.inference_id, JobState.RETRY_WAIT, JobState.READY, state="X")


def test_a_field_only_self_edge_is_allowed(ledger):
    job = admitted(ledger, JobState.RETRY_WAIT)
    same = ledger.transition(
        job.inference_id, JobState.RETRY_WAIT, JobState.RETRY_WAIT, attempts=7
    )
    assert same.state is JobState.RETRY_WAIT
    assert same.attempts == 7


def test_claimable_respects_route_timer_and_order(ledger, clock):
    admitted(ledger, JobState.READY, inference_id="first")
    admitted(ledger, JobState.READY, inference_id="second")
    admitted(ledger, JobState.READY, inference_id="other", route_id="cam-02")
    ledger.transition(
        "second", JobState.READY, JobState.READY, next_attempt_at_ms=clock.now + 10_000
    )
    assert [j.inference_id for j in ledger.claimable(None, 10)] == ["first", "other"]
    assert [j.inference_id for j in ledger.claimable("cam-02", 10)] == ["other"]
    assert [j.inference_id for j in ledger.claimable(None, 1)] == ["first"]


def test_by_state_pages_deterministically(ledger):
    for index in range(5):
        admitted(ledger, JobState.READY, inference_id=f"job-{index}")
    admitted(ledger, JobState.CLAIMED, inference_id="claimed")
    page, cursor = ledger.by_state([JobState.READY], limit=2)
    assert [j.inference_id for j in page] == ["job-0", "job-1"]
    assert cursor
    seen = list(page)
    while cursor:
        page, cursor = ledger.by_state([JobState.READY], cursor=cursor, limit=2)
        seen.extend(page)
    assert [j.inference_id for j in seen] == [f"job-{i}" for i in range(5)]
    scoped, _ = ledger.by_state([JobState.READY], route_id="cam-99")
    assert scoped == []
    assert ledger.by_state([]) == ([], None)


def test_by_state_rejects_a_malformed_cursor(ledger):
    admitted(ledger, JobState.READY)
    for bad in ("nonsense", "abc|job-1"):
        with pytest.raises(ValueError):
            ledger.by_state([JobState.READY], cursor=bad)


def test_route_generations_round_trip(ledger):
    assert ledger.route_generation("cam-01") == (None, None)
    ledger.set_route_generation("cam-01", "gen-2", None)
    assert ledger.route_generation("cam-01") == ("gen-2", None)
    ledger.set_route_generation("cam-01", "gen-2", "gen-2")
    assert ledger.route_generation("cam-01") == ("gen-2", "gen-2")


def test_kv_round_trip(ledger):
    assert ledger.kv_get("capture-watermark") is None
    ledger.kv_set("capture-watermark", "cursor-1")
    assert ledger.kv_get("capture-watermark") == "cursor-1"
    ledger.kv_set("capture-watermark", "cursor-2")
    assert ledger.kv_get("capture-watermark") == "cursor-2"
    ledger.kv_set("capture-watermark", None)
    assert ledger.kv_get("capture-watermark") is None


def test_close_is_idempotent_and_refuses_further_writes(db_path, clock):
    store = Ledger(db_path, synchronous="NORMAL", clock=clock)
    store.kv_set("a", "b")
    store.close()
    store.close()
    with pytest.raises(LedgerClosed):
        store.kv_set("a", "c")


def test_context_manager_closes(db_path, clock):
    with Ledger(db_path, synchronous="NORMAL", clock=clock) as store:
        store.kv_set("a", "b")
    with pytest.raises(LedgerClosed):
        store.kv_set("a", "c")


def test_every_state_is_reachable_along_legal_edges(ledger):
    for state in JobState:
        assert admitted(ledger, state, inference_id=f"reach-{state.value}").state is state


def test_writes_from_many_threads_are_serialized(ledger):
    for index in range(20):
        admitted(ledger, JobState.READY, inference_id=f"t-{index}")
    errors = []

    def worker(index):
        try:
            ledger.transition(f"t-{index}", JobState.READY, JobState.CLAIMED)
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    claimed, _ = ledger.by_state([JobState.CLAIMED], limit=100)
    assert len(claimed) == 20


def test_a_failing_write_closure_does_not_leave_a_transaction_open(ledger):
    with pytest.raises(LedgerConflict):
        ledger.transition("nope", JobState.READY, JobState.CLAIMED)
    ledger.kv_set("after", "still-writable")
    assert ledger.kv_get("after") == "still-writable"


def test_path_to_covers_every_state():
    assert path_to(JobState.DISCOVERED) == []
    assert path_to(JobState.COMPLETED)[-1] is JobState.COMPLETED
    reached = set(itertools.chain.from_iterable(path_to(s) for s in JobState))
    assert len(reached) == len(JobState) - 1
