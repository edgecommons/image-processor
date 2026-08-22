"""Kill-point recovery: crash after each durable step, reopen the file, and recover.

Every test here closes the ledger, reopens a second :class:`~image_processor.ledger.Ledger` on the
same database file, and calls ``recover()``. Closing and reopening is what a process crash looks
like to the next run: the WAL holds everything that committed and nothing that did not.
"""

import pytest
from ledger_support import MODEL, StepClock, admitted, build_job, row

from image_processor.ledger import Ledger, RecoveryReport, edge_key, plan_recovery
from image_processor.ledger.recovery import RecoveryMove
from image_processor.types import CleanupIntent, CompletionAction, JobState

RESULT = b'{"status":"SUCCEEDED"}'
APP_TOPIC = "ecv1/dev/image-processor/cam-01/app/inference/result"


@pytest.fixture()
def reopen(db_path, clock):
    """Reopen the state file as a fresh ledger, the way a restart does."""
    opened = []

    def _reopen(**kwargs):
        store = Ledger(db_path, synchronous="NORMAL", clock=clock, **kwargs)
        opened.append(store)
        return store

    try:
        yield _reopen
    finally:
        for store in opened:
            store.close()


def test_kill_point_after_admission(ledger, reopen):
    admitted(ledger, JobState.DISCOVERED)
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.transitions == {}
    assert store.get("job-1").state is JobState.DISCOVERED


def test_kill_point_after_claiming(ledger, reopen):
    admitted(ledger, JobState.CLAIMED)
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.count(JobState.CLAIMED, JobState.READY) == 1
    assert store.get("job-1").state is JobState.READY


def test_kill_point_after_pinning_the_model(ledger, reopen):
    admitted(ledger, JobState.WAITING_MODEL)
    ledger.close()
    store = reopen()
    assert store.recover().count(JobState.WAITING_MODEL, JobState.READY) == 1
    assert store.get("job-1").state is JobState.READY


def test_kill_point_during_inference_keeps_the_attempt_count(ledger, reopen):
    job = admitted(ledger, JobState.WAITING_MODEL)
    ledger.transition(job.inference_id, JobState.WAITING_MODEL, JobState.INFERENCING, attempts=2)
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.count(JobState.INFERENCING, JobState.READY) == 1
    recovered = store.get("job-1")
    assert recovered.state is JobState.READY
    assert recovered.attempts == 2


def test_kill_point_before_the_result_transaction_leaves_nothing_behind(ledger, reopen):
    admitted(ledger, JobState.INFERENCING)
    ledger.close()
    store = reopen()
    store.recover()
    assert store.result_bytes("job-1") is None
    assert store.outbox_for("job-1") == []
    assert store.get("job-1").state is JobState.READY


def test_kill_point_after_the_result_transaction_keeps_the_outbox_eligible(ledger, reopen):
    admitted(ledger, JobState.INFERENCING)
    ledger.commit_result(
        "job-1", RESULT, ("/spool/a.inference.json", "b" * 64), [row("job-1", APP_TOPIC, b"app")]
    )
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.transitions == {}
    assert report.publish_pending == ("job-1",)
    assert store.get("job-1").state is JobState.PUBLISH_PENDING
    assert [r.encoded_bytes for r in store.pending_outbox(10)] == [b"app"]
    assert store.result_bytes("job-1") == RESULT
    assert [(s.inference_id, s.sidecar_sha256) for s in report.sidecars] == [("job-1", "b" * 64)]


def test_kill_point_between_transport_confirmation_and_the_local_commit(ledger, reopen):
    admitted(ledger, JobState.INFERENCING)
    ledger.commit_result("job-1", RESULT, None, [row("job-1", APP_TOPIC, b"app")])
    ledger.close()
    store = reopen()
    store.recover()
    pending = store.pending_outbox(10)
    assert len(pending) == 1
    store.mark_published(pending[0].id)
    assert store.get("job-1").state is JobState.PUBLISHED


def test_kill_point_after_publication_before_the_cleanup_intent(ledger, reopen):
    admitted(ledger, JobState.PUBLISHED)
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.cleanup_pending == ()
    assert store.get("job-1").state is JobState.PUBLISHED


def test_kill_point_after_the_cleanup_intent_lists_it_for_reconciliation(ledger, reopen):
    admitted(ledger, JobState.PUBLISHED)
    stored = CleanupIntent(
        inference_id="job-1",
        action=CompletionAction.ARCHIVE,
        source_path="/spool/a.jpg",
        source_sha256="a" * 64,
        target_path="/processed/a.jpg",
        members=("/spool/a.jpg.json",),
    )
    ledger.record_cleanup_intent(stored)
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.cleanup_pending == (stored,)
    assert store.get("job-1").state is JobState.CLEANUP_PENDING


def test_kill_point_after_cleanup_completed_leaves_the_job_alone(ledger, reopen):
    admitted(ledger, JobState.CLEANUP_PENDING)
    ledger.close()
    store = reopen()
    assert store.recover().cleanup_pending == ()
    assert store.get("job-1").state is JobState.CLEANUP_PENDING


def test_retry_wait_recovers_only_once_the_timer_has_elapsed(db_path):
    clock = StepClock(start=1_000_000)
    store = Ledger(db_path, synchronous="NORMAL", clock=clock)
    try:
        for name, due in (("due", 500_000), ("not-due", 9_000_000), ("no-timer", None)):
            job = build_job(inference_id=name, model=MODEL)
            store.admit(job, 16)
            store.transition(name, JobState.DISCOVERED, JobState.READY)
            store.transition(name, JobState.READY, JobState.CLAIMED)
            store.transition(name, JobState.CLAIMED, JobState.WAITING_MODEL)
            store.transition(
                name, JobState.WAITING_MODEL, JobState.RETRY_WAIT, next_attempt_at_ms=due
            )
    finally:
        store.close()
    resumed = Ledger(db_path, synchronous="NORMAL", clock=StepClock(start=1_000_000))
    try:
        report = resumed.recover()
        assert report.count(JobState.RETRY_WAIT, JobState.READY) == 2
        assert resumed.get("due").state is JobState.READY
        assert resumed.get("no-timer").state is JobState.READY
        assert resumed.get("not-due").state is JobState.RETRY_WAIT
    finally:
        resumed.close()


def test_recovery_counts_every_transition_it_took(ledger, reopen):
    admitted(ledger, JobState.CLAIMED, inference_id="a")
    admitted(ledger, JobState.CLAIMED, inference_id="b")
    admitted(ledger, JobState.WAITING_MODEL, inference_id="c")
    admitted(ledger, JobState.INFERENCING, inference_id="d")
    admitted(ledger, JobState.COMPLETED, inference_id="done")
    ledger.close()
    store = reopen()
    report = store.recover()
    assert report.transitions == {
        edge_key(JobState.CLAIMED, JobState.READY): 2,
        edge_key(JobState.WAITING_MODEL, JobState.READY): 1,
        edge_key(JobState.INFERENCING, JobState.READY): 1,
    }
    assert report.total == 4
    assert store.get("done").state is JobState.COMPLETED


def test_recovery_is_idempotent(ledger, reopen):
    admitted(ledger, JobState.INFERENCING)
    ledger.close()
    store = reopen()
    assert store.recover().total == 1
    assert store.recover().total == 0
    assert store.get("job-1").state is JobState.READY


def test_terminal_and_blocked_jobs_are_never_restarted(ledger, reopen):
    for state in (
        JobState.QUARANTINED,
        JobState.RETAINED_FAILED,
        JobState.COMPLETED,
        JobState.BLOCKED_CONFIGURATION,
        JobState.PUBLISH_EXHAUSTED,
        JobState.INPUT_INVALID,
    ):
        admitted(ledger, state, inference_id=state.value)
    ledger.close()
    store = reopen()
    assert store.recover().transitions == {}


def test_plan_recovery_is_pure():
    rows = [
        ("a", JobState.INFERENCING.value, None),
        ("b", JobState.RETRY_WAIT.value, 50),
        ("c", JobState.RETRY_WAIT.value, 500),
        ("d", JobState.COMPLETED.value, None),
    ]
    moves = plan_recovery(rows, now_ms=100)
    assert moves == [
        RecoveryMove("a", JobState.INFERENCING, JobState.READY),
        RecoveryMove("b", JobState.RETRY_WAIT, JobState.READY),
    ]


def test_an_empty_report_reads_cleanly():
    report = RecoveryReport()
    assert report.total == 0
    assert report.count(JobState.INFERENCING, JobState.READY) == 0
    assert report.cleanup_pending == ()
