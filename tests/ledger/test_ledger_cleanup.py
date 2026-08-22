"""Cleanup intents: write-ahead recording, completion edges, and failure never being success."""

import pytest
from ledger_support import admitted

from image_processor.ledger import IllegalTransition, LedgerConflict
from image_processor.types import CleanupIntent, CompletionAction, JobState


def intent(inference_id="job-1", action=CompletionAction.ARCHIVE, target="/processed/a.jpg"):
    """Build a cleanup intent for a test."""
    return CleanupIntent(
        inference_id=inference_id,
        action=action,
        source_path="/spool/a.jpg",
        source_sha256="a" * 64,
        target_path=target,
        members=("/spool/a.jpg.json",),
    )


def test_recording_an_intent_moves_a_published_job_to_cleanup_pending(ledger):
    admitted(ledger, JobState.PUBLISHED)
    job = ledger.record_cleanup_intent(intent())
    assert job.state is JobState.CLEANUP_PENDING
    stored = ledger.cleanup_intent("job-1")
    assert stored == intent()
    assert ledger.cleanup_observed("job-1") is None
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-1"]


def test_completing_cleanup_finishes_the_job_and_records_what_was_observed(ledger):
    admitted(ledger, JobState.PUBLISHED)
    ledger.record_cleanup_intent(intent())
    job = ledger.complete_cleanup("job-1", "archived")
    assert job.state is JobState.COMPLETED
    assert ledger.cleanup_observed("job-1") == "archived"
    assert ledger.pending_cleanup(10) == []


def test_failing_cleanup_is_never_success(ledger):
    admitted(ledger, JobState.PUBLISHED)
    ledger.record_cleanup_intent(intent())
    job = ledger.fail_cleanup("job-1", "COLLISION: target holds a different object")
    assert job.state is JobState.CLEANUP_FAILED
    assert ledger.cleanup_observed("job-1") is None
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-1"]
    assert "COLLISION" in ledger.last_error("job-1")


def test_a_failed_cleanup_is_retried_by_re_recording_the_intent(ledger):
    admitted(ledger, JobState.PUBLISHED)
    ledger.record_cleanup_intent(intent())
    ledger.fail_cleanup("job-1", "fs error")
    retried = ledger.record_cleanup_intent(intent(target="/processed/a.deadbeef.jpg"))
    assert retried.state is JobState.CLEANUP_PENDING
    assert ledger.cleanup_intent("job-1").target_path == "/processed/a.deadbeef.jpg"
    assert ledger.complete_cleanup("job-1", "archived").state is JobState.COMPLETED


def test_quarantine_of_an_invalid_input_keeps_its_direct_terminal_edge(ledger):
    admitted(ledger, JobState.INPUT_INVALID)
    job = ledger.record_cleanup_intent(intent(action=CompletionAction.QUARANTINE))
    assert job.state is JobState.INPUT_INVALID
    assert ledger.complete_cleanup("job-1", "quarantined").state is JobState.QUARANTINED


def test_a_failed_quarantine_stays_retryable_rather_than_terminal(ledger):
    admitted(ledger, JobState.INPUT_INVALID)
    ledger.record_cleanup_intent(intent(action=CompletionAction.QUARANTINE))
    job = ledger.fail_cleanup("job-1", "COLLISION")
    assert job.state is JobState.INPUT_INVALID
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-1"]
    assert ledger.complete_cleanup("job-1", "quarantined").state is JobState.QUARANTINED


def test_retain_of_an_exhausted_job_keeps_its_direct_terminal_edge(ledger):
    admitted(ledger, JobState.PROCESSING_EXHAUSTED)
    job = ledger.record_cleanup_intent(intent(action=CompletionAction.RETAIN, target=None))
    assert job.state is JobState.PROCESSING_EXHAUSTED
    assert ledger.complete_cleanup("job-1", "retained").state is JobState.RETAINED_FAILED


def test_cleanup_never_runs_from_an_unrelated_state(ledger):
    admitted(ledger, JobState.READY)
    with pytest.raises(IllegalTransition):
        ledger.record_cleanup_intent(intent())
    with pytest.raises(IllegalTransition):
        ledger.complete_cleanup("job-1", "archived")


def test_cleanup_on_a_missing_job_conflicts(ledger):
    for call in (
        lambda: ledger.record_cleanup_intent(intent(inference_id="nope")),
        lambda: ledger.complete_cleanup("nope", "archived"),
        lambda: ledger.fail_cleanup("nope", "error"),
    ):
        with pytest.raises(LedgerConflict):
            call()
    assert ledger.cleanup_intent("nope") is None
    assert ledger.cleanup_observed("nope") is None


def test_pending_cleanup_is_ordered_and_limited(ledger):
    for index in range(3):
        admitted(ledger, JobState.PUBLISHED, inference_id=f"job-{index}")
        ledger.record_cleanup_intent(intent(inference_id=f"job-{index}"))
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-0", "job-1", "job-2"]
    assert len(ledger.pending_cleanup(2)) == 2
    ledger.complete_cleanup("job-0", "archived")
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-1", "job-2"]


def test_recording_an_intent_while_already_pending_is_idempotent(ledger):
    admitted(ledger, JobState.PUBLISHED)
    ledger.record_cleanup_intent(intent())
    job = ledger.record_cleanup_intent(intent())
    assert job.state is JobState.CLEANUP_PENDING
    assert len(ledger.pending_cleanup(10)) == 1
