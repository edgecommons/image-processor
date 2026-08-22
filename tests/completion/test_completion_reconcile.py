"""Reconciliation: every DESIGN.md §7 observed-state rule, one test each."""

from pathlib import Path

import pytest
from completion_support import (
    ARCHIVE,
    FAILED,
    RELATIVE,
    SPOOL,
    FakeFs,
    Policy,
    StepClock,
    admitted,
    key,
)

from image_processor.completion import Completer
from image_processor.ledger import Ledger
from image_processor.types import JobState

IMAGE = b"image-bytes"
SIDECAR = b'{"image": {"bytes": 11}}'
SOURCE = Path(SPOOL) / RELATIVE
MEMBER = Path(SPOOL) / (RELATIVE + ".json")
TARGET = Path(ARCHIVE) / RELATIVE
MEMBER_TARGET = Path(ARCHIVE) / (RELATIVE + ".json")


@pytest.fixture()
def completer(ledger, fs):
    return Completer(ledger, fs=fs)


def stored_intent(completer, ledger, fs, state=JobState.PUBLISHED, policy=None, members=()):
    """Seed the spool, admit a job, plan its completion, and persist the intent."""
    digest = fs.write(SOURCE, IMAGE)
    job = admitted(ledger, state, sha256=digest)
    intent = completer.plan(job, policy or Policy(), list(members))
    ledger.record_cleanup_intent(intent)
    return intent


def test_source_present_and_target_absent_retries_the_move(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs)
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert fs.files[key(TARGET)] == IMAGE
    assert key(SOURCE) not in fs.files


def test_source_absent_and_a_matching_target_completes(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs)
    fs.files.pop(key(SOURCE))
    fs.write(TARGET, IMAGE)
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert ledger.cleanup_observed("job-1") == "archived"
    assert ("copy", key(SOURCE)) not in fs.calls


def test_both_present_after_a_cross_filesystem_copy_verifies_then_removes(ledger):
    fs = FakeFs(devices=[SPOOL, ARCHIVE])
    completer = Completer(ledger, fs=fs)
    intent = stored_intent(completer, ledger, fs)
    fs.write(TARGET, IMAGE)
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert key(SOURCE) not in fs.files
    assert fs.files[key(TARGET)] == IMAGE


def test_a_target_with_a_different_digest_is_a_collision_failure(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs)
    fs.write(TARGET, b"a different capture")
    assert completer.reconcile(intent) is JobState.CLEANUP_FAILED
    assert "COLLISION" in ledger.last_error("job-1")
    assert key(SOURCE) in fs.files
    assert ledger.cleanup_observed("job-1") is None


def test_a_source_absent_under_a_delete_intent_completes(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs, policy=Policy(on_success="delete"))
    fs.files.pop(key(SOURCE))
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert ledger.cleanup_observed("job-1") == "already-deleted"


def test_a_delete_intent_whose_source_was_replaced_does_not_complete(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs, policy=Policy(on_success="delete"))
    fs.write(SOURCE, b"a different capture")
    assert completer.reconcile(intent) is JobState.CLEANUP_FAILED
    assert key(SOURCE) in fs.files
    assert "SOURCE_REPLACED" in ledger.last_error("job-1")


def test_a_retain_intent_whose_source_is_gone_is_a_failure(completer, ledger, fs):
    intent = stored_intent(
        completer, ledger, fs, state=JobState.PROCESSING_EXHAUSTED, policy=Policy()
    )
    fs.files.pop(key(SOURCE))
    assert completer.reconcile(intent) is JobState.PROCESSING_EXHAUSTED
    assert "SOURCE_MISSING" in ledger.last_error("job-1")
    assert ledger.cleanup_observed("job-1") is None


def test_neither_source_nor_target_present_is_lost_evidence(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs)
    fs.files.pop(key(SOURCE))
    assert completer.reconcile(intent) is JobState.CLEANUP_FAILED
    assert "EVIDENCE_LOST" in ledger.last_error("job-1")


def test_reconciling_a_failed_cleanup_returns_it_to_pending_then_completes(
    completer, ledger, fs
):
    intent = stored_intent(completer, ledger, fs)
    ledger.fail_cleanup("job-1", "COLLISION: earlier attempt")
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED
    assert completer.reconcile(intent) is JobState.COMPLETED


def test_a_half_moved_bundle_finishes_the_remaining_members(completer, ledger, fs):
    fs.write(MEMBER, SIDECAR)
    intent = stored_intent(completer, ledger, fs, members=[MEMBER])
    fs.files.pop(key(SOURCE))
    fs.write(TARGET, IMAGE)
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert fs.files[key(MEMBER_TARGET)] == SIDECAR
    assert key(MEMBER) not in fs.files


def test_reconciling_a_quarantine_takes_the_quarantined_edge(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs, state=JobState.INPUT_INVALID)
    assert Path(intent.target_path) == Path(FAILED) / RELATIVE
    assert completer.reconcile(intent) is JobState.QUARANTINED


def test_reconciliation_is_idempotent(completer, ledger, fs):
    intent = stored_intent(completer, ledger, fs)
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert completer.reconcile(intent) is JobState.COMPLETED
    assert fs.files[key(TARGET)] == IMAGE


def test_reconciliation_survives_a_restart_of_the_ledger(tmp_path, fs):
    db = tmp_path / "state.db"
    first = Ledger(db, synchronous="NORMAL", clock=StepClock())
    try:
        completer = Completer(first, fs=fs)
        stored_intent(completer, first, fs)
    finally:
        first.close()
    second = Ledger(db, synchronous="NORMAL", clock=StepClock())
    try:
        report = second.recover()
        assert [i.inference_id for i in report.cleanup_pending] == ["job-1"]
        resumed = Completer(second, fs=fs)
        assert resumed.reconcile(report.cleanup_pending[0]) is JobState.COMPLETED
        assert fs.files[key(TARGET)] == IMAGE
    finally:
        second.close()


def test_an_os_error_during_reconciliation_is_recorded_as_a_failure(completer, ledger, fs):
    import errno

    intent = stored_intent(completer, ledger, fs)
    fs.fail("makedirs", TARGET.parent, OSError(errno.EACCES, "denied"))
    assert completer.reconcile(intent) is JobState.CLEANUP_FAILED
    assert "FS_ERROR" in ledger.last_error("job-1")


def test_a_per_route_collision_override_applies_to_reconciliation(completer, ledger, fs):
    from image_processor.completion import COLLISION_SUFFIX

    intent = stored_intent(completer, ledger, fs)
    fs.write(TARGET, b"a different capture")
    assert completer.reconcile(intent, on_collision=COLLISION_SUFFIX) is JobState.COMPLETED
    suffixed_target = TARGET.with_name(
        f"{TARGET.stem}.{intent.source_sha256[:8]}{TARGET.suffix}"
    )
    assert fs.files[key(suffixed_target)] == IMAGE
