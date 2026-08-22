"""Applying cleanup: intents before mutation, atomic moves, bundles, and failure handling."""

import errno
import json
from pathlib import Path

import pytest
from completion_support import ARCHIVE, FAILED, RELATIVE, SPOOL, FakeFs, Policy, admitted, key

from image_processor.completion import (
    BUNDLE_MANIFEST_SUFFIX,
    COLLISION_SUFFIX,
    ERROR_RECORD_SUFFIX,
    CleanupError,
    Completer,
)
from image_processor.types import JobState

IMAGE = b"image-bytes"
SIDECAR = b'{"image": {"bytes": 11}}'
SOURCE = Path(SPOOL) / RELATIVE
MEMBER = Path(SPOOL) / (RELATIVE + ".json")
TARGET = Path(ARCHIVE) / RELATIVE


def seed(fs: FakeFs, member: bool = True):
    """Seed the spool with an image and, by default, its camera sidecar."""
    digest = fs.write(SOURCE, IMAGE)
    if member:
        fs.write(MEMBER, SIDECAR)
    return digest


def planned(completer, ledger, fs, state=JobState.PUBLISHED, policy=None, members=None):
    """Admit a job in ``state`` and plan its completion."""
    digest = seed(fs, member=members is not None)
    job = admitted(ledger, state, sha256=digest)
    return completer.plan(job, policy or Policy(), members or [])


@pytest.fixture()
def completer(ledger, fs):
    return Completer(ledger, fs=fs)


def test_archive_renames_within_one_filesystem(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    completer.apply(intent)
    assert key(TARGET) in fs.files
    assert key(SOURCE) not in fs.files
    assert fs.files[key(TARGET)] == IMAGE
    assert ledger.get("job-1").state is JobState.COMPLETED
    assert ledger.cleanup_observed("job-1") == "archived"
    assert ("fsync_dir", key(TARGET.parent)) in fs.calls


def test_the_intent_is_persisted_before_the_first_mutation(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    fs.fail("replace", SOURCE, OSError(errno.EACCES, "denied"))
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "FS_ERROR"
    assert ledger.cleanup_intent("job-1") == intent
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED
    assert key(SOURCE) in fs.files
    assert key(TARGET) not in fs.files


def test_a_multi_member_move_installs_the_bundle_manifest_last(completer, ledger, fs):
    intent = planned(completer, ledger, fs, members=[MEMBER])
    completer.apply(intent)
    member_target = Path(ARCHIVE) / (RELATIVE + ".json")
    manifest_path = TARGET.with_name(TARGET.name + BUNDLE_MANIFEST_SUFFIX)
    assert fs.files[key(member_target)] == SIDECAR
    manifest = json.loads(fs.files[key(manifest_path)])
    assert manifest["image"] == TARGET.name
    assert [m["path"] for m in manifest["members"]] == [TARGET.name, member_target.name]
    writes = [c for c in fs.calls if c[0] in ("replace", "write_bytes")]
    assert writes[-1] == ("write_bytes", key(manifest_path))


def test_a_single_file_move_writes_no_manifest(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    completer.apply(intent)
    assert key(TARGET.with_name(TARGET.name + BUNDLE_MANIFEST_SUFFIX)) not in fs.files


def test_a_cross_filesystem_move_copies_verifies_then_removes(ledger):
    fs = FakeFs(devices=[SPOOL, ARCHIVE])
    completer = Completer(ledger, fs=fs)
    intent = planned(completer, ledger, fs)
    completer.apply(intent)
    assert fs.files[key(TARGET)] == IMAGE
    assert key(SOURCE) not in fs.files
    assert ("copy", key(SOURCE)) in fs.calls
    assert ledger.get("job-1").state is JobState.COMPLETED


def test_a_cross_filesystem_copy_that_does_not_verify_fails_and_removes_the_copy(ledger):
    fs = FakeFs(devices=[SPOOL, ARCHIVE])
    completer = Completer(ledger, fs=fs)
    intent = planned(completer, ledger, fs)
    fs.corrupt_copy(TARGET)
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "COPY_VERIFY_FAILED"
    assert key(TARGET) not in fs.files
    assert key(SOURCE) in fs.files
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED


def test_a_colliding_target_fails_by_default(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    fs.write(TARGET, b"a different capture")
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "COLLISION"
    assert fs.files[key(TARGET)] == b"a different capture"
    assert key(SOURCE) in fs.files
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED
    assert "COLLISION" in ledger.last_error("job-1")


def test_a_colliding_target_takes_a_deterministic_suffix_when_configured(ledger, fs):
    completer = Completer(ledger, fs=fs, on_collision=COLLISION_SUFFIX)
    intent = planned(completer, ledger, fs)
    fs.write(TARGET, b"a different capture")
    completer.apply(intent)
    digest = intent.source_sha256[:8]
    suffixed_target = TARGET.with_name(f"{TARGET.stem}.{digest}{TARGET.suffix}")
    assert fs.files[key(suffixed_target)] == IMAGE
    assert fs.files[key(TARGET)] == b"a different capture"
    assert ledger.get("job-1").state is JobState.COMPLETED


def test_a_target_already_holding_the_same_object_finishes_the_move(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    fs.write(TARGET, IMAGE)
    completer.apply(intent)
    assert key(SOURCE) not in fs.files
    assert fs.files[key(TARGET)] == IMAGE
    assert ledger.get("job-1").state is JobState.COMPLETED


def test_delete_removes_the_image_and_its_members(completer, ledger, fs):
    intent = planned(
        completer, ledger, fs, policy=Policy(on_success="delete"), members=[MEMBER]
    )
    completer.apply(intent)
    assert key(SOURCE) not in fs.files
    assert key(MEMBER) not in fs.files
    assert ledger.get("job-1").state is JobState.COMPLETED
    assert ledger.cleanup_observed("job-1") == "deleted"


def test_delete_refuses_when_the_source_was_replaced(completer, ledger, fs):
    intent = planned(completer, ledger, fs, policy=Policy(on_success="delete"))
    fs.write(SOURCE, b"a different capture")
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "SOURCE_REPLACED"
    assert key(SOURCE) in fs.files
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED


def test_retain_leaves_the_file_alone(completer, ledger, fs):
    intent = planned(
        completer, ledger, fs, state=JobState.PROCESSING_EXHAUSTED, policy=Policy()
    )
    completer.apply(intent)
    assert fs.files[key(SOURCE)] == IMAGE
    assert ledger.get("job-1").state is JobState.RETAINED_FAILED
    assert ledger.cleanup_observed("job-1") == "retained"


def test_retain_fails_when_the_evidence_is_gone(completer, ledger, fs):
    intent = planned(completer, ledger, fs, state=JobState.PROCESSING_EXHAUSTED)
    fs.files.pop(key(SOURCE))
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "SOURCE_MISSING"
    assert ledger.get("job-1").state is JobState.PROCESSING_EXHAUSTED
    assert [i.inference_id for i in ledger.pending_cleanup(10)] == ["job-1"]


def test_quarantine_bundles_the_image_members_and_an_error_record(completer, ledger, fs):
    digest = seed(fs)
    job = admitted(ledger, JobState.INPUT_INVALID, sha256=digest)
    ledger.transition(
        job.inference_id,
        JobState.INPUT_INVALID,
        JobState.INPUT_INVALID,
        last_error="DECODE_FAILED: truncated JPEG",
    )
    intent = completer.plan(ledger.get("job-1"), Policy(), [MEMBER])
    completer.apply(intent)
    image_target = Path(FAILED) / RELATIVE
    record_path = image_target.with_name(image_target.name + ERROR_RECORD_SUFFIX)
    manifest_path = image_target.with_name(image_target.name + BUNDLE_MANIFEST_SUFFIX)
    assert fs.files[key(image_target)] == IMAGE
    assert fs.files[key(Path(FAILED) / (RELATIVE + ".json"))] == SIDECAR
    record = json.loads(fs.files[key(record_path)])
    assert record["inferenceId"] == "job-1"
    assert record["action"] == "quarantine"
    assert record["error"] == "DECODE_FAILED: truncated JPEG"
    assert record["source"]["sha256"] == digest
    assert record["job"]["state"] == JobState.INPUT_INVALID.value
    assert record["job"]["model"]["digest"].startswith("sha256:")
    manifest = json.loads(fs.files[key(manifest_path)])
    assert record_path.name in [m["path"] for m in manifest["members"]]
    assert ledger.get("job-1").state is JobState.QUARANTINED


def test_a_member_from_another_directory_lands_beside_the_image(completer, ledger, fs):
    stray = Path("/elsewhere/notes.txt")
    fs.write(stray, b"notes")
    intent = planned(completer, ledger, fs, members=[stray])
    completer.apply(intent)
    assert fs.files[key(TARGET.parent / "notes.txt")] == b"notes"


def test_a_nested_member_keeps_its_subtree(completer, ledger, fs):
    nested = SOURCE.parent / "thumbs" / "capture-1.jpg"
    fs.write(nested, b"thumb")
    intent = planned(completer, ledger, fs, members=[nested])
    completer.apply(intent)
    assert fs.files[key(TARGET.parent / "thumbs" / "capture-1.jpg")] == b"thumb"


def test_a_source_digest_that_changed_before_the_move_is_refused(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    fs.write(SOURCE, b"rewritten in place")
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "SOURCE_DIGEST_MISMATCH"
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED


def test_a_move_with_no_target_is_refused(completer, ledger, fs):
    from image_processor.types import CleanupIntent, CompletionAction

    digest = seed(fs)
    admitted(ledger, JobState.PUBLISHED, sha256=digest)
    broken = CleanupIntent(
        inference_id="job-1",
        action=CompletionAction.ARCHIVE,
        source_path=str(SOURCE),
        source_sha256=digest,
        target_path=None,
        members=(),
    )
    with pytest.raises(CleanupError) as caught:
        completer.apply(broken)
    assert caught.value.code == "NO_TARGET"


def test_an_unexpected_os_error_is_reported_as_a_cleanup_failure(completer, ledger, fs):
    intent = planned(completer, ledger, fs)
    fs.fail("sha256", SOURCE, OSError(errno.EIO, "read error"))
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "FS_ERROR"
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED


def test_a_member_that_vanished_before_the_move_fails_the_cleanup(completer, ledger, fs):
    missing = Path(SPOOL) / (RELATIVE + ".inference.json")
    intent = planned(completer, ledger, fs, members=[missing])
    with pytest.raises(CleanupError) as caught:
        completer.apply(intent)
    assert caught.value.code == "SOURCE_MISSING"
    assert ledger.get("job-1").state is JobState.CLEANUP_FAILED
