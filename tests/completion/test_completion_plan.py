"""Planning: deterministic targets, action selection, and path safety."""

import pytest
from completion_support import ARCHIVE, FAILED, RELATIVE, SPOOL, Policy, build_job

from image_processor.completion import CleanupError, Completer, coerce_action, safe_relative
from image_processor.completion.actions import suffixed
from image_processor.types import CompletionAction, JobState
from pathlib import Path


@pytest.fixture()
def completer(ledger, fs):
    return Completer(ledger, fs=fs)


def test_archive_target_preserves_the_relative_subtree(completer, policy):
    job = build_job(state=JobState.PUBLISHED)
    intent = completer.plan(job, policy, [])
    assert intent.action is CompletionAction.ARCHIVE
    assert Path(intent.source_path) == Path(SPOOL) / RELATIVE
    assert Path(intent.target_path) == Path(ARCHIVE) / RELATIVE
    assert intent.source_sha256 == job.source.sha256
    assert intent.members == ()


def test_quarantine_target_lands_under_the_failed_directory(completer, policy):
    intent = completer.plan(
        build_job(state=JobState.INPUT_INVALID),
        policy,
        [Path(SPOOL) / (RELATIVE + ".json")],
    )
    assert intent.action is CompletionAction.QUARANTINE
    assert Path(intent.target_path) == Path(FAILED) / RELATIVE
    assert intent.members == (str(Path(SPOOL) / (RELATIVE + ".json")),)


def test_retain_and_delete_have_no_target(completer, policy):
    exhausted = completer.plan(build_job(state=JobState.PROCESSING_EXHAUSTED), policy, [])
    assert exhausted.action is CompletionAction.RETAIN
    assert exhausted.target_path is None
    deleting = completer.plan(
        build_job(state=JobState.PUBLISHED),
        Policy(on_success="delete"),
        [],
    )
    assert deleting.action is CompletionAction.DELETE
    assert deleting.target_path is None


def test_publish_failure_uses_its_own_setting(completer, policy):
    intent = completer.plan(build_job(state=JobState.PUBLISH_EXHAUSTED), policy, [])
    assert intent.action is CompletionAction.RETAIN


def test_planning_is_deterministic(completer, policy):
    job = build_job(state=JobState.PUBLISHED)
    assert completer.plan(job, policy, []) == completer.plan(job, policy, [])


def test_a_state_with_no_configured_completion_is_refused(completer, policy):
    with pytest.raises(CleanupError) as caught:
        completer.plan(build_job(state=JobState.READY), policy, [])
    assert caught.value.code == "NO_COMPLETION_FOR_STATE"


def test_a_missing_completion_directory_is_refused(completer):
    with pytest.raises(CleanupError) as caught:
        completer.plan(
            build_job(state=JobState.PUBLISHED), Policy(archive_dir=None), []
        )
    assert caught.value.code == "COMPLETION_DIR_MISSING"
    with pytest.raises(CleanupError):
        completer.plan(build_job(state=JobState.INPUT_INVALID), Policy(failed_dir=""), [])


def test_an_unknown_action_is_refused(completer):
    with pytest.raises(CleanupError) as caught:
        completer.plan(build_job(state=JobState.PUBLISHED), Policy(on_success="incinerate"), [])
    assert caught.value.code == "UNKNOWN_COMPLETION_ACTION"


def test_coerce_action_accepts_config_and_enum_spellings():
    assert coerce_action(CompletionAction.ARCHIVE) is CompletionAction.ARCHIVE
    assert coerce_action("archive") is CompletionAction.ARCHIVE
    assert coerce_action("ARCHIVE") is CompletionAction.ARCHIVE
    assert coerce_action("retainInPlace") is CompletionAction.RETAIN
    assert coerce_action("retain-in-place") is CompletionAction.RETAIN
    assert coerce_action("retain") is CompletionAction.RETAIN


@pytest.mark.parametrize(
    "unsafe",
    ["", "   ", "/etc/passwd", "C:/windows/system32/a.jpg", "../escape.jpg", "a/../../b.jpg"],
)
def test_unsafe_relative_paths_are_refused(completer, policy, unsafe):
    with pytest.raises(CleanupError) as caught:
        completer.plan(build_job(state=JobState.PUBLISHED, relative_path=unsafe), policy, [])
    assert caught.value.code == "UNSAFE_RELATIVE_PATH"


BACKSLASH = chr(92)
WINDOWS_RELATIVE = BACKSLASH.join(["2026", "08", "22", "a.jpg"])


def test_windows_separators_normalize_to_the_same_target(completer, policy):
    forward = completer.plan(
        build_job(state=JobState.PUBLISHED, relative_path="2026/08/22/a.jpg"), policy, []
    )
    backward = completer.plan(
        build_job(state=JobState.PUBLISHED, relative_path=WINDOWS_RELATIVE), policy, []
    )
    assert Path(forward.target_path) == Path(backward.target_path)
    assert safe_relative(WINDOWS_RELATIVE).as_posix() == "2026/08/22/a.jpg"


def test_the_collision_suffix_is_deterministic():
    target = Path("/evidence/2026/a.jpg")
    assert suffixed(target, "sha256:deadbeefcafebabe") == Path("/evidence/2026/a.deadbeef.jpg")
    assert suffixed(target, "deadbeefcafebabe") == suffixed(target, "sha256:deadbeefcafebabe")
