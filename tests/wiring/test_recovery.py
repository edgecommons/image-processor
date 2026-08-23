"""Restart recovery: the ledger and the filesystem are reconciled before anything publishes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from image_processor.types import JobState
from tests.wiring.conftest import write_capture


def _run_one(app, home, corpus, name: str = "cap.png") -> str:
    """Take one image through inference and the commit, leaving it unpublished."""
    write_capture(home / "spool", name, corpus.image("anomaly-good.png"))
    app._source_of("clearance-cam-01").rescan()
    for _ in range(50):
        app._scheduler.run_once()
        if not app._scheduler.queued():
            break
    jobs, _cursor = app._ledger.by_state([JobState.PUBLISH_PENDING], None, None, 5)
    assert len(jobs) == 1
    return jobs[0].inference_id


def test_a_committed_result_whose_evidence_is_gone_returns_to_re_inference(running, home, corpus):
    inference_id = _run_one(running, home, corpus)
    sidecar = home / "spool" / "cap.png.inference.json"
    assert sidecar.is_file()
    sidecar.unlink()

    report = running._recover()

    assert [record.inference_id for record in report.sidecars] == [inference_id]
    job = running._ledger.get(inference_id)
    assert job.state is JobState.READY
    assert running._ledger.result_bytes(inference_id) is None
    assert running._ledger.outbox_for(inference_id) == []


def test_evidence_that_no_longer_matches_is_removed_and_re_inferred(running, home, corpus):
    inference_id = _run_one(running, home, corpus)
    sidecar = home / "spool" / "cap.png.inference.json"
    sidecar.write_text('{"tampered": true}', encoding="utf-8")

    running._recover()

    assert not sidecar.exists()
    assert running._ledger.get(inference_id).state is JobState.READY


def test_intact_evidence_is_adopted_and_the_result_still_publishes(running, home, corpus, gg):
    inference_id = _run_one(running, home, corpus)

    running._recover()

    assert running._ledger.get(inference_id).state is JobState.PUBLISH_PENDING
    assert running._publisher.drain_once() == 1
    assert running._ledger.get(inference_id).state is JobState.COMPLETED


def test_an_interrupted_archive_is_decided_from_what_the_filesystem_shows(running, home, corpus, gg):
    inference_id = _run_one(running, home, corpus)
    running._publisher.drain_once()
    archived = home / "processed" / "cap.png"
    assert archived.is_file()

    # the intent is still on record; reconciling it again must not undo a completed move
    intent = running._ledger.cleanup_intent(inference_id)
    running._reconcile_cleanup(intent)

    assert archived.is_file()
    assert running._ledger.get(inference_id).state is JobState.COMPLETED


def test_a_cleanup_that_cannot_run_is_reported_and_repairable(running, home, corpus, gg):
    inference_id = _run_one(running, home, corpus)
    # something else already occupies the archive target, and it is not the same object
    occupied = home / "processed" / "cap.png"
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_bytes(b"an unrelated file")

    running._publisher.drain_once()

    assert running._ledger.get(inference_id).state is JobState.CLEANUP_FAILED
    assert gg.find_event("cleanup-failed") is not None
    assert (home / "spool" / "cap.png").is_file(), "neither object was overwritten"

    occupied.unlink()
    repaired = running.retry_cleanup(None, inference_id)

    assert repaired == {"repaired": [inference_id], "stillFailed": []}
    assert running._ledger.get(inference_id).state is JobState.COMPLETED
    assert occupied.is_file()


def test_recovery_resubmits_what_it_restarted(running, home, corpus):
    write_capture(home / "spool", "later.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    jobs, _cursor = running._ledger.by_state([JobState.READY], None, None, 5)
    assert len(jobs) == 1

    running._resubmit()

    assert running._scheduler.queued() >= 1
