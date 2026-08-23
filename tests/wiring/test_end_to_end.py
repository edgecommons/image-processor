"""The whole component, in process: image in, confirmed decision out, image archived.

These are the tests that would notice a reordering of the durability chain. They run the real
configuration parser, the real ledger, the real bundle cache, a real executor cell on
``CPUExecutionProvider``, the real scheduler, and the real outputs, against a fake bus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_processor.types import JobState
from tests.wiring.conftest import write_capture


def _drain(app, attempts: int = 200) -> int:
    """Run scheduling passes until the queue is empty."""
    dispatched = 0
    for _ in range(attempts):
        dispatched += app._scheduler.run_once()
        if not app._scheduler.queued():
            break
    return dispatched


def _one_job(app):
    """Return the single job the ledger holds."""
    jobs, _cursor = app._ledger.by_state(list(JobState), None, None, 10)
    assert len(jobs) == 1, [job.inference_id for job in jobs]
    return jobs[0]


def test_a_spool_image_becomes_a_confirmed_result_and_an_archived_file(running, gg, corpus, home):
    good = corpus.image("anomaly-good.png")
    write_capture(home / "spool", "2026/08/22/cap-0001.png", good)

    assert running._source_of("clearance-cam-01").rescan() == 1
    _drain(running)
    assert running._publisher.drain_once() == 1

    job = _one_job(running)
    assert job.state is JobState.COMPLETED

    results = [item for item in gg.messaging.published if item.topic.endswith("app/inference/result")]
    assert len(results) == 1
    body = results[0].body
    assert results[0].confirmed is True
    assert body["status"] == "SUCCEEDED"
    assert body["decision"]["outcome"] == "CLEAR"
    assert body["decision"]["pass"] is True
    assert body["outputs"]["anomaly"]["anomalous"] is False
    assert body["model"]["providers"] == ["CPUExecutionProvider"]
    assert body["source"]["captureId"] == "cap-0001"
    assert body["source"]["relativePath"] == "2026/08/22/cap-0001.png"

    # the decision mirror, best effort, on the data class
    mirrored = {item.topic.rsplit("data/", 1)[-1]: item.body for item in gg.messaging.published
                if "/data/" in item.topic}
    assert set(mirrored) == {"line-clearance/pass", "line-clearance/status"}
    assert mirrored["line-clearance/pass"]["samples"][0]["value"] is True

    # the evidence sidecar, installed beside the image and named by the published result
    archived = home / "processed" / "2026/08/22/cap-0001.png"
    assert archived.is_file()
    assert not (home / "spool" / "2026/08/22/cap-0001.png").exists()
    sidecar = archived.with_name(archived.name + ".inference.json")
    assert sidecar.is_file()
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["result"]["decision"]["outcome"] == "CLEAR"
    assert body["artifacts"]["sha256"] == app_sidecar_digest(sidecar)
    # the camera sidecar travelled with the image
    assert archived.with_name(archived.name + ".json").is_file()


def app_sidecar_digest(path: Path) -> str:
    """Return the digest of an installed sidecar."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_bad_image_holds_and_still_archives(running, gg, corpus, home):
    write_capture(home / "spool", "bad.png", corpus.image("anomaly-bad.png"), capture_id="cap-bad")

    running._source_of("clearance-cam-01").rescan()
    _drain(running)
    running._publisher.drain_once()

    body = gg.messaging.bodies("app/inference/result")[0]
    assert body["decision"]["outcome"] == "HOLD"
    assert body["decision"]["pass"] is False
    assert body["outputs"]["anomaly"]["anomalous"] is True
    assert _one_job(running).state is JobState.COMPLETED


def _trigger(app, body, *, correlation_id=None, reply_to=None) -> None:
    """Deliver one message to the trigger route as its subscription would."""
    from edgecommons.messaging.message import Message, MessageHeader

    header = MessageHeader("InspectRequest", "1.0", correlation_id=correlation_id)
    header.reply_to = reply_to
    app._source_of("adhoc-inspect").on_message(Message(header=header, body=body))


def test_an_inline_trigger_image_is_staged_inferred_and_deleted(running, gg, corpus, home):
    _trigger(running, corpus.image("anomaly-good.png"))

    _drain(running)
    running._publisher.drain_once()

    body = gg.messaging.bodies("app/inference/result")[0]
    assert body["source"]["kind"] == "inline"
    assert body["decision"]["outcome"] == "CLEAR"
    assert _one_job(running).state is JobState.COMPLETED
    staged = list((home / "staging" / "adhoc").rglob("*.png"))
    assert staged == [], "an inline job deletes its staged copy on success"


def test_a_file_reference_trigger_answers_the_requester(running, gg, corpus, home):
    import hashlib

    data = corpus.image("anomaly-bad.png")
    (home / "inbox" / "batch").mkdir(parents=True)
    (home / "inbox" / "batch" / "part.png").write_bytes(data)
    _trigger(
        running,
        {
            "relativePath": "batch/part.png",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
        correlation_id="corr-7",
        reply_to="ecv1/smoke-device/inspection-ui/panel/app/inspect/reply",
    )

    _drain(running)
    running._publisher.drain_once()

    replies = [item for item in gg.messaging.published if item.topic.endswith("inspect/reply")]
    assert len(replies) == 1
    assert replies[0].body["decision"]["outcome"] == "HOLD"
    assert replies[0].body["source"]["kind"] == "reference"
    assert replies[0].body["source"]["correlationId"] == "corr-7"


def test_a_crash_after_the_commit_publishes_on_the_next_start(gg, document, corpus, home):
    from image_processor.ImageProcessor import ImageProcessor

    first = ImageProcessor(gg)
    first._supervisor.start()
    first._artifacts.reconcile()
    first._start_routes()
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    first._source_of("clearance-cam-01").rescan()
    _drain(first)
    committed = _one_job(first)
    assert committed.state is JobState.PUBLISH_PENDING
    assert gg.messaging.bodies("app/inference/result") == [], (
        "the authoritative result is published only by the outbox drain"
    )
    first.stop()

    # the process comes back with the same durable state and finishes the job
    second = ImageProcessor(gg)
    try:
        second._supervisor.start()
        report = second._recover()
        assert report.publish_pending == (committed.inference_id,)
        assert second._publisher.drain_once() == 1
        assert _one_job(second).state is JobState.COMPLETED
        assert (home / "processed" / "cap.png").is_file()
    finally:
        second.stop()


def test_a_broker_outage_leaves_the_result_pending_and_the_image_in_place(
    running, gg, corpus, home
):
    write_capture(home / "spool", "outage.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    _drain(running)

    gg.messaging.fail_confirmed = TimeoutError("no PUBACK")
    assert running._publisher.drain_once() == 0
    assert _one_job(running).state is JobState.PUBLISH_PENDING
    assert (home / "spool" / "outage.png").is_file(), "cleanup never runs before confirmation"

    gg.messaging.fail_confirmed = None
    assert running._publisher.drain_once() == 1
    assert _one_job(running).state is JobState.COMPLETED
    assert (home / "processed" / "outage.png").is_file()


def test_a_file_reference_leaves_no_staged_copy_behind(running, gg, corpus, home):
    import hashlib

    data = corpus.image("anomaly-good.png")
    (home / "inbox" / "batch").mkdir(parents=True)
    (home / "inbox" / "batch" / "part.png").write_bytes(data)
    _trigger(
        running,
        {
            "relativePath": "batch/part.png",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
    )

    _drain(running)
    running._publisher.drain_once()

    assert _one_job(running).state is JobState.COMPLETED
    assert not (home / "inbox" / "batch" / "part.png").exists(), "the route deletes on success"
    assert list((home / "staging").rglob("*.png")) == [], "the staged copy went with the job"
