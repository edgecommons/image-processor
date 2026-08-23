"""Bring-up, shutdown, and the paths that only happen when something goes wrong."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from image_processor.types import JobState
from tests.wiring.conftest import config_document, write_capture


def _drain(app) -> None:
    """Run the scheduler until the queue is empty."""
    for _ in range(200):
        app._scheduler.run_once()
        if not app._scheduler.queued():
            return


def test_run_brings_everything_up_and_stop_drains_it_in_order(app, gg, home, corpus):
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not gg.ready[-1:] == [True] and time.monotonic() < deadline:
        time.sleep(0.05)

    assert gg.ready[-1] is True
    assert app._supervisor.healthy() is True
    assert set(app._routes) == {"clearance-cam-01", "adhoc-inspect"}
    assert gg.messaging.subscriptions, "the trigger route and the camera hints are subscribed"

    app.stop()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert gg.messaging.subscriptions == {}, "every subscription was dropped on the way out"
    assert app._supervisor.cells() == []
    app.stop()  # idempotent


def test_a_camera_hint_admits_the_capture_without_waiting_for_a_walk(running, gg, home, corpus):
    from edgecommons.messaging.message import Message, MessageHeader

    body = write_capture(home / "spool", "hint.png", corpus.image("anomaly-good.png"))
    topic = "ecv1/smoke-device/camera-adapter/cam-01/app/image/captured"
    assert topic in gg.messaging.subscriptions

    gg.messaging.subscriptions[topic](topic, Message(header=MessageHeader("ImageCaptured", "1.0"), body=body))

    assert running.job_counts("clearance-cam-01") == {"READY": 1}
    assert running._source_of("clearance-cam-01").hints_accepted == 1


def test_a_hint_that_cannot_be_used_never_loses_a_job(running, gg):
    topic = "ecv1/smoke-device/camera-adapter/cam-01/app/image/captured"

    gg.messaging.subscriptions[topic](topic, object())

    assert running.job_counts() == {}


def test_a_subscription_the_broker_refuses_is_reported_and_the_walk_still_covers_it(
    home, corpus, gg
):
    from image_processor.ImageProcessor import ImageProcessor

    def _refuse(topic, callback, *args, **kwargs):
        raise RuntimeError("the broker refused the subscription")

    gg.messaging.subscribe = _refuse
    app = ImageProcessor(gg)
    try:
        app._start_routes()

        assert app._routes["clearance-cam-01"].topics == ()
        assert app._routes["adhoc-inspect"].topics == ()
    finally:
        app.stop()


def test_the_capture_status_reconciler_asks_the_camera_over_the_bus(running, gg, home, corpus):
    from edgecommons.messaging.message import Message, MessageHeader

    reconciler = running._routes["clearance-cam-01"].reconciler
    assert reconciler is not None
    body = write_capture(home / "spool", "status.png", corpus.image("anomaly-good.png"))
    gg.messaging.reply_factory = lambda topic, message: Message(
        header=MessageHeader("sb/capture-status", "1.0"),
        body={
            "ok": True,
            "result": {
                "jobs": [
                    {
                        "captureId": body["captureId"],
                        "instance": "cam-01",
                        "state": "SUCCEEDED",
                        "terminalAtMs": 1789012506010,
                        "result": body,
                    }
                ],
                "nextCursor": None,
            },
        },
    )

    assert reconciler.poll_once() == 1

    topic, request = gg.messaging.requests[0]
    assert topic == "ecv1/smoke-device/camera-adapter/cam-01/cmd/sb/capture-status"
    assert request["states"] == ["SUCCEEDED"]
    assert reconciler.lookup("status.png") is not None


def test_a_camera_that_does_not_answer_is_recorded_rather_than_raised(running, gg):
    reconciler = running._routes["clearance-cam-01"].reconciler

    assert reconciler.poll_once() == 0
    assert reconciler.last_error is not None


def test_an_undecodable_image_fails_holds_and_is_quarantined(running, gg, home):
    write_capture(home / "spool", "broken.png", b"this is not an image at all")

    running._source_of("clearance-cam-01").rescan()
    _drain(running)

    body = gg.messaging.bodies("app/inference/result")[0]
    assert body["status"] == "FAILED"
    assert body["decision"]["outcome"] == "HOLD"
    assert body["error"]["class"] == "permanent"
    assert gg.find_event("inference-failed") is not None
    # a permanent failure is a property of the input, so it takes the invalid-input action
    assert (home / "failed" / "broken.png").is_file()
    jobs, _cursor = running._ledger.by_state([JobState.RETAINED_FAILED], None, None, 5)
    assert len(jobs) == 1


def test_a_failed_trigger_job_still_answers_the_requester(running, gg, home):
    from edgecommons.messaging.message import Message, MessageHeader

    header = MessageHeader("InspectRequest", "1.0", correlation_id="corr-9")
    header.reply_to = "ecv1/smoke-device/inspection-ui/panel/app/inspect/reply"
    running._source_of("adhoc-inspect").on_message(
        Message(header=header, body=b"this is not an image at all")
    )

    _drain(running)

    replies = [item for item in gg.messaging.published if item.topic.endswith("inspect/reply")]
    assert replies[0].body["status"] == "FAILED"
    assert replies[0].body["decision"]["outcome"] == "HOLD"


def test_a_result_over_the_budget_writes_evidence_even_when_the_route_configures_none(
    home, corpus, gg, monkeypatch
):
    from image_processor.ImageProcessor import ImageProcessor
    from image_processor.outputs import ResultLimits

    document = config_document(home, corpus, write_sidecar=False, trigger=False)
    gg.config_manager.document = document
    app = ImageProcessor(gg, limits=ResultLimits(max_items=100, max_body_bytes=64))
    try:
        app._supervisor.start()
        app._artifacts.reconcile()
        app._start_routes()
        write_capture(home / "spool", "big.png", corpus.image("anomaly-good.png"))
        app._source_of("clearance-cam-01").rescan()
        _drain(app)
        app._publisher.drain_once()

        body = gg.messaging.bodies("app/inference/result")[0]
        # the route asked for no evidence, but a body over the budget is only publishable
        # alongside the full result, so the sidecar is written anyway and named in the message
        assert body["artifacts"]["localRelativePath"].endswith(".inference.json")
        assert body["artifacts"]["sha256"]
        assert (home / "processed" / "big.png.inference.json").is_file()
    finally:
        app.stop()


def test_evidence_that_cannot_be_installed_commits_nothing(running, gg, home, corpus, monkeypatch):
    from image_processor.outputs import sidecar as sidecar_module

    def _refuse(src, dst):
        raise OSError("the volume is read only")

    monkeypatch.setattr(sidecar_module, "replace", _refuse)
    write_capture(home / "spool", "noevidence.png", corpus.image("anomaly-good.png"))

    running._source_of("clearance-cam-01").rescan()
    _drain(running)

    assert gg.find_event("evidence-failed") is not None
    assert gg.messaging.bodies("app/inference/result") == []
    # nothing was committed, so the job is still in flight and recovery will retry it
    assert running._ledger.get(running.list_jobs(None, None, None, 5)[0][0]["inferenceId"])


def test_publication_that_gives_up_reports_and_keeps_the_image(running, gg, home, corpus):
    write_capture(home / "spool", "gone.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    _drain(running)
    gg.messaging.fail_confirmed = TimeoutError("no PUBACK")

    for _ in range(running._config.publish.max_attempts):
        running._publisher.drain_once()

    event = gg.find_event("publish-exhausted")
    assert event is not None
    assert event["context"]["configuredAction"] == "retain"
    assert (home / "spool" / "gone.png").is_file()
    inference_id = event["context"]["inferenceId"]
    assert running._ledger.get(inference_id).state is JobState.PUBLISH_EXHAUSTED

    gg.messaging.fail_confirmed = None
    outcome = running.retry_publication(None, inference_id)

    assert outcome == {"returned": [inference_id], "published": 1}
    assert running._ledger.get(inference_id).state is JobState.COMPLETED


def test_stored_bytes_that_are_not_an_envelope_still_publish(running, gg):
    from image_processor.ledger import OutboxRow

    running._publish_confirmed("ecv1/dev/image-processor/r/app/inference/result", b"not-protobuf", 1.0)

    published = gg.messaging.published[-1]
    assert published.confirmed is True
    assert published.topic.endswith("app/inference/result")


def test_a_result_for_a_route_that_no_longer_exists_is_reported_not_raised(running, gg, corpus):
    from dataclasses import replace

    from tests.outputs.conftest import make_job, make_result

    job = make_job(route_id="gone", source=replace(make_job().source, route_id="gone"))

    running._on_result(job, make_result())

    assert gg.messaging.bodies("app/inference/result") == []


def test_the_result_pipeline_reports_its_own_failure(running, gg, monkeypatch, home, corpus):
    from tests.outputs.conftest import make_result

    def _boom(*args, **kwargs):
        raise RuntimeError("the body builder blew up")

    monkeypatch.setattr("image_processor.ImageProcessor.build_result_body", _boom)
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()

    _drain(running)

    event = gg.find_event("inference-failed")
    assert event["context"]["code"] == "RESULT_PIPELINE_FAILED"


def test_a_route_whose_root_disappeared_is_not_reachable(running, home):
    import shutil

    shutil.rmtree(home / "spool")

    status = {item.route_id: item for item in running.route_statuses()}["clearance-cam-01"]

    assert status.source_reachable is False
    assert status.connected is False


def test_the_module_helpers_answer_for_the_shapes_they_meet():
    from image_processor.ImageProcessor import (
        _body_of,
        _command_answer,
        _completion_measure,
        _credentials_document,
        _file_size,
        _tree_size,
    )
    from image_processor.types import CompletionAction

    assert _body_of({"body": {"a": 1}}) == {"a": 1}
    assert _body_of({"a": 1}) == {"a": 1}
    assert _command_answer(None) is None
    assert _command_answer({"ok": True, "result": {"jobs": []}}) == {"jobs": []}
    assert _command_answer({"ok": True, "result": None}) == {}
    assert _command_answer({"ok": False, "error": {"code": "NOPE", "message": "no"}}) == {
        "errorCode": "NOPE",
        "errorMessage": "no",
    }
    assert _command_answer({"ok": False}) == {"errorCode": "COMMAND_FAILED", "errorMessage": ""}
    assert _command_answer({"jobs": []}) == {"jobs": []}
    assert _credentials_document('{"aws_access_key_id": "A"}') == {"aws_access_key_id": "A"}
    assert _credentials_document("a-token") == {"bearerToken": "a-token"}
    assert _credentials_document("[1]") == {"bearerToken": "[1]"}
    assert _completion_measure(CompletionAction.ARCHIVE) == "archived"
    assert _completion_measure(CompletionAction.DELETE) == "deleted"
    assert _completion_measure(CompletionAction.QUARANTINE) == "quarantined"
    assert _completion_measure(CompletionAction.RETAIN) == "retained"
    assert _file_size(Path("nope-not-here")) == 0.0
    assert _tree_size(Path("nope-not-here")) == 0.0


def test_a_component_that_cannot_read_its_state_is_not_ready(running):
    running._ledger.close()

    assert running._state_writable() is False
    assert running._health.evaluate().failed is True


def test_a_cache_that_cannot_be_read_is_reported(running, monkeypatch):
    monkeypatch.setattr(
        running._cache, "list", lambda: (_ for _ in ()).throw(OSError("the volume is gone"))
    )

    assert running._cache_verified() is False


def test_an_unknown_route_cannot_be_overridden(running):
    from edgecommons.command_inbox import CommandException

    with pytest.raises(CommandException):
        running.set_activation_override("nope", True)


def test_the_component_answers_its_own_identity_even_without_one(home, corpus):
    from image_processor.ImageProcessor import ImageProcessor
    from tests.wiring.conftest import FakeGg

    gg = FakeGg(config_document(home, corpus, trigger=False))

    def _boom():
        raise RuntimeError("no identity yet")

    gg.config_manager.get_thing_name = _boom
    app = ImageProcessor(gg)
    try:
        assert app._device() == "unknown"
    finally:
        app.stop()
