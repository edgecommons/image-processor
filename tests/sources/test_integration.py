"""The camera-bound route end to end: reconciled status feeding the authoritative walk."""

from __future__ import annotations

from image_processor.sources.camera import CaptureStatusReconciler, capture_status_topic
from image_processor.sources.spool import SpoolSource
from tests.sources.conftest import sha256_of, spool_route, status_job, write_capture
from tests.sources.test_camera import FakeCamera, KeyValue

DATA = b"a capture the camera has already committed"


def test_a_camera_status_route_admits_only_what_the_camera_confirmed(tmp_path, events):
    confirmed = write_capture(
        tmp_path, "2026/08/22/confirmed.jpg", DATA, sidecar=False, capture_id="cap_ok"
    )
    write_capture(tmp_path, "2026/08/22/unconfirmed.jpg", DATA + b"!", sidecar=False)

    camera = FakeCamera([[status_job(confirmed)]])
    reconciler = CaptureStatusReconciler(
        route_id="clearance-cam-01",
        root=tmp_path,
        topic=capture_status_topic("dallas-01", "camera-adapter", "cam-01"),
        request=camera,
        kv_get=KeyValue().get,
        kv_set=lambda key, value: None,
        instance="cam-01",
    )
    source = SpoolSource(
        spool_route(tmp_path, mode="cameraStatus", camera={}),
        events,
        status_lookup=reconciler.lookup,
    )
    reconciler._on_verified = lambda record: source.nudge()

    assert source.rescan() == 0

    reconciler.poll_once()

    assert source.rescan() == 1
    assert events.paths == ["2026/08/22/confirmed.jpg"]
    assert events.discovered_calls[0][1].capture_id == "cap_ok"


def test_a_hint_and_the_reconciler_agree_on_one_job(tmp_path, events):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    camera = FakeCamera([[status_job(body)]])
    reconciler = CaptureStatusReconciler(
        route_id="clearance-cam-01",
        root=tmp_path,
        topic="t",
        request=camera,
        kv_get=lambda key: None,
        kv_set=lambda key, value: None,
        instance="cam-01",
    )
    source = SpoolSource(
        spool_route(tmp_path, mode="cameraStatus", camera={}),
        events,
        status_lookup=reconciler.lookup,
    )

    source.on_hint(body)
    reconciler.poll_once()
    source.rescan()

    assert len(events.discovered_calls) == 1
    assert events.discovered_calls[0][1].sha256 == sha256_of(DATA)
