"""Capture-status reconciliation against a fake camera that pages exactly as the real one does."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_processor.sources.camera import (
    CAPTURE_NOT_FOUND,
    DEFAULT_PAGE_LIMIT,
    CaptureStatusReconciler,
    ReconcileError,
    capture_status_topic,
    image_captured_topic,
    reply_body,
)
from tests.sources.conftest import sha256_of, status_job, write_capture

DATA = b"one capture worth of image bytes"


class FakeCamera:
    """A ``sb/capture-status`` responder that pages its list the way camera-adapter does.

    It records every request body it was handed, so a test can assert on the exact request shape
    as well as on what the reconciler did with the answers.
    """

    def __init__(self, pages, per_capture=None) -> None:
        self.pages = list(pages)
        self.per_capture = dict(per_capture or {})
        self.requests: list = []

    def __call__(self, topic, body, timeout):
        self.requests.append((topic, dict(body), timeout))
        if "captureId" in body:
            answer = self.per_capture.get(body["captureId"])
            return answer if answer is not None else {"errorCode": CAPTURE_NOT_FOUND,
                                                      "errorMessage": "no such capture"}
        cursor = body.get("cursor")
        index = 0 if cursor is None else int(cursor)
        if index >= len(self.pages):
            return {"jobs": [], "nextCursor": None}
        jobs = self.pages[index]
        next_cursor = str(index + 1) if index + 1 < len(self.pages) else None
        return {"jobs": jobs, "nextCursor": next_cursor}


class KeyValue:
    """The ledger key-value pair the reconciler persists its watermark through."""

    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key):
        """Read a key."""
        return self.store.get(key)

    def set(self, key, value) -> None:
        """Write a key."""
        self.store[key] = value


def reconciler(root: Path, camera, kv, **kwargs) -> CaptureStatusReconciler:
    """Build a reconciler over ``root`` answering from ``camera``."""
    kwargs.setdefault("instance", "cam-01")
    return CaptureStatusReconciler(
        route_id="clearance-cam-01",
        root=root,
        topic=capture_status_topic("dallas-01", "camera-adapter", "cam-01"),
        request=camera,
        kv_get=kv.get,
        kv_set=kv.set,
        **kwargs,
    )


def test_the_topics_follow_the_uns_grammar():
    assert capture_status_topic("dallas-01", "camera-adapter", "cam-01") == (
        "ecv1/dallas-01/camera-adapter/cam-01/cmd/sb/capture-status"
    )
    assert capture_status_topic("dallas-01", "camera-adapter") == (
        "ecv1/dallas-01/camera-adapter/cmd/sb/capture-status"
    )
    assert image_captured_topic("dallas-01", "camera-adapter", "cam-01") == (
        "ecv1/dallas-01/camera-adapter/cam-01/app/image/captured"
    )


def test_a_sweep_follows_every_next_cursor_to_the_end(tmp_path):
    jobs = []
    for index in range(5):
        relative = f"2026/08/22/frame-{index}.jpg"
        body = write_capture(
            tmp_path, relative, DATA + str(index).encode(), sidecar=False,
            capture_id=f"cap_{index}",
        )
        jobs.append(status_job(body, terminal_at_ms=1789012506000 + index))
    camera = FakeCamera([jobs[0:2], jobs[2:4], jobs[4:5]])
    kv = KeyValue()
    seen = []
    subject = reconciler(tmp_path, camera, kv, page_limit=2, on_verified=seen.append)

    assert subject.poll_once() == 5

    assert subject.pages_read == 3
    assert [request[1].get("cursor") for request in camera.requests] == [None, "1", "2"]
    assert camera.requests[0][1]["states"] == ["SUCCEEDED"]
    assert camera.requests[0][1]["instance"] == "cam-01"
    assert camera.requests[0][1]["limit"] == 2
    assert [record.relative_path for record in seen] == [
        f"2026/08/22/frame-{index}.jpg" for index in range(5)
    ]


def test_a_capture_seen_twice_in_one_sweep_is_absorbed_once(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    job = status_job(body)
    camera = FakeCamera([[job, job], [job]])
    kv = KeyValue()
    subject = reconciler(tmp_path, camera, kv)

    assert subject.poll_once() == 1
    assert subject.verified_count == 1


def test_the_watermark_stops_a_later_sweep_re_announcing_what_it_already_did(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    camera = FakeCamera([[status_job(body, terminal_at_ms=1789012506010)]])
    kv = KeyValue()
    seen = []
    subject = reconciler(tmp_path, camera, kv, on_verified=seen.append)
    assert subject.poll_once() == 1

    fresh = reconciler(tmp_path, camera, kv, on_verified=seen.append)

    assert fresh.poll_once() == 0
    assert len(seen) == 1
    stored = json.loads(kv.store[subject.kv_key])
    assert stored == {"terminalAtMs": 1789012506010, "captureIds": [body["captureId"]]}


def test_a_capture_newer_than_the_watermark_still_lands(tmp_path):
    first = write_capture(tmp_path, "a.jpg", DATA, sidecar=False, capture_id="cap_a")
    second = write_capture(tmp_path, "b.jpg", DATA + b"!", sidecar=False, capture_id="cap_b")
    kv = KeyValue()
    subject = reconciler(
        tmp_path, FakeCamera([[status_job(first, terminal_at_ms=100)]]), kv
    )
    subject.poll_once()

    later = reconciler(
        tmp_path,
        FakeCamera(
            [[status_job(first, terminal_at_ms=100), status_job(second, terminal_at_ms=200)]]
        ),
        kv,
    )

    assert later.poll_once() == 1
    assert set(later.records()) == {"b.jpg"}


def test_two_captures_sharing_a_terminal_time_are_both_remembered(tmp_path):
    first = write_capture(tmp_path, "a.jpg", DATA, sidecar=False, capture_id="cap_a")
    second = write_capture(tmp_path, "b.jpg", DATA + b"!", sidecar=False, capture_id="cap_b")
    jobs = [status_job(first, terminal_at_ms=500), status_job(second, terminal_at_ms=500)]
    kv = KeyValue()

    assert reconciler(tmp_path, FakeCamera([jobs]), kv).poll_once() == 2

    stored = json.loads(kv.store[list(kv.store)[0]])
    assert stored["captureIds"] == ["cap_a", "cap_b"]
    assert reconciler(tmp_path, FakeCamera([jobs]), kv).poll_once() == 0


def test_an_expired_capture_answers_not_found_rather_than_failing(tmp_path):
    camera = FakeCamera([[]])
    subject = reconciler(tmp_path, camera, KeyValue())

    assert subject.lookup_capture("cap_long_gone") is None
    assert subject.last_error is None


def test_a_capture_still_within_retention_reads_back(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    job = status_job(body)
    camera = FakeCamera([[]], per_capture={body["captureId"]: job})
    subject = reconciler(tmp_path, camera, KeyValue())

    assert subject.lookup_capture(body["captureId"]) == job


def test_a_refused_lookup_is_recorded_as_an_error(tmp_path):
    camera = FakeCamera([[]], per_capture={"cap_1": {"errorCode": "DEVICE_UNAVAILABLE"}})
    subject = reconciler(tmp_path, camera, KeyValue())

    assert subject.lookup_capture("cap_1") is None
    assert subject.last_error == "DEVICE_UNAVAILABLE"


def test_a_refused_page_stops_the_sweep_and_is_recorded(tmp_path):
    def refuse(topic, body, timeout):
        return {"errorCode": "DEVICE_UNAVAILABLE", "errorMessage": "reload draining"}

    subject = reconciler(tmp_path, refuse, KeyValue())

    assert subject.poll_once() == 0
    assert subject.last_error == "DEVICE_UNAVAILABLE"
    assert subject.pages_read == 0


def test_a_request_that_raises_stops_the_sweep_without_raising(tmp_path):
    def explode(topic, body, timeout):
        raise TimeoutError("no reply within the deadline")

    subject = reconciler(tmp_path, explode, KeyValue())

    assert subject.poll_once() == 0
    assert subject.last_error == "TimeoutError"


def test_an_unreadable_reply_is_recorded(tmp_path):
    subject = reconciler(tmp_path, lambda topic, body, timeout: object(), KeyValue())

    assert subject.poll_once() == 0
    assert subject.last_error == "UNREADABLE_REPLY"


@pytest.mark.parametrize(
    "mutate, expect_rejected",
    [
        (lambda job: job.update(result=None), True),
        (lambda job: job["result"].update(image=None), True),
        (lambda job: job["result"]["image"].update(relativePath="../../escape.jpg"), True),
        (lambda job: job["result"]["image"].update(sha256="0" * 64), True),
        (lambda job: job.update(state="FAILED"), False),
        (lambda job: job.update(captureId=None), False),
    ],
)
def test_a_record_that_does_not_verify_is_never_admitted(tmp_path, mutate, expect_rejected):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    job = status_job(body)
    mutate(job)
    subject = reconciler(tmp_path, FakeCamera([[job]]), KeyValue())

    assert subject.poll_once() == 0
    assert subject.records() == {}
    assert (subject.rejected_count > 0) is expect_rejected


def test_a_page_holding_something_that_is_not_a_job_is_ignored(tmp_path):
    subject = reconciler(tmp_path, FakeCamera([["not a job", 17, None]]), KeyValue())

    assert subject.poll_once() == 0


def test_a_page_limit_the_camera_would_refuse_is_refused_here_first(tmp_path):
    for bad in (0, 1001):
        with pytest.raises(ReconcileError) as caught:
            reconciler(tmp_path, FakeCamera([[]]), KeyValue(), page_limit=bad)
        assert caught.value.code == "PAGE_LIMIT_OUT_OF_RANGE"


def test_a_component_scope_sweep_omits_the_instance_filter(tmp_path):
    camera = FakeCamera([[]])
    subject = reconciler(tmp_path, camera, KeyValue(), instance=None)

    subject.poll_once()

    assert "instance" not in camera.requests[0][1]
    assert camera.requests[0][1]["limit"] == DEFAULT_PAGE_LIMIT


def test_paging_stops_at_the_ceiling_rather_than_looping_forever(tmp_path, monkeypatch):
    monkeypatch.setattr("image_processor.sources.camera.MAX_PAGES_PER_SWEEP", 3)

    def endless(topic, body, timeout):
        return {"jobs": [], "nextCursor": "always more"}

    subject = reconciler(tmp_path, endless, KeyValue())

    assert subject.poll_once() == 0
    assert subject.pages_read == 3


def test_a_corrupt_or_unreadable_watermark_starts_from_the_beginning(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    kv = KeyValue()
    kv.store["image-processor/capture-status-watermark/clearance-cam-01"] = "{not json"

    subject = reconciler(tmp_path, FakeCamera([[status_job(body)]]), kv)

    assert subject.poll_once() == 1


def test_a_key_value_pair_that_fails_never_stops_the_sweep(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)

    def broken_get(key):
        raise RuntimeError("the ledger is not open yet")

    def broken_set(key, value):
        raise RuntimeError("the ledger is not open yet")

    subject = CaptureStatusReconciler(
        route_id="r",
        root=tmp_path,
        topic="t",
        request=FakeCamera([[status_job(body)]]),
        kv_get=broken_get,
        kv_set=broken_set,
    )

    assert subject.poll_once() == 1


def test_a_subscriber_that_raises_does_not_lose_the_record(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)

    def explode(record):
        raise RuntimeError("the spool source is not started")

    subject = reconciler(tmp_path, FakeCamera([[status_job(body)]]), KeyValue(),
                         on_verified=explode)

    assert subject.poll_once() == 1
    assert subject.lookup("frame.jpg").identity.sha256 == sha256_of(DATA)


def test_the_background_thread_sweeps_and_stops(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    subject = reconciler(
        tmp_path, FakeCamera([[status_job(body)]]), KeyValue(), interval_secs=0.01
    )

    subject.start()
    subject.start()
    try:
        deadline = __import__("time").monotonic() + 5
        while __import__("time").monotonic() < deadline and not subject.records():
            __import__("time").sleep(0.01)
    finally:
        subject.stop()
        subject.stop()

    assert subject.lookup("frame.jpg") is not None


def test_a_sweep_that_raises_does_not_kill_the_thread(tmp_path, monkeypatch):
    subject = reconciler(tmp_path, FakeCamera([[]]), KeyValue(), interval_secs=0.01)
    calls = []

    def explode():
        calls.append(1)
        raise RuntimeError("the broker went away")

    monkeypatch.setattr(subject, "poll_once", explode)
    subject.start()
    try:
        deadline = __import__("time").monotonic() + 5
        while __import__("time").monotonic() < deadline and len(calls) < 2:
            __import__("time").sleep(0.01)
    finally:
        subject.stop()

    assert len(calls) >= 2


@pytest.mark.parametrize(
    "reply, expected",
    [
        ({"jobs": []}, {"jobs": []}),
        (None, None),
        (object(), None),
    ],
)
def test_reply_body_reads_what_the_transport_hands_back(reply, expected):
    assert reply_body(reply) == expected


def test_reply_body_unwraps_a_core_message():
    class Message:
        def get_body(self):
            return {"jobs": []}

    class Envelope:
        body = {"jobs": [1]}

    assert reply_body(Message()) == {"jobs": []}
    assert reply_body(Envelope()) == {"jobs": [1]}


def test_a_record_with_no_terminal_time_is_absorbed_once_and_never_watermarked(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    job = status_job(body)
    job["terminalAtMs"] = None
    camera = FakeCamera([[job]])
    kv = KeyValue()
    subject = reconciler(tmp_path, camera, kv)

    assert subject.poll_once() == 1
    assert subject.poll_once() == 0
    assert json.loads(kv.store[subject.kv_key]) == {"terminalAtMs": 0, "captureIds": []}


def test_a_record_older_than_the_watermark_is_skipped(tmp_path):
    old = write_capture(tmp_path, "old.jpg", DATA, sidecar=False, capture_id="cap_old")
    new = write_capture(tmp_path, "new.jpg", DATA + b"!", sidecar=False, capture_id="cap_new")
    kv = KeyValue()
    reconciler(tmp_path, FakeCamera([[status_job(new, terminal_at_ms=900)]]), kv).poll_once()

    later = reconciler(
        tmp_path,
        FakeCamera([[status_job(old, terminal_at_ms=100),
                     status_job(new, terminal_at_ms=900)]]),
        kv,
    )

    assert later.poll_once() == 0


def test_a_lookup_whose_request_fails_answers_nothing(tmp_path):
    def explode(topic, body, timeout):
        raise TimeoutError("no reply")

    subject = reconciler(tmp_path, explode, KeyValue())

    assert subject.lookup_capture("cap_1") is None
    assert subject.last_error == "TimeoutError"
