"""The four readiness modes, and the rule that a camera route never falls back to guessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_processor.types import SourceKind
from image_processor.sources.readiness import (
    CAMERA_SIDECAR,
    CAMERA_STATUS,
    MARKER,
    STABILITY,
    CameraStatusReadiness,
    MarkerReadiness,
    Readiness,
    ReadinessError,
    StabilityReadiness,
    identity_from_capture,
    parse_timestamp_ms,
    read_sidecar,
)
from tests.sources.conftest import sha256_of, spool_route, write_capture

DATA = b"a small but complete jpeg-shaped payload"


class FakeClock:
    """A monotonic clock a test advances by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_camera_sidecar_admits_a_capture_written_sidecar_first(tmp_path):
    write_capture(tmp_path, "2026/08/22/frame.jpg", DATA)
    rule = Readiness.for_route(spool_route(tmp_path, mode=CAMERA_SIDECAR))

    verdict = rule.ready(tmp_path / "2026/08/22/frame.jpg", "2026/08/22/frame.jpg")

    assert verdict.ready
    assert verdict.reason == "SIDECAR_VERIFIED"
    identity = verdict.identity
    assert identity.kind is SourceKind.SPOOL
    assert identity.relative_path == "2026/08/22/frame.jpg"
    assert identity.bytes == len(DATA)
    assert identity.sha256 == sha256_of(DATA)
    assert identity.capture_id == "cap_018f9c2b0001"
    assert identity.camera_id == "cam-01"
    assert identity.correlation_id == "corr-018f9c2b"
    assert identity.captured_at_ms == 1787393704512


def test_camera_sidecar_holds_an_image_whose_sidecar_has_not_arrived(tmp_path):
    (tmp_path / "frame.jpg").write_bytes(DATA)
    rule = Readiness.for_route(spool_route(tmp_path, mode=CAMERA_SIDECAR))

    verdict = rule.ready(tmp_path / "frame.jpg", "frame.jpg")

    assert not verdict.ready
    assert verdict.reason == "SIDECAR_MISSING"


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda body: body["image"].update(bytes=len(DATA) + 5), "SIDECAR_BYTES_MISMATCH"),
        (lambda body: body["image"].update(sha256="0" * 64), "SIDECAR_SHA256_MISMATCH"),
        (lambda body: body.pop("image"), "SIDECAR_IMAGE_ELEMENT_MISSING"),
        (lambda body: body["image"].update(bytes="lots"), "SIDECAR_DECLARED_BYTES_INVALID"),
        (lambda body: body["image"].update(sha256=17), "SIDECAR_DECLARED_SHA256_INVALID"),
    ],
)
def test_camera_sidecar_refuses_a_sidecar_that_does_not_match_the_file(tmp_path, mutate, reason):
    body = write_capture(tmp_path, "frame.jpg", DATA)
    mutate(body)
    (tmp_path / "frame.jpg.json").write_text(json.dumps(body), encoding="utf-8")
    rule = Readiness.for_route(spool_route(tmp_path, mode=CAMERA_SIDECAR))

    verdict = rule.ready(tmp_path / "frame.jpg", "frame.jpg")

    assert not verdict.ready
    assert verdict.reason == reason


@pytest.mark.parametrize("content", [b"{not json", b"[]", b"\xff\xfe not utf8"])
def test_read_sidecar_refuses_a_sidecar_that_is_not_a_json_object(tmp_path, content):
    (tmp_path / "frame.jpg").write_bytes(DATA)
    (tmp_path / "frame.jpg.json").write_bytes(content)

    assert read_sidecar(tmp_path / "frame.jpg") is None


def test_marker_waits_for_the_companion_file(tmp_path):
    (tmp_path / "frame.jpg").write_bytes(DATA)
    rule = Readiness.for_route(spool_route(tmp_path, mode=MARKER, marker_suffix=".done"))

    assert rule.ready(tmp_path / "frame.jpg", "frame.jpg").reason == "MARKER_MISSING"

    (tmp_path / "frame.jpg.done").write_text("", encoding="utf-8")
    verdict = rule.ready(tmp_path / "frame.jpg", "frame.jpg")

    assert verdict.ready
    assert verdict.identity is None
    assert verdict.reason == "MARKER_PRESENT"


def test_marker_requires_a_configured_suffix(tmp_path):
    with pytest.raises(ReadinessError) as caught:
        Readiness.for_route(spool_route(tmp_path, mode=MARKER))
    assert caught.value.code == "MARKER_SUFFIX_REQUIRED"


def test_stability_admits_a_file_that_has_held_still_for_the_quiet_period(tmp_path):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    clock = FakeClock()
    rule = Readiness.for_route(
        spool_route(tmp_path, mode=STABILITY, quiet_secs=5), clock=clock
    )

    assert rule.ready(target, "frame.jpg").reason == "QUIET_PENDING"
    clock.advance(4)
    assert rule.ready(target, "frame.jpg").reason == "QUIET_PENDING"
    clock.advance(2)
    verdict = rule.ready(target, "frame.jpg")

    assert verdict.ready
    assert verdict.identity is None
    assert verdict.reason == "QUIET_ELAPSED"


def test_stability_restarts_the_quiet_period_when_the_file_moves(tmp_path):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    clock = FakeClock()
    rule = StabilityReadiness("route", quiet_secs=5, clock=clock)

    rule.ready(target, "frame.jpg")
    clock.advance(4)
    target.write_bytes(DATA + b"more data arrived")
    assert rule.ready(target, "frame.jpg").reason == "QUIET_PENDING"
    clock.advance(4)
    assert rule.ready(target, "frame.jpg").reason == "QUIET_PENDING"
    clock.advance(2)
    assert rule.ready(target, "frame.jpg").ready


def test_stability_forgets_a_file_that_disappeared(tmp_path):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    rule = StabilityReadiness("route", quiet_secs=1, clock=FakeClock())

    rule.ready(target, "frame.jpg")
    target.unlink()

    assert rule.ready(target, "frame.jpg").reason == "MISSING"
    rule.prune(set())
    assert rule._first_seen == {}


def test_stability_prunes_timers_for_files_no_longer_present(tmp_path):
    (tmp_path / "a.jpg").write_bytes(DATA)
    (tmp_path / "b.jpg").write_bytes(DATA)
    rule = StabilityReadiness("route", quiet_secs=1, clock=FakeClock())
    rule.ready(tmp_path / "a.jpg", "a.jpg")
    rule.ready(tmp_path / "b.jpg", "b.jpg")

    rule.prune({"a.jpg"})

    assert set(rule._first_seen) == {"a.jpg"}


def test_camera_status_admits_only_a_verified_record(tmp_path):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    target = tmp_path / "frame.jpg"
    records = {}
    rule = Readiness.for_route(
        spool_route(tmp_path, mode=CAMERA_STATUS, camera={}),
        status_lookup=records.get,
    )

    assert rule.ready(target, "frame.jpg").reason == "NO_VERIFIED_CAPTURE"

    from image_processor.sources.camera import CaptureRecord
    from image_processor.sources.staging import stat_signature

    records["frame.jpg"] = CaptureRecord(
        capture_id=body["captureId"],
        relative_path="frame.jpg",
        identity=identity_from_capture(
            "clearance-cam-01", "frame.jpg", body, len(DATA), sha256_of(DATA)
        ),
        signature=stat_signature(target),
        terminal_at_ms=1789012506010,
    )
    verdict = rule.ready(target, "frame.jpg")

    assert verdict.ready
    assert verdict.reason == "CAPTURE_STATUS_VERIFIED"
    assert verdict.identity.capture_id == body["captureId"]


def test_camera_status_refuses_a_file_that_moved_since_it_was_verified(tmp_path):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    from image_processor.sources.camera import CaptureRecord

    record = CaptureRecord(
        capture_id="cap_1",
        relative_path="frame.jpg",
        identity=None,
        signature=(1, 1),
        terminal_at_ms=None,
    )
    rule = CameraStatusReadiness("route", {"frame.jpg": record}.get)

    assert rule.ready(target, "frame.jpg").reason == "CHANGED_SINCE_VERIFICATION"

    target.unlink()
    assert rule.ready(target, "frame.jpg").reason == "MISSING"


def test_camera_status_needs_a_reconciler(tmp_path):
    with pytest.raises(ReadinessError) as caught:
        Readiness.for_route(spool_route(tmp_path, mode=CAMERA_STATUS, camera={}))
    assert caught.value.code == "CAMERA_STATUS_REQUIRES_RECONCILER"


def test_a_camera_bound_route_may_not_fall_back_to_stability(tmp_path):
    with pytest.raises(ReadinessError) as caught:
        Readiness.for_route(spool_route(tmp_path, mode=STABILITY, camera={}))
    assert caught.value.code == "STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE"


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ReadinessError) as caught:
        Readiness.for_route(spool_route(tmp_path, mode="whenItFeelsRight"))
    assert caught.value.code == "UNKNOWN_READINESS_MODE"


def test_the_default_mode_follows_whether_the_route_is_camera_bound(tmp_path):
    camera_bound = Readiness.for_route(spool_route(tmp_path, mode=None, camera={}))
    plain = Readiness.for_route(spool_route(tmp_path, mode=None))

    assert camera_bound.mode == CAMERA_SIDECAR
    assert plain.mode == STABILITY


def test_a_camera_block_naming_nothing_does_not_bind_the_route(tmp_path):
    route = spool_route(tmp_path, mode=STABILITY, camera={"component": "", "instance": ""})

    assert Readiness.for_route(route).mode == STABILITY


def test_companion_suffixes_name_what_the_walk_must_skip(tmp_path):
    sidecar = Readiness.for_route(spool_route(tmp_path, mode=CAMERA_SIDECAR))
    marker = Readiness.for_route(spool_route(tmp_path, mode=MARKER, marker_suffix=".ok"))
    stability = Readiness.for_route(spool_route(tmp_path, mode=STABILITY))

    assert sidecar.companion_suffixes() == (".json",)
    assert marker.companion_suffixes() == (".ok",)
    assert stability.companion_suffixes() == ()
    assert isinstance(stability.strategy, StabilityReadiness)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-08-22T10:15:04.512Z", 1787393704512),
        ("2026-08-22T10:15:04Z", 1787393704000),
        ("2026-08-22T10:15:04.123456789Z", 1787393704123),
        ("2026-08-22T12:15:04.512+02:00", 1787393704512),
        ("2026-08-22T08:15:04.512-02:00", 1787393704512),
        (1787393704512, 1787393704512),
        ("not a timestamp", None),
        (None, None),
        ("2026-13-45T99:99:99Z", None),
        (True, None),
    ],
)
def test_parse_timestamp_ms_reads_what_chrono_writes(value, expected):
    assert parse_timestamp_ms(value) == expected


def test_verify_declared_image_reports_a_file_that_is_not_there(tmp_path):
    from image_processor.sources.readiness import verify_declared_image

    image = {"bytes": len(DATA), "sha256": sha256_of(DATA)}

    assert verify_declared_image(tmp_path / "absent.jpg", image)[2] == "MISSING"


def test_verify_declared_image_reports_a_file_it_cannot_read(tmp_path, monkeypatch):
    from image_processor.sources import readiness as readiness_module

    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)

    def refuse(path, chunk=1 << 20):
        raise PermissionError("another writer holds the file")

    monkeypatch.setattr(readiness_module, "sha256_file", refuse)
    image = {"bytes": len(DATA), "sha256": sha256_of(DATA)}

    assert readiness_module.verify_declared_image(target, image)[2] == "MISSING"
