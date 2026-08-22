"""Fixtures for the input-source suites: camera-shaped spool fixtures and recording sinks.

The camera fixtures build the real thing rather than a convenient subset. camera-adapter's
terminal body, its ``<image>.json`` sidecar, and the ``result`` element of a capture-status record
are the same document (``camera-adapter/src/actor.rs`` asserts the sidecar equals the announced
body), so one builder produces all three and every test that reads camera provenance reads the
same shape the camera actually writes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def sha256_of(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def terminal_body(
    relative_path: str,
    data: bytes,
    *,
    root: Path = Path("/var/spool/camera-adapter/cam-01"),
    capture_id: str = "cap_018f9c2b0001",
    camera_id: str = "cam-01",
    correlation_id: str = "corr-018f9c2b",
    persisted_at: str = "2026-08-22T10:15:04.512Z",
    with_sidecar: bool = True,
    image_overrides: dict | None = None,
) -> dict:
    """Build one camera-adapter schema-v1 terminal body for an image.

    The field names are camera-adapter's own (``src/messages.rs``: ``TerminalBody`` and
    ``ImageArtifact``), including the ``absolutePath`` this component must never follow.
    """
    absolute = str(root / relative_path)
    image = {
        "absolutePath": absolute,
        "relativePath": relative_path,
        "fileUri": Path(absolute).as_uri() if os.path.isabs(absolute) else absolute,
        "contentType": "image/jpeg",
        "encoding": "jpeg",
        "bytes": len(data),
        "sha256": sha256_of(data),
    }
    if with_sidecar:
        image["metadataSidecarRelativePath"] = relative_path + ".json"
    if image_overrides:
        image.update(image_overrides)
    return {
        "schemaVersion": 1,
        "eventId": "evt_" + capture_id,
        "captureId": capture_id,
        "cameraId": camera_id,
        "correlationId": correlation_id,
        "trigger": "command",
        "captureProfile": "default",
        "captureMode": "still",
        "timestamps": {
            "requestedAt": "2026-08-22T10:15:04.000Z",
            "frameReceivedAt": "2026-08-22T10:15:04.400Z",
            "persistedAt": persisted_at,
            "cameraFrameTimestampQuality": "adapterReceipt",
        },
        "durationsMs": {"queue": 3, "acquisition": 380, "persistence": 20, "total": 512},
        "image": image,
        "camera": {"backend": "sim", "model": "sim-1", "warnings": []},
        "metadata": {},
        "backendMetadata": {},
    }


def write_capture(
    root: Path,
    relative_path: str,
    data: bytes,
    *,
    sidecar: bool = True,
    body_overrides: dict | None = None,
    **kwargs,
) -> dict:
    """Write one capture into a spool the way camera-adapter writes it.

    The sidecar is written and flushed first, then the image is installed. Any test that reads the
    image before the sidecar exists is testing an ordering camera-adapter never produces.
    """
    body = terminal_body(relative_path, data, root=root, **kwargs)
    if body_overrides:
        body.update(body_overrides)
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if sidecar:
        companion = target.with_name(target.name + ".json")
        companion.write_text(json.dumps(body), encoding="utf-8")
    target.write_bytes(data)
    return body


def status_job(
    body: dict,
    *,
    state: str = "SUCCEEDED",
    terminal_at_ms: int = 1789012506010,
    instance: str = "cam-01",
) -> dict:
    """Build one ``sb/capture-status`` job element around a terminal body.

    The element's fields are the ones the camera's messaging reference documents for a single-job
    body: ``captureId``, ``instance``, ``state``, ``acceptedAtMs``, ``terminalAtMs``,
    ``captureGroupId``, ``errorCode``, ``errorMessage``, and ``result``.
    """
    return {
        "captureId": body["captureId"],
        "instance": instance,
        "state": state,
        "acceptedAtMs": terminal_at_ms - 512,
        "terminalAtMs": terminal_at_ms,
        "captureGroupId": None,
        "errorCode": None,
        "errorMessage": None,
        "result": body,
    }


class RecordingEvents:
    """A ``SourceEvents`` that keeps what it was told."""

    def __init__(self) -> None:
        self.discovered_calls: list = []
        self.invalid_calls: list = []

    def discovered(self, route_id, source, staged_path) -> None:
        """Record one discovered input."""
        self.discovered_calls.append((route_id, source, staged_path))

    def invalid(self, route_id, relative_path, reason) -> None:
        """Record one refused input."""
        self.invalid_calls.append((route_id, relative_path, reason))

    @property
    def paths(self) -> list:
        """The relative paths announced, in order."""
        return [call[1].relative_path for call in self.discovered_calls]

    @property
    def reasons(self) -> list:
        """The refusal reasons reported, in order."""
        return [call[2] for call in self.invalid_calls]


@pytest.fixture
def events() -> RecordingEvents:
    """A recording ``SourceEvents``."""
    return RecordingEvents()


def spool_route(
    root: Path,
    *,
    route_id: str = "clearance-cam-01",
    include=("**/*.jpg",),
    exclude=(),
    mode: str = "cameraSidecar",
    quiet_secs: float | None = None,
    marker_suffix: str | None = None,
    camera: dict | None = None,
) -> SimpleNamespace:
    """Build a spool route configuration with the field names ``config.schema.json`` uses."""
    readiness = SimpleNamespace(mode=mode, quietSecs=quiet_secs, markerSuffix=marker_suffix)
    camera_block = None
    if camera is not None:
        camera_block = SimpleNamespace(
            component=camera.get("component", "camera-adapter"),
            instance=camera.get("instance", "cam-01"),
            subscribeAnnouncements=camera.get("subscribeAnnouncements", True),
            reconcileCaptureStatusSecs=camera.get("reconcileCaptureStatusSecs", 30),
        )
    return SimpleNamespace(
        id=route_id,
        source=SimpleNamespace(
            kind="spool",
            root=str(root),
            include=include,
            exclude=exclude,
            readiness=readiness,
            camera=camera_block,
        ),
    )


def trigger_route(
    *,
    route_id: str = "adhoc-inspect",
    subscribe=("ecv1/+/inspection-ui/+/app/inspect/request",),
    file_root: Path | None = None,
    inline_staging: Path | None = None,
    max_inline_bytes: int | None = None,
) -> SimpleNamespace:
    """Build a trigger route configuration with the field names ``config.schema.json`` uses."""
    return SimpleNamespace(
        id=route_id,
        source=SimpleNamespace(
            kind="trigger",
            subscribe=subscribe,
            fileRoot=str(file_root) if file_root is not None else None,
            inlineStaging=str(inline_staging) if inline_staging is not None else None,
            maxInlineBytes=max_inline_bytes,
        ),
    )


class FakeObserver:
    """A stand-in for a ``watchdog`` observer that hands the handler back to the test."""

    def __init__(self) -> None:
        self.handler = None
        self.watched: list = []
        self.started = False
        self.stopped = False

    def schedule(self, handler, path, recursive=False) -> None:
        """Record the scheduled watch."""
        self.handler = handler
        self.watched.append((path, recursive))

    def start(self) -> None:
        """Record the start."""
        self.started = True

    def stop(self) -> None:
        """Record the stop."""
        self.stopped = True

    def join(self, timeout=None) -> None:
        """Accept a join."""

    def fire(self) -> None:
        """Deliver one filesystem event to the scheduled handler."""
        self.handler.dispatch(SimpleNamespace(src_path="whatever", event_type="modified"))
