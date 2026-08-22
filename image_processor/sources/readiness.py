"""Readiness strategies: the four ways a spool file proves it is finished (DESIGN.md 4.1).

A file appearing in a directory is not an input. It is an input once something proves the writer
is done with it, and the proof is source-specific:

* ``cameraSidecar`` reads the camera's own metadata sidecar and checks its declared size and
  digest against the file. camera-adapter installs the sidecar and flushes it *before* the image
  becomes visible, so an image that is visible next to its sidecar is complete by construction.
  Regulated routes require this mode.
* ``cameraStatus`` waits for a ``SUCCEEDED`` record that the capture-status reconciler has already
  verified against the file.
* ``marker`` waits for a companion file the writer creates last.
* ``stability`` waits for size and mtime to stop moving. It is the only mode that infers rather
  than verifies, so it is refused on a camera-bound route: the camera offers proof, and a route
  that has proof available never settles for a guess.

Each strategy answers with a ``ReadyVerdict``. The modes that verify a digest return the full
``SourceIdentity`` they proved, including capture provenance; the modes that only prove quiescence
return no identity and leave the digest to the caller.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from image_processor.types import SourceIdentity, SourceKind

from image_processor.sources.staging import (
    SourceError,
    config_field,
    sha256_file,
    stat_signature,
)

#: The four readiness modes DESIGN.md 4.1 defines.
CAMERA_SIDECAR = "cameraSidecar"
CAMERA_STATUS = "cameraStatus"
MARKER = "marker"
STABILITY = "stability"
MODES = (CAMERA_SIDECAR, CAMERA_STATUS, MARKER, STABILITY)

#: Quiet period a ``stability`` route uses when configuration does not name one.
DEFAULT_QUIET_SECS = 5.0

#: Suffix the camera appends to an image to name its metadata sidecar.
SIDECAR_SUFFIX = ".json"


class ReadinessError(SourceError):
    """A readiness mode cannot be built for this route."""


@dataclass(frozen=True)
class ReadyVerdict:
    """One readiness answer for one file.

    Attributes:
        ready: Whether the file is finished and may be admitted.
        identity: The verified identity when the mode proved a digest, otherwise None.
        reason: A stable SCREAMING_SNAKE token naming the verdict, for metrics and logs.
    """

    ready: bool
    identity: Optional[SourceIdentity]
    reason: str

    @staticmethod
    def no(reason: str) -> "ReadyVerdict":
        """Return a not-ready verdict carrying ``reason``."""
        return ReadyVerdict(False, None, reason)

    @staticmethod
    def yes(identity: Optional[SourceIdentity], reason: str) -> "ReadyVerdict":
        """Return a ready verdict, with the proven identity when the mode has one."""
        return ReadyVerdict(True, identity, reason)


@runtime_checkable
class ReadinessStrategy(Protocol):
    """One readiness rule."""

    mode: str

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file.

        Args:
            path: The absolute path of the candidate file.
            relative_path: Its normalized, forward-slashed path under the route root.
        """


_TIMESTAMP = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?"
    r"(?:([Zz])|([+-])(\d{2}):?(\d{2}))?$"
)


def parse_timestamp_ms(value: Any) -> Optional[int]:
    """Parse an RFC 3339 timestamp into epoch milliseconds, or return None.

    camera-adapter serializes its timestamps through chrono, which emits a ``Z`` suffix and a
    variable number of subsecond digits. Parsing them here rather than through
    ``datetime.fromisoformat`` keeps the accepted grammar the same on every supported runtime.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if not isinstance(value, str):
        return None
    match = _TIMESTAMP.match(value.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(match.group(index)) for index in range(1, 7))
    fraction = match.group(7) or ""
    microseconds = int((fraction + "000000")[:6]) if fraction else 0
    try:
        moment = datetime(
            year, month, day, hour, minute, second, microseconds, tzinfo=timezone.utc
        )
    except ValueError:
        return None
    if match.group(9):
        offset = timedelta(hours=int(match.group(10)), minutes=int(match.group(11)))
        moment = moment - offset if match.group(9) == "+" else moment + offset
    return int(moment.timestamp() * 1000)


def sidecar_path(path: Path) -> Path:
    """Return the camera metadata sidecar path for an image path."""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def read_sidecar(path: Path) -> Optional[dict]:
    """Read and parse the camera metadata sidecar beside ``path``.

    Returns:
        The parsed sidecar document, or None when it is absent, unreadable, or not a JSON object.
    """
    companion = sidecar_path(path)
    try:
        raw = companion.read_bytes()
    except OSError:
        return None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def verify_declared_image(
    path: Path,
    image: Any,
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Check a camera ``image`` element against the bytes on disk.

    The ``image`` element is the one camera-adapter publishes and writes: it carries ``bytes`` and
    a lowercase ``sha256`` of the exact installed file. Both are checked, size first, so a
    still-growing file is rejected without paying for a hash.

    Args:
        path: The image file.
        image: The ``image`` element from a sidecar, a hint body, or a capture-status record.

    Returns:
        ``(bytes, sha256, reason)``. ``reason`` is None on success; on failure it is the stable
        token naming what did not match and the other two elements are None.
    """
    if not isinstance(image, dict):
        return None, None, "IMAGE_ELEMENT_MISSING"
    declared_bytes = image.get("bytes")
    declared_sha = image.get("sha256")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        return None, None, "DECLARED_BYTES_INVALID"
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        return None, None, "DECLARED_SHA256_INVALID"
    signature = stat_signature(path)
    if signature is None:
        return None, None, "MISSING"
    if signature[0] != declared_bytes:
        return None, None, "BYTES_MISMATCH"
    try:
        actual = sha256_file(path)
    except OSError:
        return None, None, "MISSING"
    if actual != declared_sha.lower():
        return None, None, "SHA256_MISMATCH"
    return declared_bytes, actual, None


def identity_from_capture(
    route_id: str,
    relative_path: str,
    document: dict,
    size: int,
    sha256: str,
    kind: SourceKind = SourceKind.SPOOL,
) -> SourceIdentity:
    """Build a ``SourceIdentity`` from a camera terminal body and a verified digest.

    The terminal body is the same document in all three places it appears: the ``ImageCaptured``
    announcement, the ``<image>.json`` sidecar, and the ``result`` element of a capture-status
    record. Reading provenance the same way from all three is what lets a hint, a sidecar, and a
    reconciled record produce the same job identity.
    """
    timestamps = document.get("timestamps")
    persisted_at = timestamps.get("persistedAt") if isinstance(timestamps, dict) else None
    return SourceIdentity(
        kind=kind,
        route_id=route_id,
        relative_path=relative_path,
        bytes=size,
        sha256=sha256,
        capture_id=_text(document.get("captureId")),
        camera_id=_text(document.get("cameraId")),
        correlation_id=_text(document.get("correlationId")),
        captured_at_ms=parse_timestamp_ms(persisted_at),
    )


def _text(value: Any) -> Optional[str]:
    """Return a non-empty string, or None."""
    return value if isinstance(value, str) and value else None


class CameraSidecarReadiness:
    """Ready when the camera metadata sidecar parses and matches the file.

    camera-adapter writes ``<image>.json``, flushes it, and only then makes the image visible
    (``camera-adapter/src/storage.rs``). A visible image beside a matching sidecar is therefore a
    finished capture, and the sidecar carries the capture provenance the result and the evidence
    sidecar need.
    """

    mode = CAMERA_SIDECAR

    def __init__(self, route_id: str) -> None:
        self._route_id = route_id

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file against its sidecar."""
        document = read_sidecar(path)
        if document is None:
            return ReadyVerdict.no("SIDECAR_MISSING")
        size, digest, reason = verify_declared_image(path, document.get("image"))
        if reason is not None:
            return ReadyVerdict.no(f"SIDECAR_{reason}")
        identity = identity_from_capture(
            self._route_id, relative_path, document, int(size), str(digest)
        )
        return ReadyVerdict.yes(identity, "SIDECAR_VERIFIED")


class CameraStatusReadiness:
    """Ready when the capture-status reconciler holds a verified ``SUCCEEDED`` record.

    The reconciler verified the file against the record's declared size and digest when it read the
    page, so this strategy re-stats rather than re-hashing: a file whose size or mtime moved since
    verification is no longer the file that was proven, and it waits for the next sweep.
    """

    mode = CAMERA_STATUS

    def __init__(self, route_id: str, lookup: Callable[[str], Any]) -> None:
        self._route_id = route_id
        self._lookup = lookup

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file against the reconciled capture records."""
        record = self._lookup(relative_path)
        if record is None:
            return ReadyVerdict.no("NO_VERIFIED_CAPTURE")
        signature = stat_signature(path)
        if signature is None:
            return ReadyVerdict.no("MISSING")
        if signature != record.signature:
            return ReadyVerdict.no("CHANGED_SINCE_VERIFICATION")
        return ReadyVerdict.yes(record.identity, "CAPTURE_STATUS_VERIFIED")


class MarkerReadiness:
    """Ready when the writer's companion marker file exists.

    The marker proves the writer finished but says nothing about the bytes, so the caller hashes
    the file itself and re-stats around the hash.
    """

    mode = MARKER

    def __init__(self, route_id: str, suffix: str) -> None:
        if not suffix:
            raise ReadinessError(
                "MARKER_SUFFIX_REQUIRED", "readiness.markerSuffix names the companion file"
            )
        self._route_id = route_id
        self.suffix = suffix

    def marker_for(self, path: Path) -> Path:
        """Return the marker path for an image path."""
        return path.with_name(path.name + self.suffix)

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file by the presence of its marker."""
        if not self.marker_for(path).is_file():
            return ReadyVerdict.no("MARKER_MISSING")
        return ReadyVerdict.yes(None, "MARKER_PRESENT")


class StabilityReadiness:
    """Ready when size and mtime have not moved for ``quietSecs``.

    This is the only mode that infers completion instead of verifying it, so it is refused on
    camera-bound routes. The clock is injected: a test advances it rather than sleeping, and the
    quiet period is measured against a monotonic clock so a wall-clock step never shortens it.
    """

    mode = STABILITY

    def __init__(
        self,
        route_id: str,
        quiet_secs: float = DEFAULT_QUIET_SECS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._route_id = route_id
        self.quiet_secs = float(quiet_secs)
        self._clock = clock
        self._first_seen: dict = {}

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file by how long it has held still."""
        signature = stat_signature(path)
        if signature is None:
            self._first_seen.pop(relative_path, None)
            return ReadyVerdict.no("MISSING")
        now = self._clock()
        previous = self._first_seen.get(relative_path)
        if previous is None or previous[0] != signature:
            self._first_seen[relative_path] = (signature, now)
            return ReadyVerdict.no("QUIET_PENDING")
        if now - previous[1] >= self.quiet_secs:
            return ReadyVerdict.yes(None, "QUIET_ELAPSED")
        return ReadyVerdict.no("QUIET_PENDING")

    def prune(self, keep) -> None:
        """Forget the quiet timers of files that are no longer in the spool."""
        for relative_path in [key for key in self._first_seen if key not in keep]:
            del self._first_seen[relative_path]


def is_camera_bound(source: Any) -> bool:
    """Report whether a spool source names a camera to integrate with."""
    camera = config_field(source, "camera", default=None)
    if camera is None:
        return False
    component = config_field(camera, "component", default=None)
    instance = config_field(camera, "instance", default=None)
    return bool(component or instance)


class Readiness:
    """The readiness rule a route uses, selected from its configuration.

    Attributes:
        mode: The selected mode name, one of ``MODES``.
    """

    def __init__(self, strategy: ReadinessStrategy) -> None:
        self._strategy = strategy

    @property
    def mode(self) -> str:
        """The selected readiness mode."""
        return self._strategy.mode

    @property
    def strategy(self) -> ReadinessStrategy:
        """The strategy object, for callers that need its mode-specific detail."""
        return self._strategy

    @staticmethod
    def for_route(
        route: Any,
        *,
        status_lookup: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> "Readiness":
        """Build the readiness rule for one spool route.

        Args:
            route: The route configuration. Its ``source`` carries ``readiness`` and ``camera``.
            status_lookup: Resolves a relative path to a verified capture record. The
                ``cameraStatus`` mode requires it.
            clock: Monotonic clock for the ``stability`` mode.

        Returns:
            The readiness rule.

        Raises:
            ReadinessError: The mode is unknown, ``stability`` was asked for on a camera-bound
                route, ``cameraStatus`` was asked for with no reconciler, or ``marker`` was asked
                for with no suffix.
        """
        route_id = str(config_field(route, "id", "route_id"))
        source = config_field(route, "source")
        settings = config_field(source, "readiness", default={})
        camera_bound = is_camera_bound(source)
        default_mode = CAMERA_SIDECAR if camera_bound else STABILITY
        mode = str(config_field(settings, "mode", default=default_mode))
        if mode not in MODES:
            raise ReadinessError("UNKNOWN_READINESS_MODE", f"{mode!r} is not a readiness mode")
        if mode == STABILITY and camera_bound:
            raise ReadinessError(
                "STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE",
                "a camera-bound route verifies its inputs rather than inferring quiescence",
            )
        if mode == CAMERA_SIDECAR:
            return Readiness(CameraSidecarReadiness(route_id))
        if mode == CAMERA_STATUS:
            if status_lookup is None:
                raise ReadinessError(
                    "CAMERA_STATUS_REQUIRES_RECONCILER",
                    "the cameraStatus mode reads verified capture-status records",
                )
            return Readiness(CameraStatusReadiness(route_id, status_lookup))
        if mode == MARKER:
            suffix = str(config_field(settings, "markerSuffix", default=""))
            return Readiness(MarkerReadiness(route_id, suffix))
        quiet_secs = float(config_field(settings, "quietSecs", default=DEFAULT_QUIET_SECS))
        return Readiness(StabilityReadiness(route_id, quiet_secs, clock))

    def ready(self, path: Path, relative_path: str) -> ReadyVerdict:
        """Judge one file."""
        return self._strategy.ready(path, relative_path)

    def companion_suffixes(self) -> tuple:
        """Return the suffixes this mode's companion files carry.

        The spool walk skips them: a sidecar or a marker sits beside an image in the same
        directory, and it is metadata about an input, never an input.
        """
        if self.mode == CAMERA_SIDECAR:
            return (SIDECAR_SUFFIX,)
        if self.mode == MARKER:
            return (self._strategy.suffix,)
        return ()

    def prune(self, keep) -> None:
        """Forget per-file state for paths no longer present."""
        prune = getattr(self._strategy, "prune", None)
        if prune is not None:
            prune(keep)
