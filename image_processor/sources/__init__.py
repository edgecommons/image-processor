"""Input sources: how an image becomes a job (DESIGN.md 4, LLD 7).

Two kinds of source feed the ledger, and they answer the same question two different ways.

``SpoolSource`` watches a directory the component owns. Filesystem state is authoritative there:
a deterministic walk decides what exists, a readiness rule decides what is finished, and a digest
decides what it is. OS notifications and camera announcements only make that walk happen sooner.

``TriggerSource`` accepts images that arrive as messages, either inline within the envelope's
64 KiB binary cap or as a reference to a file under a configured root. Both are verified and
copied into processor-owned staging, so a trigger job is an ordinary file job by the time the
ledger sees it.

Both report through ``SourceEvents``, which the application implements. A source never writes to
the ledger, never publishes, and never decides what happens to a file afterward: it reports what
it found and what it refused.

The configuration protocols below name the fields these classes read, spelled as
``config.schema.json`` spells them (DESIGN.md 11). They are structural: any object or mapping
carrying those names works, so the concrete configuration dataclasses satisfy them without
importing anything from this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from image_processor.types import SourceIdentity


@runtime_checkable
class SourceEvents(Protocol):
    """What a source reports to the application.

    Both methods are called from the source's own thread and must not block: the walk that found
    the input is waiting behind them.
    """

    def discovered(
        self, route_id: str, source: SourceIdentity, staged_path: Optional[Path]
    ) -> None:
        """Report one verified, finished input.

        Args:
            route_id: The route that owns the input.
            source: Its verified identity and provenance.
            staged_path: The immutable processor-owned copy, when the source made one. A spool
                input has none: the component owns the spool and the cell reads the file in place.
        """

    def invalid(self, route_id: str, relative_path: str, reason: str) -> None:
        """Report an input that can never be admitted as it stands.

        Args:
            route_id: The route that owns the input.
            relative_path: The path as configured or as the message declared it. Empty when the
                input never had one, as an inline body does not.
            reason: A stable SCREAMING_SNAKE token. It carries no path and no digest.
        """


class ReadinessConfig(Protocol):
    """The ``source.readiness`` block."""

    mode: str
    quietSecs: float
    markerSuffix: str


class CameraConfig(Protocol):
    """The ``source.camera`` block of a camera-bound spool route."""

    component: str
    instance: str
    subscribeAnnouncements: bool
    reconcileCaptureStatusSecs: float


class SpoolSourceConfig(Protocol):
    """A ``source`` with ``kind: spool``."""

    root: str
    include: Any
    exclude: Any
    readiness: ReadinessConfig
    camera: CameraConfig


class TriggerSourceConfig(Protocol):
    """A ``source`` with ``kind: trigger``."""

    subscribe: Any
    fileRoot: str
    inlineStaging: str
    maxInlineBytes: int


class RouteConfig(Protocol):
    """One entry of ``component.instances[]``, as the sources read it."""

    id: str
    source: Any


from image_processor.sources.camera import (  # noqa: E402
    CaptureRecord,
    CaptureStatusReconciler,
    capture_status_topic,
    image_captured_topic,
)
from image_processor.sources.readiness import (  # noqa: E402
    CameraSidecarReadiness,
    CameraStatusReadiness,
    MarkerReadiness,
    Readiness,
    ReadinessError,
    ReadyVerdict,
    StabilityReadiness,
)
from image_processor.sources.spool import SpoolSource, SpoolError  # noqa: E402
from image_processor.sources.staging import (  # noqa: E402
    MAX_INLINE_BYTES,
    PathError,
    SourceError,
    StagingError,
    stage_bytes,
    stage_copy,
)
from image_processor.sources.trigger import TriggerError, TriggerSource  # noqa: E402

__all__ = [
    "CameraConfig",
    "CameraSidecarReadiness",
    "CameraStatusReadiness",
    "CaptureRecord",
    "CaptureStatusReconciler",
    "MAX_INLINE_BYTES",
    "MarkerReadiness",
    "PathError",
    "Readiness",
    "ReadinessConfig",
    "ReadinessError",
    "ReadyVerdict",
    "RouteConfig",
    "SourceError",
    "SourceEvents",
    "SpoolError",
    "SpoolSource",
    "SpoolSourceConfig",
    "StabilityReadiness",
    "StagingError",
    "TriggerError",
    "TriggerSource",
    "TriggerSourceConfig",
    "capture_status_topic",
    "image_captured_topic",
    "stage_bytes",
    "stage_copy",
]
