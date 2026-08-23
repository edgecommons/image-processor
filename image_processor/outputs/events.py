"""Operator events on the ``evt`` class (DESIGN.md 12.3, LLD 8).

``evt`` carries conditions an operator acts on: a model that will not verify or warm up, a
required GPU that is not there, a route that has fallen behind, an executor that keeps recycling,
a publish backlog approaching its bound, evidence or cleanup that failed, and disk or GPU
pressure. Success is not an event -- a component that emits one message per processed image on
the operator channel is a component whose operator stops reading it.

Every payload here is bounded: a stable type, a short message, and a context of scalars. No image
bytes, no tensors, no credentials, and no unbounded model output ever reach this class
(DESIGN.md 15). A condition that is a state rather than an occurrence -- a required GPU missing, a
route degraded -- is raised and cleared as an alarm, so a console can show it as a live condition
instead of a stream of repeats.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from edgecommons.facades import Severity

logger = logging.getLogger(__name__)

#: Every event type this component emits. The set is closed so a consumer can enumerate it.
EVENT_TYPES = (
    "model-staging-failed",
    "model-warmup-failed",
    "model-activated",
    "executor-unavailable",
    "executor-recycled",
    "route-degraded",
    "queue-age-exceeded",
    "publish-backlog",
    "publish-exhausted",
    "evidence-failed",
    "cleanup-failed",
    "input-rejected",
    "inference-failed",
    "disk-pressure",
    "gpu-pressure",
)

#: The bound on any single context string.
MAX_CONTEXT_CHARS = 512


def _bounded(context: Optional[dict]) -> dict:
    """Return a context of scalars, with every string bounded.

    Args:
        context: The caller structured data, or ``None``.

    Returns:
        A new dictionary carrying only scalars, each string cut to
        :data:`MAX_CONTEXT_CHARS`.
    """
    out: dict = {}
    for key, value in (context or {}).items():
        if isinstance(value, str):
            out[str(key)] = value[:MAX_CONTEXT_CHARS]
        elif isinstance(value, (bool, int, float)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)[:MAX_CONTEXT_CHARS]
    return out


class RouteEvents:
    """Typed operator events over the ``evt`` facade.

    Args:
        gg: The EdgeCommons handle. A route condition is emitted through
            ``gg.instance(routeId).events()`` so it carries the route own instance token; a
            component-wide condition is emitted at component scope through ``gg.events()``.
    """

    def __init__(self, gg: Any) -> None:
        """Build the event helpers over the EdgeCommons handle."""
        self._gg = gg
        self._active: dict = {}
        self.counters = {"emitted": 0, "raised": 0, "cleared": 0, "failed": 0}

    # -- the facade --------------------------------------------------------------------

    def _facade(self, route_id: Optional[str]):
        """Return the events facade for a route, or the component-scope one."""
        if route_id:
            return self._gg.instance(route_id).events()
        return self._gg.events()

    def emit(
        self,
        route_id: Optional[str],
        event_type: str,
        message: str,
        context: Optional[dict] = None,
        severity: Severity = Severity.WARNING,
    ) -> bool:
        """Emit one condition.

        Args:
            route_id: The route it concerns, or ``None`` for the component as a whole.
            event_type: One of :data:`EVENT_TYPES`.
            message: Short operator text.
            context: Bounded structured detail.
            severity: The severity, which derives the channel.

        Returns:
            Whether the event reached the bus.
        """
        try:
            self._facade(route_id).emit(event_type, message, _bounded(context), severity=severity)
        except Exception as exc:  # noqa: BLE001 - reporting a condition never fails the work
            self.counters["failed"] += 1
            logger.warning("could not emit %s: %s", event_type, exc, exc_info=True)
            return False
        self.counters["emitted"] += 1
        return True

    def alarm(
        self,
        route_id: Optional[str],
        event_type: str,
        active: bool,
        message: str = "",
        context: Optional[dict] = None,
        severity: Severity = Severity.CRITICAL,
    ) -> bool:
        """Raise or clear a stateful condition, publishing only on a transition.

        A condition sampled every few seconds -- a missing GPU, a degraded route -- would
        otherwise fill the operator channel with repeats of a fact that has not changed.

        Args:
            route_id: The route it concerns, or ``None`` for the component as a whole.
            event_type: One of :data:`EVENT_TYPES`.
            active: Whether the condition holds now.
            message: Short operator text, used on the raise.
            context: Bounded structured detail.
            severity: The severity, which derives the channel for both the raise and the clear.

        Returns:
            Whether this call published a transition.
        """
        key = (route_id or "", event_type)
        if bool(active) == bool(self._active.get(key, False)):
            return False
        try:
            facade = self._facade(route_id)
            if active:
                facade.raise_alarm(event_type, message, _bounded(context), severity=severity)
                self.counters["raised"] += 1
            else:
                facade.clear_alarm(event_type, _bounded(context), severity=severity)
                self.counters["cleared"] += 1
        except Exception as exc:  # noqa: BLE001 - reporting a condition never fails the work
            self.counters["failed"] += 1
            logger.warning("could not publish alarm %s: %s", event_type, exc, exc_info=True)
            return False
        self._active[key] = bool(active)
        return True

    def active_alarms(self) -> tuple:
        """Return the conditions currently raised, as ``(route_id, event_type)`` pairs."""
        return tuple(sorted(key for key, value in self._active.items() if value))

    # -- typed conditions --------------------------------------------------------------

    def model_staging_failed(self, route_id: Optional[str], model: str, error: str) -> bool:
        """A model bundle could not be fetched, verified, or extracted."""
        return self.emit(
            route_id,
            "model-staging-failed",
            f"model {model} could not be staged",
            {"model": model, "error": error},
            Severity.CRITICAL,
        )

    def model_warmup_failed(self, route_id: Optional[str], model: str, error: str) -> bool:
        """A staged bundle failed its golden warmup, so no route switched to it."""
        return self.emit(
            route_id,
            "model-warmup-failed",
            f"model {model} failed warmup and was not activated",
            {"model": model, "error": error},
            Severity.CRITICAL,
        )

    def model_activated(self, route_id: str, model: str, digest: str) -> bool:
        """A route switched to a new model generation. This is the completion of a
        long-running operation, which DESIGN.md 13 reports as an event."""
        return self.emit(
            route_id,
            "model-activated",
            f"route {route_id} is running {model}",
            {"model": model, "digest": digest},
            Severity.INFO,
        )

    def executor_unavailable(self, active: bool, reason: str = "") -> bool:
        """No healthy executor cell can serve the routes that require one."""
        return self.alarm(
            None,
            "executor-unavailable",
            active,
            "no healthy executor cell is available",
            {"reason": reason},
        )

    def executor_recycled(self, cell_id: str, reason: str, count: int) -> bool:
        """An executor cell was drained and restarted."""
        return self.emit(
            None,
            "executor-recycled",
            f"executor cell {cell_id} was recycled",
            {"cell": cell_id, "reason": reason, "recycleCount": count},
            Severity.WARNING,
        )

    def route_degraded(self, route_id: str, active: bool, reason: str = "") -> bool:
        """A route cannot execute: its model is not active, or its source is unreachable."""
        return self.alarm(
            route_id,
            "route-degraded",
            active,
            f"route {route_id} is degraded",
            {"reason": reason},
        )

    def queue_age_exceeded(
        self, route_id: str, active: bool, oldest_secs: float, threshold_secs: float, queued: int
    ) -> bool:
        """The oldest queued job on a route has waited past the configured threshold."""
        return self.alarm(
            route_id,
            "queue-age-exceeded",
            active,
            f"route {route_id} has work older than {threshold_secs:g}s",
            {
                "oldestSecs": round(float(oldest_secs), 3),
                "thresholdSecs": float(threshold_secs),
                "queued": int(queued),
            },
            Severity.WARNING,
        )

    def publish_backlog(self, active: bool, pending: int, capacity: int) -> bool:
        """The outbox is approaching its configured capacity."""
        return self.alarm(
            None,
            "publish-backlog",
            active,
            "the publication backlog is approaching capacity",
            {"pending": int(pending), "capacity": int(capacity)},
            Severity.WARNING,
        )

    def publish_exhausted(self, route_id: str, inference_id: str, error: str, action: str) -> bool:
        """A result spent its publication budget. The input is retained for an operator retry."""
        return self.emit(
            route_id,
            "publish-exhausted",
            f"{inference_id} could not be published and is retained",
            {"inferenceId": inference_id, "error": error, "configuredAction": action},
            Severity.CRITICAL,
        )

    def evidence_failed(self, route_id: str, inference_id: str, error: str) -> bool:
        """The evidence sidecar could not be installed, so nothing was committed."""
        return self.emit(
            route_id,
            "evidence-failed",
            f"the evidence sidecar for {inference_id} could not be installed",
            {"inferenceId": inference_id, "error": error},
            Severity.CRITICAL,
        )

    def cleanup_failed(self, route_id: str, inference_id: str, action: str, error: str) -> bool:
        """A completion action failed. The job is ``CLEANUP_FAILED``, never success."""
        return self.emit(
            route_id,
            "cleanup-failed",
            f"the {action} of {inference_id} failed",
            {"inferenceId": inference_id, "action": action, "error": error},
            Severity.CRITICAL,
        )

    def input_rejected(self, route_id: str, relative_path: str, reason: str) -> bool:
        """An input can never be admitted as it stands."""
        return self.emit(
            route_id,
            "input-rejected",
            f"route {route_id} refused an input",
            {"relativePath": relative_path, "reason": reason},
            Severity.WARNING,
        )

    def inference_failed(self, route_id: str, inference_id: str, code: str, message: str) -> bool:
        """An inference ended without a result. Its decision is HOLD."""
        return self.emit(
            route_id,
            "inference-failed",
            f"inference {inference_id} failed",
            {"inferenceId": inference_id, "code": code, "error": message},
            Severity.CRITICAL,
        )

    def disk_pressure(self, active: bool, path: str, free_mib: int, floor_mib: int) -> bool:
        """The state or cache filesystem is running out of room."""
        return self.alarm(
            None,
            "disk-pressure",
            active,
            "the component filesystem is low on free space",
            {"path": path, "freeMiB": int(free_mib), "floorMiB": int(floor_mib)},
            Severity.WARNING,
        )

    def gpu_pressure(self, active: bool, device: str, free_mib: int, needed_mib: int) -> bool:
        """A device cannot admit the models the routes need."""
        return self.alarm(
            None,
            "gpu-pressure",
            active,
            f"device {device} cannot admit a required model",
            {"device": device, "freeMiB": int(free_mib), "neededMiB": int(needed_mib)},
            Severity.WARNING,
        )
