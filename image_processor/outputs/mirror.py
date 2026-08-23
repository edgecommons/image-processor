"""The decision mirror: a few normalized readings on the ``data`` class (DESIGN.md 12.2, LLD 8).

The mirror exists so an ordinary telemetry consumer -- a historian, a dashboard, an HMI tag --
can read a line-clearance verdict without learning this component result body. It publishes the
configured ``decisionSignals`` through the ``data()`` facade, which mints
``ecv1/{device}/image-processor/{routeId}/data/<signalId>`` and imposes the
``SouthboundSignalUpdate`` body.

It is best effort, deliberately and permanently. It is derived from the already-committed result,
it is not cleanup-gating, and a failure to publish it never fails a job (D-IP-6). A consumer
enforcing a safety gate subscribes to ``app/inference/result`` instead: any missing, failed,
stale, or degraded inference is not clear, and a mirror that silently stopped updating looks
exactly like one that keeps reporting the last value.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from edgecommons.facades import Quality

from image_processor.engine.decision import resolve_path

logger = logging.getLogger(__name__)

#: Value types a signal sample carries. A document is not a reading.
_SCALARS = (bool, int, float, str)


class DecisionMirror:
    """Publishes a route configured decision signals from a committed result body.

    Args:
        gg: The EdgeCommons handle. The mirror uses ``gg.instance(routeId).data()``, so every
            reading carries the route own instance token.
        on_error: Called with ``(route_id, signal_id, error)`` when one signal cannot be
            published. The mirror still returns normally.
    """

    def __init__(self, gg: Any, *, on_error: Optional[Any] = None) -> None:
        """Build the mirror over the EdgeCommons handle."""
        self._gg = gg
        self._on_error = on_error
        self.counters = {"published": 0, "unresolved": 0, "unsupported": 0, "failed": 0}

    def publish(self, route_id: str, decision_signals: Iterable, result_body: dict) -> int:
        """Mirror one result body onto the route configured signals.

        Args:
            route_id: The route that produced the result.
            decision_signals: The route ``outputs.decisionSignals`` entries, each carrying an
                ``id`` and a JSONPath ``value``.
            result_body: The committed result body.

        Returns:
            How many readings were published.
        """
        signals = list(decision_signals or ())
        if not signals:
            return 0
        quality = Quality.GOOD if result_body.get("status") == "SUCCEEDED" else Quality.BAD
        published = 0
        for signal in signals:
            signal_id = str(getattr(signal, "id", "") or "")
            expression = str(getattr(signal, "value", "") or "")
            value = resolve_path(result_body, expression)
            if value is None:
                self.counters["unresolved"] += 1
                logger.debug("route %s: %s resolved nothing", route_id, expression)
                continue
            if not isinstance(value, _SCALARS):
                self.counters["unsupported"] += 1
                logger.debug(
                    "route %s: %s resolved a %s, which is not a reading",
                    route_id,
                    expression,
                    type(value).__name__,
                )
                continue
            if self._publish_one(route_id, signal_id, value, quality):
                published += 1
        return published

    def _publish_one(self, route_id: str, signal_id: str, value: Any, quality: Quality) -> bool:
        """Publish one reading, swallowing a failure the way a mirror must."""
        try:
            self._gg.instance(route_id).data().publish(signal_id, value, quality)
        except Exception as exc:  # noqa: BLE001 - the mirror never fails a job
            self.counters["failed"] += 1
            logger.warning(
                "route %s could not mirror %s: %s", route_id, signal_id, exc, exc_info=True
            )
            if self._on_error is not None:
                try:
                    self._on_error(route_id, signal_id, str(exc))
                except Exception:  # noqa: BLE001 - reporting a mirror failure cannot fail a job
                    logger.debug("the mirror error callback failed", exc_info=True)
            return False
        self.counters["published"] += 1
        return True
