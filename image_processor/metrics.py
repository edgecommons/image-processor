"""The eight metric groups (DESIGN.md 12.3, LLD 8).

Metrics answer the operational questions that events must not be used for, because they happen
per image: how much was discovered, how deep the queue is, what the cache and the device hold,
how long inference took, how the outbox and the completions are doing, and how much disk is
left. They are defined once through ``MetricBuilder``, so the same names, measures, and
dimensions reach a log file, CloudWatch, Prometheus, or the reserved ``metric`` class depending
only on ``metricEmission.target``.

Cardinality is a deliberate constraint: no file name, capture id, model version, or inference id
is ever a dimension. Those identify one image, and a dimension that identifies one image turns a
metric backend into an unbounded index of them. They belong in the result, the sidecar, the
events, and the command replies, all of which are bounded by something other than traffic.

Three kinds of measure feed a group, and they are flushed together on the interval: counters the
component increments as work happens, averages it observes per job, and gauges sampled from the
live subsystems at flush time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from edgecommons.metrics.metric_builder import MetricBuilder

logger = logging.getLogger(__name__)

#: How often the counters are flushed, in seconds.
DEFAULT_INTERVAL_SECS = 60

#: The metric groups and their measures, as DESIGN.md 12.3 names them.
METRIC_GROUPS = {
    "ImageProcessorDiscovery": (
        ("discovered", "Count"),
        ("rejected", "Count"),
        ("rescans", "Count"),
        ("nudges", "Count"),
        ("hintsAccepted", "Count"),
        ("hintsRejected", "Count"),
        ("hintsUnmapped", "Count"),
        ("captureRecords", "Count"),
        ("triggersAccepted", "Count"),
        ("triggersRejected", "Count"),
    ),
    "ImageProcessorQueue": (
        ("admitted", "Count"),
        ("queued", "Count"),
        ("retryWaiting", "Count"),
        ("dispatched", "Count"),
        ("oldestAgeSecs", "Seconds"),
        ("pausedRoutes", "Count"),
    ),
    "ImageProcessorModelCache": (
        ("staged", "Count"),
        ("activated", "Count"),
        ("stagingFailures", "Count"),
        ("warmupFailures", "Count"),
        ("rollbacks", "Count"),
        ("cachedBundles", "Count"),
        ("routesStaging", "Count"),
    ),
    "ImageProcessorGpu": (
        ("residentModels", "Count"),
        ("residentMiB", "Megabytes"),
        ("loads", "Count"),
        ("evictions", "Count"),
        ("recycles", "Count"),
        ("healthyCells", "Count"),
    ),
    "ImageProcessorInference": (
        ("succeeded", "Count"),
        ("failed", "Count"),
        ("retried", "Count"),
        ("exhausted", "Count"),
        ("blocked", "Count"),
        ("queueMs", "Milliseconds"),
        ("inferenceMs", "Milliseconds"),
        ("totalMs", "Milliseconds"),
    ),
    "ImageProcessorOutbox": (
        ("pending", "Count"),
        ("published", "Count"),
        ("attempted", "Count"),
        ("failed", "Count"),
        ("exhausted", "Count"),
        ("reservedBytes", "Bytes"),
    ),
    "ImageProcessorCompletion": (
        ("completed", "Count"),
        ("archived", "Count"),
        ("deleted", "Count"),
        ("quarantined", "Count"),
        ("retained", "Count"),
        ("failed", "Count"),
        ("mirrored", "Count"),
    ),
    "ImageProcessorDisk": (
        ("stateDbBytes", "Bytes"),
        ("modelCacheBytes", "Bytes"),
        ("stagingBytes", "Bytes"),
        ("freeMiB", "Megabytes"),
    ),
}

#: Measures whose interval value is a mean of what was observed, not a sum.
AVERAGED = frozenset({"queueMs", "inferenceMs", "totalMs", "oldestAgeSecs"})


class ProcessorMetrics:
    """Defines the metric groups, accumulates the interval, and flushes it.

    Args:
        gg: The EdgeCommons handle, for the metric service and the config manager.
        interval_secs: How often the accumulated interval is emitted.
        gauges: Returns ``{group: {measure: value}}`` sampled at flush time. It is called on the
            flush thread, so it must not block.
    """

    def __init__(
        self,
        gg: Any,
        *,
        interval_secs: float = DEFAULT_INTERVAL_SECS,
        gauges: Optional[Callable[[], dict]] = None,
    ) -> None:
        """Build the accumulator. Nothing is defined until :meth:`define` runs."""
        self._gg = gg
        self._emitter = gg.get_metrics()
        self.interval_secs = float(interval_secs)
        self._gauges = gauges
        self._lock = threading.Lock()
        self._counters: dict = {}
        self._samples: dict = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.flushes = 0

    # -- definition --------------------------------------------------------------------

    def define(self) -> "ProcessorMetrics":
        """Define every group with the metric service.

        Returns:
            This accumulator.
        """
        config_manager = None
        accessor = getattr(self._gg, "get_config_manager", None)
        if callable(accessor):
            try:
                config_manager = accessor()
            except Exception:  # noqa: BLE001 - a bring-up without configuration still defines
                config_manager = None
        for name, measures in METRIC_GROUPS.items():
            builder = MetricBuilder.create(name)
            if config_manager is not None:
                builder = builder.with_config(config_manager)
            for measure, unit in measures:
                builder = builder.add_measure(measure, unit, 60)
            self._emitter.define_metric(builder.build())
        return self

    # -- accumulation ------------------------------------------------------------------

    def incr(self, group: str, measure: str, value: float = 1.0) -> None:
        """Add to a counter for this interval.

        Args:
            group: The metric group.
            measure: The measure inside it.
            value: How much to add.
        """
        with self._lock:
            self._counters[(group, measure)] = self._counters.get((group, measure), 0.0) + float(
                value
            )

    def observe(self, group: str, measure: str, value: float) -> None:
        """Record one observation of an averaged measure.

        Args:
            group: The metric group.
            measure: The measure inside it.
            value: The observation.
        """
        with self._lock:
            total, count = self._samples.get((group, measure), (0.0, 0))
            self._samples[(group, measure)] = (total + float(value), count + 1)

    def snapshot(self) -> dict:
        """Read the interval and reset it.

        Returns:
            ``{group: {measure: value}}`` with counters summed and observations averaged.
        """
        with self._lock:
            counters, self._counters = self._counters, {}
            samples, self._samples = self._samples, {}
        out: dict = {}
        for (group, measure), value in counters.items():
            out.setdefault(group, {})[measure] = float(value)
        for (group, measure), (total, count) in samples.items():
            out.setdefault(group, {})[measure] = float(total / count) if count else 0.0
        return out

    # -- emission ----------------------------------------------------------------------

    def flush(self) -> int:
        """Emit one interval: the accumulated counters merged with the sampled gauges.

        A group with nothing to say is still emitted with zeros for its counters, because a
        counter that stops appearing looks like a component that stopped rather than one that had
        a quiet minute.

        Returns:
            How many groups were emitted.
        """
        values = self.snapshot()
        if self._gauges is not None:
            try:
                for group, measures in (self._gauges() or {}).items():
                    values.setdefault(group, {}).update(
                        {str(key): float(value) for key, value in measures.items()}
                    )
            except Exception:  # noqa: BLE001 - a sampling failure must not lose the counters
                logger.warning("the metric gauges could not be sampled", exc_info=True)
        emitted = 0
        for group, measures in METRIC_GROUPS.items():
            observed = values.get(group, {})
            payload = {
                measure: float(observed.get(measure, 0.0))
                for measure, _unit in measures
                if measure in observed or measure not in AVERAGED
            }
            if not payload:
                continue
            try:
                self._emitter.emit_metric(group, payload)
                emitted += 1
            except Exception:  # noqa: BLE001 - one bad target must not stop the rest
                logger.warning("could not emit %s", group, exc_info=True)
        self.flushes += 1
        return emitted

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> "ProcessorMetrics":
        """Start the flush thread.

        Returns:
            This accumulator.
        """
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="image-processor-metrics", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the flush thread, emitting the interval that was in progress.

        Args:
            timeout_s: How long to wait for the thread.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.debug("the final metric flush failed", exc_info=True)

    def _loop(self) -> None:
        """Flush on the interval until stopped."""
        while not self._stop.wait(self.interval_secs):
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - the thread outlives one bad flush
                logger.warning("a metric flush failed", exc_info=True)


__all__ = [
    "AVERAGED",
    "DEFAULT_INTERVAL_SECS",
    "METRIC_GROUPS",
    "ProcessorMetrics",
]
