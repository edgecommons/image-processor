"""ImageProcessor -- a processing component.

A **processor** subscribes to messages, transforms them, and forwards the result. This scaffold
wires that shape end to end; the transformation itself lives in ``image_processor/pipeline.py``, which is where
your code goes.

.. code-block:: text

    subscribe(filter) --> bounded queue --> one thread per route --> publish
                                               (Pipeline)           local | northbound

Each entry of ``component.instances[]`` is **one route**: topic filters, a pipeline of stages, and a
target. Routes are independent -- one thread each -- so a slow route cannot stall another, and the
per-key state inside a stage needs no lock.

Why a processor uses ``get_messaging()`` and not ``data()``
-----------------------------------------------------------
Worth reading twice, because it is the mistake this archetype invites. The ``data()`` facade is for a
component that *produces* readings: it mints its own topic from a signal id and imposes the
``SouthboundSignalUpdate`` body. A processor is **payload-agnostic** -- it republishes what it was
handed, on a topic its route names. Routing that through ``data()`` would rewrite both the topic and
the body, which is exactly what a republisher must not do. So: raw ``gg.get_messaging()``, and topics
from config.

Two guards that are not optional
--------------------------------
* **Self-echo.** A processor that publishes onto a class it also subscribes to will consume its own
  output, reprocess it, republish it, and saturate the device.
  :func:`~app.pipeline.is_self_echo` drops anything carrying our own identity. (``main.py`` also asks
  the transport not to echo, but only Greengrass IPC can honour that -- an MQTT broker redelivers our
  own publishes to our own wildcard subscription regardless. The guard is what actually holds.)
* **Identity restamp.** What we publish is *ours*. Without the restamp the fleet cannot tell who
  emitted a message -- and the self-echo guard downstream cannot work either.
"""
import logging
import queue
import threading
import time

from edgecommons.config.manager.configuration_change_listener import (
    ConfigurationChangeListener,
)
from edgecommons.facades import Severity
from edgecommons.messaging.qos import Qos
from edgecommons.metrics.metric_builder import MetricBuilder

from image_processor.pipeline import ProcMsg, is_self_echo, parse_route

logger = logging.getLogger("ImageProcessor")

#: The metric this component emits each interval.
METRIC_NAME = "processorThroughput"
#: How often the counters above are flushed as a metric, in seconds.
METRIC_INTERVAL_SECS = 60


def _now_ms() -> int:
    return int(time.time() * 1000)


class Stats:
    """Counters, reported as a metric each interval.

    ``dropped`` is the one that must never be invisible: a processor that silently discards messages
    is worse than one that crashes.
    """

    MEASURES = ("received", "published", "dropped", "errors")

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {m: 0 for m in self.MEASURES}

    def incr(self, measure: str, n: int = 1) -> None:
        with self._lock:
            self._counts[measure] += n

    def swap(self) -> dict:
        """Read the counters and reset them -- the interval semantics the metric measures declare."""
        with self._lock:
            snapshot = {k: float(v) for k, v in self._counts.items()}
            self._counts = {m: 0 for m in self.MEASURES}
        return snapshot


class ImageProcessor(ConfigurationChangeListener):
    """One route per ``component.instances[]`` entry; one worker thread per route."""

    def __init__(self, gg):
        self._gg = gg
        self._cm = gg.get_config_manager()
        self._cm.add_config_change_listener(self)
        self._messaging = gg.get_messaging()
        self._metrics = gg.get_metrics()
        self._stats = Stats()
        self._stop = threading.Event()
        self._threads = []

        self._metrics.define_metric(
            MetricBuilder.create(METRIC_NAME)
            .with_config(self._cm)
            .add_measure("received", "Count", 60)
            .add_measure("published", "Count", 60)
            .add_measure("dropped", "Count", 60)
            .add_measure("errors", "Count", 60)
            .build()
        )

        # One route per instance. A malformed route is skipped with a warning rather than killing the
        # component -- but if *every* route is malformed there is nothing to run, and failing loudly
        # beats idling silently.
        defaults = (self._cm.get_global_config() or {}).get("defaults", {})
        self._routes = []
        for instance_id in self._cm.get_instance_ids():
            try:
                self._routes.append(parse_route(self._cm.get_instance_config(instance_id) or {}, defaults))
            except ValueError as e:
                logger.warning("skipping malformed route `%s`: %s", instance_id, e)
        if not self._routes:
            raise ValueError("no valid routes in component.instances[]")

        # Our own identity, captured once: the self-echo guard compares against it per message.
        identity = self._cm.get_component_identity()
        self._me = (identity.path, identity.component)

        # --- instance connectivity: ONE provider, TWO surfaces. Whatever it returns is pushed into
        # the `state` keepalive's instances[] on every tick AND returned by the built-in `status`
        # verb when a console asks — so whoever watches and whoever asks can never get different
        # answers.
        gg.set_instance_connectivity_provider(self.instance_connectivity)

    def instance_connectivity(self) -> list:
        """The per-instance connectivity this component reports.

        A processor owns **no southbound connections** — its routes are subscriptions on a bus the
        library already reports on, not links to a device — so it reports **no instances**. That is
        a real answer, not a missing one: with no instances the ``instances[]`` section is omitted
        and ``status`` answers exactly as ``ping`` does. The provider is registered anyway, so the
        seam is visible the day a route of yours does own a connection.

        When one does (an enrichment stage that calls a database or an upstream API), return one
        entry per connection::

            from edgecommons.heartbeat.instance_connectivity import InstanceConnectivity

            return [
                InstanceConnectivity.of("enrich", self._db.is_connected(), "postgres://plant-db")
                .with_state("ONLINE")
                .with_attributes({"pool": self._db.pool_size()})
            ]

        ``connected`` is the **normalized** flag — always present, so a console renders a health dot
        without knowing this component's vocabulary. ``state`` is that vocabulary (``ONLINE`` /
        ``CONNECTING`` / ``BACKOFF`` / ``DISABLED``: a boolean cannot tell "reconnecting" from "gave
        up"). ``attributes`` is the open bag for domain data, so what only you understand never
        destabilizes the two fields everyone reads.

        Keep it cheap and non-blocking — it is sampled on every keepalive tick (5 s by default).
        """
        return []

    def on_configuration_change(self, configuration) -> bool:
        logger.info("configuration changed")
        return True

    def run(self):  # pragma: no cover - live-runtime seam: subscribes over a real transport, spawns the per-route worker threads, and blocks on the metric-flush loop until shutdown; exercised by the HOST/GREENGRASS smoke, not offline unit tests (edgecommons AGENTS.md validation matrix). The testable pieces it calls (_handler, _dispatch, _emit_metrics) have their own tests.
        """Subscribe every route, start its worker, then flush metrics until shutdown."""
        for route in self._routes:
            q = queue.Queue(maxsize=route.max_queue)
            for topic_filter in route.subscribe:
                # max_concurrency=1: the transport dispatches this route's messages in order, and the
                # route's own thread is the only thing that touches its pipeline state.
                self._messaging.subscribe(topic_filter, self._handler(q), 1, route.max_queue)
                logger.info("[%s] subscribed: %s", route.id, topic_filter)

            t = threading.Thread(
                target=self._run_route, args=(route, q), name=f"route-{route.id}", daemon=True
            )
            t.start()
            self._threads.append(t)

        self._gg.set_ready(True)
        while not self._stop.wait(METRIC_INTERVAL_SECS):
            self._emit_metrics()
        self._emit_metrics()

    def stop(self):
        """Stop the routes and drop their subscriptions. Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        for route in self._routes:
            for topic_filter in route.subscribe:
                try:
                    self._messaging.unsubscribe(topic_filter)
                except Exception as e:  # noqa: BLE001 - shutdown must not raise
                    logger.debug("[%s] unsubscribe failed: %s", route.id, e)
        for t in self._threads:
            t.join(timeout=5)

    # --- the inbound seam ------------------------------------------------------------------------

    def _handler(self, q: "queue.Queue"):
        """The subscription callback: guard, count, and hand the message to the route's queue."""

        def on_message(topic, msg):
            if is_self_echo(msg, *self._me):
                return  # our own output; consuming it would loop forever
            self._stats.incr("received")
            try:
                # put_nowait, never put: a full queue must DROP and be COUNTED, not block the
                # transport's dispatch thread.
                q.put_nowait(ProcMsg(topic, msg))
            except queue.Full:
                self._stats.incr("dropped")

        return on_message

    # --- one route's thread ----------------------------------------------------------------------

    def _run_route(self, route, q: "queue.Queue"):  # pragma: no cover - live-runtime seam: the per-route worker's infinite queue-drain/tick loop, driven by real inbound traffic and wall-clock ticks; exercised by the HOST/GREENGRASS smoke, not offline unit tests. The per-message work it delegates to (_dispatch + the pure Pipeline) is unit-tested directly.
        """Two things happen here, and they are the archetype: a message arrived -> run the pipeline;
        the tick came due -> let stateful stages emit.

        The tick is checked against the clock on every pass, not merely when the queue times out: a
        route that never stops receiving would otherwise never tick, and its windows would never
        close -- the failure mode of a busy processor is exactly the one you cannot afford to have.
        """
        pipeline = route.build_pipeline()
        tick_secs = route.tick_ms / 1000.0
        next_tick = time.monotonic() + tick_secs

        while not self._stop.is_set():
            try:
                m = q.get(timeout=max(0.0, next_tick - time.monotonic()))
            except queue.Empty:
                pass
            else:
                self._dispatch(route, pipeline.run([m]))

            if time.monotonic() >= next_tick:
                self._dispatch(route, pipeline.run([], _now_ms()))
                next_tick = time.monotonic() + tick_secs

        # A final tick on the way out, so a half-full window is emitted rather than silently lost.
        self._dispatch(route, pipeline.run([], _now_ms()))
        logger.info("[%s] route stopped", route.id)

    def _dispatch(self, route, out):
        for m in out:
            # Restamp identity: what we publish is OURS, not the producer's. new_message() stamps the
            # config-resolved identity with this route's instance token.
            msg = (
                self._gg.instance(route.id)
                .new_message(m.msg.header.name, m.msg.header.version)
                .with_payload(m.msg.body)
                .build()
            )
            try:
                if route.target == "northbound":
                    self._messaging.publish_northbound(route.publish_topic, msg, Qos.AT_LEAST_ONCE)
                else:
                    self._messaging.publish(route.publish_topic, msg)
                self._stats.incr("published")
            except Exception as e:  # noqa: BLE001 - one bad publish must not kill the route
                self._stats.incr("errors")
                logger.warning("[%s] publish failed: %s", route.id, e)
                self._gg.instance(route.id).events().emit(
                    "publish-failed",
                    f"route {route.id} could not publish",
                    {"route": route.id, "topic": route.publish_topic, "reason": str(e)},
                    severity=Severity.WARNING,
                )

    def _emit_metrics(self):
        try:
            self._metrics.emit_metric(METRIC_NAME, self._stats.swap())
        except Exception as e:  # noqa: BLE001
            logger.warning("metric emit failed: %s", e)
