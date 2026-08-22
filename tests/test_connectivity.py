"""What this processor reports about its instances, plus the library-facing wiring's testable seams.

`test_pipeline.py` covers the payload-agnostic core; this covers what the app wires into the library:
the instance-connectivity provider, the route parsing in `__init__`, the inbound handler's
guard/count/drop, the dispatch restamp+publish (and its failure event), the metric flush, and stop.
The app is handed the framework facade, so recording stand-ins for it are enough — no broker, no
transport, no threads (the two infinite worker loops, `run()`/`_run_route()`, are the live-runtime
seam, validated by the HOST/GREENGRASS smoke). Run them with `pytest`.
"""
import queue

import pytest

from image_processor.pipeline import ProcMsg, parse_route
from image_processor.ImageProcessor import METRIC_NAME, Stats, _now_ms, ImageProcessor


# --- the framework stand-ins: they record what the app publishes / emits ------------------------


class FakeIdentity:
    path = "factory-1/gw-01"
    component = "my-processor"


class FakeHeader:
    def __init__(self, name="Rollup", version="1.0"):
        self.name = name
        self.version = version


class FakeMsg:
    """The duck type the library's Message satisfies: `.header`, `.body`, `.get_identity()`."""

    def __init__(self, body=None, identity=None):
        self.header = FakeHeader()
        self.body = body if body is not None else {}
        self._identity = identity

    def get_identity(self):
        return self._identity


class RecordingBuilder:
    def __init__(self, sink, name, version):
        self._sink = sink
        self._msg = FakeMsg()
        self._msg.header = FakeHeader(name, version)

    def with_payload(self, body):
        self._msg.body = body
        return self

    def build(self):
        return self._msg


class RecordingEvents:
    def __init__(self, sink):
        self._sink = sink

    def emit(self, type, message, context, severity=None):
        self._sink.events.append((type, message, context))


class RecordingInstance:
    """`gg.instance(route.id)` — mints restamped messages and this instance's events facade."""

    def __init__(self, sink):
        self._sink = sink

    def new_message(self, name, version):
        return RecordingBuilder(self._sink, name, version)

    def events(self):
        return RecordingEvents(self._sink)


class FakeGg:
    """Records the app's registrations, publishes, events and metrics; getters are otherwise inert.

    `routes` maps instance id -> its raw `component.instances[]` config, so a test can inject a
    malformed route. `publish_raises` forces the transport to fail, to exercise the dispatch error
    path. `metric_raises` forces the metric emit to fail.
    """

    def __init__(self, routes=None, publish_raises=False, metric_raises=False,
                 unsubscribe_raises=False):
        self._routes = routes if routes is not None else {
            "temps": {"id": "temps", "subscribe": ["ecv1/+/+/+/data/#"], "publishTopic": "t"}
        }
        self._publish_raises = publish_raises
        self._metric_raises = metric_raises
        self._unsubscribe_raises = unsubscribe_raises
        self.connectivity_provider = None
        self.published = []
        self.published_northbound = []
        self.events = []
        self.metrics = []
        self.subscribed = []
        self.unsubscribed = []
        self.ready = False

    # config manager
    def get_config_manager(self):
        return self

    def add_config_change_listener(self, listener):
        pass

    def get_global_config(self):
        return {}

    def get_instance_ids(self):
        return list(self._routes)

    def get_instance_config(self, instance_id):
        return self._routes[instance_id]

    def get_component_identity(self):
        return FakeIdentity()

    # messaging
    def get_messaging(self):
        return self

    def subscribe(self, topic_filter, handler, max_concurrency, max_queue):
        self.subscribed.append(topic_filter)

    def unsubscribe(self, topic_filter):
        if self._unsubscribe_raises:
            raise RuntimeError("already gone")
        self.unsubscribed.append(topic_filter)

    def publish(self, topic, msg):
        if self._publish_raises:
            raise RuntimeError("broker down")
        self.published.append((topic, msg))

    def publish_northbound(self, topic, msg, qos):
        if self._publish_raises:
            raise RuntimeError("broker down")
        self.published_northbound.append((topic, msg))

    # metrics
    def get_metrics(self):
        return self

    def define_metric(self, metric):
        pass

    def emit_metric(self, name, values):
        if self._metric_raises:
            raise RuntimeError("metrics sink down")
        self.metrics.append((name, values))

    # instance facade + lifecycle
    def instance(self, instance_id):
        return RecordingInstance(self)

    def set_instance_connectivity_provider(self, provider):
        self.connectivity_provider = provider

    def set_ready(self, ready):
        self.ready = ready


# --- instance connectivity -----------------------------------------------------------------------


def test_the_component_registers_an_instance_connectivity_provider():
    # ONE provider, TWO surfaces: the library pushes this sample into every `state` keepalive's
    # instances[] AND returns it from the built-in `status` verb. A console that subscribes and a
    # console that asks cannot get different answers.
    gg = FakeGg()

    ImageProcessor(gg)

    assert gg.connectivity_provider is not None


def test_a_processor_owns_no_connections_so_it_reports_no_instances():
    # A route is a subscription, not a link to a device. No instances -> no instances[] section ->
    # `status` answers exactly as `ping`. That is a real answer, not a missing one. Once a stage of
    # yours does own a connection, report it here — and assert that a configured-but-down one is
    # still reported.
    gg = FakeGg()

    app = ImageProcessor(gg)

    assert app.instance_connectivity() == []
    assert gg.connectivity_provider() == []


def test_a_config_change_is_accepted_by_the_listener():
    app = ImageProcessor(FakeGg())
    assert app.on_configuration_change({"component": {"global": {}}}) is True


# --- route parsing in __init__ -------------------------------------------------------------------


def test_a_malformed_route_is_skipped_but_the_valid_ones_still_run():
    gg = FakeGg(routes={
        "good": {"id": "good", "subscribe": ["ecv1/+/+/+/data/#"], "publishTopic": "t"},
        "bad": {"id": "bad", "publishTopic": "t", "nosuchkey": 1},  # rejected by parse_route
    })

    app = ImageProcessor(gg)

    # The good route survived; the malformed one was skipped with a warning, not fatal.
    assert [r.id for r in app._routes] == ["good"]


def test_a_component_with_no_valid_routes_fails_loudly_rather_than_idling():
    gg = FakeGg(routes={"bad": {"id": "bad", "publishTopic": "t", "nosuchkey": 1}})
    with pytest.raises(ValueError, match="no valid routes"):
        ImageProcessor(gg)


# --- Stats: the interval counters ----------------------------------------------------------------


def test_stats_count_and_reset_on_swap():
    stats = Stats()
    stats.incr("received", 3)
    stats.incr("dropped")
    snapshot = stats.swap()
    assert snapshot == {"received": 3.0, "published": 0.0, "dropped": 1.0, "errors": 0.0}
    # swap reset the interval: a second swap sees only what accrued since.
    assert stats.swap()["received"] == 0.0


# --- the inbound handler: guard, count, bounded-drop ---------------------------------------------


def test_the_handler_drops_our_own_echo_counts_receipts_and_drops_on_a_full_queue():
    app = ImageProcessor(FakeGg())
    q = queue.Queue(maxsize=1)
    on_message = app._handler(q)

    # Our own output is dropped by the self-echo guard — not received, not queued.
    on_message("t", FakeMsg(identity=FakeIdentity()))
    assert q.qsize() == 0
    assert app._stats.swap()["received"] == 0.0

    # A foreign message is counted and queued.
    on_message("t", FakeMsg(identity=None))
    assert q.qsize() == 1

    # The queue is full now: the next arrival must DROP and be COUNTED, never block.
    on_message("t", FakeMsg(identity=None))
    snap = app._stats.swap()
    assert snap["received"] == 2.0
    assert snap["dropped"] == 1.0


# --- dispatch: identity restamp, target routing, failure event -----------------------------------


def _out_batch():
    return [ProcMsg("t", FakeMsg(body={"v": 1}))]


def test_dispatch_publishes_locally_and_restamps_identity():
    gg = FakeGg()
    app = ImageProcessor(gg)
    route = parse_route({"id": "temps", "publishTopic": "out", "target": "local"})

    app._dispatch(route, _out_batch())

    assert len(gg.published) == 1
    topic, msg = gg.published[0]
    assert topic == "out"
    assert msg.body == {"v": 1}, "the payload rode through the restamp"
    assert app._stats.swap()["published"] == 1.0


def test_dispatch_routes_a_northbound_target_to_the_northbound_publish():
    gg = FakeGg()
    app = ImageProcessor(gg)
    route = parse_route({"id": "temps", "publishTopic": "out", "target": "northbound"})

    app._dispatch(route, _out_batch())

    assert len(gg.published_northbound) == 1
    assert not gg.published


def test_a_failed_publish_is_counted_and_raises_an_event_rather_than_killing_the_route():
    gg = FakeGg(publish_raises=True)
    app = ImageProcessor(gg)
    route = parse_route({"id": "temps", "publishTopic": "out", "target": "local"})

    app._dispatch(route, _out_batch())

    assert app._stats.swap()["errors"] == 1.0
    assert [e[0] for e in gg.events] == ["publish-failed"]


# --- metric flush + stop -------------------------------------------------------------------------


def test_emit_metrics_flushes_the_counters_and_survives_a_sink_outage():
    gg = FakeGg()
    app = ImageProcessor(gg)
    app._stats.incr("received", 2)
    app._emit_metrics()
    assert gg.metrics[-1][0] == METRIC_NAME
    assert gg.metrics[-1][1]["received"] == 2.0

    # A metric-sink outage must not propagate out of the flush.
    gg2 = FakeGg(metric_raises=True)
    app2 = ImageProcessor(gg2)
    app2._emit_metrics()  # does not raise


def test_stop_is_idempotent_and_drops_every_subscription():
    gg = FakeGg()
    app = ImageProcessor(gg)
    # Pretend run() had subscribed this route's filters (stop drops whatever the routes declare).
    app.stop()
    assert gg.unsubscribed == ["ecv1/+/+/+/data/#"]
    # A second stop is a no-op — no double unsubscribe.
    app.stop()
    assert gg.unsubscribed == ["ecv1/+/+/+/data/#"]


def test_stop_survives_an_unsubscribe_that_fails():
    # Shutdown must not raise even if a subscription is already gone.
    app = ImageProcessor(FakeGg(unsubscribe_raises=True))
    app.stop()  # does not propagate the unsubscribe failure


def test_now_ms_is_a_monotonic_wall_clock_in_milliseconds():
    a = _now_ms()
    assert isinstance(a, int)
    assert _now_ms() >= a
