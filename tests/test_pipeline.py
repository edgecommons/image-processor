"""The processor's invariants, tested without a broker, a transport, or the library.

`image_processor/pipeline.py` is deliberately payload-agnostic and library-free -- it works on a duck-typed
message (`.header`, `.body`, `.get_identity()`), which is exactly the shape of
`edgecommons.messaging.message.Message`. So the stages, the tick, the self-echo guard and the route
parser are all testable as pure logic, in-process, in milliseconds. Run them with `pytest`.
"""
import pytest

from image_processor.pipeline import (
    DEFAULT_MAX_QUEUE,
    DEFAULT_TICK_MS,
    CountPerTick,
    FieldEquals,
    Pipeline,
    ProcMsg,
    build_stage,
    is_self_echo,
    parse_route,
    pluck,
)


# --- the message stand-in: the same duck type the library's Message satisfies -------------------


class FakeIdentity:
    def __init__(self, path, component):
        self.path = path
        self.component = component


class FakeHeader:
    def __init__(self, name="T", version="1.0"):
        self.name = name
        self.version = version


class FakeMsg:
    def __init__(self, body, identity=None):
        self.header = FakeHeader()
        self.body = body
        self.identity = identity

    def get_identity(self):
        return self.identity


def msg(body, identity=None):
    return ProcMsg("ecv1/gw/x/main/data/t", FakeMsg(body, identity))


# --- stages: 0..N out, and the tick --------------------------------------------------------------


def test_a_filter_stage_drops_what_does_not_match():
    p = Pipeline([FieldEquals("quality", "GOOD")])

    assert len(p.run([msg({"quality": "GOOD"})])) == 1
    assert p.run([msg({"quality": "BAD"})]) == [], "a filter that does not match emits nothing"


def test_a_stateful_stage_emits_on_the_tick_not_on_arrival():
    p = Pipeline([CountPerTick()])

    # Three messages arrive: nothing goes downstream yet.
    for _ in range(3):
        assert p.run([msg({"v": 1})]) == []

    # The tick closes the window and emits one rollup.
    out = p.run([], now_ms=1_000)
    assert len(out) == 1
    assert out[0].msg.body["count"] == 3

    # A second tick with nothing accumulated emits nothing -- an empty window is not an event.
    assert p.run([], now_ms=2_000) == []


def test_stages_chain_and_a_tick_flows_through_the_rest_of_the_pipeline():
    # Filter, then count. A window closing in stage 2 is emitted on the same pass.
    p = Pipeline([FieldEquals("quality", "GOOD"), CountPerTick()])

    p.run([msg({"quality": "GOOD"})])
    p.run([msg({"quality": "BAD"})])  # filtered out before it reaches the counter

    out = p.run([], now_ms=1_000)
    assert len(out) == 1
    assert out[0].msg.body["count"] == 1, "only the GOOD message reached the counter"
    assert out[0].msg.body["last"] == {"quality": "GOOD"}


def test_a_rollup_does_not_mutate_the_message_it_rolled_up():
    stage = CountPerTick()
    original = msg({"v": 1})
    stage.process(original)

    out = stage.on_tick(1_000)

    assert out[0].msg.body == {"count": 1, "last": {"v": 1}}
    assert original.msg.body == {"v": 1}, "the carrier is copied, not rewritten in place"


def test_an_empty_pipeline_is_a_pass_through_republisher():
    m = msg({"v": 1})
    assert Pipeline([]).run([m]) == [m]


def test_pluck_walks_a_dotted_path():
    body = {"signal": {"id": "temp-1"}}
    assert pluck(body, "signal.id") == "temp-1"
    assert pluck(body, "signal.nope") is None
    assert pluck(body, "nope.nope") is None
    assert pluck("not-an-object", "signal.id") is None


# --- the self-echo guard -------------------------------------------------------------------------


def test_the_self_echo_guard_drops_our_own_output():
    # THE guard. Without it, a processor that publishes onto a class it also subscribes to consumes
    # its own output, reprocesses it, republishes it, and saturates the device.
    mine = FakeMsg({}, FakeIdentity("factory-1/gw-01", "my-processor"))
    assert is_self_echo(mine, "factory-1/gw-01", "my-processor") is True


def test_the_self_echo_guard_keeps_everyone_elses_messages():
    other_component = FakeMsg({}, FakeIdentity("factory-1/gw-01", "modbus-adapter"))
    other_device = FakeMsg({}, FakeIdentity("factory-1/gw-02", "my-processor"))
    anonymous = FakeMsg({}, identity=None)

    assert is_self_echo(other_component, "factory-1/gw-01", "my-processor") is False
    assert is_self_echo(other_device, "factory-1/gw-01", "my-processor") is False
    assert is_self_echo(anonymous, "factory-1/gw-01", "my-processor") is False


# --- route config --------------------------------------------------------------------------------


def test_a_route_parses_from_its_instance_config():
    route = parse_route(
        {
            "id": "temps",
            "subscribe": ["ecv1/+/+/+/data/#"],
            "publishTopic": "ecv1/gw01/proc/main/data/rollup",
            "target": "northbound",
            "pipeline": [
                {"fieldEquals": {"path": "signal.id", "value": "temp-1"}},
                {"countPerTick": {}},
            ],
            "tickMs": 5000,
        }
    )

    assert route.id == "temps"
    assert route.target == "northbound"
    assert len(route.pipeline) == 2
    assert route.tick_ms == 5_000
    assert route.max_queue == DEFAULT_MAX_QUEUE, "the queue is bounded by default"


def test_the_defaults_are_the_common_case():
    route = parse_route({"id": "r", "publishTopic": "t"})

    assert route.target == "local", "the device-local bus is the common target"
    assert route.pipeline == [], "no stages == a pass-through republisher"
    assert route.tick_ms == DEFAULT_TICK_MS


def test_global_defaults_apply_to_a_route_that_does_not_override_them():
    defaults = {"tickMs": 1000, "maxQueue": 8}

    inherited = parse_route({"id": "r", "publishTopic": "t"}, defaults)
    assert (inherited.tick_ms, inherited.max_queue) == (1000, 8)

    overridden = parse_route({"id": "r", "publishTopic": "t", "tickMs": 50}, defaults)
    assert (overridden.tick_ms, overridden.max_queue) == (50, 8)


def test_an_unknown_config_key_is_rejected_rather_than_ignored():
    # A typo'd route key is a mistake, not a no-op. A knob that silently does nothing is the worst
    # kind of bug to find in the field.
    with pytest.raises(ValueError, match="unknown route key"):
        parse_route({"id": "r", "publishTopic": "t", "pipelnie": []})


def test_a_bad_route_is_rejected_at_config_time_not_on_the_first_message():
    with pytest.raises(ValueError):
        parse_route({"id": "r"})  # no publishTopic
    with pytest.raises(ValueError):
        parse_route({"id": "r", "publishTopic": "t", "target": "sideways"})
    with pytest.raises(ValueError):
        parse_route({"id": "r", "publishTopic": "t", "maxQueue": 0})
    with pytest.raises(ValueError, match="unknown stage"):
        parse_route({"id": "r", "publishTopic": "t", "pipeline": [{"nosuchstage": {}}]})


def test_a_stage_is_a_single_key_object():
    assert isinstance(build_stage({"countPerTick": {}}), CountPerTick)
    with pytest.raises(ValueError):
        build_stage({"fieldEquals": {"path": "a", "value": 1}, "countPerTick": {}})
    with pytest.raises(ValueError):
        build_stage({"fieldEquals": {"path": "a"}})  # `value` is required


def test_a_stage_with_no_args_is_the_empty_object_but_a_non_object_arg_is_rejected():
    # `{"countPerTick": null}` is the natural JSON for an argument-less stage -- accept it as `{}`.
    assert isinstance(build_stage({"countPerTick": None}), CountPerTick)
    # But an argument block that is present and not an object is a config mistake, not a no-op.
    with pytest.raises(ValueError, match="takes an object of arguments"):
        build_stage({"fieldEquals": 5})


def test_build_pipeline_materializes_the_route_stages():
    route = parse_route({
        "id": "r", "publishTopic": "t",
        "pipeline": [{"fieldEquals": {"path": "a", "value": 1}}, {"countPerTick": {}}],
    })
    pipeline = route.build_pipeline()
    # An empty pipeline is a pass-through; this one has two real stages.
    assert pipeline.run([]) == []
    assert len(pipeline._stages) == 2


def test_an_instance_that_is_not_an_object_or_has_a_malformed_field_is_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        parse_route("not-an-object")
    with pytest.raises(ValueError, match="subscribe"):
        parse_route({"id": "r", "publishTopic": "t", "subscribe": ["ok", ""]})
    with pytest.raises(ValueError, match="pipeline"):
        parse_route({"id": "r", "publishTopic": "t", "pipeline": "not-a-list"})
