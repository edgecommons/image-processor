"""Operator events: bounded, typed, and quiet about things that have not changed."""

from __future__ import annotations

import pytest
from edgecommons.facades import Severity

from image_processor.outputs.events import EVENT_TYPES, RouteEvents


class RecordingEvents:
    """An ``events()`` facade that records, or fails on command."""

    def __init__(self, log: list, instance, error=None) -> None:
        self.log = log
        self.instance = instance
        self.error = error

    def emit(self, event_type, message=None, context=None, severity=None) -> None:
        """Record one event."""
        if self.error is not None:
            raise self.error
        self.log.append((self.instance, "emit", event_type, message, context, str(severity)))

    def raise_alarm(self, event_type, message=None, context=None, severity=None) -> None:
        """Record one alarm raise."""
        self.log.append((self.instance, "raise", event_type, message, context, str(severity)))

    def clear_alarm(self, event_type, context=None, severity=None) -> None:
        """Record one alarm clear."""
        self.log.append((self.instance, "clear", event_type, None, context, str(severity)))


class Gg:
    """Just enough handle for the event helpers."""

    def __init__(self, error=None) -> None:
        self.log: list = []
        self.error = error

    def events(self):
        """The component-scope facade."""
        return RecordingEvents(self.log, None, self.error)

    def instance(self, instance_id):
        """The instance-scoped handle."""
        outer = self

        class _Handle:
            def events(self):
                return RecordingEvents(outer.log, instance_id, outer.error)

        return _Handle()


def test_a_route_condition_carries_the_route_instance_token():
    gg = Gg()

    RouteEvents(gg).model_staging_failed("cam-01", "line-clearance", "DIGEST_MISMATCH")

    instance, kind, event_type, message, context, severity = gg.log[0]
    assert instance == "cam-01"
    assert kind == "emit"
    assert event_type == "model-staging-failed"
    assert context == {"model": "line-clearance", "error": "DIGEST_MISMATCH"}
    assert severity == str(Severity.CRITICAL)


def test_a_component_condition_has_no_instance_token():
    gg = Gg()

    RouteEvents(gg).executor_recycled("gpu0-0", "CUDA error", 3)

    assert gg.log[0][0] is None
    assert gg.log[0][2] == "executor-recycled"


def test_an_alarm_publishes_only_on_a_transition():
    gg = Gg()
    events = RouteEvents(gg)

    assert events.route_degraded("cam-01", True, "no model") is True
    assert events.route_degraded("cam-01", True, "no model") is False
    assert events.route_degraded("cam-01", False) is True
    assert events.route_degraded("cam-01", False) is False

    assert [entry[1] for entry in gg.log] == ["raise", "clear"]
    assert events.active_alarms() == ()


def test_two_routes_hold_their_own_alarm_state():
    gg = Gg()
    events = RouteEvents(gg)

    events.route_degraded("cam-01", True)
    events.route_degraded("cam-02", True)
    events.route_degraded("cam-01", False)

    assert events.active_alarms() == (("cam-02", "route-degraded"),)


def test_context_values_are_bounded_and_scalar():
    gg = Gg()

    RouteEvents(gg).input_rejected("cam-01", "x" * 1000, "DIGEST_MISMATCH")

    context = gg.log[0][4]
    assert len(context["relativePath"]) == 512
    assert context["reason"] == "DIGEST_MISMATCH"


def test_a_non_scalar_context_value_is_rendered_as_text():
    gg = Gg()

    RouteEvents(gg).emit(None, "disk-pressure", "m", {"paths": ["/a", "/b"], "n": None})

    assert gg.log[0][4] == {"paths": "['/a', '/b']", "n": None}


def test_a_broker_failure_never_propagates():
    gg = Gg(RuntimeError("the broker is gone"))
    events = RouteEvents(gg)

    assert events.inference_failed("cam-01", "01k", "DECODE_FAILED", "truncated") is False
    assert events.counters["failed"] == 1


def test_a_failed_alarm_is_not_recorded_as_raised():
    class _Failing(Gg):
        def instance(self, instance_id):
            outer = self

            class _Handle:
                def events(self):
                    raise RuntimeError("no facade")

            return _Handle()

    events = RouteEvents(_Failing())

    assert events.route_degraded("cam-01", True) is False
    assert events.active_alarms() == ()


@pytest.mark.parametrize(
    "call",
    [
        lambda events: events.model_warmup_failed("r", "m", "e"),
        lambda events: events.model_activated("r", "m", "sha256:ab"),
        lambda events: events.executor_unavailable(True, "no cell"),
        lambda events: events.queue_age_exceeded("r", True, 400.0, 300.0, 12),
        lambda events: events.publish_backlog(True, 900, 1000),
        lambda events: events.publish_exhausted("r", "01k", "timeout", "retain"),
        lambda events: events.evidence_failed("r", "01k", "disk full"),
        lambda events: events.cleanup_failed("r", "01k", "archive", "collision"),
        lambda events: events.disk_pressure(True, "/var/lib", 100, 512),
        lambda events: events.gpu_pressure(True, "0", 100, 4096),
    ],
)
def test_every_typed_condition_emits_a_declared_event_type(call):
    gg = Gg()
    events = RouteEvents(gg)

    assert call(events) is True
    assert gg.log[0][2] in EVENT_TYPES
