"""The decision mirror: derived, bounded, and best effort."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from image_processor.outputs.mirror import DecisionMirror


@dataclass(frozen=True)
class Signal:
    """One configured decision signal."""

    id: str
    value: str


class RecordingData:
    """A ``data()`` facade that records, or fails on command."""

    def __init__(self, error=None) -> None:
        self.published: list = []
        self.error = error

    def publish(self, signal_path, value, quality=None) -> None:
        """Record one reading."""
        if self.error is not None:
            raise self.error
        self.published.append((signal_path, value, str(quality)))


class Gg:
    """Just enough handle for the mirror."""

    def __init__(self, data) -> None:
        self.data_facade = data
        self.instances: list = []

    def instance(self, instance_id):
        """Return the instance handle, recording which token was used."""
        self.instances.append(instance_id)
        outer = self

        class _Handle:
            def data(self):
                return outer.data_facade

        return _Handle()


BODY = {
    "status": "SUCCEEDED",
    "decision": {"outcome": "CLEAR", "pass": True, "confidence": 0.997},
    "outputs": {"classes": [{"label": "clear", "index": 0, "score": 0.9}]},
}


def test_each_signal_publishes_the_value_its_path_names():
    data = RecordingData()
    mirror = DecisionMirror(Gg(data))

    published = mirror.publish(
        "clearance-cam-01",
        [
            Signal("line-clearance/pass", "$.decision.pass"),
            Signal("line-clearance/confidence", "$.decision.confidence"),
            Signal("line-clearance/status", "$.status"),
        ],
        BODY,
    )

    assert published == 3
    assert data.published == [
        ("line-clearance/pass", True, "Quality.GOOD"),
        ("line-clearance/confidence", 0.997, "Quality.GOOD"),
        ("line-clearance/status", "SUCCEEDED", "Quality.GOOD"),
    ]


def test_the_reading_of_a_failed_result_is_marked_bad():
    data = RecordingData()
    mirror = DecisionMirror(Gg(data))

    mirror.publish("r", [Signal("s", "$.status")], {"status": "FAILED", "decision": {}})

    assert data.published == [("s", "FAILED", "Quality.BAD")]


def test_a_path_that_resolves_nothing_publishes_nothing():
    data = RecordingData()
    mirror = DecisionMirror(Gg(data))

    assert mirror.publish("r", [Signal("s", "$.nope.missing")], BODY) == 0
    assert data.published == []
    assert mirror.counters["unresolved"] == 1


def test_a_path_that_resolves_a_document_is_not_a_reading():
    data = RecordingData()
    mirror = DecisionMirror(Gg(data))

    assert mirror.publish("r", [Signal("s", "$.outputs.classes")], BODY) == 0
    assert mirror.counters["unsupported"] == 1


def test_a_failing_publish_is_counted_and_never_raised():
    seen: list = []
    mirror = DecisionMirror(
        Gg(RecordingData(RuntimeError("the broker is gone"))),
        on_error=lambda route, signal, error: seen.append((route, signal)),
    )

    assert mirror.publish("r", [Signal("s", "$.status")], BODY) == 0
    assert mirror.counters["failed"] == 1
    assert seen == [("r", "s")]


def test_a_route_with_no_signals_touches_the_bus_at_all():
    gg = Gg(RecordingData())
    mirror = DecisionMirror(gg)

    assert mirror.publish("r", [], BODY) == 0
    assert gg.instances == []


def test_a_failing_error_callback_is_swallowed():
    def _boom(route, signal, error):
        raise RuntimeError("the reporter is broken too")

    mirror = DecisionMirror(Gg(RecordingData(RuntimeError("gone"))), on_error=_boom)

    assert mirror.publish("r", [Signal("s", "$.status")], BODY) == 0


def test_the_reading_carries_the_route_instance_token():
    gg = Gg(RecordingData())

    DecisionMirror(gg).publish("clearance-cam-01", [Signal("s", "$.status")], BODY)

    assert gg.instances == ["clearance-cam-01"]
