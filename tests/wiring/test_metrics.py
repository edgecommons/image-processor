"""The metric groups: what they carry, what they never carry, and how they flush."""

from __future__ import annotations

import time

import pytest

from image_processor.metrics import AVERAGED, METRIC_GROUPS, ProcessorMetrics
from tests.wiring.conftest import FakeGg


@pytest.fixture
def metrics(gg) -> ProcessorMetrics:
    """A metric accumulator over the fake handle."""
    return ProcessorMetrics(gg, interval_secs=0.01)


def test_every_designed_group_is_defined_with_its_measures(metrics, gg):
    metrics.define()

    assert set(gg.metrics.defined) == set(METRIC_GROUPS)
    discovery = gg.metrics.defined["ImageProcessorDiscovery"]
    assert set(discovery.get_measures()) == {
        name for name, _unit in METRIC_GROUPS["ImageProcessorDiscovery"]
    }


def test_no_measure_is_dimensioned_by_anything_that_identifies_one_image(metrics, gg):
    metrics.define()

    for metric in gg.metrics.defined.values():
        keys = {key.lower() for key in metric.get_dimensions()}
        assert not keys & {"file", "filename", "path", "captureid", "inferenceid", "version"}


def test_counters_sum_over_the_interval_and_reset(metrics, gg):
    metrics.define()
    metrics.incr("ImageProcessorInference", "succeeded")
    metrics.incr("ImageProcessorInference", "succeeded", 2)

    metrics.flush()
    assert gg.metrics.last("ImageProcessorInference")["succeeded"] == 3.0

    metrics.flush()
    assert gg.metrics.last("ImageProcessorInference")["succeeded"] == 0.0


def test_observations_average_over_the_interval(metrics, gg):
    metrics.define()
    metrics.observe("ImageProcessorInference", "inferenceMs", 10.0)
    metrics.observe("ImageProcessorInference", "inferenceMs", 20.0)

    metrics.flush()

    assert gg.metrics.last("ImageProcessorInference")["inferenceMs"] == 15.0


def test_an_averaged_measure_with_no_observation_is_not_emitted_as_zero(metrics, gg):
    metrics.define()
    metrics.incr("ImageProcessorInference", "succeeded")

    metrics.flush()

    values = gg.metrics.last("ImageProcessorInference")
    assert "inferenceMs" not in values
    assert set(AVERAGED) & set(values) == set()


def test_gauges_are_sampled_at_flush(gg):
    metrics = ProcessorMetrics(gg, gauges=lambda: {"ImageProcessorQueue": {"queued": 7}})
    metrics.define()

    metrics.flush()

    assert gg.metrics.last("ImageProcessorQueue")["queued"] == 7.0


def test_a_failing_gauge_never_loses_the_counters(gg):
    def _boom():
        raise RuntimeError("the ledger is busy")

    metrics = ProcessorMetrics(gg, gauges=_boom)
    metrics.define()
    metrics.incr("ImageProcessorOutbox", "published", 4)

    metrics.flush()

    assert gg.metrics.last("ImageProcessorOutbox")["published"] == 4.0


def test_a_failing_target_does_not_stop_the_other_groups(gg):
    metrics = ProcessorMetrics(gg)
    metrics.define()
    original = gg.metrics.emit_metric

    def _emit(name, values):
        if name == "ImageProcessorDiscovery":
            raise RuntimeError("the target is gone")
        original(name, values)

    gg.metrics.emit_metric = _emit

    emitted = metrics.flush()

    assert emitted == len(METRIC_GROUPS) - 1


def test_the_flush_thread_emits_on_the_interval_and_once_more_on_the_way_out(gg):
    metrics = ProcessorMetrics(gg, interval_secs=0.01)
    metrics.define()
    metrics.start()
    deadline = time.monotonic() + 5
    while metrics.flushes < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    flushes = metrics.flushes

    metrics.stop(timeout_s=5)

    assert flushes >= 2
    assert metrics.flushes == flushes + 1


def test_a_bring_up_without_a_configuration_manager_still_defines(gg):
    class _Bare:
        def get_metrics(self):
            return gg.metrics

    ProcessorMetrics(_Bare()).define()

    assert set(gg.metrics.defined) == set(METRIC_GROUPS)
