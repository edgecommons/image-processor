"""Readiness and failure: ready is a claim about work, and failure is stronger than degradation."""

from __future__ import annotations

import pytest

from image_processor.connectivity import RouteStatus
from image_processor.health import Health


def status(route_id: str = "cam-01", **overrides) -> RouteStatus:
    """Build one route status, connected unless the test says otherwise."""
    base = {
        "route_id": route_id,
        "enabled": True,
        "source_reachable": True,
        "desired_generation": "sha256:ab",
        "active_generation": "sha256:ab",
        "executor_healthy": True,
    }
    base.update(overrides)
    return RouteStatus(**base)


def build(**overrides) -> Health:
    """Build a health evaluator whose every input is healthy unless overridden."""
    base = {
        "statuses": lambda: [status()],
        "state_writable": lambda: True,
        "cache_verified": lambda: True,
        "outbox_pending": lambda: 0,
        "outbox_capacity": 100,
        "requires_executor": lambda: True,
        "executor_healthy": lambda: True,
    }
    base.update(overrides)
    return Health(**base)


def test_a_healthy_component_is_ready():
    report = build().evaluate()

    assert report.ready is True
    assert bool(report) is True
    assert report.failed is False
    assert report.reasons == ()


def test_a_component_with_no_enabled_route_is_not_ready():
    report = build(statuses=lambda: [status(enabled=False)]).evaluate()

    assert report.ready is False
    assert report.failed is False
    assert "NO_ENABLED_ROUTE" in report.reasons


def test_one_degraded_route_among_several_degrades_rather_than_fails():
    report = build(
        statuses=lambda: [status("cam-01"), status("cam-02", active_generation=None)]
    ).evaluate()

    assert report.failed is False
    assert report.degraded_routes == ("cam-02",)
    assert report.ready is True


def test_no_route_that_can_execute_is_a_failure():
    report = build(statuses=lambda: [status(active_generation=None)]).evaluate()

    assert report.ready is False
    assert report.failed is True
    assert "NO_ROUTE_CAN_EXECUTE" in report.reasons


def test_losing_durable_state_is_a_failure():
    report = build(state_writable=lambda: False).evaluate()

    assert report.failed is True
    assert "STATE_NOT_WRITABLE" in report.reasons


def test_an_unverified_model_cache_is_not_ready_but_is_not_a_failure():
    report = build(cache_verified=lambda: False).evaluate()

    assert report.ready is False
    assert report.failed is False
    assert "MODEL_CACHE_UNVERIFIED" in report.reasons


def test_a_missing_executor_fails_only_when_a_route_needs_one():
    needed = build(executor_healthy=lambda: False).evaluate()
    assert needed.failed is True
    assert "NO_HEALTHY_EXECUTOR" in needed.reasons

    not_needed = build(executor_healthy=lambda: False, requires_executor=lambda: False).evaluate()
    assert not_needed.ready is True


def test_a_full_outbox_fails_closed():
    report = build(outbox_pending=lambda: 100).evaluate()

    assert report.failed is True
    assert "PUBLISH_BACKLOG_EXCEEDED" in report.reasons


def test_a_check_that_cannot_run_fails_closed():
    def _boom():
        raise RuntimeError("the ledger is gone")

    report = build(state_writable=_boom).evaluate()

    assert report.failed is True
    assert "STATE_NOT_WRITABLE" in report.reasons


def test_an_unreadable_route_status_is_a_failure():
    def _boom():
        raise RuntimeError("the configuration is gone")

    report = build(statuses=_boom).evaluate()

    assert report.failed is True
    assert "ROUTE_STATUS_UNAVAILABLE" in report.reasons


def test_an_unreadable_outbox_is_treated_as_empty():
    def _boom():
        raise RuntimeError("busy")

    report = build(outbox_pending=_boom).evaluate()

    assert "PUBLISH_BACKLOG_EXCEEDED" not in report.reasons


def test_an_unreadable_executor_requirement_fails_closed():
    def _boom():
        raise RuntimeError("busy")

    report = build(requires_executor=_boom, executor_healthy=lambda: False).evaluate()

    assert "NO_HEALTHY_EXECUTOR" in report.reasons


def test_applying_pushes_the_verdict_into_the_library(gg):
    health = build()

    report = health.apply(gg)

    assert gg.ready == [True]
    assert health.last is report


def test_a_readiness_flag_that_cannot_be_set_does_not_raise():
    class _Broken:
        def set_ready(self, ready):
            raise RuntimeError("no readiness state")

    assert build().apply(_Broken()).ready is True
