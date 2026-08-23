"""Per-route connectivity: one sample, two surfaces, and a token that says why."""

from __future__ import annotations

import pytest

from image_processor.connectivity import (
    DEGRADED,
    DISABLED,
    ONLINE,
    STAGING,
    ConnectivityProvider,
    RouteStatus,
    route_connectivity,
)


def status(**overrides) -> RouteStatus:
    """Build a connected route status unless the test says otherwise."""
    base = {
        "route_id": "clearance-cam-01",
        "enabled": True,
        "source_reachable": True,
        "source_detail": "/var/spool/camera-adapter/cam-01",
        "desired_generation": "sha256:ab",
        "active_generation": "sha256:ab",
        "executor_healthy": True,
        "queued": 3,
        "oldest_age_secs": 1.25,
    }
    base.update(overrides)
    return RouteStatus(**base)


def test_a_route_that_can_decide_right_now_is_connected_and_online():
    element = route_connectivity(status())

    assert element.instance == "clearance-cam-01"
    assert element.connected is True
    assert element.state == ONLINE
    assert element.detail == "/var/spool/camera-adapter/cam-01"
    assert element.attributes["activeGeneration"] == "sha256:ab"
    assert element.attributes["queued"] == 3
    assert element.attributes["oldestAgeSecs"] == 1.25


def test_a_route_whose_configuration_is_ahead_reports_staging_and_keeps_serving():
    reported = status(desired_generation="sha256:cd")

    assert reported.staging is True
    # DESIGN.md 9: while the two generations differ the route reports STAGING and stays on the
    # last known good model, so it can still decide.
    assert reported.connected is True
    assert route_connectivity(reported).state == STAGING


def test_a_route_that_has_never_activated_a_model_is_not_connected():
    reported = status(desired_generation="sha256:cd", active_generation=None)

    assert reported.staging is True
    assert reported.connected is False


def test_a_disabled_route_says_so_rather_than_looking_broken():
    assert route_connectivity(status(enabled=False)).state == DISABLED


def test_a_paused_route_is_not_connected():
    assert status(paused=True).connected is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_reachable": False},
        {"executor_healthy": False},
        {"active_generation": None, "desired_generation": None},
    ],
)
def test_anything_that_stops_a_decision_degrades_the_route(overrides):
    reported = status(**overrides)

    assert reported.connected is False
    assert route_connectivity(reported).state == DEGRADED


def test_the_most_recent_condition_is_the_detail_line():
    element = route_connectivity(status(last_error="MODEL_STAGING_FAILED: bad digest"))

    assert element.detail == "MODEL_STAGING_FAILED: bad digest"


def test_the_provider_renders_every_route():
    provider = ConnectivityProvider(lambda: [status(), status(route_id="adhoc-inspect")])

    elements = provider()

    assert [element.instance for element in elements] == ["clearance-cam-01", "adhoc-inspect"]


def test_a_sampling_failure_reports_no_instances_rather_than_failing_the_keepalive():
    def _boom():
        raise RuntimeError("the ledger is busy")

    assert ConnectivityProvider(_boom)() == []


def test_the_element_serializes_to_the_wire_shape():
    element = route_connectivity(status())

    document = element.to_dict()

    assert document["instance"] == "clearance-cam-01"
    assert document["connected"] is True
    assert document["state"] == ONLINE
    assert document["attributes"]["desiredGeneration"] == "sha256:ab"
