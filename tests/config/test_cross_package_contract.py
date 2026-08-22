"""WP1's configuration and the packages that consume it are one contract.

Each assertion here fails the moment a default, a code, or a field name drifts apart from the
package that reads it, which is the whole point of the config package owning these numbers.
"""

import json

import pytest

from image_processor.completion.actions import (
    COLLISION_FAIL,
    COLLISION_SUFFIX,
)
from image_processor.completion.actions import CompletionPolicy as CompleterPolicy
from image_processor.config import (
    MAX_INLINE_BYTES,
    CollisionPolicy,
    ReadinessMode,
    parse_component_config,
)
from image_processor.ledger.ledger import DEFAULT_RESERVE_BUDGET_BYTES
from image_processor.sources.readiness import Readiness, ReadinessError
from image_processor.sources.spool import DEFAULT_DEBOUNCE_SECS, DEFAULT_RESCAN_INTERVAL_SECS
from image_processor.sources.trigger import MAX_INLINE_BYTES as TRIGGER_MAX_INLINE_BYTES
from image_processor.types import SourceKind


def parsed(global_config, instances):
    return parse_component_config(global_config, instances)


# --- WP3: the completion policy -------------------------------------------------------------


def test_a_routes_completion_policy_satisfies_the_completer_protocol(
    global_config, spool_route, trigger_route
):
    config = parsed(global_config, [spool_route, trigger_route])
    for route in config.routes:
        policy = route.completion
        assert isinstance(policy, CompleterPolicy), route.id
        for attribute in CompleterPolicy.__annotations__:
            assert getattr(policy, attribute, None) is not None or attribute in (
                "archive_dir",
                "failed_dir",
            ), f"{route.id}.{attribute}"


def test_a_spool_route_completes_under_its_own_root(global_config, spool_route):
    route = parsed(global_config, [spool_route]).routes[0]
    assert route.completion.source_root == route.source.root
    assert route.completion_for(SourceKind.SPOOL).source_root == route.source.root


def test_a_trigger_route_completes_inline_images_under_its_staging(global_config, trigger_route):
    route = parsed(global_config, [trigger_route]).routes[0]
    assert route.completion.source_root == route.source.inline_staging
    assert route.completion_for(SourceKind.INLINE).source_root == route.source.inline_staging


def test_a_trigger_route_completes_a_referenced_file_under_its_file_root(
    global_config, trigger_route
):
    """A reference resolves under `fileRoot`, which is not where an inline image is staged."""
    route = parsed(global_config, [trigger_route]).routes[0]
    resolved = route.completion_for(SourceKind.REFERENCE)
    assert resolved.source_root == route.source.file_root
    assert resolved.source_root != route.completion.source_root
    assert resolved.on_success is route.completion.on_success


def test_a_spool_route_ignores_the_reference_kind(global_config, spool_route):
    route = parsed(global_config, [spool_route]).routes[0]
    assert route.completion_for(SourceKind.REFERENCE) is route.completion


def test_the_defaults_template_carries_no_root(global_config):
    """`completionDefaults` is inherited from, not run under."""
    assert parsed(global_config, []).completion_defaults.source_root is None


def test_the_collision_policies_are_the_ones_the_completer_implements():
    assert {item.value for item in CollisionPolicy} == {COLLISION_FAIL, COLLISION_SUFFIX}
    assert CollisionPolicy.FAIL == COLLISION_FAIL
    assert CollisionPolicy.SUFFIX == COLLISION_SUFFIX


def test_a_route_may_choose_the_suffix_policy(global_config, spool_route):
    global_config["completionDefaults"] = {"onCollision": "suffix"}
    inherited = parsed(global_config, [spool_route]).routes[0]
    assert inherited.completion.on_collision is CollisionPolicy.SUFFIX

    spool_route["completion"]["onCollision"] = "fail"
    overridden = parsed(global_config, [spool_route]).routes[0]
    assert overridden.completion.on_collision is CollisionPolicy.FAIL
    assert parsed(global_config, []).completion_defaults.on_collision is CollisionPolicy.SUFFIX


# --- WP3: the reservation budget ------------------------------------------------------------


def test_the_reserve_budget_default_is_the_ledgers(global_config):
    publish = parsed(global_config, []).publish
    assert publish.outbox_reserve_budget_bytes == DEFAULT_RESERVE_BUDGET_BYTES
    assert publish.outbox_reserve_budget_mib == 256


def test_the_reserve_budget_converts_to_the_bytes_the_ledger_takes(global_config):
    global_config["publish"] = {"outboxReserveBudgetMiB": 4}
    assert parsed(global_config, []).publish.outbox_reserve_budget_bytes == 4 * 1024 * 1024


# --- WP5: discovery cadence and the inline cap ----------------------------------------------


def test_the_discovery_defaults_are_the_spool_sources(global_config):
    discovery = parsed(global_config, []).discovery
    assert discovery.rescan_secs == DEFAULT_RESCAN_INTERVAL_SECS
    assert discovery.debounce_secs == DEFAULT_DEBOUNCE_SECS


def test_discovery_can_be_tuned(global_config):
    global_config["discovery"] = {"rescanSecs": 5, "debounceMs": 0}
    discovery = parsed(global_config, []).discovery
    assert discovery.rescan_secs == 5
    assert discovery.debounce_ms == 0
    assert discovery.debounce_secs == 0.0


@pytest.mark.parametrize(
    "block, expected",
    [
        ({"rescanSecs": 0}, "INVALID_VALUE"),
        ({"debounceMs": -1}, "INVALID_VALUE"),
        ({"rescanSecs": 1.5}, "INVALID_TYPE"),
        ({"nope": 1}, "UNKNOWN_KEY"),
    ],
    ids=["rescan-is-positive", "debounce-is-not-negative", "rescan-is-an-integer", "unknown-key"],
)
def test_a_malformed_discovery_block_is_refused(global_config, block, expected):
    from image_processor.config import ConfigError

    global_config["discovery"] = block
    with pytest.raises(ConfigError) as caught:
        parsed(global_config, [])
    assert caught.value.code == expected


def test_the_inline_cap_is_the_one_the_trigger_source_clamps_to(config_schema):
    assert MAX_INLINE_BYTES == TRIGGER_MAX_INLINE_BYTES == 65536
    trigger = config_schema["$defs"]["triggerSource"]["properties"]["maxInlineBytes"]
    assert trigger["maximum"] == TRIGGER_MAX_INLINE_BYTES


# --- WP5: readiness ---------------------------------------------------------------------------


def test_the_readiness_modes_are_the_ones_the_source_builds(config_schema):
    documented = set(config_schema["$defs"]["readiness"]["properties"]["mode"]["enum"])
    assert documented == {mode.value for mode in ReadinessMode}


def test_the_validator_refuses_what_the_readiness_rule_refuses(global_config, spool_route):
    """Both packages reject the same configuration, under the same code."""
    from image_processor.config import ConfigError, validate_candidate
    from edgecommons.config import ConfigurationValidationPhase

    spool_route["source"]["readiness"] = {"mode": "stability"}
    candidate = {"component": {"global": global_config, "instances": [spool_route]}}
    verdict = validate_candidate(candidate, None, ConfigurationValidationPhase.INITIAL)
    assert not verdict.accepted

    route = parsed(global_config, [spool_route]).routes[0]
    with pytest.raises(ReadinessError) as caught:
        Readiness.for_route(route)
    assert verdict.code == caught.value.code == "STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE"


def test_a_configured_enum_stringifies_to_its_wire_spelling():
    """`sources/` normalizes configuration fields with `str()`, so the member name must not leak."""
    assert str(ReadinessMode.CAMERA_SIDECAR) == "cameraSidecar"
    assert str(CollisionPolicy.SUFFIX) == "suffix"
    assert json.dumps({"mode": ReadinessMode.MARKER}) == '{"mode": "marker"}'


@pytest.mark.parametrize(
    "mode, camera",
    [
        ("cameraSidecar", True),
        ("cameraStatus", True),
        ("marker", False),
        ("stability", False),
    ],
)
def test_the_readiness_rule_builds_from_a_parsed_route(global_config, spool_route, mode, camera):
    if not camera:
        spool_route["source"].pop("camera")
    spool_route["source"]["readiness"] = {"mode": mode, "markerSuffix": ".done", "quietSecs": 2}
    route = parsed(global_config, [spool_route]).routes[0]
    readiness = Readiness.for_route(route, status_lookup=lambda relative: None)
    assert readiness.mode == mode
