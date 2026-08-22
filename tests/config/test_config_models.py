"""`parse_component_config`: defaults, inheritance, secret references, and refusals."""

from pathlib import Path

import pytest

from image_processor.config import (
    MAX_INLINE_BYTES,
    CollisionPolicy,
    ConfigError,
    ReadinessMode,
    SpoolSourceConfig,
    TriggerSourceConfig,
    parse_component_config,
)
from image_processor.types import CompletionAction, ModelRef, SecretRef

DIGEST = "sha256:" + "ab" * 32


def parse(global_cfg=None, instances=None):
    return parse_component_config(global_cfg, instances)


def raises(global_cfg=None, instances=None) -> str:
    with pytest.raises(ConfigError) as caught:
        parse_component_config(global_cfg, instances)
    return caught.value.code


def test_an_empty_configuration_takes_every_default():
    config = parse()
    assert str(config.paths.state_db).endswith("state.db")
    assert config.runtime.providers == ("CUDAExecutionProvider",)
    assert config.runtime.required_provider == "CUDAExecutionProvider"
    assert config.runtime.allow_cpu_only is False
    assert config.runtime.executor_cells_per_gpu == 1
    assert config.gpu.devices == ("0",)
    assert config.gpu.resident_memory_budget_percent == 80
    assert config.gpu.reserve_mib == 2048
    assert config.scheduler.max_batch_size == 1
    assert config.scheduler.hot_ttl_secs == 120
    assert config.scheduler.max_attempts == 5
    assert config.publish.confirmation_timeout_secs == 10
    assert config.publish.require_confirmation_before_cleanup is True
    assert config.signing.required is False
    assert config.model_sources.allowed_schemes == ("s3", "https", "file")
    assert config.model_sources.verify_tls is True
    assert config.models == ()
    assert config.routes == ()


def test_the_default_completion_policy_matches_the_schema(config_schema):
    defaults = parse().completion_defaults
    documented = config_schema["properties"]["completionDefaults"]["properties"]
    assert documented["onSuccess"]["default"] == "archive"
    assert documented["onInvalidInput"]["default"] == "quarantine"
    assert documented["onOperationalFailure"]["default"] == "retainInPlace"
    assert documented["onPublishFailure"]["default"] == "retainInPlace"
    assert defaults.on_success is CompletionAction.ARCHIVE
    assert defaults.on_invalid_input is CompletionAction.QUARANTINE
    assert defaults.on_operational_failure is CompletionAction.RETAIN
    assert defaults.on_publish_failure is CompletionAction.RETAIN
    assert defaults.on_collision is CollisionPolicy.FAIL
    assert defaults.mutates is True


SCHEMA_DEFAULT_BLOCKS = {
    "paths": {"stateDb": "state_db", "modelCache": "model_cache", "staging": "staging"},
    "runtime": {
        "providers": "providers",
        "requiredProvider": "required_provider",
        "allowCpuOnly": "allow_cpu_only",
        "executorCellsPerGpu": "executor_cells_per_gpu",
        "loadConcurrencyPerGpu": "load_concurrency_per_gpu",
    },
    "gpu": {
        "devices": "devices",
        "residentMemoryBudgetPercent": "resident_memory_budget_percent",
        "reserveMiB": "reserve_mib",
    },
    "scheduler": {
        "maxBatchSize": "max_batch_size",
        "maxBatchLatencyMs": "max_batch_latency_ms",
        "hotTtlSecs": "hot_ttl_secs",
        "minResidencySecs": "min_residency_secs",
        "maxAttempts": "max_attempts",
        "retryBackoffSecs": "retry_backoff_secs",
        "maxRetryBackoffSecs": "max_retry_backoff_secs",
        "queueAgeWarningSecs": "queue_age_warning_secs",
    },
    "publish": {
        "confirmationTimeoutSecs": "confirmation_timeout_secs",
        "requireConfirmationBeforeCleanup": "require_confirmation_before_cleanup",
        "maxAttempts": "max_attempts",
        "outboxCapacity": "outbox_capacity",
        "outboxReserveBudgetMiB": "outbox_reserve_budget_mib",
    },
    "discovery": {"rescanSecs": "rescan_secs", "debounceMs": "debounce_ms"},
    "modelSources": {
        "allowedSchemes": "allowed_schemes",
        "allowedUriPrefixes": "allowed_uri_prefixes",
        "verifyTls": "verify_tls",
    },
}

BLOCK_ATTRIBUTES = {
    "paths": "paths",
    "runtime": "runtime",
    "gpu": "gpu",
    "scheduler": "scheduler",
    "publish": "publish",
    "discovery": "discovery",
    "modelSources": "model_sources",
}


@pytest.mark.parametrize("block", sorted(SCHEMA_DEFAULT_BLOCKS))
def test_the_schema_defaults_and_the_parsed_defaults_agree(config_schema, block):
    """The schema documents each default; the parser applies it. They are one contract."""
    parsed = getattr(parse(), BLOCK_ATTRIBUTES[block])
    properties = config_schema["properties"][block]["properties"]
    for schema_name, attribute in SCHEMA_DEFAULT_BLOCKS[block].items():
        assert "default" in properties[schema_name], f"{block}.{schema_name} has no default"
        expected = properties[schema_name]["default"]
        actual = getattr(parsed, attribute)
        if isinstance(actual, Path):
            assert actual == Path(expected), f"{block}.{schema_name}"
        elif isinstance(actual, tuple):
            assert list(actual) == expected, f"{block}.{schema_name}"
        elif isinstance(actual, bool) or expected is None:
            assert actual == expected, f"{block}.{schema_name}"
        elif isinstance(actual, (int, float)):
            assert float(actual) == float(expected), f"{block}.{schema_name}"
        else:
            assert actual == expected, f"{block}.{schema_name}"


def test_a_route_inherits_the_completion_defaults_it_does_not_set(spool_route, global_config):
    global_config["completionDefaults"] = {"onSuccess": "delete", "onInvalidInput": "retainInPlace"}
    spool_route["completion"] = {
        "onSuccess": "archive",
        "archiveDir": spool_route["completion"]["archiveDir"],
    }
    route = parse(global_config, [spool_route]).routes[0]
    assert route.completion.on_success is CompletionAction.ARCHIVE
    assert route.completion.on_invalid_input is CompletionAction.RETAIN
    assert route.completion.failed_dir is None


def test_a_route_that_only_retains_claims_no_root(spool_route, global_config):
    global_config["completionDefaults"] = {
        "onSuccess": "retainInPlace",
        "onInvalidInput": "retainInPlace",
    }
    spool_route.pop("completion")
    route = parse(global_config, [spool_route]).routes[0]
    assert route.completion.mutates is False
    assert route.mutating_roots == ()
    assert route.output_dirs == ()


def test_a_spool_route_takes_its_readiness_and_camera_defaults(spool_route, global_config):
    spool_route["source"]["readiness"] = {"mode": "marker"}
    route = parse(global_config, [spool_route]).routes[0]
    assert isinstance(route.source, SpoolSourceConfig)
    assert route.is_spool and not route.is_trigger
    assert route.source.readiness.mode is ReadinessMode.MARKER
    assert route.source.readiness.marker_suffix == ".done"
    assert route.source.readiness.quiet_secs == 5
    assert route.source.include[0] == "**/*.jpg"
    assert route.source.exclude == ()
    assert route.source.camera.subscribe_announcements is True
    assert route.source.camera.reconcile_capture_status_secs == 30
    assert route.enabled is True
    assert route.reprocess_existing_on_model_change is False


def test_a_spool_route_without_a_camera_is_allowed(spool_route, global_config):
    spool_route["source"].pop("camera")
    spool_route["source"]["readiness"] = {"mode": "stability", "quietSecs": 2.5}
    route = parse(global_config, [spool_route]).routes[0]
    assert route.source.camera is None
    assert route.source.readiness.quiet_secs == 2.5


def test_a_trigger_route_is_capped_at_the_binary_body_limit(trigger_route, global_config):
    route = parse(global_config, [trigger_route]).routes[0]
    assert isinstance(route.source, TriggerSourceConfig)
    assert route.is_trigger and not route.is_spool
    assert route.source.max_inline_bytes == MAX_INLINE_BYTES
    assert route.input_root == route.source.file_root


def test_a_route_lookup_finds_its_model_and_misses_an_unknown_one(candidate):
    component = candidate["component"]
    config = parse(component["global"], component["instances"])
    assert config.model_entry(config.routes[0].model_ref) is config.models[0]
    assert config.model_entry(ModelRef("other", "1", DIGEST)) is None
    assert config.route("nope") is None
    assert len(config.enabled_routes) == 2
    assert config.models[0].ref == ModelRef("line-clearance-cam-01", "2026.08.20", DIGEST)
    assert config.models[0].activation.require_warmup is True


def test_a_disabled_route_is_parsed_but_claims_no_work(spool_route, global_config):
    spool_route["enabled"] = False
    config = parse(global_config, [spool_route])
    assert config.routes[0].enabled is False
    assert config.enabled_routes == ()


def test_a_secret_reference_is_carried_as_a_value_and_never_opened(global_config):
    global_config["models"][0]["credentials"] = {
        "$secret": "model-source/approved-models",
        "field": "accessKeyId",
    }
    global_config["signing"] = {
        "required": True,
        "trustedKeys": [
            {"keyId": "pharma-model-publisher-1",
             "publicKey": {"$secret": "model-signing/publisher-1"}},
            {"keyId": "inline-key", "publicKey": "MCowBQYDK2VwAyEA"},
        ],
    }
    config = parse(global_config, [])
    assert config.models[0].credentials_ref == SecretRef("model-source/approved-models",
                                                        "accessKeyId")
    assert config.models[0].credentials_ref.to_config() == {
        "$secret": "model-source/approved-models",
        "field": "accessKeyId",
    }
    trusted = config.signing.key("pharma-model-publisher-1")
    assert trusted.public_key == SecretRef("model-signing/publisher-1")
    assert config.signing.key("inline-key").public_key == "MCowBQYDK2VwAyEA"
    assert config.signing.key("absent") is None


def test_a_model_without_credentials_uses_ambient_ones(global_config):
    global_config["models"][0].pop("credentials")
    assert parse(global_config, []).models[0].credentials_ref is None


@pytest.mark.parametrize(
    "value",
    [
        {"$secret": ""},
        {"$secret": "name", "extra": 1},
        {"$secret": "name", "field": ""},
        {"$secret": 7},
    ],
    ids=["empty-name", "unknown-key", "empty-field", "non-string-name"],
)
def test_a_malformed_secret_reference_is_refused(global_config, value):
    global_config["models"][0]["credentials"] = value
    assert raises(global_config, []) == "INVALID_SECRET_REF"


def test_a_secret_reference_reports_what_it_is():
    assert SecretRef.is_ref({"$secret": "a"}) is True
    assert SecretRef.is_ref({"secret": "a"}) is False
    assert SecretRef.is_ref("a") is False
    with pytest.raises(ValueError):
        SecretRef.parse({"secret": "a"})


# --- refusals ------------------------------------------------------------------------------


def test_an_error_code_must_be_a_stable_operator_facing_code():
    with pytest.raises(ValueError):
        ConfigError("not a code", "message")
    error = ConfigError("UNKNOWN_KEY", "global has unknown keys: nope")
    assert error.code == "UNKNOWN_KEY"
    assert "nope" in str(error)


@pytest.mark.parametrize(
    "global_cfg, expected",
    [
        pytest.param({"nope": 1}, "UNKNOWN_KEY", id="unknown-global-key"),
        pytest.param([], "INVALID_TYPE", id="global-is-not-an-object"),
        pytest.param({"paths": {"stateDb": "state.db"}}, "PATH_NOT_ABSOLUTE", id="relative-path"),
        pytest.param({"paths": {"stateDb": 7}}, "INVALID_TYPE", id="path-is-not-a-string"),
        pytest.param({"runtime": {"providers": []}}, "INVALID_VALUE", id="no-providers"),
        pytest.param({"runtime": {"providers": [7]}}, "INVALID_TYPE", id="provider-not-a-string"),
        pytest.param({"runtime": {"providers": "cuda"}}, "INVALID_TYPE", id="providers-not-a-list"),
        pytest.param(
            {"runtime": {"providers": ["MagicExecutionProvider"]}},
            "INVALID_VALUE",
            id="unknown-provider",
        ),
        pytest.param(
            {"runtime": {"requiredProvider": "Magic"}}, "INVALID_VALUE", id="unknown-required"
        ),
        pytest.param({"runtime": {"allowCpuOnly": "yes"}}, "INVALID_TYPE", id="not-a-boolean"),
        pytest.param({"gpu": {"reserveMiB": -1}}, "INVALID_VALUE", id="below-minimum"),
        pytest.param(
            {"gpu": {"residentMemoryBudgetPercent": 101}}, "INVALID_VALUE", id="above-maximum"
        ),
        pytest.param({"gpu": {"reserveMiB": 1.5}}, "INVALID_TYPE", id="not-an-integer"),
        pytest.param({"gpu": {"reserveMiB": True}}, "INVALID_TYPE", id="boolean-is-not-a-number"),
        pytest.param(
            {"publish": {"confirmationTimeoutSecs": "soon"}}, "INVALID_TYPE", id="not-a-number"
        ),
        pytest.param({"models": {}}, "INVALID_TYPE", id="models-is-not-an-array"),
        pytest.param(
            {"modelSources": {"allowedSchemes": ["ftp"]}}, "INVALID_VALUE", id="unknown-scheme"
        ),
    ],
)
def test_a_malformed_global_section_is_refused(global_cfg, expected):
    assert raises(global_cfg, []) == expected


def test_a_missing_required_model_field_is_refused(global_config):
    global_config["models"][0].pop("uri")
    assert raises(global_config, []) == "MISSING_FIELD"


def test_two_models_may_not_declare_the_same_version_twice(global_config):
    global_config["models"].append(dict(global_config["models"][0]))
    assert raises(global_config, []) == "DUPLICATE_MODEL_ID"


def test_two_trusted_keys_may_not_share_an_id(global_config):
    global_config["signing"] = {
        "trustedKeys": [
            {"keyId": "k", "publicKey": "a"},
            {"keyId": "k", "publicKey": "b"},
        ]
    }
    assert raises(global_config, []) == "DUPLICATE_KEY_ID"


def test_a_trusted_key_needs_real_key_material(global_config):
    global_config["signing"] = {"trustedKeys": [{"keyId": "k", "publicKey": 7}]}
    assert raises(global_config, []) == "INVALID_TYPE"


def test_instances_must_be_an_array(global_config):
    assert raises(global_config, {}) == "INVALID_TYPE"


def test_two_routes_may_not_share_an_id(global_config, spool_route):
    import copy as _copy

    assert raises(global_config, [spool_route, _copy.deepcopy(spool_route)]) == "DUPLICATE_ROUTE_ID"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(lambda r: r.pop("id"), "MISSING_FIELD", id="no-id"),
        pytest.param(lambda r: r.update(id="Clearance"), "INVALID_VALUE", id="id-is-not-a-token"),
        pytest.param(lambda r: r.update(nope=1), "UNKNOWN_KEY", id="unknown-route-key"),
        pytest.param(lambda r: r.pop("source"), "MISSING_FIELD", id="no-source"),
        pytest.param(lambda r: r.update(source=[]), "INVALID_TYPE", id="source-is-not-an-object"),
        pytest.param(
            lambda r: r["source"].update(kind="folder"), "INVALID_SOURCE_KIND", id="unknown-kind"
        ),
        pytest.param(lambda r: r.pop("modelRef"), "MISSING_FIELD", id="no-model-ref"),
        pytest.param(
            lambda r: r["modelRef"].update(digest="sha256:xyz"), "INVALID_VALUE", id="bad-digest"
        ),
        pytest.param(lambda r: r.update(priority=-1), "INVALID_VALUE", id="negative-priority"),
        pytest.param(lambda r: r["source"].pop("readiness"), "MISSING_FIELD", id="no-readiness"),
        pytest.param(
            lambda r: r["source"]["readiness"].update(mode="magic"),
            "INVALID_VALUE",
            id="unknown-readiness-mode",
        ),
        pytest.param(
            lambda r: r["source"].pop("root"), "MISSING_FIELD", id="spool-needs-a-root"
        ),
        pytest.param(
            lambda r: r["source"]["camera"].pop("instance"),
            "MISSING_FIELD",
            id="camera-needs-an-instance",
        ),
        pytest.param(
            lambda r: r["source"]["camera"].update(instance="Cam_01"),
            "INVALID_VALUE",
            id="camera-instance-is-a-token",
        ),
        pytest.param(
            lambda r: r["outputs"]["decisionSignals"].append({"id": "a/b", "value": "decision"}),
            "INVALID_DECISION_SIGNAL",
            id="signal-value-is-a-jsonpath",
        ),
        pytest.param(
            lambda r: r["outputs"]["decisionSignals"].append(
                {"id": "line-clearance/pass", "value": "$.x"}
            ),
            "INVALID_DECISION_SIGNAL",
            id="duplicate-signal-id",
        ),
        pytest.param(
            lambda r: r["outputs"]["decisionSignals"].append({"id": "a/b", "value": "$.x", "z": 1}),
            "UNKNOWN_KEY",
            id="unknown-signal-key",
        ),
        pytest.param(
            lambda r: r["outputs"]["decisionSignals"].append({"id": "a b", "value": "$.x"}),
            "INVALID_VALUE",
            id="signal-id-shape",
        ),
        pytest.param(
            lambda r: r["completion"].update(onSuccess="shred"),
            "INVALID_VALUE",
            id="unknown-completion-action",
        ),
        pytest.param(
            lambda r: r["completion"].update(onCollision="overwrite"),
            "INVALID_VALUE",
            id="collisions-never-overwrite",
        ),
        pytest.param(
            lambda r: r["completion"].update(onCollision="rename"),
            "INVALID_VALUE",
            id="collision-policies-are-closed",
        ),
        pytest.param(
            lambda r: r["completion"].update(archiveDir="processed"),
            "PATH_NOT_ABSOLUTE",
            id="relative-archive-dir",
        ),
    ],
)
def test_a_malformed_spool_route_is_refused(global_config, spool_route, mutate, expected):
    mutate(spool_route)
    assert raises(global_config, [spool_route]) == expected


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda r: r["source"].update(maxInlineBytes=65537),
            "INLINE_LIMIT_EXCEEDED",
            id="above-the-binary-body-cap",
        ),
        pytest.param(
            lambda r: r["source"].update(subscribe=[]), "INVALID_VALUE", id="no-topic-filters"
        ),
        pytest.param(
            lambda r: r["source"].pop("fileRoot"), "MISSING_FIELD", id="no-file-root"
        ),
        pytest.param(
            lambda r: r["source"].pop("inlineStaging"), "MISSING_FIELD", id="no-inline-staging"
        ),
        pytest.param(
            lambda r: r["source"].update(nope=1), "UNKNOWN_KEY", id="unknown-source-key"
        ),
    ],
)
def test_a_malformed_trigger_route_is_refused(global_config, trigger_route, mutate, expected):
    mutate(trigger_route)
    assert raises(global_config, [trigger_route]) == expected


def test_a_route_may_lower_its_inline_ceiling(global_config, trigger_route):
    trigger_route["source"]["maxInlineBytes"] = 4096
    assert parse(global_config, [trigger_route]).routes[0].source.max_inline_bytes == 4096


def test_a_non_string_where_a_string_belongs_is_refused(global_config):
    global_config["models"][0]["id"] = 7
    assert raises(global_config, []) == "INVALID_TYPE"
