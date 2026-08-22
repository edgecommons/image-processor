"""`validate_candidate`: the cross-field rules a schema cannot express."""

import copy

import pytest
from edgecommons.config import ConfigurationValidationPhase as Phase

from image_processor.config import (
    CONFIG_SCHEMA,
    INFERENCE_RESULT_SCHEMA,
    MODEL_BUNDLE_MANIFEST_SCHEMA,
    VALIDATOR_NAME,
    ConfigError,
    load_schema,
    register,
    schema_path,
    validate_candidate,
)


def verdict(candidate, current=None, phase=Phase.INITIAL):
    return validate_candidate(candidate, current, phase)


def rejection(candidate, current=None, phase=Phase.INITIAL) -> str:
    result = verdict(candidate, current, phase)
    assert not result.accepted, "the candidate was accepted"
    return result.code


def test_the_shipped_candidate_is_accepted(candidate):
    assert verdict(candidate).accepted


def test_a_document_with_no_component_section_is_accepted():
    """A deployment that configures no routes is empty, not malformed; readiness reports it."""
    assert verdict({"logging": {"level": "INFO"}}).accepted


def test_a_candidate_must_be_an_object():
    assert rejection(["not", "a", "document"]) == "CANDIDATE_NOT_OBJECT"


def test_the_component_section_must_be_an_object():
    assert rejection({"component": []}) == "INVALID_TYPE"


def test_the_instances_list_must_be_an_array(candidate):
    candidate["component"]["instances"] = {"id": "x"}
    assert rejection(candidate) == "INVALID_TYPE"


# --- schema layer --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c["component"]["global"].update(nope=1), id="unknown-global-key"),
        pytest.param(
            lambda c: c["component"]["instances"][0].update(nope=1), id="unknown-route-key"
        ),
        pytest.param(
            lambda c: c["component"]["instances"][0]["source"].update(kind="folder"),
            id="unknown-source-kind",
        ),
        pytest.param(
            lambda c: c["component"]["instances"][0]["completion"].update(onSuccess="shred"),
            id="unknown-completion-action",
        ),
        pytest.param(
            lambda c: c["component"]["global"]["models"][0].update(digest="sha256:short"),
            id="malformed-digest",
        ),
        pytest.param(
            lambda c: c["component"]["instances"][1]["source"].update(maxInlineBytes=65537),
            id="inline-above-the-cap",
        ),
    ],
)
def test_the_component_schema_rejects_a_malformed_document(candidate, mutate):
    mutate(candidate)
    assert rejection(candidate) == "SCHEMA_INVALID"


# --- model resolution ----------------------------------------------------------------------


def test_a_route_must_name_a_declared_model(candidate):
    candidate["component"]["instances"][0]["modelRef"]["version"] = "2026.08.21"
    assert rejection(candidate) == "UNRESOLVED_MODEL_REF"


def test_a_model_uri_scheme_must_be_allowed(candidate):
    candidate["component"]["global"]["modelSources"] = {"allowedSchemes": ["https"]}
    assert rejection(candidate) == "MODEL_URI_SCHEME_NOT_ALLOWED"


def test_a_model_uri_must_match_the_allowlist(candidate):
    candidate["component"]["global"]["modelSources"] = {
        "allowedUriPrefixes": ["s3://other-bucket/"]
    }
    assert rejection(candidate) == "MODEL_URI_NOT_ALLOWED"


def test_a_model_uri_inside_the_allowlist_is_accepted(candidate):
    candidate["component"]["global"]["modelSources"] = {
        "allowedSchemes": ["s3"],
        "allowedUriPrefixes": ["s3://approved-models/"],
    }
    assert verdict(candidate).accepted


def test_required_signing_needs_a_trusted_key(candidate):
    candidate["component"]["global"]["signing"] = {"required": True}
    assert rejection(candidate) == "NO_TRUSTED_KEYS"

    candidate["component"]["global"]["signing"]["trustedKeys"] = [
        {"keyId": "pharma-model-publisher-1", "publicKey": {"$secret": "model-signing/one"}}
    ]
    assert verdict(candidate).accepted


# --- ownership of roots --------------------------------------------------------------------


def test_two_mutating_routes_may_not_claim_overlapping_roots(candidate, workspace):
    second = copy.deepcopy(candidate["component"]["instances"][0])
    second["id"] = "clearance-cam-01-copy"
    candidate["component"]["instances"].append(second)
    assert rejection(candidate) == "OVERLAPPING_MUTATING_ROOTS"


def test_a_mutating_route_may_not_claim_a_parent_of_another_root(candidate, workspace):
    second = copy.deepcopy(candidate["component"]["instances"][0])
    second["id"] = "clearance-all"
    second["source"]["root"] = str(workspace / "spool")
    candidate["component"]["instances"].append(second)
    assert rejection(candidate) == "OVERLAPPING_MUTATING_ROOTS"


def test_a_trigger_file_root_is_a_mutating_root_too(candidate, workspace):
    candidate["component"]["instances"][1]["source"]["fileRoot"] = str(
        workspace / "spool" / "cam-01"
    )
    assert rejection(candidate) == "OVERLAPPING_MUTATING_ROOTS"


def test_routes_that_only_retain_may_read_the_same_root(candidate, workspace):
    candidate["component"]["global"]["completionDefaults"] = {
        "onSuccess": "retainInPlace",
        "onInvalidInput": "retainInPlace",
    }
    for instance in candidate["component"]["instances"]:
        instance.pop("completion", None)
    second = copy.deepcopy(candidate["component"]["instances"][0])
    second["id"] = "clearance-cam-01-shadow"
    candidate["component"]["instances"].append(second)
    assert verdict(candidate).accepted


def test_an_archive_directory_may_not_sit_inside_a_mutating_root(candidate, workspace):
    candidate["component"]["instances"][0]["completion"]["archiveDir"] = str(
        workspace / "spool" / "cam-01" / "processed"
    )
    assert rejection(candidate) == "OUTPUT_INSIDE_SOURCE_ROOT"


def test_staging_may_not_sit_inside_a_mutating_root(candidate, workspace):
    candidate["component"]["global"]["paths"]["staging"] = str(
        workspace / "spool" / "cam-01" / "staging"
    )
    candidate["component"]["instances"][1]["source"]["inlineStaging"] = str(
        workspace / "spool" / "cam-01" / "staging" / "adhoc"
    )
    assert rejection(candidate) == "OUTPUT_INSIDE_SOURCE_ROOT"


def test_a_path_that_walks_upward_is_compared_after_folding(candidate, workspace):
    candidate["component"]["instances"][0]["completion"]["archiveDir"] = str(
        workspace / "spool" / "cam-01" / ".." / "cam-01" / "processed"
    )
    assert rejection(candidate) == "OUTPUT_INSIDE_SOURCE_ROOT"


# --- completion policy ---------------------------------------------------------------------


def test_archiving_needs_somewhere_to_archive_to(candidate):
    candidate["component"]["instances"][0]["completion"].pop("archiveDir")
    assert rejection(candidate) == "MISSING_ARCHIVE_DIR"


def test_quarantining_without_a_directory_quarantines_in_place(candidate):
    """DESIGN.md §11's trigger route inherits `quarantine` and sets no `failedDir`."""
    candidate["component"]["instances"][0]["completion"].pop("failedDir")
    assert verdict(candidate).accepted


def test_a_completion_directory_must_be_creatable(candidate, workspace):
    blocker = workspace / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    candidate["component"]["instances"][0]["completion"]["archiveDir"] = str(blocker / "processed")
    assert rejection(candidate) == "COMPLETION_DIR_NOT_CREATABLE"


def test_a_completion_target_that_is_a_file_is_refused(candidate, workspace):
    blocker = workspace / "processed-file"
    blocker.write_text("", encoding="utf-8")
    candidate["component"]["instances"][0]["completion"]["archiveDir"] = str(blocker)
    assert rejection(candidate) == "COMPLETION_DIR_NOT_CREATABLE"


def test_a_completion_directory_that_does_not_exist_yet_is_accepted(candidate, workspace):
    candidate["component"]["instances"][0]["completion"]["archiveDir"] = str(
        workspace / "processed" / "2026" / "08"
    )
    assert verdict(candidate).accepted


# --- readiness and staging -------------------------------------------------------------------


def test_a_camera_bound_route_may_not_infer_finalization_from_stability(candidate):
    candidate["component"]["instances"][0]["source"]["readiness"] = {"mode": "stability"}
    assert rejection(candidate) == "STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE"


def test_reading_capture_status_needs_a_camera_binding(candidate):
    source = candidate["component"]["instances"][0]["source"]
    source["readiness"] = {"mode": "cameraStatus"}
    source.pop("camera")
    assert rejection(candidate) == "CAMERA_BINDING_REQUIRED"

    source["camera"] = {"component": "camera-adapter", "instance": "cam-01"}
    assert verdict(candidate).accepted


def test_a_route_without_a_camera_may_use_stability(candidate):
    source = candidate["component"]["instances"][0]["source"]
    source.pop("camera")
    source["readiness"] = {"mode": "stability", "quietSecs": 3}
    assert verdict(candidate).accepted


def test_inline_images_stage_inside_the_processors_own_staging_tree(candidate, workspace):
    candidate["component"]["instances"][1]["source"]["inlineStaging"] = str(
        workspace / "inspection" / "staging"
    )
    assert rejection(candidate) == "INLINE_STAGING_NOT_CONTAINED"


def test_inline_staging_may_not_sit_inside_the_routes_own_file_root(candidate, workspace):
    candidate["component"]["global"]["paths"]["staging"] = str(workspace / "inspection")
    candidate["component"]["instances"][1]["source"]["inlineStaging"] = str(
        workspace / "inspection" / "staging"
    )
    candidate["component"]["instances"][1]["completion"] = {"onSuccess": "retainInPlace"}
    assert rejection(candidate) == "INLINE_STAGING_NOT_CONTAINED"


# --- provider policy -------------------------------------------------------------------------


def test_the_required_provider_must_be_offered(candidate):
    candidate["component"]["global"]["runtime"] = {
        "providers": ["CUDAExecutionProvider"],
        "requiredProvider": "TensorrtExecutionProvider",
    }
    assert rejection(candidate) == "PROVIDER_POLICY_UNSATISFIED"


def test_cpu_only_execution_is_a_deliberate_setting(candidate):
    candidate["component"]["global"]["runtime"] = {
        "providers": ["CPUExecutionProvider"],
        "requiredProvider": "CPUExecutionProvider",
    }
    assert rejection(candidate) == "PROVIDER_POLICY_UNSATISFIED"


def test_a_cpu_requirement_alongside_a_gpu_still_needs_the_opt_in(candidate):
    candidate["component"]["global"]["runtime"] = {
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "requiredProvider": "CPUExecutionProvider",
    }
    assert rejection(candidate) == "PROVIDER_POLICY_UNSATISFIED"


def test_no_required_provider_accepts_anything_offered(candidate):
    candidate["component"]["global"]["runtime"] = {
        "providers": ["CUDAExecutionProvider"],
        "requiredProvider": None,
    }
    assert verdict(candidate).accepted


# --- reload ------------------------------------------------------------------------------------


def test_the_ledger_cannot_move_under_a_running_process(candidate, workspace):
    current = copy.deepcopy(candidate)
    candidate["component"]["global"]["paths"]["stateDb"] = str(workspace / "state" / "moved.db")
    assert rejection(candidate, current, Phase.RELOAD) == "IMMUTABLE_PATH_CHANGED"


def test_the_model_cache_cannot_move_under_a_running_process(candidate, workspace):
    current = copy.deepcopy(candidate)
    (workspace / "models2").mkdir()
    candidate["component"]["global"]["paths"]["modelCache"] = str(workspace / "models2")
    assert rejection(candidate, current, Phase.RELOAD) == "IMMUTABLE_PATH_CHANGED"


def test_the_same_paths_reload_cleanly(candidate):
    current = copy.deepcopy(candidate)
    candidate["component"]["instances"][0]["priority"] = 200
    assert verdict(candidate, current, Phase.RELOAD).accepted


def test_the_initial_generation_ignores_whatever_came_before(candidate, workspace):
    current = copy.deepcopy(candidate)
    candidate["component"]["global"]["paths"]["stateDb"] = str(workspace / "state" / "moved.db")
    assert verdict(candidate, current, Phase.INITIAL).accepted


@pytest.mark.parametrize(
    "current",
    [None, {}, {"component": []}, {"component": {"global": {"nope": 1}}}],
    ids=["no-current", "empty", "component-not-an-object", "unparseable-current"],
)
def test_a_reload_with_nothing_comparable_judges_the_candidate_alone(candidate, current):
    assert verdict(candidate, current, Phase.RELOAD).accepted


# --- registration and schema loading -----------------------------------------------------------


def test_registering_the_validator_hands_it_to_the_builder():
    class FakeBuilder:
        def __init__(self):
            self.registered = {}

        def configuration_validator(self, name, validator):
            self.registered[name] = validator
            return self

    builder = FakeBuilder()
    assert register(builder) is builder
    assert builder.registered == {VALIDATOR_NAME: validate_candidate}


@pytest.mark.parametrize(
    "relative", [CONFIG_SCHEMA, INFERENCE_RESULT_SCHEMA, MODEL_BUNDLE_MANIFEST_SCHEMA]
)
def test_every_shipped_schema_loads(relative):
    assert schema_path(relative).is_file()
    assert load_schema(relative)["$schema"].endswith("2020-12/schema")


def test_a_missing_schema_fails_closed():
    with pytest.raises(ConfigError) as caught:
        schema_path("schemas/not-shipped.schema.json")
    assert caught.value.code == "SCHEMA_UNAVAILABLE"


def test_a_schema_that_is_not_json_fails_closed(tmp_path, monkeypatch):
    import image_processor.config.validate as module

    broken = tmp_path / "broken.schema.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(module, "schema_path", lambda relative: broken)
    module.load_schema.cache_clear()
    try:
        with pytest.raises(ConfigError) as caught:
            module.load_schema("broken.schema.json")
        assert caught.value.code == "SCHEMA_UNAVAILABLE"
    finally:
        module.load_schema.cache_clear()
