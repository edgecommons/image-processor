"""Every example DESIGN.md publishes validates against the contract that governs it."""

import copy

import pytest
from jsonschema import Draft202012Validator

from image_processor.config import parse_component_config
from image_processor.types import CompletionAction

HEX256 = "ab" * 32
DIGEST = "sha256:" + HEX256


def _route_validator(config_schema):
    return Draft202012Validator(
        {
            "$schema": config_schema["$schema"],
            "$id": config_schema["$id"] + "#route",
            "$ref": "#/$defs/route",
            "$defs": config_schema["$defs"],
        }
    )


def _errors(validator, instance):
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(instance)]


def test_the_design_global_example_validates(config_schema, design_config_example):
    validator = Draft202012Validator(config_schema)
    global_cfg = design_config_example["component"]["global"]
    assert _errors(validator, global_cfg) == []


def test_every_design_route_example_validates(config_schema, design_config_example):
    validator = _route_validator(config_schema)
    instances = design_config_example["component"]["instances"]
    assert [instance["id"] for instance in instances] == ["clearance-cam-01", "adhoc-inspect"]
    for instance in instances:
        assert _errors(validator, instance) == [], instance["id"]


def test_the_design_example_parses_into_the_configuration_it_describes(design_config_example):
    component = design_config_example["component"]
    config = parse_component_config(component["global"], component["instances"])

    camera_route = config.route("clearance-cam-01")
    assert camera_route.priority == 100
    assert camera_route.is_spool
    assert camera_route.source.camera.instance == "cam-01"
    assert camera_route.source.readiness.mode.value == "cameraSidecar"
    assert [signal.id for signal in camera_route.outputs.decision_signals] == [
        "line-clearance/pass",
        "line-clearance/confidence",
        "line-clearance/status",
    ]
    # The route sets only onSuccess; everything else comes from completionDefaults.
    assert camera_route.completion.on_success is CompletionAction.ARCHIVE
    assert camera_route.completion.on_invalid_input is CompletionAction.QUARANTINE
    assert camera_route.completion.on_operational_failure is CompletionAction.RETAIN

    trigger_route = config.route("adhoc-inspect")
    assert trigger_route.is_trigger
    assert trigger_route.source.max_inline_bytes == 65536
    assert trigger_route.completion.on_success is CompletionAction.DELETE
    assert trigger_route.outputs.write_result_sidecar is False

    assert config.model_entry(camera_route.model_ref) is config.models[0]
    assert config.models[0].credentials_ref.name == "model-source/approved-models"
    assert config.signing.trusted_keys[0].public_key.name == "model-signing/publisher-1"


def test_the_design_result_example_validates(result_schema, design_result_example):
    validator = Draft202012Validator(result_schema)
    assert _errors(validator, design_result_example) == []
    assert design_result_example["status"] == "SUCCEEDED"
    assert design_result_example["decision"]["outcome"] == "CLEAR"


@pytest.mark.parametrize(
    "mutate, expect",
    [
        pytest.param(lambda b: b.pop("decision"), "decision", id="succeeded-needs-a-decision"),
        pytest.param(lambda b: b.pop("outputs"), "outputs", id="succeeded-needs-outputs"),
        pytest.param(lambda b: b.pop("timingsMs"), "timingsMs", id="timings-are-required"),
        pytest.param(lambda b: b.update(schemaVersion="2.0"), "1.0", id="version-is-pinned"),
        pytest.param(lambda b: b.update(status="PARTIAL"), "PARTIAL", id="status-is-closed"),
        pytest.param(lambda b: b.update(unexpected=1), "unexpected", id="unknown-keys-rejected"),
        pytest.param(
            lambda b: b["source"].update(kind="stream"), "stream", id="source-kind-is-closed"
        ),
        pytest.param(
            lambda b: b["source"].update(sha256="not-a-digest"), "not-a-digest", id="digest-shape"
        ),
        pytest.param(
            lambda b: b["model"].pop("providers"),
            "providers",
            id="the-actual-provider-is-required",
        ),
    ],
)
def test_the_result_contract_rejects_a_malformed_body(
    result_schema, design_result_example, mutate, expect
):
    body = copy.deepcopy(design_result_example)
    mutate(body)
    messages = _errors(Draft202012Validator(result_schema), body)
    assert messages, "the malformed body was accepted"
    assert any(expect in message for message in messages), messages


def test_a_failed_result_carries_an_error_and_never_clears(result_schema, design_result_example):
    body = copy.deepcopy(design_result_example)
    body["status"] = "FAILED"
    body.pop("decision")
    body.pop("outputs")
    assert _errors(Draft202012Validator(result_schema), body), "FAILED without an error was accepted"

    body["error"] = {"code": "DECODE_FAILED", "message": "the image could not be decoded",
                     "class": "permanent"}
    assert _errors(Draft202012Validator(result_schema), body) == []

    body["decision"] = {"outcome": "CLEAR", "pass": True}
    assert _errors(Draft202012Validator(result_schema), body), "a failed result cleared the gate"

    body["decision"] = {"outcome": "HOLD", "pass": False}
    assert _errors(Draft202012Validator(result_schema), body) == []


def test_a_truncated_result_must_name_its_sidecar(result_schema, design_result_example):
    body = copy.deepcopy(design_result_example)
    body["outputs"]["truncated"] = True
    body.pop("artifacts", None)
    assert _errors(Draft202012Validator(result_schema), body), "a silent truncation was accepted"

    body["artifacts"] = {
        "evidenceId": "01KZ8Q4M7N3P5R7T9V1X3Z5B7D",
        "localRelativePath": "cam-01/cap-0001.inference.json",
        "sha256": HEX256,
        "bytes": 40960,
    }
    assert _errors(Draft202012Validator(result_schema), body) == []
