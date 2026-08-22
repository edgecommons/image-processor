"""The model bundle manifest contract (DESIGN.md §8, LLD `BundleManifest`)."""

import copy

import pytest
from jsonschema import Draft202012Validator

HEX256 = "ab" * 32


def manifest() -> dict:
    """A complete, valid manifest for a detection bundle."""
    return {
        "schemaVersion": 1,
        "modelId": "line-clearance-cam-01",
        "version": "2026.08.20",
        "files": {
            "manifest.json": HEX256,
            "model.onnx": HEX256,
            "labels.json": HEX256,
            "warmup/input-01.bin": HEX256,
            "warmup/expected-01.json": HEX256,
        },
        "minOnnxRuntime": "1.20.0",
        "providersPermitted": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "providerPolicy": "requireListed",
        "inputs": [{"name": "images", "dtype": "float32", "shape": ["N", 3, 640, 640]}],
        "outputs": [{"name": "output", "dtype": "float32", "shape": ["N", 8400, 85]}],
        "dynamicBatch": True,
        "family": "detection",
        "familyParams": {
            "decoder": "yolox",
            "strides": [8, 16, 32],
            "numClasses": 80,
            "scoreThreshold": 0.25,
            "iouThreshold": 0.45,
            "maxDetections": 100,
        },
        "preprocess": {
            "resize": {"mode": "letterbox", "width": 640, "height": 640, "padValue": [114, 114, 114]},
            "normalize": {"scale": 1.0, "mean": [0, 0, 0], "std": [1, 1, 1]},
            "layout": "NCHW",
            "colorOrder": "RGB",
            "dtype": "float32",
        },
        "decisionRules": {
            "pass": "$.detections[?(@.label=='foreign-object')] == []",
            "confidence": "$.detections[0].score",
            "threshold": 0.5,
            "outcome": {"whenPass": "CLEAR", "whenFail": "HOLD", "whenUnevaluable": "HOLD"},
        },
        "maxResultItems": 100,
        "estimatedDeviceMiB": 900,
        "measuredDeviceMiB": {"sm_120": {"loadPeakMiB": 1400, "steadyMiB": 880}},
        "warmup": [
            {
                "input": "warmup/input-01.bin",
                "expected": "warmup/expected-01.json",
                "dtype": "float32",
                "shape": [1, 3, 640, 640],
                "compare": "tensors",
            }
        ],
        "tolerances": {"absolute": 0.0001, "relative": 0.001, "boxIou": 0.9},
        "compatibilityKeys": {"precision": "fp16", "shapeProfile": "1x3x640x640"},
        "provenance": {
            "publisher": "EdgeCommons reference models",
            "publishedAt": "2026-08-20T09:00:00Z",
            "license": "Apache-2.0",
        },
        "keyId": "pharma-model-publisher-1",
        "transformVersion": "1",
    }


def _errors(manifest_schema, instance):
    validator = Draft202012Validator(manifest_schema)
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(instance)]


def test_a_complete_manifest_validates(manifest_schema):
    assert _errors(manifest_schema, manifest()) == []


@pytest.mark.parametrize(
    "family, params",
    [
        ("classification", {"topK": 5, "activation": "softmax", "labelsFile": "labels.json"}),
        ("segmentation", {"numClasses": 21, "threshold": 0.5, "ignoreIndex": 0}),
        ("anomaly", {"threshold": 0.7, "higherIsAnomalous": True, "maxRegions": 5}),
    ],
)
def test_every_task_family_has_its_own_parameters(manifest_schema, family, params):
    body = manifest()
    body["family"] = family
    body["familyParams"] = params
    assert _errors(manifest_schema, body) == []


def test_a_family_rejects_another_familys_parameters(manifest_schema):
    body = manifest()
    body["family"] = "classification"
    body["familyParams"] = {"iouThreshold": 0.45}
    assert _errors(manifest_schema, body), "detection parameters were accepted on a classifier"


@pytest.mark.parametrize(
    "mutate, expect",
    [
        pytest.param(lambda m: m.pop("files"), "files", id="files-are-required"),
        pytest.param(lambda m: m.pop("providerPolicy"), "providerPolicy", id="policy-required"),
        pytest.param(lambda m: m.pop("transformVersion"), "transformVersion", id="transform"),
        pytest.param(lambda m: m.update(schemaVersion=2), "schemaVersion", id="version-is-pinned"),
        pytest.param(lambda m: m.update(family="pose"), "pose", id="families-are-closed"),
        pytest.param(lambda m: m.update(extra=1), "extra", id="unknown-keys-rejected"),
        pytest.param(
            lambda m: m["files"].update({"../escape": HEX256}), "escape", id="no-traversal"
        ),
        pytest.param(
            lambda m: m["files"].update({"/etc/passwd": HEX256}), "passwd", id="no-absolute-member"
        ),
        pytest.param(
            lambda m: m["files"].update({"model.onnx": "not-a-digest"}),
            "not-a-digest",
            id="digest-shape",
        ),
        pytest.param(
            lambda m: m.update(providersPermitted=["MagicExecutionProvider"]),
            "Magic",
            id="providers-are-closed",
        ),
        pytest.param(
            lambda m: m["decisionRules"]["outcome"].update(whenFail="CLEAR"),
            "CLEAR",
            id="a-failed-rule-never-clears",
        ),
        pytest.param(
            lambda m: m["decisionRules"].update(confidence="decision.confidence"),
            "decision.confidence",
            id="confidence-is-a-jsonpath",
        ),
        pytest.param(
            lambda m: m["preprocess"].update(layout="CHWN"), "CHWN", id="layout-is-closed"
        ),
        pytest.param(
            lambda m: m["preprocess"]["resize"].update(mode="squash"), "squash", id="resize-closed"
        ),
        pytest.param(lambda m: m.pop("decisionRules"), "decisionRules", id="rules-are-required"),
        pytest.param(
            lambda m: m.update(maxResultItems=0), "0", id="result-cardinality-is-positive"
        ),
    ],
)
def test_the_manifest_contract_rejects_a_malformed_bundle(manifest_schema, mutate, expect):
    body = manifest()
    mutate(body)
    messages = _errors(manifest_schema, body)
    assert messages, "the malformed manifest was accepted"
    assert any(expect in message for message in messages), messages


def test_an_unsigned_bundle_names_no_key(manifest_schema):
    body = manifest()
    body["keyId"] = None
    assert _errors(manifest_schema, body) == []


def test_a_dynamic_axis_may_be_named(manifest_schema):
    body = manifest()
    body["inputs"] = [{"name": "images", "dtype": "float32", "shape": ["N", 3, "height", "width"]}]
    assert _errors(manifest_schema, body) == []
