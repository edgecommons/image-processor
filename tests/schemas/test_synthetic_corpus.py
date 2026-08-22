"""The synthetic corpus and the shipped contracts are the same contract.

`tests/fixtures/build.py` writes seven bundles whose answers are computable by hand, and the
engine reads them. Both facts are worthless if the bundles are not ones the component would
accept, so every manifest is validated against the shipped bundle contract and every known
answer is assembled into a result body and validated against the shipped wire contract.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

HEX256 = "ab" * 32
DIGEST = "sha256:" + HEX256


def _fixture_builder(component_root: Path):
    spec = importlib.util.spec_from_file_location(
        "wp1_fixture_build", component_root / "tests" / "fixtures" / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def corpus(component_root, tmp_path_factory):
    """Build the synthetic corpus once and return its root and its record."""
    root = tmp_path_factory.mktemp("corpus")
    record = _fixture_builder(component_root).build(root / "out")
    return root / "out", record


@pytest.fixture(scope="session")
def bundle_names(corpus):
    return sorted(corpus[1]["bundles"])


def _errors(validator, instance):
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(instance)]


def test_the_corpus_covers_every_task_family(corpus, bundle_names):
    families = {corpus[1]["bundles"][name]["family"] for name in bundle_names}
    assert families == {"classification", "detection", "segmentation", "anomaly"}
    assert len(bundle_names) == 7


def test_every_synthetic_manifest_validates(corpus, bundle_names, manifest_schema):
    """A bundle the fixtures build is a bundle the component would stage."""
    root, record = corpus
    validator = Draft202012Validator(manifest_schema)
    for name in bundle_names:
        document = json.loads(
            (root / record["bundles"][name]["path"] / "manifest.json").read_text(encoding="utf-8")
        )
        assert _errors(validator, document) == [], name


def test_every_synthetic_manifest_declares_what_the_loader_reads(corpus, bundle_names):
    """The manifest keys are the ones `bundles/manifest.py` maps onto `BundleManifest`."""
    from image_processor.bundles.manifest import parse_manifest

    root, record = corpus
    for name in bundle_names:
        document = json.loads(
            (root / record["bundles"][name]["path"] / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = parse_manifest(document)
        assert manifest.min_onnxruntime, name
        assert manifest.estimated_device_mib > 0, name
        assert manifest.provider_policy in ("requireListed", "preferListed"), name
        assert manifest.family.value == record["bundles"][name]["family"], name


def _result_body(record, name, case) -> dict:
    """Assemble the wire body WP6 builds from one known answer."""
    bundle = record["bundles"][name]
    expected = case["expected"]
    decision = case["decision"]
    outputs = {"family": bundle["family"], "truncated": False}
    outputs.update(expected)
    return {
        "schemaVersion": "1.0",
        "inferenceId": "01KZ8Q4M7N3P5R7T9V1X3Z5B7D",
        "routeId": "clearance-cam-01",
        "status": "SUCCEEDED",
        "source": {
            "kind": "spool",
            "relativePath": case["image"],
            "bytes": 4096,
            "sha256": HEX256,
        },
        "model": {
            "id": bundle["modelId"],
            "version": bundle["version"],
            "digest": DIGEST,
            "runtime": "onnxruntime",
            "providers": ["CPUExecutionProvider"],
            "gpu": None,
            "transformVersion": "1",
        },
        "decision": {
            "outcome": decision["outcome"],
            "pass": decision["passed"],
            "confidence": decision["confidence"],
            "threshold": decision["threshold"],
            "rule": decision["rule"],
        },
        "outputs": outputs,
        "timingsMs": {"queue": 0.0, "modelLoad": 0.0, "preprocess": 1.0, "inference": 2.0,
                      "postprocess": 0.5, "total": 3.5},
    }


def test_every_known_answer_assembles_into_a_valid_result(corpus, bundle_names, result_schema):
    """The normalized output each family produces is one the wire contract accepts."""
    _root, record = corpus
    validator = Draft202012Validator(result_schema)
    seen = 0
    for name in bundle_names:
        for case in record["bundles"][name]["cases"]:
            body = _result_body(record, name, case)
            assert _errors(validator, body) == [], f"{name}: {case['image']}"
            seen += 1
    assert seen >= len(bundle_names), "every bundle contributes at least one case"


def test_a_held_answer_never_clears(corpus, bundle_names):
    """The corpus exercises both verdicts, and a failing rule never reports CLEAR."""
    _root, record = corpus
    outcomes = {
        case["decision"]["outcome"]
        for name in bundle_names
        for case in record["bundles"][name]["cases"]
    }
    assert "CLEAR" in outcomes
    assert outcomes - {"CLEAR", "HOLD", "FAIL"} == set()
    for name in bundle_names:
        for case in record["bundles"][name]["cases"]:
            decision = case["decision"]
            assert decision["passed"] is (decision["outcome"] == "CLEAR"), name
