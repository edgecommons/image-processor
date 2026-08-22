"""Manifest schema validation and per-file digest verification (DESIGN.md section 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from image_processor.bundles import (
    BundleError,
    extract_tarball,
    load_manifest,
    load_schema,
    parse_manifest,
    resolve_model_path,
    validate_document,
)
from image_processor.types import Family


@pytest.fixture
def extracted(tmp_path: Path, build_bundle) -> Path:
    """Extract a freshly built bundle and return its directory."""
    built = build_bundle()
    dest = tmp_path / "extracted"
    extract_tarball(built.archive, dest)
    return dest


def rewrite(bundle_dir: Path, mutate: Callable[[Dict[str, Any]], None]) -> None:
    """Apply a change to an extracted bundle's manifest.json."""
    path = bundle_dir / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_good_bundle_loads_every_design_field(extracted: Path, schema_path: Path) -> None:
    manifest = load_manifest(extracted, schema_path)
    assert manifest.schema_version == 1
    assert manifest.model_id == "line-clearance-cam-01"
    assert manifest.version == "2026.08.20"
    assert manifest.min_onnxruntime == "1.18.0"
    assert manifest.providers_permitted == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert manifest.provider_policy == "requireListed"
    assert manifest.inputs[0].name == "images"
    assert manifest.inputs[0].shape == ("N", 3, 224, 224)
    assert manifest.outputs[0].dtype == "float32"
    assert manifest.dynamic_batch is True
    assert manifest.family is Family.CLASSIFICATION
    assert manifest.family_params["topK"] == 2
    assert manifest.preprocess["layout"] == "NCHW"
    assert manifest.decision_rules["threshold"] == 0.8
    assert manifest.max_result_items == 10
    assert manifest.estimated_device_mib == 512
    assert manifest.warmup[0]["input"] == "warmup/input-01.bin"
    assert manifest.tolerances["absolute"] == 0.001
    assert manifest.compatibility_keys["gpuClass"] == "sm_86"
    assert manifest.provenance["publisher"] == "pharma-mlops"
    assert manifest.key_id == "pharma-model-publisher-1"
    assert manifest.transform_version == "2026.08.20-1"
    assert set(manifest.files) == {
        "model.onnx",
        "labels.json",
        "transforms.json",
        "result.schema.json",
        "model-card.json",
        "warmup/input-01.bin",
        "warmup/expected-01.json",
    }
    assert resolve_model_path(extracted, manifest) == extracted / "model.onnx"


def test_a_missing_manifest_is_refused(tmp_path: Path, schema_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BundleError) as caught:
        load_manifest(empty, schema_path)
    assert caught.value.code == "MANIFEST_MISSING"


def test_a_manifest_that_is_not_json_is_refused(extracted: Path, schema_path: Path) -> None:
    (extracted / "manifest.json").write_bytes(b"{not json")
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"


def test_the_schema_is_enforced(extracted: Path, schema_path: Path) -> None:
    rewrite(extracted, lambda document: document.pop("providerPolicy"))
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"


def test_a_tensor_shape_is_enforced(extracted: Path, schema_path: Path) -> None:
    rewrite(extracted, lambda document: document["inputs"][0].update({"shape": [{"dim": 3}]}))
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"


def test_a_tampered_file_is_caught(extracted: Path, schema_path: Path) -> None:
    (extracted / "labels.json").write_text('["clear", "hold", "extra"]', encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "FILE_DIGEST_MISMATCH"


def test_digest_verification_can_be_skipped_for_a_cache_read(extracted: Path, schema_path: Path) -> None:
    (extracted / "labels.json").write_text('["clear", "hold", "extra"]', encoding="utf-8")
    assert load_manifest(extracted, schema_path, verify_files=False).model_id


def test_a_declared_file_that_is_gone_is_caught(extracted: Path, schema_path: Path) -> None:
    (extracted / "warmup" / "input-01.bin").unlink()
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "FILE_MISSING"


def test_an_undeclared_file_is_refused(extracted: Path, schema_path: Path) -> None:
    (extracted / "extra-weights.bin").write_bytes(b"unverified payload")
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "FILE_UNDECLARED"


def test_a_manifest_cannot_declare_its_own_digest(extracted: Path, schema_path: Path) -> None:
    rewrite(
        extracted,
        lambda document: document["files"].update({"manifest.json": "0" * 64}),
    )
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"


def test_a_files_key_cannot_escape_the_bundle(extracted: Path, schema_path: Path) -> None:
    rewrite(extracted, lambda document: document["files"].update({"../secrets.env": "0" * 64}))
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"
    # The shipped schema's `bundlePath` pattern refuses the key before the loader's own
    # traversal guard sees it, so the diagnostic names the key rather than the guard.
    assert "../secrets.env" in caught.value.message


def test_a_missing_schema_file_is_reported(extracted: Path, tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, tmp_path / "no-such-schema.json")
    assert caught.value.code == "SCHEMA_UNAVAILABLE"


def test_a_schema_that_is_not_json_is_reported(extracted: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.schema.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        load_manifest(extracted, broken)
    assert caught.value.code == "SCHEMA_UNAVAILABLE"


def test_an_invalid_schema_is_reported(extracted: Path, tmp_path: Path) -> None:
    broken = tmp_path / "invalid.schema.json"
    broken.write_text(json.dumps({"type": "not-a-json-type"}), encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        validate_document({"modelId": "x"}, broken)
    assert caught.value.code == "SCHEMA_UNAVAILABLE"


def test_the_schema_is_cached_by_path_and_timestamp(schema_path: Path) -> None:
    assert load_schema(schema_path) is load_schema(schema_path)


def test_parse_manifest_rejects_a_non_object() -> None:
    with pytest.raises(BundleError) as caught:
        parse_manifest(["not", "an", "object"])  # type: ignore[arg-type]
    assert caught.value.code == "MANIFEST_INVALID"


def test_parse_manifest_rejects_an_unknown_family() -> None:
    with pytest.raises(BundleError) as caught:
        parse_manifest({"family": "pose-estimation", "files": {"m.onnx": "0" * 64}})
    assert caught.value.code == "MANIFEST_INVALID"
    assert "task family" in caught.value.message


def test_parse_manifest_requires_a_family() -> None:
    with pytest.raises(BundleError) as caught:
        parse_manifest({"files": {"m.onnx": "0" * 64}})
    assert caught.value.code == "MANIFEST_INVALID"


def test_parse_manifest_requires_the_identity_fields() -> None:
    with pytest.raises(BundleError) as caught:
        parse_manifest({"family": "anomaly", "files": {"m.onnx": "0" * 64}})
    assert caught.value.code == "MANIFEST_INVALID"
    assert "schemaVersion" in caught.value.message


@pytest.mark.parametrize(
    "files", [{}, "model.onnx", {"model.onnx": 5}]
)
def test_parse_manifest_checks_the_files_map(files) -> None:
    document = {"schemaVersion": 1, "modelId": "m", "version": "1", "family": "anomaly", "files": files}
    with pytest.raises(BundleError) as caught:
        parse_manifest(document)
    assert caught.value.code == "MANIFEST_INVALID"


@pytest.mark.parametrize(
    "tensors",
    ["not a list", [["name"]], [{"name": "x", "dtype": "float32"}], [{"name": "x", "dtype": "f", "shape": "224"}]],
)
def test_parse_manifest_checks_tensor_specs(tensors) -> None:
    document = {
        "schemaVersion": 1,
        "modelId": "m",
        "version": "1",
        "family": "anomaly",
        "files": {"model.onnx": "0" * 64},
        "inputs": tensors,
    }
    with pytest.raises(BundleError) as caught:
        parse_manifest(document)
    assert caught.value.code == "MANIFEST_INVALID"


def test_optional_fields_fall_back_to_empty_values() -> None:
    manifest = parse_manifest(
        {
            "schemaVersion": 2,
            "modelId": "minimal",
            "version": "0.1",
            "family": "anomaly",
            "files": {"model.onnx": "0" * 64},
        }
    )
    assert manifest.providers_permitted == []
    assert manifest.warmup == []
    assert manifest.key_id is None
    assert manifest.dynamic_batch is False
    assert manifest.transform_version == ""


def test_resolve_model_path_accepts_a_single_named_graph(tmp_path: Path) -> None:
    manifest = parse_manifest(
        {
            "schemaVersion": 1,
            "modelId": "m",
            "version": "1",
            "family": "detection",
            "files": {"graphs/detector.onnx": "0" * 64, "labels.json": "1" * 64},
        }
    )
    assert resolve_model_path(tmp_path, manifest) == tmp_path / "graphs" / "detector.onnx"


@pytest.mark.parametrize(
    "files",
    [{"labels.json": "0" * 64}, {"a.onnx": "0" * 64, "b.onnx": "1" * 64}],
)
def test_resolve_model_path_needs_exactly_one_graph(tmp_path: Path, files) -> None:
    manifest = parse_manifest(
        {"schemaVersion": 1, "modelId": "m", "version": "1", "family": "detection", "files": files}
    )
    with pytest.raises(BundleError) as caught:
        resolve_model_path(tmp_path, manifest)
    assert caught.value.code == "MODEL_FILE_MISSING"
