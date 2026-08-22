"""The corpus builder itself: determinism, bundle shape, and camera fidelity (D-IP-17)."""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.fixtures.build import (
    DEFAULT_OUT,
    build,
    load_bundle_manifest,
    main,
    manifest_from_document,
)
from image_processor.types import BundleManifest, Family


def _digests(root):
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_rebuilding_in_place_reproduces_every_byte(tmp_path):
    root = tmp_path / "corpus"
    build(root)
    before = _digests(root)
    build(root)
    assert _digests(root) == before


def test_two_builds_agree_on_the_oracle_wherever_they_are_written(tmp_path):
    assert build(tmp_path / "a") == build(tmp_path / "b")


def test_a_spool_sidecar_records_where_its_image_actually_is(tmp_path):
    document = build(tmp_path / "corpus")
    record = document["spool"][0]
    body = json.loads((tmp_path / "corpus" / record["sidecar"]).read_text(encoding="utf-8"))
    installed = tmp_path / "corpus" / record["path"]
    assert body["image"]["absolutePath"] == installed.resolve().as_posix()
    assert body["image"]["fileUri"] == installed.resolve().as_uri()


def test_a_different_seed_changes_only_the_pseudo_random_fixtures(tmp_path):
    default = build(tmp_path / "a")
    other = build(tmp_path / "b", seed=99)
    assert default["images"]["quadrant-red.png"]["sha256"] == other["images"]["quadrant-red.png"]["sha256"]
    assert default["images"]["detect-scene.png"]["sha256"] != other["images"]["detect-scene.png"]["sha256"]


def test_every_bundle_carries_the_four_members_and_matching_digests(corpus):
    for key, bundle in corpus.expected["bundles"].items():
        directory = corpus.path(bundle["path"])
        document = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert set(document["files"]) == {"model.onnx", "labels.json", "transforms.json"}, key
        for member, digest in document["files"].items():
            assert hashlib.sha256((directory / member).read_bytes()).hexdigest() == digest, key
        transforms = json.loads((directory / "transforms.json").read_text(encoding="utf-8"))
        assert transforms["preprocess"] == document["preprocess"], key
        assert transforms["transformVersion"] == document["transformVersion"], key
        labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        assert isinstance(labels, list) and labels, key


def test_the_manifest_document_maps_onto_the_shared_dataclass(corpus):
    manifest = corpus.manifest("synthetic-detection-grid-1.0.0")
    assert isinstance(manifest, BundleManifest)
    assert manifest.family is Family.DETECTION
    assert manifest.model_id == "synthetic-detection-grid"
    assert manifest.inputs[0].name == "images"
    assert manifest.inputs[0].shape == (1, 3, 64, 64)
    assert manifest.outputs[0].shape == (1, 84, 8)
    assert manifest.key_id is None


def test_a_manifest_may_be_rebuilt_from_its_own_document(corpus):
    directory = corpus.path(corpus.expected["bundles"]["synthetic-anomaly-map-1.0.0"]["path"])
    document = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_from_document(document) == load_bundle_manifest(directory)


def test_every_recorded_file_is_present_with_the_recorded_size_and_digest(corpus):
    records = list(corpus.expected["images"].values())
    records += corpus.expected["badInputs"]
    for record in records:
        data = corpus.read(record["path"])
        assert len(data) == record["bytes"], record["path"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]


def test_the_bad_input_set_covers_every_hostile_shape(corpus):
    names = {record["path"].split("/")[-1] for record in corpus.expected["badInputs"]}
    assert {
        "corrupt.jpg",
        "truncated.jpg",
        "zero-byte.jpg",
        "wrong-extension.png",
        "not-an-image.png",
        "bomb-dims.png",
        "bomb-pixels.png",
        "bomb-declared.png",
        "sixteen-bit.tiff",
        "sixteen-bit.png",
        "animated.gif",
        "multipage.tiff",
    } <= names


def test_a_decompression_bomb_is_tiny_on_disk_and_enormous_in_its_header(corpus):
    record = next(entry for entry in corpus.expected["badInputs"] if entry["path"].endswith("bomb-declared.png"))
    assert record["bytes"] < 200


def test_a_spool_capture_is_a_camera_shaped_pair(corpus):
    for record in corpus.expected["spool"]:
        image = corpus.path(record["path"])
        sidecar = corpus.path(record["sidecar"])
        assert image.exists() and sidecar.exists(), record["path"]
        assert sidecar.name == image.name + ".json"
        body = json.loads(sidecar.read_text(encoding="utf-8"))
        assert body["schemaVersion"] == 1
        assert body["captureId"] == record["captureId"]
        assert body["cameraId"] == record["cameraId"]
        assert body["correlationId"] == record["correlationId"]
        assert body["image"]["bytes"] == image.stat().st_size == record["bytes"]
        assert body["image"]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert body["image"]["relativePath"] == record["relativePath"]
        assert body["image"]["metadataSidecarRelativePath"] == record["relativePath"] + ".json"


def test_the_sidecar_carries_the_camera_terminal_body_fields(corpus):
    body = json.loads(corpus.path(corpus.expected["spool"][0]["sidecar"]).read_text(encoding="utf-8"))
    assert set(body) >= {
        "schemaVersion",
        "eventId",
        "captureId",
        "cameraId",
        "correlationId",
        "trigger",
        "captureProfile",
        "captureMode",
        "timestamps",
        "durationsMs",
        "image",
        "frame",
        "camera",
        "metadata",
    }
    assert body["trigger"]["type"] == "schedule"
    assert "scheduleId" in body["trigger"]
    assert body["captureMode"] == "simulated"
    assert body["timestamps"]["cameraFrameTimestampQuality"] == "adapter-receive"
    assert body["camera"]["backend"] == "sim"
    assert body["frame"]["pixelFormat"] == "JPEG"
    assert body["image"]["encoding"] == "jpeg"
    assert body["image"]["contentType"] == "image/jpeg"
    assert "failure" not in body


def test_a_spool_image_under_the_camera_root_resolves_by_its_relative_path(corpus):
    record = corpus.expected["spool"][0]
    root = corpus.path(record["cameraRoot"])
    assert (root / record["relativePath"]).resolve() == corpus.path(record["path"]).resolve()


def test_the_command_line_entry_point_builds_into_the_named_directory(tmp_path, capsys):
    assert main(["--out", str(tmp_path / "cli"), "--seed", "7"]) == 0
    assert (tmp_path / "cli" / "expected.json").exists()
    assert "built 7 bundles" in capsys.readouterr().out


def test_the_default_target_is_inside_the_fixtures_directory():
    assert DEFAULT_OUT.name == "out"
    assert DEFAULT_OUT.parent.name == "fixtures"
