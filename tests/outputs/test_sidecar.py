"""The evidence sidecar: its content, its ordering, and what it refuses to overwrite."""

from __future__ import annotations

import hashlib
import json

import pytest

from image_processor.outputs import sidecar as sidecar_module
from image_processor.outputs.sidecar import (
    SIDECAR_SUFFIX,
    SidecarError,
    encode_document,
    sidecar_document,
    sidecar_path_for,
    write_sidecar,
)
from image_processor.outputs.result import build_result_body
from tests.outputs.conftest import make_job, make_result


class _Manifest:
    """The manifest fields the evidence document binds."""

    provider_policy = "requireListed"
    min_onnxruntime = "1.17.0"
    transform_version = "1.0.0"
    max_result_items = 100


def _document(job=None, body=None) -> dict:
    """Build one evidence document."""
    job = job or make_job(staged_path="/var/spool/cam-01/cap.jpg")
    body = body or build_result_body(job, make_result())
    return sidecar_document(
        job,
        body,
        evidence_id=job.inference_id,
        config_generation=7,
        manifest=_Manifest(),
        written_at_ms=1789012506010,
    )


def test_the_evidence_binds_the_identity_the_result_alone_does_not_carry():
    document = _document()

    assert document["schemaVersion"] == "1.0"
    assert document["evidenceId"] == document["inferenceId"] == "01k5inference0001"
    assert document["routeId"] == "clearance-cam-01"
    assert document["configGeneration"] == 7
    assert document["writtenAtMs"] == 1789012506010
    assert document["providerPolicy"] == "requireListed"
    assert document["minOnnxruntime"] == "1.17.0"
    assert document["transformVersion"] == "1.0.0"
    assert document["stagedPath"] == "/var/spool/cam-01/cap.jpg"
    assert document["result"]["decision"]["outcome"] == "CLEAR"


def test_the_evidence_path_sits_beside_the_image(tmp_path):
    assert sidecar_path_for(tmp_path / "a" / "cap.jpg") == tmp_path / "a" / (
        "cap.jpg" + SIDECAR_SUFFIX
    )


def test_installing_writes_a_temporary_file_flushes_and_renames(tmp_path, monkeypatch):
    order = []
    target = tmp_path / "deep" / "cap.jpg.inference.json"
    monkeypatch.setattr(sidecar_module, "fsync", lambda handle: order.append("flush"))
    real_replace = sidecar_module.replace

    def _replace(src, dst):
        order.append("install")
        assert order == ["flush", "install"], "the bytes are flushed before they are installed"
        assert not target.exists()
        real_replace(src, dst)

    monkeypatch.setattr(sidecar_module, "replace", _replace)

    installed = write_sidecar(target, _document())

    assert order == ["flush", "install"]
    assert installed.path == target
    assert installed.bytes == target.stat().st_size
    assert installed.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_a_failed_rename_leaves_no_sidecar_and_no_temporary(tmp_path, monkeypatch):
    def _boom(src, dst):
        raise OSError("the volume went away")

    monkeypatch.setattr(sidecar_module, "replace", _boom)
    target = tmp_path / "cap.jpg.inference.json"

    with pytest.raises(SidecarError) as failure:
        write_sidecar(target, _document())

    assert failure.value.code == "SIDECAR_WRITE_FAILED"
    assert not target.exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_an_identical_sidecar_is_adopted_so_a_retried_commit_is_idempotent(tmp_path):
    target = tmp_path / "cap.jpg.inference.json"
    document = _document()

    first = write_sidecar(target, document)
    second = write_sidecar(target, document)

    assert first == second


def test_a_different_sidecar_is_a_collision_not_an_overwrite(tmp_path):
    target = tmp_path / "cap.jpg.inference.json"
    write_sidecar(target, _document())
    other = _document(job=make_job(inference_id="01k5inference0002"))

    with pytest.raises(SidecarError) as failure:
        write_sidecar(target, other)

    assert failure.value.code == "SIDECAR_COLLISION"
    assert json.loads(target.read_text(encoding="utf-8"))["inferenceId"] == "01k5inference0001"


def test_a_document_json_cannot_express_is_refused():
    with pytest.raises(SidecarError) as failure:
        encode_document({"score": float("inf")})

    assert failure.value.code == "SIDECAR_NOT_SERIALIZABLE"


def test_an_unwritable_directory_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sidecar_module.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no"))
    )

    with pytest.raises(SidecarError) as failure:
        write_sidecar(tmp_path / "x" / "cap.inference.json", _document())

    assert failure.value.code == "SIDECAR_DIR_UNWRITABLE"
