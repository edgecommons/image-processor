"""The bundle authoring tool: key generation, packing, signing, and the staging round trip."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from image_processor.bundles import (
    BundleCache,
    BundleError,
    sha256_file,
    sign_manifest,
    stage_bundle,
    verify_manifest_signature,
)
from tools.make_bundle import build_manifest_document, collect_files, main, make_bundle

from .conftest import manifest_document, write_source


def test_gen_key_writes_a_usable_keypair(tmp_path: Path, capsys) -> None:
    key_path = tmp_path / "keys" / "publisher-1.pem"
    assert main(["--gen-key", str(key_path)]) == 0

    output = capsys.readouterr().out
    assert str(key_path) in output
    private = key_path.read_bytes()
    public_raw = key_path.with_name("publisher-1.pem.pub").read_bytes()
    assert key_path.with_name("publisher-1.pem.pub.pem").read_bytes().startswith(b"-----BEGIN")
    assert len(public_raw) == 32
    verify_manifest_signature(b"payload", sign_manifest(b"payload", private), public_raw)


def test_gen_key_refuses_to_overwrite(tmp_path: Path, capsys) -> None:
    key_path = tmp_path / "publisher-1.pem"
    assert main(["--gen-key", str(key_path)]) == 0
    assert main(["--gen-key", str(key_path)]) == 2
    assert "KEY_EXISTS" in capsys.readouterr().err


def test_an_encrypted_private_key_round_trips(tmp_path: Path) -> None:
    key_path = tmp_path / "publisher-1.pem"
    assert main(["--gen-key", str(key_path), "--key-password", "line-clearance"]) == 0
    assert b"ENCRYPTED" in key_path.read_bytes()

    source = write_source(tmp_path / "src")
    out = tmp_path / "dist" / "bundle.tar.gz"
    argv = [
        str(source),
        "--out",
        str(out),
        "--key",
        str(key_path),
        "--key-id",
        "pharma-model-publisher-1",
        "--key-password",
        "line-clearance",
    ]
    assert main(argv) == 0
    assert out.is_file()


def test_the_round_trip_from_gen_key_to_a_cached_bundle(
    tmp_path: Path, schema_path: Path, capsys
) -> None:
    key_path = tmp_path / "publisher-1.pem"
    main(["--gen-key", str(key_path)])
    capsys.readouterr()

    source = write_source(tmp_path / "src", manifest_document(keyId=None))
    authored = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    out = tmp_path / "dist" / "line-clearance-2026.08.20.tar.gz"
    argv = [
        str(source),
        "--out",
        str(out),
        "--key",
        str(key_path),
        "--key-id",
        "pharma-model-publisher-1",
        "--schema",
        str(schema_path),
    ]
    assert main(argv) == 0
    digest = capsys.readouterr().out.strip()
    assert digest == "sha256:" + sha256_file(out)

    cache = BundleCache(tmp_path / "models", schema_path)
    public_raw = key_path.with_name("publisher-1.pem.pub").read_bytes()
    cached = stage_bundle(
        uri=str(out),
        digest=digest,
        staging_root=tmp_path / "staging",
        cache=cache,
        schema_path=schema_path,
        signing_required=True,
        trusted_keys={"pharma-model-publisher-1": public_raw},
        model_id=authored["modelId"],
        version=authored["version"],
    )

    assert cached.manifest.key_id == "pharma-model-publisher-1"
    assert cached.manifest.model_id == authored["modelId"]
    assert cached.manifest.version == authored["version"]
    assert cached.manifest.family.value == authored["family"]
    assert cached.manifest.preprocess == authored["preprocess"]
    assert cached.manifest.decision_rules == authored["decisionRules"]
    assert cached.manifest.warmup == authored["warmup"]
    assert cached.manifest.tolerances == authored["tolerances"]
    assert cached.manifest.compatibility_keys == authored["compatibilityKeys"]
    assert cached.manifest.provenance == authored["provenance"]
    assert cached.manifest.transform_version == authored["transformVersion"]
    assert cached.manifest.files == collect_files(source)
    assert cached.model_path.read_bytes() == (source / "model.onnx").read_bytes()


def test_the_digest_is_reproducible(tmp_path: Path, schema_path: Path) -> None:
    source = write_source(tmp_path / "src")
    first = make_bundle(source, tmp_path / "one.tar.gz", schema_path=schema_path)
    second = make_bundle(source, tmp_path / "two.tar.gz", schema_path=schema_path)
    assert first == second
    assert make_bundle(source, tmp_path / "one.tar") == make_bundle(source, tmp_path / "two.tar")


def test_an_unsigned_bundle_carries_no_signature(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src")
    out = tmp_path / "unsigned.tar"
    make_bundle(source, out)
    with tarfile.open(out) as tar:
        names = tar.getnames()
    assert names[0] == "manifest.json"
    assert "manifest.sig" not in names


def test_gzip_can_be_forced(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src")
    out = tmp_path / "forced.tar"
    make_bundle(source, out, compress=True)
    assert out.read_bytes()[:2] == b"\x1f\x8b"


def test_signing_without_a_key_id_is_refused(tmp_path: Path, signing_key) -> None:
    source = write_source(tmp_path / "src", manifest_document(keyId=None))
    with pytest.raises(BundleError) as caught:
        make_bundle(source, tmp_path / "out.tar", key=signing_key[0])
    assert caught.value.code == "KEY_ID_MISSING"


def test_the_source_directory_must_exist_and_hold_files(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        make_bundle(tmp_path / "absent", tmp_path / "out.tar")
    assert caught.value.code == "BUNDLE_EMPTY"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BundleError) as caught:
        make_bundle(empty, tmp_path / "out.tar")
    assert caught.value.code == "BUNDLE_EMPTY"


@pytest.mark.parametrize("content", ["{ not json", '["not", "an", "object"]'])
def test_an_unusable_authored_manifest_is_refused(tmp_path: Path, content: str) -> None:
    source = write_source(tmp_path / "src")
    (source / "manifest.json").write_text(content, encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        build_manifest_document(source)
    assert caught.value.code == "MANIFEST_INVALID"


def test_the_schema_gate_stops_an_incomplete_manifest(
    tmp_path: Path, schema_path: Path, capsys
) -> None:
    document = manifest_document()
    document.pop("transformVersion")
    source = write_source(tmp_path / "src", document)
    argv = [str(source), "--out", str(tmp_path / "out.tar"), "--schema", str(schema_path)]
    assert main(argv) == 2
    assert "MANIFEST_INVALID" in capsys.readouterr().err


def test_the_manifest_is_still_parsed_without_a_schema(tmp_path: Path, capsys) -> None:
    source = write_source(tmp_path / "src", manifest_document(family="pose-estimation"))
    assert main([str(source), "--out", str(tmp_path / "out.tar")]) == 2
    assert "MANIFEST_INVALID" in capsys.readouterr().err


def test_a_missing_key_file_is_an_io_error(tmp_path: Path, capsys) -> None:
    source = write_source(tmp_path / "src")
    argv = [str(source), "--out", str(tmp_path / "out.tar"), "--key", str(tmp_path / "absent.pem")]
    assert main(argv) == 2
    assert "IO_ERROR" in capsys.readouterr().err


def test_src_and_out_are_required(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main([str(tmp_path)])
    assert caught.value.code == 2


def test_collect_files_skips_the_manifest_and_signature(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src")
    (source / "manifest.sig").write_bytes(b"x" * 64)
    files = collect_files(source)
    assert "manifest.json" not in files
    assert "manifest.sig" not in files
    assert files["model.onnx"] == sha256_file(source / "model.onnx")


def test_extra_manifest_fields_can_be_supplied(tmp_path: Path) -> None:
    source = write_source(tmp_path / "src")
    document = build_manifest_document(source, key_id="k1", overrides={"version": "2026.09.01"})
    assert document["keyId"] == "k1"
    assert document["version"] == "2026.09.01"
    assert "model.onnx" in document["files"]
