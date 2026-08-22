"""Staging a model bundle end to end (DESIGN.md section 9 steps 2 to 6)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Tuple

import pytest

from image_processor.bundles import (
    BundleCache,
    BundleError,
    extract_tarball,
    sha256_file,
    stage_bundle,
)

from .conftest import file_member, manifest_document, write_tar


@pytest.fixture
def cache(tmp_path: Path, schema_path: Path) -> BundleCache:
    """Return an empty cache for the staging tests."""
    return BundleCache(tmp_path / "models", schema_path)


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    """Return the staging root staging attempts run under."""
    return tmp_path / "staging"


def repack(tmp_path: Path, archive: Path, name: str, mutate: Callable[[Path], None]) -> Tuple[Path, str]:
    """Rebuild a bundle after changing its extracted contents.

    Returns:
        The new archive and its ``sha256:<hex>`` digest.
    """
    work = tmp_path / f"work-{name}"
    extract_tarball(archive, work)
    mutate(work)
    members = [
        file_member(path.relative_to(work).as_posix(), path.read_bytes())
        for path in sorted(work.rglob("*"))
        if path.is_file()
    ]
    out = tmp_path / f"{name}.tar"
    write_tar(out, members)
    return out, "sha256:" + sha256_file(out)


def stage(built, cache: BundleCache, staging: Path, schema_path: Path, **kwargs):
    """Stage a built bundle with this suite's defaults."""
    return stage_bundle(
        uri=str(built.archive),
        digest=built.digest,
        staging_root=staging,
        cache=cache,
        schema_path=schema_path,
        **kwargs,
    )


def test_a_signed_bundle_stages_into_the_cache(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle(compress=True)
    cached = stage(built, cache, staging, schema_path, signing_required=True, trusted_keys=built.trusted)

    assert cached.digest == built.digest
    assert cached.manifest.model_id == "line-clearance-cam-01"
    assert cached.model_path.read_bytes() == (built.source / "model.onnx").read_bytes()
    assert cache.get(built.digest, verify=True) is not None
    assert list(staging.iterdir()) == []


def test_a_cached_bundle_is_not_downloaded_again(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    stage(built, cache, staging, schema_path)
    built.archive.unlink()

    cached = stage(built, cache, staging, schema_path)
    assert cached.manifest.version == "2026.08.20"


def test_a_cached_bundle_that_no_longer_verifies_is_staged_again(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle, caplog
) -> None:
    built = build_bundle()
    cached = stage(built, cache, staging, schema_path)
    (cached.root / "labels.json").write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        restaged = stage(built, cache, staging, schema_path)
    assert restaged.manifest.model_id == "line-clearance-cam-01"
    assert cache.get(built.digest, verify=True) is not None
    assert "no longer verifies" in caplog.text


def test_a_tampered_tarball_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(built.archive),
            digest="sha256:" + "00" * 32,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "DIGEST_MISMATCH"
    assert cache.list() == []
    assert list(staging.iterdir()) == []


def test_signing_can_be_required(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    unsigned = build_bundle(sign=False)
    with pytest.raises(BundleError) as caught:
        stage(unsigned, cache, staging, schema_path, signing_required=True, trusted_keys=unsigned.trusted)
    assert caught.value.code == "SIGNATURE_MISSING"


def test_an_unsigned_bundle_stages_when_signing_is_not_required(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    unsigned = build_bundle(sign=False)
    assert stage(unsigned, cache, staging, schema_path).digest == unsigned.digest


def test_an_untrusted_key_id_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path, signing_required=True, trusted_keys={"other-key": b"x" * 32})
    assert caught.value.code == "UNTRUSTED_KEY"


def test_a_signed_bundle_without_a_key_id_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()

    def drop_key_id(work: Path) -> None:
        document = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        document.pop("keyId")
        (work / "manifest.json").write_text(json.dumps(document), encoding="utf-8")

    archive, digest = repack(tmp_path, built.archive, "no-key-id", drop_key_id)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest=digest,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
            signing_required=True,
            trusted_keys=built.trusted,
        )
    assert caught.value.code == "UNTRUSTED_KEY"


def _tamper_manifest(work: Path) -> None:
    """Change the manifest after it was signed, leaving the old signature in place."""
    document = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    document["estimatedDeviceMiB"] = 1
    (work / "manifest.json").write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def test_a_broken_signature_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    archive, digest = repack(tmp_path, built.archive, "tampered", _tamper_manifest)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest=digest,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
            signing_required=True,
            trusted_keys=built.trusted,
        )
    assert caught.value.code == "BAD_SIGNATURE"


def test_a_broken_signature_is_refused_even_when_signing_is_optional(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    archive, digest = repack(tmp_path, built.archive, "tampered-optional", _tamper_manifest)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest=digest,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
            signing_required=False,
            trusted_keys=built.trusted,
        )
    assert caught.value.code == "BAD_SIGNATURE"


def test_an_unknown_signing_key_is_logged_when_signing_is_optional(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle, caplog
) -> None:
    built = build_bundle()
    with caplog.at_level(logging.WARNING):
        cached = stage(built, cache, staging, schema_path, trusted_keys={"another-key": b"z" * 32})
    assert cached.digest == built.digest
    assert "not configured as trusted" in caplog.text


def test_a_tampered_payload_file_is_caught(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()

    def swap_labels(work: Path) -> None:
        (work / "labels.json").write_text('["hold", "clear"]', encoding="utf-8")

    archive, digest = repack(tmp_path, built.archive, "swapped", swap_labels)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest=digest,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "FILE_DIGEST_MISMATCH"
    assert cache.list() == []


def test_a_path_traversal_member_never_reaches_the_cache(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()

    def add_escape(work: Path) -> None:
        (work / "escape.txt").write_bytes(b"owned")

    archive, _digest = repack(tmp_path, built.archive, "escaping", add_escape)
    members = [
        file_member("manifest.json", (tmp_path / "work-escaping" / "manifest.json").read_bytes()),
        file_member("../escape.txt", b"owned"),
    ]
    evil = write_tar(tmp_path / "evil.tar", members)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(evil),
            digest="sha256:" + sha256_file(evil),
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"
    assert not (tmp_path / "escape.txt").exists()
    assert list(staging.iterdir()) == []


def test_an_archive_without_a_manifest_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path
) -> None:
    archive = write_tar(tmp_path / "headless.tar", [file_member("model.onnx", b"weights")])
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest="sha256:" + sha256_file(archive),
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "MANIFEST_MISSING"


def test_a_manifest_at_a_nested_path_does_not_count(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    extract_tarball(built.archive, tmp_path / "nested-src")
    members = [
        file_member(f"inner/{path.relative_to(tmp_path / 'nested-src').as_posix()}", path.read_bytes())
        for path in sorted((tmp_path / "nested-src").rglob("*"))
        if path.is_file()
    ]
    archive = write_tar(tmp_path / "nested.tar", members)
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest="sha256:" + sha256_file(archive),
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "MANIFEST_MISSING"


def test_the_bundle_must_be_the_model_configuration_asked_for(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path, model_id="some-other-model")
    assert caught.value.code == "MANIFEST_MISMATCH"

    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path, model_id="line-clearance-cam-01", version="2026.01.01")
    assert caught.value.code == "MANIFEST_MISMATCH"

    assert stage(
        built, cache, staging, schema_path, model_id="line-clearance-cam-01", version="2026.08.20"
    )


def test_a_bundle_no_provider_can_run_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path, available_providers=["TensorrtExecutionProvider"])
    assert caught.value.code == "PROVIDER_UNSUPPORTED"

    assert stage(built, cache, staging, schema_path, available_providers=["CUDAExecutionProvider"])


def test_a_bundle_with_no_provider_list_skips_the_provider_check(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    permissive = build_bundle(manifest=manifest_document(providersPermitted=["CPUExecutionProvider"]))
    assert stage(permissive, cache, staging, schema_path, available_providers=None)


def test_extra_validators_run_before_promotion(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle
) -> None:
    built = build_bundle()
    seen = []

    def refuse(manifest) -> None:
        seen.append(manifest.family)
        raise BundleError("FAMILY_UNSUPPORTED", "no task family can interpret this head")

    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path, validators=[refuse])
    assert caught.value.code == "FAMILY_UNSUPPORTED"
    assert seen and cache.list() == []


def test_the_signed_manifest_must_be_the_extracted_manifest(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle, monkeypatch
) -> None:
    built = build_bundle()
    real_read = __import__(
        "image_processor.bundles.fetch", fromlist=["read_member_bytes"]
    ).read_member_bytes

    def swapped(path, names, *args, **kwargs):
        found = dict(real_read(path, names, *args, **kwargs))
        found["manifest.json"] = found["manifest.json"].replace(b"line-clearance", b"other-model")
        return found

    monkeypatch.setattr("image_processor.bundles.fetch.read_member_bytes", swapped)
    with pytest.raises(BundleError) as caught:
        stage(built, cache, staging, schema_path)
    assert caught.value.code == "MANIFEST_INVALID"
    assert "differs from the one the signature covered" in caught.value.message


def test_a_bundle_stages_over_https(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle, tls_server
) -> None:
    built = build_bundle(compress=True)
    shutil.copyfile(built.archive, tls_server.root / "line-clearance.tar.gz")

    cached = stage_bundle(
        uri=tls_server.url("line-clearance.tar.gz"),
        digest=built.digest,
        staging_root=staging,
        cache=cache,
        schema_path=schema_path,
        signing_required=True,
        trusted_keys=built.trusted,
        allowed_prefixes=[f"https://localhost:{tls_server.port}/"],
        ssl_context=tls_server.client_context(),
    )
    assert cached.digest == built.digest
    assert list(staging.iterdir()) == []


def test_an_unreachable_source_leaves_nothing_behind(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path
) -> None:
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(tmp_path / "absent.tar.gz"),
            digest="sha256:" + "11" * 32,
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "FETCH_FAILED"
    assert list(staging.iterdir()) == []
    assert cache.list() == []


def test_a_manifest_that_is_not_json_stops_staging_before_extraction(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path
) -> None:
    archive = write_tar(tmp_path / "bad-json.tar", [file_member("manifest.json", b"{ not json")])
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest="sha256:" + sha256_file(archive),
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "MANIFEST_INVALID"


def test_a_manifest_that_is_not_an_object_is_refused(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path
) -> None:
    archive = write_tar(tmp_path / "list.tar", [file_member("manifest.json", b'["nope"]')])
    with pytest.raises(BundleError) as caught:
        stage_bundle(
            uri=str(archive),
            digest="sha256:" + sha256_file(archive),
            staging_root=staging,
            cache=cache,
            schema_path=schema_path,
        )
    assert caught.value.code == "MANIFEST_INVALID"


def test_a_trusted_signature_is_verified_even_when_signing_is_optional(
    tmp_path: Path, cache: BundleCache, staging: Path, schema_path: Path, build_bundle, caplog
) -> None:
    built = build_bundle()
    with caplog.at_level(logging.WARNING):
        cached = stage(built, cache, staging, schema_path, trusted_keys=built.trusted)
    assert cached.manifest.key_id == "pharma-model-publisher-1"
    assert "not configured as trusted" not in caplog.text
