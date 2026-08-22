"""Content-addressed cache behavior: promotion, idempotency, and garbage collection."""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path

import pytest

from image_processor.bundles import BundleCache, BundleError, extract_tarball
from image_processor.bundles.cache import TEMP_MAX_AGE_SECS, TEMP_PREFIX

from .conftest import manifest_document


@pytest.fixture
def cache(tmp_path: Path, schema_path: Path) -> BundleCache:
    """Return an empty cache rooted in the test's temporary directory."""
    return BundleCache(tmp_path / "models", schema_path)


def stage(built, tmp_path: Path, name: str = "staged") -> Path:
    """Extract a built bundle into a staging directory ready for promotion."""
    dest = tmp_path / "staging" / name
    extract_tarball(built.archive, dest)
    return dest


def test_promote_puts_the_bundle_under_its_digest(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    built = build_bundle()
    staged = stage(built, tmp_path)
    cached = cache.promote(staged, built.digest)

    assert cached.digest == built.digest
    assert cached.root == cache.root / built.digest.split(":")[1]
    assert cached.model_path == cached.root / "model.onnx"
    assert cached.manifest.model_id == "line-clearance-cam-01"
    assert not staged.exists()
    assert cache.metadata_path(built.digest).is_file()
    metadata = cache.metadata(built.digest)
    assert metadata["modelId"] == "line-clearance-cam-01"
    assert metadata["modelPath"] == "model.onnx"
    assert metadata["declaredFiles"] == 7


def test_promote_is_idempotent(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    built = build_bundle()
    first = cache.promote(stage(built, tmp_path, "one"), built.digest)
    second_staging = stage(built, tmp_path, "two")
    second = cache.promote(second_staging, built.digest)

    assert second.root == first.root
    assert second.manifest == first.manifest
    assert not second_staging.exists()
    assert len(cache.list()) == 1


def test_promote_completes_an_interrupted_promotion(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    built = build_bundle()
    cache.promote(stage(built, tmp_path, "one"), built.digest)
    cache.metadata_path(built.digest).unlink()
    assert cache.get(built.digest) is None

    cache.promote(stage(built, tmp_path, "two"), built.digest)
    assert cache.get(built.digest) is not None


def test_promote_replaces_a_cached_bundle_that_no_longer_loads(
    tmp_path: Path, cache: BundleCache, build_bundle
) -> None:
    built = build_bundle()
    cached = cache.promote(stage(built, tmp_path, "one"), built.digest)
    (cached.root / "manifest.json").write_bytes(b"{ corrupt")

    replaced = cache.promote(stage(built, tmp_path, "two"), built.digest)
    assert replaced.manifest.model_id == "line-clearance-cam-01"
    assert cache.get(built.digest, verify=True) is not None


def test_get_reports_a_cached_bundle_that_no_longer_verifies(
    tmp_path: Path, cache: BundleCache, build_bundle
) -> None:
    built = build_bundle()
    cached = cache.promote(stage(built, tmp_path), built.digest)
    (cached.root / "labels.json").write_text("[]", encoding="utf-8")

    assert cache.get(built.digest) is not None
    with pytest.raises(BundleError) as caught:
        cache.get(built.digest, verify=True)
    assert caught.value.code == "FILE_DIGEST_MISMATCH"


def test_get_and_metadata_report_nothing_for_an_uncached_digest(cache: BundleCache) -> None:
    absent = "sha256:" + "ab" * 32
    assert cache.get(absent) is None
    assert cache.metadata(absent) is None
    assert cache.list() == []


def test_list_skips_an_unreadable_entry(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    good = build_bundle(name="good")
    bad = build_bundle(name="bad", manifest=manifest_document(modelId="anomaly-cam-02"))
    cache.promote(stage(good, tmp_path, "good"), good.digest)
    broken = cache.promote(stage(bad, tmp_path, "bad"), bad.digest)
    (broken.root / "manifest.json").write_bytes(b"{ corrupt")

    listed = cache.list()
    assert [item.digest for item in listed] == [good.digest]


def test_metadata_survives_a_corrupt_metadata_file(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    built = build_bundle()
    cache.promote(stage(built, tmp_path), built.digest)
    cache.metadata_path(built.digest).write_text("{ corrupt", encoding="utf-8")
    assert cache.metadata(built.digest) is None


def test_gc_removes_only_what_is_not_pinned(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    keep = build_bundle(name="keep")
    drop = build_bundle(name="drop", manifest=manifest_document(modelId="detector-cam-03"))
    cache.promote(stage(keep, tmp_path, "keep"), keep.digest)
    cache.promote(stage(drop, tmp_path, "drop"), drop.digest)

    removed = cache.gc({keep.digest})
    assert removed == [drop.digest]
    assert cache.get(keep.digest) is not None
    assert cache.get(drop.digest) is None
    assert not cache.metadata_path(drop.digest).exists()


def test_gc_accepts_bare_hex_pins(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    built = build_bundle()
    cache.promote(stage(built, tmp_path), built.digest)
    assert cache.gc([built.digest.split(":")[1]]) == []
    assert cache.get(built.digest) is not None


def test_gc_removes_orphan_metadata(cache: BundleCache) -> None:
    orphan = "cd" * 32
    (cache.root / f"{orphan}.json").write_text(json.dumps({"digest": orphan}), encoding="utf-8")
    assert cache.gc(set()) == [f"sha256:{orphan}"]
    assert not (cache.root / f"{orphan}.json").exists()


def test_gc_ignores_unrelated_entries(cache: BundleCache) -> None:
    (cache.root / "README.txt").write_text("not a bundle", encoding="utf-8")
    (cache.root / "not-a-digest").mkdir()
    assert cache.gc(set()) == []
    assert (cache.root / "README.txt").exists()
    assert (cache.root / "not-a-digest").is_dir()


def test_gc_leaves_a_running_promotion_alone_and_clears_a_stale_one(cache: BundleCache) -> None:
    fresh = cache.root / (TEMP_PREFIX + "fresh")
    stale = cache.root / (TEMP_PREFIX + "stale")
    fresh.mkdir()
    stale.mkdir()
    old = time.time() - TEMP_MAX_AGE_SECS - 60
    os.utime(stale, (old, old))

    assert cache.gc(set()) == []
    assert fresh.is_dir()
    assert not stale.exists()


def test_promote_copies_when_staging_is_on_another_filesystem(
    tmp_path: Path, cache: BundleCache, build_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_bundle()
    staged = stage(built, tmp_path)
    real_replace = os.replace
    calls = {"n": 0}

    def fake_replace(src, dst, *args, **kwargs):
        if Path(src) == staged:
            calls["n"] += 1
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("image_processor.bundles.cache.os.replace", fake_replace)
    cached = cache.promote(staged, built.digest)

    assert calls["n"] == 1
    assert cached.model_path.read_bytes() == (built.source / "model.onnx").read_bytes()
    assert not staged.exists()
    assert not any(entry.name.startswith(TEMP_PREFIX) for entry in cache.root.iterdir())


def test_promote_reports_a_move_it_cannot_make(
    tmp_path: Path, cache: BundleCache, build_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_bundle()
    staged = stage(built, tmp_path)

    def deny(src, dst, *args, **kwargs):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr("image_processor.bundles.cache.os.replace", deny)
    with pytest.raises(BundleError) as caught:
        cache.promote(staged, built.digest)
    assert caught.value.code == "PROMOTE_FAILED"


def test_promote_reports_a_copy_it_cannot_make(
    tmp_path: Path, cache: BundleCache, build_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = build_bundle()
    staged = stage(built, tmp_path)

    def cross_device(src, dst, *args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def deny_copy(src, dst, *args, **kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr("image_processor.bundles.cache.os.replace", cross_device)
    monkeypatch.setattr("image_processor.bundles.cache.shutil.copytree", deny_copy)
    with pytest.raises(BundleError) as caught:
        cache.promote(staged, built.digest)
    assert caught.value.code == "PROMOTE_FAILED"


def test_promote_refuses_a_directory_that_is_not_a_bundle(tmp_path: Path, cache: BundleCache) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BundleError) as caught:
        cache.promote(empty, "sha256:" + "ef" * 32)
    assert caught.value.code == "MANIFEST_MISSING"


@pytest.mark.skipif(os.name != "nt", reason="read-only files only block deletion on Windows")
def test_gc_removes_a_bundle_with_read_only_files(tmp_path: Path, cache: BundleCache, build_bundle) -> None:
    import stat as stat_module

    built = build_bundle()
    cached = cache.promote(stage(built, tmp_path), built.digest)
    (cached.root / "model.onnx").chmod(stat_module.S_IREAD)

    assert cache.gc(set()) == [built.digest]
    assert not cached.root.exists()
