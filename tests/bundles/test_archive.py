"""Digest and extraction safety tests (DESIGN.md section 15, LLD section 4)."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from image_processor.bundles import (
    BundleError,
    ExtractLimits,
    digest_hex,
    extract_tarball,
    normalize_digest,
    read_member_bytes,
    sha256_file,
    verify_tarball_digest,
)
from image_processor.bundles.archive import (
    _member_kind,
    _resolved_target,
    _safe_relative_name,
)

from .conftest import file_member, link_member, write_tar


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"model bytes" * 5000
    target = tmp_path / "model.onnx"
    target.write_bytes(payload)
    assert sha256_file(target, chunk=1024) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_reports_unreadable(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        sha256_file(tmp_path / "absent.onnx")
    assert caught.value.code == "FILE_UNREADABLE"


@pytest.mark.parametrize(
    "value",
    ["sha256:" + "ab" * 32, "AB" * 32, "  sha256:" + "AB" * 32 + "  "],
)
def test_normalize_digest_accepts_both_forms(value: str) -> None:
    assert normalize_digest(value) == "sha256:" + "ab" * 32
    assert digest_hex(value) == "ab" * 32


@pytest.mark.parametrize("value", ["md5:" + "ab" * 16, "ab" * 16, "", 17])
def test_normalize_digest_rejects_anything_else(value: object) -> None:
    with pytest.raises(BundleError) as caught:
        normalize_digest(value)  # type: ignore[arg-type]
    assert caught.value.code == "DIGEST_FORMAT"


def test_verify_tarball_digest(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar"
    archive.write_bytes(b"not really a tarball")
    verify_tarball_digest(archive, "sha256:" + sha256_file(archive))
    with pytest.raises(BundleError) as caught:
        verify_tarball_digest(archive, "sha256:" + "00" * 32)
    assert caught.value.code == "DIGEST_MISMATCH"


@pytest.mark.parametrize("compress", [False, True])
def test_extract_round_trips_a_real_bundle(tmp_path: Path, build_bundle, compress: bool) -> None:
    built = build_bundle(compress=compress)
    dest = tmp_path / "out"
    written = extract_tarball(built.archive, dest)
    names = {path.relative_to(dest).as_posix() for path in written}
    assert "manifest.json" in names
    assert "manifest.sig" in names
    assert "warmup/input-01.bin" in names
    assert (dest / "model.onnx").read_bytes() == (built.source / "model.onnx").read_bytes()


def test_extract_creates_missing_parent_directories(tmp_path: Path) -> None:
    directory = tarfile.TarInfo("engines/")
    directory.type = tarfile.DIRTYPE
    archive = write_tar(
        tmp_path / "nested.tar",
        [(directory, None), file_member("engines/sm_86/engine.plan", b"plan")],
    )
    written = extract_tarball(archive, tmp_path / "out")
    assert written == [tmp_path / "out" / "engines" / "sm_86" / "engine.plan"]
    assert (tmp_path / "out" / "engines").is_dir()


def test_extract_rejects_a_member_that_collides_with_an_earlier_one(tmp_path: Path) -> None:
    archive = write_tar(
        tmp_path / "collide.tar",
        [file_member("engines", b"a file, not a directory"), file_member("engines/x.plan", b"p")],
    )
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_extract_normalizes_dot_slash_names(tmp_path: Path) -> None:
    archive = write_tar(tmp_path / "dot.tar", [file_member("./manifest.json", b"{}")])
    written = extract_tarball(archive, tmp_path / "out")
    assert written == [tmp_path / "out" / "manifest.json"]


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "nested/../../escape.txt", "/etc/shadow", "C:/Windows/System32/x.dll", "back" + chr(92) + "slash.bin"],
)
def test_extract_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    archive = write_tar(tmp_path / "unsafe.tar", [file_member(name, b"payload")])
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    "member",
    [
        link_member("evil.onnx", "/etc/passwd", tarfile.SYMTYPE),
        link_member("evil.onnx", "model.onnx", tarfile.LNKTYPE),
        (tarfile.TarInfo("dev/null"), None),
    ],
)
def test_extract_rejects_non_regular_members(tmp_path: Path, member) -> None:
    info, payload = member
    if info.type == tarfile.REGTYPE:
        info.type = tarfile.CHRTYPE
    archive = write_tar(tmp_path / "special.tar", [(info, payload)])
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_extract_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = write_tar(
        tmp_path / "dup.tar",
        [file_member("model.onnx", b"first"), file_member("model.onnx", b"second")],
    )
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_extract_enforces_the_member_count_limit(tmp_path: Path) -> None:
    members = [file_member(f"chunk-{index}.bin", b"x") for index in range(5)]
    archive = write_tar(tmp_path / "many.tar", members)
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out", ExtractLimits(max_members=3))
    assert caught.value.code == "ARCHIVE_LIMIT"


def test_extract_enforces_the_per_member_limit(tmp_path: Path) -> None:
    archive = write_tar(tmp_path / "big.tar", [file_member("model.onnx", b"x" * 4096)])
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out", ExtractLimits(max_member_bytes=1024))
    assert caught.value.code == "ARCHIVE_LIMIT"


def test_extract_enforces_the_total_limit(tmp_path: Path) -> None:
    members = [file_member(f"part-{index}.bin", b"y" * 4096) for index in range(4)]
    archive = write_tar(tmp_path / "total.tar", members)
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out", ExtractLimits(max_total_bytes=5000))
    assert caught.value.code == "ARCHIVE_LIMIT"


def test_extract_stops_a_decompression_bomb(tmp_path: Path) -> None:
    archive = write_tar(
        tmp_path / "bomb.tar.gz", [file_member("zeros.bin", b"\0" * (8 << 20))], compress=True
    )
    assert archive.stat().st_size < 1 << 20
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_LIMIT"
    assert "expands by more than" in caught.value.message


def test_extract_rejects_a_file_that_is_not_a_tarball(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar"
    archive.write_bytes(b"this is not a tarball at all")
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNREADABLE"


def test_extract_reports_a_truncated_archive(tmp_path: Path, build_bundle) -> None:
    built = build_bundle(compress=True)
    payload = built.archive.read_bytes()
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(BundleError) as caught:
        extract_tarball(truncated, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNREADABLE"


def test_extract_reports_an_unreadable_path(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        extract_tarball(tmp_path / "missing.tar", tmp_path / "out")
    assert caught.value.code in ("ARCHIVE_UNREADABLE", "FILE_UNREADABLE")


def test_read_member_bytes_returns_only_what_it_finds(tmp_path: Path, build_bundle) -> None:
    built = build_bundle()
    found = read_member_bytes(built.archive, ("manifest.json", "manifest.sig", "absent.json"))
    assert set(found) == {"manifest.json", "manifest.sig"}
    assert found["manifest.json"].startswith(b"{")
    assert len(found["manifest.sig"]) == 64


def test_read_member_bytes_enforces_its_own_cap(tmp_path: Path) -> None:
    archive = write_tar(tmp_path / "wide.tar", [file_member("manifest.json", b"z" * 4096)])
    with pytest.raises(BundleError) as caught:
        read_member_bytes(archive, ("manifest.json",), max_bytes=1024)
    assert caught.value.code == "ARCHIVE_LIMIT"


def test_read_member_bytes_rejects_unsafe_members(tmp_path: Path) -> None:
    archive = write_tar(
        tmp_path / "linked.tar",
        [link_member("manifest.json", "/etc/passwd", tarfile.SYMTYPE)],
    )
    with pytest.raises(BundleError) as caught:
        read_member_bytes(archive, ("manifest.json",))
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_read_member_bytes_skips_directories_and_other_files(tmp_path: Path) -> None:
    archive = write_tar(
        tmp_path / "mixed.tar",
        [
            (tarfile.TarInfo("warmup/"), None),
            file_member("labels.json", b"[]"),
            file_member("manifest.json", b"{}"),
        ],
    )
    assert dict(read_member_bytes(archive, ("manifest.json",))) == {"manifest.json": b"{}"}


def test_read_member_bytes_on_a_corrupt_archive(tmp_path: Path, build_bundle) -> None:
    built = build_bundle(compress=True)
    payload = built.archive.read_bytes()
    truncated = tmp_path / "half.tar.gz"
    truncated.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(BundleError) as caught:
        read_member_bytes(truncated, ("labels.json",))
    assert caught.value.code == "ARCHIVE_UNREADABLE"


def test_safe_relative_name_rejects_nul_and_empty_names() -> None:
    for name in ["", "./", "/", "x" + chr(0) + "y"]:
        with pytest.raises(BundleError) as caught:
            _safe_relative_name(name)
        assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_resolved_target_refuses_to_leave_the_destination(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        _resolved_target(tmp_path.resolve(), PurePosixPath("..") / "outside.bin")
    assert caught.value.code == "ARCHIVE_UNSAFE_MEMBER"


def test_gzip_magic_decides_the_mode_not_the_name(tmp_path: Path) -> None:
    plain = write_tar(tmp_path / "misnamed.tar.gz", [file_member("labels.json", b"[]")])
    assert extract_tarball(plain, tmp_path / "out-plain")
    compressed = tmp_path / "misnamed.tar"
    compressed.write_bytes(gzip.compress((tmp_path / "misnamed.tar.gz").read_bytes()))
    assert extract_tarball(compressed, tmp_path / "out-gz")


@pytest.mark.parametrize(
    "kind, word",
    [
        (tarfile.CHRTYPE, "character device"),
        (tarfile.BLKTYPE, "block device"),
        (tarfile.FIFOTYPE, "fifo"),
    ],
)
def test_every_special_member_type_is_named_in_the_error(tmp_path: Path, kind, word: str) -> None:
    info = tarfile.TarInfo("device")
    info.type = kind
    archive = write_tar(tmp_path / f"{word.replace(' ', '-')}.tar", [(info, None)])
    with pytest.raises(BundleError) as caught:
        extract_tarball(archive, tmp_path / "out")
    assert word in caught.value.message


def test_a_directory_is_not_an_archive(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        extract_tarball(tmp_path, tmp_path / "out")
    assert caught.value.code == "ARCHIVE_UNREADABLE"


def test_a_filesystem_failure_while_extracting_is_reported(tmp_path: Path, monkeypatch) -> None:
    from image_processor.bundles.archive import _make_directory, _write_member

    def deny_mkdir(self, *args, **kwargs):
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    with pytest.raises(BundleError) as caught:
        _make_directory(tmp_path / "engines", "engines/")
    assert caught.value.code == "EXTRACT_FAILED"

    def deny_open(*args, **kwargs):
        raise OSError(28, "no space left on device")

    monkeypatch.setattr("builtins.open", deny_open)
    with pytest.raises(BundleError) as caught:
        _write_member(None, tmp_path / "model.onnx", "model.onnx", None)
    assert caught.value.code == "EXTRACT_FAILED"


def test_an_unrecognized_member_type_is_named_generically() -> None:
    info = tarfile.TarInfo("odd")
    info.type = b"X"
    assert _member_kind(info) == "special file"
