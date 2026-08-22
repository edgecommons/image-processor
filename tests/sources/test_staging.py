"""Staging, path containment, and the primitives every source trusts before it reads a file."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_processor.sources import staging
from image_processor.sources.staging import (
    ConfigFieldError,
    PathError,
    SourceError,
    StagingError,
    classify_path,
    config_field,
    normalize_relative,
    plain_digest,
    relative_to_root,
    resolve_under_root,
    sha256_bytes,
    sha256_file,
    stage_bytes,
    stage_copy,
    staged_path_for,
    stat_signature,
)

DATA = b"the exact bytes of one small image"
DIGEST = sha256_bytes(DATA)


def test_stage_copy_names_the_target_by_digest_and_verifies_it(tmp_path):
    source = tmp_path / "in" / "frame.jpg"
    source.parent.mkdir()
    source.write_bytes(DATA)

    staged = stage_copy(source, tmp_path / "staging", DIGEST)

    assert staged == staged_path_for(tmp_path / "staging", DIGEST, ".jpg")
    assert staged.read_bytes() == DATA
    assert sha256_file(staged) == DIGEST


def test_stage_copy_is_idempotent_and_does_not_rewrite(tmp_path):
    source = tmp_path / "frame.jpg"
    source.write_bytes(DATA)
    first = stage_copy(source, tmp_path / "staging", DIGEST)
    signature = stat_signature(first)

    second = stage_copy(source, tmp_path / "staging", DIGEST)

    assert second == first
    assert stat_signature(second) == signature


def test_stage_copy_repairs_a_target_holding_the_wrong_bytes(tmp_path):
    source = tmp_path / "frame.jpg"
    source.write_bytes(DATA)
    target = staged_path_for(tmp_path / "staging", DIGEST, ".jpg")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")

    staged = stage_copy(source, tmp_path / "staging", DIGEST)

    assert staged.read_bytes() == DATA


def test_stage_copy_refuses_bytes_that_do_not_match_the_digest(tmp_path):
    source = tmp_path / "frame.jpg"
    source.write_bytes(b"different bytes entirely")

    with pytest.raises(StagingError) as caught:
        stage_copy(source, tmp_path / "staging", DIGEST)

    assert caught.value.code == "DIGEST_MISMATCH"
    assert not staged_path_for(tmp_path / "staging", DIGEST, ".jpg").exists()


def test_stage_bytes_writes_and_reuses_the_digest_named_file(tmp_path):
    first = stage_bytes(DATA, tmp_path / "staging", DIGEST, ".img")
    second = stage_bytes(DATA, tmp_path / "staging", DIGEST, ".img")

    assert first == second
    assert first.read_bytes() == DATA


def test_stage_bytes_refuses_a_digest_that_is_not_the_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "sha256_file", lambda path, chunk=1: "0" * 64)

    with pytest.raises(StagingError):
        stage_bytes(DATA, tmp_path / "staging", DIGEST, ".img")


def test_plain_digest_accepts_both_spellings_and_refuses_anything_else():
    assert plain_digest(DIGEST) == DIGEST
    assert plain_digest("sha256:" + DIGEST.upper()) == DIGEST
    for bad in ("", "sha256:zz", DIGEST[:-1], 17):
        with pytest.raises(SourceError):
            plain_digest(bad)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026/08/22/frame.jpg", "2026/08/22/frame.jpg"),
        ("./2026/frame.jpg", "2026/frame.jpg"),
        ("2026" + chr(92) + "08" + chr(92) + "frame.jpg", "2026/08/frame.jpg"),
        ("a//b/frame.jpg", "a/b/frame.jpg"),
    ],
)
def test_normalize_relative_normalizes_a_usable_path(value, expected):
    assert normalize_relative(value) == expected


@pytest.mark.parametrize(
    "value, code",
    [
        ("", "MALFORMED_RELATIVE_PATH"),
        (None, "MALFORMED_RELATIVE_PATH"),
        ("/etc/passwd", "ABSOLUTE_PATH"),
        ("C:/Windows/system.ini", "ABSOLUTE_PATH"),
        ("../../etc/passwd", "PATH_ESCAPE"),
        ("a/../../b.jpg", "PATH_ESCAPE"),
        (".", "MALFORMED_RELATIVE_PATH"),
    ],
)
def test_normalize_relative_refuses_anything_that_could_leave_a_root(value, code):
    with pytest.raises(PathError) as caught:
        normalize_relative(value)
    assert caught.value.code == code


def test_resolve_under_root_returns_a_contained_path(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "frame.jpg").write_bytes(DATA)

    resolved = resolve_under_root(tmp_path, "a/frame.jpg")

    assert resolved.read_bytes() == DATA
    assert relative_to_root(staging.real_root(tmp_path), resolved) == "a/frame.jpg"


def test_resolve_under_root_refuses_an_escape(tmp_path):
    with pytest.raises(PathError) as caught:
        resolve_under_root(tmp_path / "root", "../outside/frame.jpg")
    assert caught.value.code == "PATH_ESCAPE"


def test_classify_path_accepts_a_regular_file_and_names_what_is_missing(tmp_path):
    good = tmp_path / "frame.jpg"
    good.write_bytes(DATA)

    assert classify_path(good) is None
    assert classify_path(tmp_path / "absent.jpg") == "MISSING"
    assert classify_path(tmp_path) == "DIRECTORY"


def test_classify_path_refuses_a_windows_reparse_point(tmp_path, monkeypatch):
    """A junction is refused through the stat seam.

    Creating one needs a privilege the CI account does not hold, so the reparse attribute is
    injected instead. The rejection has to hold on the attribute, not on the ability to make one.
    """
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    real_lstat = os.lstat

    def reparse_lstat(path):
        result = real_lstat(path)
        if Path(path) == target:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                st_reparse_tag=0xA0000003,
            )
        return result

    monkeypatch.setattr(staging, "lstat", reparse_lstat)
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False)

    assert classify_path(target) == "REPARSE_POINT"


def test_classify_path_refuses_a_device_file(tmp_path, monkeypatch):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    monkeypatch.setattr(
        staging,
        "lstat",
        lambda path: SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_size=0, st_mtime_ns=0),
    )

    assert classify_path(target) == "DEVICE_FILE"


def test_classify_path_refuses_a_fifo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging,
        "lstat",
        lambda path: SimpleNamespace(st_mode=stat.S_IFIFO | 0o666, st_size=0, st_mtime_ns=0),
    )

    assert classify_path(tmp_path / "pipe") == "NOT_REGULAR_FILE"


def test_classify_path_refuses_a_symlink(tmp_path):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    link = tmp_path / "link.jpg"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account cannot create symlinks: {exc}")

    assert classify_path(link) == "SYMLINK"


def test_stat_signature_reports_absence_and_change(tmp_path):
    target = tmp_path / "frame.jpg"
    assert stat_signature(target) is None
    target.write_bytes(DATA)
    first = stat_signature(target)
    target.write_bytes(DATA + b"more")
    assert stat_signature(target) != first


def test_config_field_reads_camel_case_snake_case_objects_and_mappings():
    camel = SimpleNamespace(quietSecs=7)
    snake = SimpleNamespace(quiet_secs=9)
    mapping = {"quietSecs": 11}

    assert config_field(camel, "quietSecs") == 7
    assert config_field(snake, "quietSecs") == 9
    assert config_field(mapping, "quietSecs") == 11
    assert config_field({"quiet_secs": 13}, "quietSecs") == 13


def test_config_field_treats_a_null_field_as_absent_but_keeps_false():
    assert config_field(SimpleNamespace(markerSuffix=None), "markerSuffix", default=".done") == (
        ".done"
    )
    assert config_field({"subscribeAnnouncements": False}, "subscribeAnnouncements") is False


def test_config_field_requires_a_field_with_no_default():
    with pytest.raises(ConfigFieldError):
        config_field(SimpleNamespace(), "root")


def test_sha256_file_streams_a_file_larger_than_one_chunk(tmp_path):
    target = tmp_path / "big.bin"
    payload = os.urandom(3 * 1024)
    target.write_bytes(payload)

    assert sha256_file(target, chunk=512) == sha256_bytes(payload)


def test_sync_directory_tolerates_a_path_it_cannot_open(tmp_path):
    staging._sync_directory(tmp_path / "absent")


def test_classify_path_refuses_a_symlink_through_the_stat_seam(tmp_path, monkeypatch):
    """The symlink rejection is proven without the privilege to create one."""
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    monkeypatch.setattr(
        staging,
        "lstat",
        lambda path: SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_size=0, st_mtime_ns=0),
    )

    assert classify_path(target) == "SYMLINK"


def test_resolve_under_root_refuses_a_path_that_resolves_outside(tmp_path, monkeypatch):
    """A link inside the root that points out of it is caught after resolution, not before."""
    root = tmp_path / "spool"
    root.mkdir()
    escape = tmp_path / "elsewhere" / "secret.jpg"
    real = staging.realpath

    def resolve(path):
        return str(escape) if Path(path).name == "link.jpg" else real(path)

    monkeypatch.setattr(staging, "realpath", resolve)

    with pytest.raises(PathError) as caught:
        resolve_under_root(root, "link.jpg")
    assert caught.value.code == "PATH_ESCAPE"


def test_sync_directory_flushes_where_the_platform_allows_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(staging.os, "open", lambda path, flags: 99)
    monkeypatch.setattr(staging.os, "fsync", lambda handle: calls.append(("fsync", handle)))
    monkeypatch.setattr(staging.os, "close", lambda handle: calls.append(("close", handle)))

    staging._sync_directory(tmp_path)

    assert calls == [("fsync", 99), ("close", 99)]


def test_sync_directory_closes_the_handle_even_when_the_flush_fails(tmp_path, monkeypatch):
    calls = []

    def refuse(handle):
        raise OSError("directory flush is not supported here")

    monkeypatch.setattr(staging.os, "open", lambda path, flags: 99)
    monkeypatch.setattr(staging.os, "fsync", refuse)
    monkeypatch.setattr(staging.os, "close", lambda handle: calls.append(handle))

    staging._sync_directory(tmp_path)

    assert calls == [99]
