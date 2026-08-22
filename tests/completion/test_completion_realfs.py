"""RealFs against a real directory, including the Windows directory-fsync no-op."""

import errno
import hashlib
import os
from pathlib import Path

import pytest

from image_processor.completion import RealFs


@pytest.fixture()
def fs() -> RealFs:
    return RealFs()


def test_makedirs_exists_and_size(fs, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    assert fs.exists(nested) is False
    fs.makedirs(nested)
    fs.makedirs(nested)
    assert fs.exists(nested) is True
    target = nested / "f.bin"
    fs.write_bytes(target, b"0123456789")
    assert fs.size(target) == 10
    assert fs.exists(target) is True


def test_write_bytes_installs_atomically_and_leaves_no_temp(fs, tmp_path):
    target = tmp_path / "evidence.json"
    fs.write_bytes(target, b"first")
    fs.write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(p.name for p in tmp_path.iterdir()) == ["evidence.json"]


def test_sha256_matches_hashlib(fs, tmp_path):
    payload = os.urandom(3 * (1 << 20) + 7)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert fs.sha256(target) == hashlib.sha256(payload).hexdigest()


def test_replace_is_atomic_and_overwrites(fs, tmp_path):
    source = tmp_path / "src.jpg"
    target = tmp_path / "dst.jpg"
    source.write_bytes(b"image")
    target.write_bytes(b"stale")
    fs.replace(source, target)
    assert target.read_bytes() == b"image"
    assert source.exists() is False


def test_copy_then_remove(fs, tmp_path):
    source = tmp_path / "src.jpg"
    target = tmp_path / "copy.jpg"
    source.write_bytes(b"image")
    fs.copy(source, target)
    assert target.read_bytes() == b"image"
    fs.remove(source)
    assert source.exists() is False


def test_replace_of_a_missing_source_raises(fs, tmp_path):
    with pytest.raises(OSError) as caught:
        fs.replace(tmp_path / "gone.jpg", tmp_path / "dst.jpg")
    assert caught.value.errno == errno.ENOENT


def test_fsync_dir_is_best_effort(fs, tmp_path):
    fs.fsync_dir(tmp_path)


def test_fsync_dir_uses_a_directory_handle_off_windows(fs, tmp_path, monkeypatch):
    opened = {}
    real_open, real_close = os.open, os.close

    def fake_open(path, flags):
        opened["path"] = path
        return real_open(str(tmp_path / "probe"), os.O_CREAT | os.O_WRONLY)

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", lambda fd: opened.setdefault("synced", fd))
    monkeypatch.setattr(os, "close", real_close)
    fs.fsync_dir(tmp_path)
    assert opened["path"] == str(tmp_path)
    assert "synced" in opened


def test_fsync_dir_is_a_no_op_on_windows(fs, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    def explode(*args, **kwargs):
        raise AssertionError("no directory handle on Windows")

    monkeypatch.setattr(os, "open", explode)
    fs.fsync_dir(tmp_path)


def test_a_real_archive_move_round_trips(fs, tmp_path):
    spool = tmp_path / "spool" / "2026" / "08" / "22"
    archive = tmp_path / "processed" / "2026" / "08" / "22"
    fs.makedirs(spool)
    fs.makedirs(archive)
    source = spool / "capture-1.jpg"
    source.write_bytes(b"image-bytes")
    digest = fs.sha256(source)
    fs.replace(source, archive / "capture-1.jpg")
    assert fs.sha256(archive / "capture-1.jpg") == digest
    assert Path(archive / "capture-1.jpg").read_bytes() == b"image-bytes"
