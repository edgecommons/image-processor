"""The pinned-asset fetch tool: verification, idempotency, selection, and archive safety."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import fetch_test_assets as fetch
from tools.fetch_test_assets import AssetError

PAYLOAD = b"a model would be here" * 64


def test_fetch_downloads_and_verifies(
    tmp_path, served, publish, asset_entry, write_manifest
):
    part = publish("model.onnx", PAYLOAD)
    path = write_manifest([asset_entry("model-a", part)])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 0
    landed = tmp_path / "cache" / "model-a" / "model.onnx"
    assert landed.read_bytes() == PAYLOAD
    assert not list((tmp_path / "cache" / "model-a").glob("*.part"))


def test_a_second_run_downloads_nothing(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("model.onnx", PAYLOAD)
    path = write_manifest([asset_entry("model-a", part)])
    argv = ["--manifest", str(path), "--cache", str(tmp_path / "cache")]

    assert fetch.main(argv) == 0
    assert "1 downloaded, 0 cached" in capsys.readouterr().out
    assert fetch.main(argv) == 0
    assert "0 downloaded, 1 cached" in capsys.readouterr().out


def test_a_wrong_digest_fails_and_removes_the_file(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("model.onnx", PAYLOAD)
    part["sha256"] = "0" * 64
    path = write_manifest([asset_entry("model-a", part)])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "the manifest pins sha256:" in capsys.readouterr().err
    assert not (tmp_path / "cache" / "model-a" / "model.onnx").exists()


def test_a_wrong_byte_count_fails(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("model.onnx", PAYLOAD)
    part["bytes"] = len(PAYLOAD) + 1
    path = write_manifest([asset_entry("model-a", part)])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "bytes)" in capsys.readouterr().err


def test_a_corrupted_cached_file_is_replaced(
    tmp_path, served, publish, asset_entry, write_manifest
):
    part = publish("model.onnx", PAYLOAD)
    path = write_manifest([asset_entry("model-a", part)])
    argv = ["--manifest", str(path), "--cache", str(tmp_path / "cache")]
    assert fetch.main(argv) == 0

    landed = tmp_path / "cache" / "model-a" / "model.onnx"
    landed.write_bytes(b"x" * len(PAYLOAD))
    assert fetch.main(argv) == 0
    assert landed.read_bytes() == PAYLOAD


def test_a_missing_url_fails(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    _, base = served
    part = publish("model.onnx", PAYLOAD)
    entry = asset_entry("model-a", part)
    entry["uri"] = base + "absent.onnx"
    path = write_manifest([entry])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "download failed" in capsys.readouterr().err


def test_only_selects_by_id(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    first = asset_entry("model-a", publish("a.onnx", b"aaaa"))
    second = asset_entry("model-b", publish("b.onnx", b"bbbb"))
    path = write_manifest([first, second])

    assert fetch.main(
        ["--manifest", str(path), "--cache", str(tmp_path / "cache"), "--only", "model-b"]
    ) == 0
    output = capsys.readouterr().out
    assert "model-b" in output and "model-a" not in output
    assert not (tmp_path / "cache" / "model-a").exists()


def test_only_rejects_an_unknown_id(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    path = write_manifest([asset_entry("model-a", publish("a.onnx", b"aaaa"))])

    assert fetch.main(
        ["--manifest", str(path), "--cache", str(tmp_path / "cache"), "--only", "model-z"]
    ) == 1
    assert "unknown asset id(s): model-z" in capsys.readouterr().err


def test_optional_assets_are_skipped_unless_asked_for(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    plain = asset_entry("model-a", publish("a.onnx", b"aaaa"))
    heavy = asset_entry("model-b", publish("b.onnx", b"bbbb"), optional=True)
    path = write_manifest([plain, heavy])
    argv = ["--manifest", str(path), "--cache", str(tmp_path / "cache")]

    assert fetch.main(argv) == 0
    assert "model-b" not in capsys.readouterr().out
    assert fetch.main(argv + ["--include-optional"]) == 0
    assert "model-b" in capsys.readouterr().out


def test_only_selects_an_optional_asset(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    heavy = asset_entry("model-b", publish("b.onnx", b"bbbb"), optional=True)
    path = write_manifest([heavy])

    assert fetch.main(
        ["--manifest", str(path), "--cache", str(tmp_path / "cache"), "--only", "model-b"]
    ) == 0
    assert "model-b" in capsys.readouterr().out


def test_list_prints_the_corpus_and_its_state(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("a.onnx", b"a" * 4096)
    path = write_manifest([asset_entry("model-a", part, optional=True)])
    argv = ["--manifest", str(path), "--cache", str(tmp_path / "cache"), "--include-optional"]

    assert fetch.main(argv + ["--list"]) == 0
    listing = capsys.readouterr().out
    assert "model-a" in listing and "missing" in listing and "(optional)" in listing

    assert fetch.main(argv) == 0
    capsys.readouterr()
    assert fetch.main(argv + ["--list"]) == 0
    assert "cached" in capsys.readouterr().out


def test_a_multi_file_asset_reports_a_partial_cache(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    _, base = served
    files = [publish(f"{index}.jpg", bytes([index]) * 32) for index in range(3)]
    entry = {
        "id": "dataset-slice",
        "kind": "dataset",
        "license": "CC BY 4.0",
        "source": "the test server",
        "files": [dict(item, uri=base + item["name"]) for item in files],
    }
    path = write_manifest([entry])
    cache = tmp_path / "cache"

    assert fetch.main(["--manifest", str(path), "--cache", str(cache)]) == 0
    assert "0 downloaded, 3 cached" not in capsys.readouterr().out
    (cache / "dataset-slice" / "1.jpg").unlink()

    assert fetch.main(["--manifest", str(path), "--cache", str(cache), "--list"]) == 0
    assert "partial (2/3)" in capsys.readouterr().out


def _tar_with(tmp_path: Path, name: str, member_name: str, payload: bytes = b"payload") -> Path:
    """Build a tar archive holding one file member under a chosen name."""
    archive = tmp_path / name
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        handle.addfile(info, __import__("io").BytesIO(payload))
    return archive


def test_a_tar_asset_is_unpacked_once(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    source = _tar_with(tmp_path, "corpus.tar", "corpus/one.txt")
    part = publish("corpus.tar", source.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar")])
    argv = ["--manifest", str(path), "--cache", str(tmp_path / "cache")]

    assert fetch.main(argv) == 0
    assert "unpacked" in capsys.readouterr().out
    assert (tmp_path / "cache" / "dataset-a" / "extracted" / "corpus" / "one.txt").is_file()

    assert fetch.main(argv) == 0
    assert "unpacked" not in capsys.readouterr().out


def test_a_gzip_tar_asset_is_unpacked(
    tmp_path, served, publish, asset_entry, write_manifest
):
    archive = tmp_path / "corpus.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("nested/two.txt")
        info.size = 4
        handle.addfile(info, __import__("io").BytesIO(b"data"))
    part = publish("corpus.tgz", archive.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar.gz")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 0
    assert (tmp_path / "cache" / "dataset-a" / "extracted" / "nested" / "two.txt").read_bytes() == b"data"


def test_a_zip_asset_is_unpacked(
    tmp_path, served, publish, asset_entry, write_manifest
):
    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("inner/three.txt", "zip")
    part = publish("corpus.zip", archive.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="zip")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 0
    assert (tmp_path / "cache" / "dataset-a" / "extracted" / "inner" / "three.txt").read_text() == "zip"


@pytest.mark.parametrize(
    "member,reason",
    [
        ("/etc/passwd", "absolute path"),
        ("C:/windows/system32/x", "absolute path"),
        ("../escape.txt", "traverses out"),
        ("nested/../../escape.txt", "traverses out"),
        (r"..\escape.txt", "traverses out"),
    ],
)
def test_an_unsafe_tar_member_is_refused(
    tmp_path, served, publish, asset_entry, write_manifest, capsys, member, reason
):
    source = _tar_with(tmp_path, "evil.tar", member)
    part = publish("evil.tar", source.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert reason in capsys.readouterr().err
    assert not (tmp_path.parent / "escape.txt").exists()


def test_a_symlink_member_is_refused(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        handle.addfile(info)
    part = publish("link.tar", archive.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "is a link" in capsys.readouterr().err


def test_a_device_member_is_refused(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    archive = tmp_path / "device.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("dev/null")
        info.type = tarfile.CHRTYPE
        handle.addfile(info)
    part = publish("device.tar", archive.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "neither a file nor a directory" in capsys.readouterr().err


def test_an_unsafe_zip_member_is_refused(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    part = publish("evil.zip", archive.read_bytes())
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="zip")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "traverses out" in capsys.readouterr().err


def test_a_broken_archive_is_reported(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("corpus.tar", b"not a tar at all")
    path = write_manifest([asset_entry("dataset-a", part, kind="dataset", extract="tar")])

    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path / "cache")]) == 1
    assert "cannot unpack" in capsys.readouterr().err


def test_a_missing_manifest_is_reported(tmp_path, capsys):
    assert fetch.main(["--manifest", str(tmp_path / "absent.json"), "--cache", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_a_manifest_that_is_not_json_is_reported(tmp_path, capsys):
    path = tmp_path / "assets.json"
    path.write_text("{not json", encoding="utf-8")
    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path)]) == 1
    assert "is not valid JSON" in capsys.readouterr().err


def test_a_manifest_without_an_assets_list_is_reported(tmp_path, capsys):
    path = tmp_path / "assets.json"
    path.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path)]) == 1
    assert "must hold an object with an assets list" in capsys.readouterr().err


def test_duplicate_ids_are_refused(
    tmp_path, served, publish, asset_entry, write_manifest, capsys
):
    part = publish("a.onnx", b"aaaa")
    path = write_manifest([asset_entry("model-a", part), asset_entry("model-a", part)])
    assert fetch.main(["--manifest", str(path), "--cache", str(tmp_path)]) == 1
    assert "duplicate asset ids: model-a" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"kind": "weights"}, "kind must be"),
        ({"extract": "rar"}, "extract must be one of"),
        ({"sha256": "abc"}, "not 64 hexadecimal"),
        ({"bytes": "12"}, "must be a non-negative integer"),
        ({"bytes": True}, "must be a non-negative integer"),
        ({"name": "../escape"}, "not a safe relative name"),
        ({"name": "/absolute"}, "not a safe relative name"),
        ({"name": "C:/drive"}, "not a safe relative name"),
        ({"files": []}, "files must be a non-empty list"),
    ],
)
def test_an_unusable_entry_is_refused(mutation, message):
    entry = {
        "id": "model-a",
        "kind": "model",
        "license": "Apache-2.0",
        "source": "s",
        "uri": "https://example.invalid/a.onnx",
        "sha256": "0" * 64,
        "bytes": 4,
    }
    entry.update(mutation)
    with pytest.raises(AssetError) as raised:
        fetch.parse_asset(entry)
    assert message in raised.value.message


def test_a_missing_required_field_is_named():
    with pytest.raises(AssetError) as raised:
        fetch.parse_asset({"id": "model-a"})
    assert "missing 'kind'" in raised.value.message


def test_a_non_object_entry_is_refused():
    with pytest.raises(AssetError):
        fetch.parse_asset(["not", "an", "object"])
    with pytest.raises(AssetError) as raised:
        fetch.parse_asset(
            {
                "id": "model-a",
                "kind": "model",
                "license": "Apache-2.0",
                "source": "s",
                "files": ["not an object"],
            }
        )
    assert "must be an object" in raised.value.message


def test_two_files_cannot_share_a_name():
    part = {"uri": "https://example.invalid/a.jpg", "sha256": "0" * 64, "bytes": 1, "name": "a.jpg"}
    with pytest.raises(AssetError) as raised:
        fetch.parse_asset(
            {
                "id": "dataset-a",
                "kind": "dataset",
                "license": "CC BY 4.0",
                "source": "s",
                "files": [part, dict(part)],
            }
        )
    assert "share one name" in raised.value.message


def test_a_name_falls_back_to_the_url(tmp_path):
    parsed = fetch.parse_asset(
        {
            "id": "model-a",
            "kind": "model",
            "license": "Apache-2.0",
            "source": "s",
            "uri": "https://example.invalid/deep/path/graph.onnx",
            "sha256": "0" * 64,
            "bytes": 4,
        }
    )
    assert parsed.parts[0].name == "graph.onnx"
    assert parsed.total_bytes == 4


def test_human_size_scales():
    assert fetch.human_size(512) == "512 B"
    assert fetch.human_size(2048) == "2.0 KiB"
    assert fetch.human_size(5 * 2**20) == "5.0 MiB"
    assert fetch.human_size(3 * 2**30) == "3.0 GiB"


def test_safe_member_name_keeps_a_relative_path():
    assert fetch.safe_member_name("a/./b/c.txt", "x") == Path("a") / "b" / "c.txt"
    with pytest.raises(AssetError):
        fetch.safe_member_name("./", "x")


def test_the_pinned_corpus_manifest_parses():
    """tests/assets.json is the real corpus; it has to parse and hold permissive licenses."""
    assets = fetch.load_assets(fetch.DEFAULT_MANIFEST)
    assert {entry.id for entry in assets} >= {
        "model-mobilenetv2-12",
        "model-resnet50-v1-12",
        "model-yolox-nano",
        "model-yolox-s",
        "model-ssd-mobilenetv1-12",
        "model-fcn-resnet50-12",
        "dataset-imagenette2-160",
        "dataset-coco-val2017-slice",
        "dataset-visa",
    }
    for entry in assets:
        assert entry.parts, entry.id
        assert "AGPL" not in entry.license and "NC" not in entry.license.replace("ONC", "")
    default = sum(entry.total_bytes for entry in assets if not entry.optional)
    assert default < 2 * 2**30, f"the default fetch is {default} bytes"
