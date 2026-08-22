"""Path comparison and filesystem probing, the two rules that behave differently per platform."""

import os
from pathlib import Path

import pytest

import image_processor.config.validate as validate


def test_a_path_folds_out_its_own_navigation():
    assert validate._tokens(Path("/var/spool/./cam-01")) == validate._tokens(
        Path("/var/spool/cam-01")
    )
    assert validate._tokens(Path("/var/spool/cam-01/../cam-01")) == validate._tokens(
        Path("/var/spool/cam-01")
    )
    assert validate._tokens(Path("")) == validate._tokens(Path("."))


def test_a_path_that_walks_above_its_root_stops_there():
    assert validate._tokens(Path("/var/../../..")) == validate._tokens(Path("/"))


def test_containment_is_directional():
    parent = Path("/var/spool")
    child = Path("/var/spool/cam-01")
    assert validate._contains(parent, child)
    assert not validate._contains(child, parent)
    assert validate._contains(parent, parent)
    assert validate._overlaps(parent, child)
    assert validate._overlaps(child, parent)
    assert not validate._overlaps(Path("/var/spool"), Path("/var/lib"))


def test_windows_paths_compare_case_insensitively(monkeypatch):
    monkeypatch.setattr(validate.os, "name", "nt")
    assert validate._contains(Path("C:/Spool"), Path("c:/spool/cam-01"))


def test_posix_paths_compare_case_sensitively(monkeypatch):
    monkeypatch.setattr(validate.os, "name", "posix")
    assert not validate._contains(Path("/Spool"), Path("/spool/cam-01"))
    assert validate._contains(Path("/spool"), Path("/spool/cam-01"))


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("s3://bucket/key.tar.gz", "s3"),
        ("https://example.com/key.tar.gz", "https"),
        ("file:///models/key.tar.gz", "file"),
        ("/models/key.tar.gz", "file"),
    ],
)
def test_a_model_uri_reports_its_scheme(uri, expected):
    assert validate._scheme_of(uri) == expected


def test_an_unwritable_ancestor_blocks_a_completion_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(validate.os, "access", lambda path, mode: False)
    problem = validate._directory_problem(tmp_path / "processed" / "2026")
    assert "not writable" in problem


def test_a_path_with_no_existing_ancestor_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert validate._directory_problem(tmp_path / "a" / "b") == "no ancestor directory exists"


def test_an_ancestor_that_is_a_file_blocks_a_completion_directory(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    assert "is not a directory" in validate._directory_problem(blocker / "processed")


def test_an_existing_writable_directory_is_usable(tmp_path):
    assert validate._directory_problem(tmp_path) is None
    assert validate._directory_problem(tmp_path / "new") is None
