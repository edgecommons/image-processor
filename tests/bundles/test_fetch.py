"""Bundle sources: local paths, https with TLS and an allow-list, and s3 as an optional extra."""

from __future__ import annotations

import shutil
import ssl
import sys
import types
from pathlib import Path

import pytest

from image_processor.bundles import (
    BundleError,
    HttpsFetcher,
    LocalFileFetcher,
    S3Fetcher,
    check_https_uri,
    fetcher_for,
    local_path_for,
    normalize_https_uri,
    parse_s3_uri,
    sha256_file,
)
from image_processor.bundles.fetch import _https_headers, _s3_client_kwargs


def test_a_local_path_is_copied(tmp_path: Path, build_bundle) -> None:
    built = build_bundle()
    dest = tmp_path / "staging" / "bundle.tar"
    dest.parent.mkdir(parents=True)
    written = LocalFileFetcher().fetch(str(built.archive), dest, None)
    assert written == dest
    assert sha256_file(dest) == built.digest.split(":")[1]


def test_a_file_url_is_copied(tmp_path: Path, build_bundle) -> None:
    built = build_bundle()
    dest = tmp_path / "staging" / "bundle.tar"
    dest.parent.mkdir(parents=True)
    LocalFileFetcher().fetch(built.archive.as_uri(), dest, None)
    assert sha256_file(dest) == built.digest.split(":")[1]


def test_a_missing_local_source_fails(tmp_path: Path) -> None:
    with pytest.raises(BundleError) as caught:
        LocalFileFetcher().fetch(str(tmp_path / "absent.tar"), tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"


def test_a_local_copy_failure_is_reported(tmp_path: Path, build_bundle, monkeypatch) -> None:
    built = build_bundle()

    def deny(src, dst, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(shutil, "copyfile", deny)
    with pytest.raises(BundleError) as caught:
        LocalFileFetcher().fetch(str(built.archive), tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"


def test_local_path_for_handles_localhost_and_hosted_urls() -> None:
    assert local_path_for("relative/path.tar") == Path("relative/path.tar")
    assert local_path_for("file://localhost/models/x.tar").as_posix().endswith("/models/x.tar")
    assert local_path_for("file://fileserver/models/x.tar").as_posix().startswith("//fileserver")


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("https://Models.Example.COM/a/b/../c.tar", "https://models.example.com/a/c.tar"),
        ("https://models.example.com/dir/", "https://models.example.com/dir/"),
        ("https://models.example.com", "https://models.example.com"),
    ],
)
def test_url_normalization(uri: str, expected: str) -> None:
    assert normalize_https_uri(uri) == expected


def test_the_https_policy_accepts_an_allow_listed_prefix() -> None:
    allowed = ["https://models.example.com/approved/"]
    assert check_https_uri("https://models.example.com/approved/m.tar.gz", allowed)


def test_the_https_policy_cannot_be_walked_out_of() -> None:
    allowed = ["https://models.example.com/approved/"]
    with pytest.raises(BundleError) as caught:
        check_https_uri("https://models.example.com/approved/../secret/m.tar.gz", allowed)
    assert caught.value.code == "URI_NOT_ALLOWED"


@pytest.mark.parametrize(
    "uri",
    [
        "http://models.example.com/m.tar",
        "https://operator:secret@models.example.com/m.tar",
        "https:///m.tar",
        "s3://bucket/key",
    ],
)
def test_the_https_policy_refuses_anything_else(uri: str) -> None:
    with pytest.raises(BundleError) as caught:
        check_https_uri(uri, None)
    assert caught.value.code == "URI_NOT_ALLOWED"


def test_an_empty_allow_list_admits_nothing() -> None:
    with pytest.raises(BundleError) as caught:
        check_https_uri("https://models.example.com/m.tar", [])
    assert caught.value.code == "URI_NOT_ALLOWED"


def test_no_allow_list_admits_any_https_url() -> None:
    assert check_https_uri("https://models.example.com/m.tar", None)


def test_an_https_bundle_downloads_over_verified_tls(tmp_path: Path, build_bundle, tls_server) -> None:
    built = build_bundle(compress=True)
    shutil.copyfile(built.archive, tls_server.root / "model.tar.gz")
    fetcher = HttpsFetcher(
        allowed_prefixes=[f"https://localhost:{tls_server.port}/"],
        ssl_context=tls_server.client_context(),
    )
    dest = tmp_path / "bundle.tar"
    fetcher.fetch(tls_server.url("model.tar.gz"), dest, {"bearerToken": "token"})
    assert sha256_file(dest) == built.digest.split(":")[1]


def test_an_untrusted_certificate_stops_the_download(tmp_path: Path, build_bundle, tls_server) -> None:
    built = build_bundle()
    shutil.copyfile(built.archive, tls_server.root / "model.tar")
    fetcher = HttpsFetcher(ssl_context=ssl.create_default_context())
    with pytest.raises(BundleError) as caught:
        fetcher.fetch(tls_server.url("model.tar"), tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"
    assert not (tmp_path / "bundle.tar").exists()
    assert not (tmp_path / "bundle.tar.part").exists()


def test_a_missing_object_is_a_fetch_failure(tmp_path: Path, tls_server) -> None:
    fetcher = HttpsFetcher(ssl_context=tls_server.client_context())
    with pytest.raises(BundleError) as caught:
        fetcher.fetch(tls_server.url("absent.tar"), tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"


def test_a_download_over_the_size_limit_is_refused(tmp_path: Path, build_bundle, tls_server) -> None:
    built = build_bundle()
    shutil.copyfile(built.archive, tls_server.root / "model.tar")
    fetcher = HttpsFetcher(ssl_context=tls_server.client_context(), max_bytes=64)
    with pytest.raises(BundleError) as caught:
        fetcher.fetch(tls_server.url("model.tar"), tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"
    assert not (tmp_path / "bundle.tar.part").exists()


def test_a_redirect_is_followed_only_inside_the_allow_list(
    tmp_path: Path, build_bundle, tls_server
) -> None:
    built = build_bundle()
    shutil.copyfile(built.archive, tls_server.root / "model.tar")
    tls_server.redirects["/approved/model.tar"] = tls_server.url("model.tar")
    tls_server.redirects["/leaky/model.tar"] = "http://models.example.com/model.tar"
    context = tls_server.client_context()

    permissive = HttpsFetcher(ssl_context=context)
    permissive.fetch(tls_server.url("approved/model.tar"), tmp_path / "ok.tar", None)
    assert sha256_file(tmp_path / "ok.tar") == built.digest.split(":")[1]

    with pytest.raises(BundleError) as caught:
        permissive.fetch(tls_server.url("leaky/model.tar"), tmp_path / "leak.tar", None)
    assert caught.value.code == "URI_NOT_ALLOWED"


def test_request_headers_carry_the_configured_credentials() -> None:
    assert _https_headers(None)["User-Agent"].startswith("edgecommons")
    assert _https_headers({"bearerToken": "abc"})["Authorization"] == "Bearer abc"
    basic = _https_headers({"username": "sam", "password": "hunter2"})["Authorization"]
    assert basic.startswith("Basic ")
    assert _https_headers({"headers": {"X-Model-Channel": "approved"}})["X-Model-Channel"] == "approved"


@pytest.mark.parametrize(
    "uri, expected",
    [("s3://approved-models/line/2026.tar.gz", ("approved-models", "line/2026.tar.gz"))],
)
def test_s3_uris_are_split(uri: str, expected) -> None:
    assert parse_s3_uri(uri) == expected


@pytest.mark.parametrize("uri", ["s3://bucket-only", "s3:///key-only"])
def test_a_malformed_s3_uri_is_refused(uri: str) -> None:
    with pytest.raises(BundleError) as caught:
        parse_s3_uri(uri)
    assert caught.value.code == "URI_NOT_ALLOWED"


def test_s3_credentials_map_onto_boto3_arguments() -> None:
    assert _s3_client_kwargs(None) == {}
    mapped = _s3_client_kwargs(
        {
            "accessKeyId": "AKIA",
            "secretAccessKey": "secret",
            "sessionToken": "token",
            "region": "us-east-1",
            "endpointUrl": "https://s3.local",
        }
    )
    assert mapped == {
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "region_name": "us-east-1",
        "endpoint_url": "https://s3.local",
    }


def test_s3_without_the_extra_reports_s3_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(BundleError) as caught:
        S3Fetcher().fetch("s3://bucket/key.tar", tmp_path / "bundle.tar", None)
    assert caught.value.code == "S3_UNAVAILABLE"


class _FakeS3Client:
    """A boto3 stand-in that copies a local file, so the S3 path is exercised without AWS."""

    def __init__(self, source: Path, kwargs: dict) -> None:
        self.source = source
        self.kwargs = kwargs
        self.calls = []

    def head_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803 - boto3's parameter names
        return {"ContentLength": self.source.stat().st_size}

    def download_file(self, Bucket: str, Key: str, filename: str) -> None:  # noqa: N803
        self.calls.append((Bucket, Key, filename))
        shutil.copyfile(self.source, filename)


def _install_fake_boto3(monkeypatch, source: Path, seen: dict) -> None:
    """Put a fake boto3 module in place for the duration of a test."""

    def client(name: str, **kwargs):
        seen["service"] = name
        seen["kwargs"] = kwargs
        seen["client"] = _FakeS3Client(source, kwargs)
        return seen["client"]

    module = types.ModuleType("boto3")
    module.client = client
    monkeypatch.setitem(sys.modules, "boto3", module)


def test_an_s3_bundle_downloads_with_explicit_credentials(tmp_path: Path, build_bundle, monkeypatch) -> None:
    built = build_bundle()
    seen: dict = {}
    _install_fake_boto3(monkeypatch, built.archive, seen)
    dest = tmp_path / "bundle.tar"

    S3Fetcher().fetch(
        "s3://approved-models/line-clearance/2026.08.20.tar",
        dest,
        {"accessKeyId": "AKIA", "secretAccessKey": "secret"},
    )

    assert sha256_file(dest) == built.digest.split(":")[1]
    assert seen["service"] == "s3"
    assert seen["kwargs"]["aws_access_key_id"] == "AKIA"
    assert seen["client"].calls[0][:2] == ("approved-models", "line-clearance/2026.08.20.tar")


def test_an_s3_object_over_the_size_limit_is_refused(tmp_path: Path, build_bundle, monkeypatch) -> None:
    built = build_bundle()
    _install_fake_boto3(monkeypatch, built.archive, {})
    with pytest.raises(BundleError) as caught:
        S3Fetcher(max_bytes=32).fetch("s3://bucket/key.tar", tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"


def test_an_s3_failure_is_reported(tmp_path: Path, monkeypatch) -> None:
    module = types.ModuleType("boto3")

    def client(name: str, **kwargs):
        raise RuntimeError("no credentials")

    module.client = client
    monkeypatch.setitem(sys.modules, "boto3", module)
    with pytest.raises(BundleError) as caught:
        S3Fetcher().fetch("s3://bucket/key.tar", tmp_path / "bundle.tar", None)
    assert caught.value.code == "FETCH_FAILED"


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("/var/lib/models/x.tar", LocalFileFetcher),
        ("models/x.tar", LocalFileFetcher),
        ("file:///var/lib/models/x.tar", LocalFileFetcher),
        ("C:" + chr(92) + "models" + chr(92) + "x.tar", LocalFileFetcher),
        ("https://models.example.com/x.tar", HttpsFetcher),
        ("s3://bucket/x.tar", S3Fetcher),
    ],
)
def test_fetcher_for_picks_the_scheme(uri: str, expected) -> None:
    assert isinstance(fetcher_for(uri), expected)


@pytest.mark.parametrize("uri", ["ftp://models.example.com/x.tar", "http://models.example.com/x.tar"])
def test_fetcher_for_refuses_an_unapproved_scheme(uri: str) -> None:
    with pytest.raises(BundleError) as caught:
        fetcher_for(uri)
    assert caught.value.code == "URI_NOT_ALLOWED"
