"""Bundle sources and the staging pipeline (DESIGN.md section 9, LLD section 4).

Configuration names a model by id, version, digest, and URI; this module turns that into a
verified bundle in the content-addressed cache. A bundle comes from a local path, an ``https://``
URL with TLS verification and allow-listed prefixes, or an ``s3://`` object read with configured
or ambient credentials (DESIGN.md section 15). ``boto3`` is an optional extra: without it an
``s3://`` URI fails with ``S3_UNAVAILABLE`` rather than breaking the component.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import posixpath
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..types import CachedBundle
from .archive import (
    BundleError,
    ExtractLimits,
    extract_tarball,
    normalize_digest,
    read_member_bytes,
    verify_tarball_digest,
)
from .cache import BundleCache
from .manifest import MANIFEST_NAME, SIGNATURE_NAME, load_manifest
from .signature import verify_manifest_signature

logger = logging.getLogger(__name__)

#: Default network timeout for an https download, in seconds.
DEFAULT_TIMEOUT_SECS = 60.0

#: Read size for streaming a download to disk.
DOWNLOAD_CHUNK_BYTES = 1 << 20

#: The file name a fetched bundle takes inside the staging directory.
ARCHIVE_NAME = "bundle.tar"


class Fetcher(Protocol):
    """Copies a bundle from one source scheme to a local file."""

    def fetch(self, uri: str, dest: Path, credentials: Optional[Mapping[str, Any]]) -> Path:
        """Fetch ``uri`` into ``dest``.

        Args:
            uri: The bundle location.
            dest: The local file to write. Its parent already exists.
            credentials: Resolved credentials for the source, or ``None`` for ambient ones.

        Returns:
            The path written, which is ``dest``.

        Raises:
            BundleError: ``FETCH_FAILED``, ``URI_NOT_ALLOWED``, or ``S3_UNAVAILABLE``.
        """


def normalize_https_uri(uri: str) -> str:
    """Return ``uri`` with its path normalized, so that allow-list checks cannot be tricked.

    Args:
        uri: The URL to normalize.

    Returns:
        The URL with ``.`` and ``..`` path segments resolved.
    """
    parts = urllib.parse.urlsplit(uri)
    path = posixpath.normpath(parts.path) if parts.path else ""
    if parts.path.endswith("/") and not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment)
    )


def check_https_uri(uri: str, allowed_prefixes: Optional[Sequence[str]] = None) -> str:
    """Apply the https download policy to a URL.

    TLS is always verified by the fetcher; this is the rest of the policy: the scheme must be
    ``https``, the URL must not carry credentials, and, when the deployment configures an
    allow-list, the normalized URL must start with one of its prefixes. Passing ``None`` means
    no allow-list is configured and any ``https`` URL is accepted; passing an empty sequence is
    an allow-list that admits nothing.

    Args:
        uri: The URL to check.
        allowed_prefixes: The configured allow-list, or ``None`` when none is configured.

    Returns:
        The normalized URL to download.

    Raises:
        BundleError: ``URI_NOT_ALLOWED``.
    """
    parts = urllib.parse.urlsplit(uri)
    if parts.scheme.lower() != "https":
        raise BundleError("URI_NOT_ALLOWED", f"{parts.scheme or uri!r} is not an https URL")
    if not parts.hostname:
        raise BundleError("URI_NOT_ALLOWED", "the https URL has no host")
    if parts.username or parts.password:
        raise BundleError("URI_NOT_ALLOWED", "credentials in the URL are not accepted")
    normalized = normalize_https_uri(uri)
    if allowed_prefixes is None:
        return normalized
    for prefix in allowed_prefixes:
        if normalized.startswith(normalize_https_uri(prefix)):
            return normalized
    raise BundleError(
        "URI_NOT_ALLOWED", f"{normalized} does not start with an allow-listed https prefix"
    )


def local_path_for(uri: str) -> Path:
    """Resolve a plain path or a ``file://`` URL to a local path.

    Args:
        uri: A filesystem path or a ``file://`` URL.

    Returns:
        The local path the URI names.
    """
    if not uri.lower().startswith("file://"):
        return Path(uri)
    parts = urllib.parse.urlsplit(uri)
    path = urllib.request.url2pathname(parts.path)
    if parts.netloc and parts.netloc.lower() != "localhost":
        return Path("//" + parts.netloc + path.replace(os.sep, "/"))
    return Path(path)


def _install(temp: Path, dest: Path) -> Path:
    """Move a completed download onto its final name."""
    os.replace(temp, dest)
    return dest


class LocalFileFetcher:
    """Copies a bundle from a local path or a ``file://`` URL."""

    def fetch(self, uri: str, dest: Path, credentials: Optional[Mapping[str, Any]] = None) -> Path:
        """Copy the bundle into the staging directory.

        Args:
            uri: A filesystem path or ``file://`` URL.
            dest: The local file to write.
            credentials: Ignored; a local path needs none.

        Returns:
            The path written.

        Raises:
            BundleError: ``FETCH_FAILED`` when the source is missing or unreadable.
        """
        source = local_path_for(uri)
        if not source.is_file():
            raise BundleError("FETCH_FAILED", f"{source} is not a readable file")
        temp = dest.with_name(dest.name + ".part")
        try:
            shutil.copyfile(source, temp)
        except OSError as exc:
            raise BundleError("FETCH_FAILED", f"cannot copy {source}: {exc}") from exc
        return _install(temp, dest)


class _PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Applies the https policy to every redirect target, not only to the first URL."""

    def __init__(self, allowed_prefixes: Optional[Sequence[str]]) -> None:
        """Initialize the handler.

        Args:
            allowed_prefixes: The configured allow-list, or ``None`` when none is configured.
        """
        self._allowed_prefixes = allowed_prefixes

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Check the redirect target against the policy before following it."""
        check_https_uri(newurl, self._allowed_prefixes)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpsFetcher:
    """Downloads a bundle over https with certificate verification and an allow-list."""

    def __init__(
        self,
        allowed_prefixes: Optional[Sequence[str]] = None,
        timeout_secs: float = DEFAULT_TIMEOUT_SECS,
        max_bytes: Optional[int] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> None:
        """Initialize the fetcher.

        Args:
            allowed_prefixes: URL prefixes the deployment allows, or ``None`` for no allow-list.
            timeout_secs: Socket timeout for the request.
            max_bytes: Largest download to accept, or ``None`` for no limit beyond the digest
                check that follows.
            ssl_context: TLS context. Defaults to the system trust store with hostname and
                certificate verification on.
        """
        self._allowed_prefixes = allowed_prefixes
        self._timeout_secs = timeout_secs
        self._max_bytes = max_bytes
        self._ssl_context = ssl_context or ssl.create_default_context()

    def fetch(self, uri: str, dest: Path, credentials: Optional[Mapping[str, Any]] = None) -> Path:
        """Stream the bundle to disk.

        Args:
            uri: The ``https://`` URL of the bundle.
            dest: The local file to write.
            credentials: Optional ``headers`` mapping, ``bearerToken``, or ``username`` and
                ``password`` for basic authentication.

        Returns:
            The path written.

        Raises:
            BundleError: ``URI_NOT_ALLOWED`` when the URL or a redirect target fails the policy,
                ``FETCH_FAILED`` on any transport, TLS, or size failure.
        """
        url = check_https_uri(uri, self._allowed_prefixes)
        request = urllib.request.Request(url, headers=_https_headers(credentials))
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self._ssl_context),
            _PolicyRedirectHandler(self._allowed_prefixes),
        )
        temp = dest.with_name(dest.name + ".part")
        try:
            with opener.open(request, timeout=self._timeout_secs) as response:
                self._stream(response, temp)
        except BundleError:
            _discard(temp)
            raise
        except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
            _discard(temp)
            raise BundleError("FETCH_FAILED", f"cannot download the bundle: {exc}") from exc
        return _install(temp, dest)

    def _stream(self, response, temp: Path) -> None:
        """Write a response body to ``temp``, enforcing ``max_bytes``."""
        total = 0
        with open(temp, "wb") as out:
            while True:
                block = response.read(DOWNLOAD_CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if self._max_bytes is not None and total > self._max_bytes:
                    raise BundleError(
                        "FETCH_FAILED", f"the download exceeds the {self._max_bytes}-byte limit"
                    )
                out.write(block)


def _discard(path: Path) -> None:
    """Remove a partial download, ignoring a file that was never created."""
    try:
        path.unlink()
    except OSError:
        pass


def _https_headers(credentials: Optional[Mapping[str, Any]]) -> dict:
    """Build request headers from resolved credentials."""
    headers = {"User-Agent": "edgecommons-image-processor"}
    if not credentials:
        return headers
    extra = credentials.get("headers")
    if isinstance(extra, Mapping):
        headers.update({str(key): str(value) for key, value in extra.items()})
    token = credentials.get("bearerToken") or credentials.get("bearer_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    user = credentials.get("username")
    password = credentials.get("password")
    if user and password:
        pair = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {pair}"
    return headers


def parse_s3_uri(uri: str) -> tuple:
    """Split an ``s3://bucket/key`` URI.

    Args:
        uri: The object URI.

    Returns:
        A ``(bucket, key)`` tuple.

    Raises:
        BundleError: ``URI_NOT_ALLOWED`` when the URI names no bucket or no key.
    """
    parts = urllib.parse.urlsplit(uri)
    bucket = parts.netloc
    key = parts.path.lstrip("/")
    if not bucket or not key:
        raise BundleError("URI_NOT_ALLOWED", f"{uri} is not an s3://bucket/key URI")
    return bucket, key


def _s3_client_kwargs(credentials: Optional[Mapping[str, Any]]) -> dict:
    """Map resolved credentials onto boto3 client arguments, or return none for ambient ones."""
    if not credentials:
        return {}
    aliases = {
        "aws_access_key_id": ("accessKeyId", "access_key_id", "aws_access_key_id"),
        "aws_secret_access_key": ("secretAccessKey", "secret_access_key", "aws_secret_access_key"),
        "aws_session_token": ("sessionToken", "session_token", "aws_session_token"),
        "region_name": ("region", "regionName", "region_name"),
        "endpoint_url": ("endpointUrl", "endpoint_url"),
    }
    kwargs = {}
    for target, names in aliases.items():
        for name in names:
            if credentials.get(name):
                kwargs[target] = credentials[name]
                break
    return kwargs


class S3Fetcher:
    """Downloads a bundle from Amazon S3 with configured or ambient credentials."""

    def __init__(self, max_bytes: Optional[int] = None) -> None:
        """Initialize the fetcher.

        Args:
            max_bytes: Largest object to accept, or ``None`` for no limit beyond the digest
                check that follows.
        """
        self._max_bytes = max_bytes

    def fetch(self, uri: str, dest: Path, credentials: Optional[Mapping[str, Any]] = None) -> Path:
        """Download the object into the staging directory.

        Args:
            uri: The ``s3://bucket/key`` URI.
            dest: The local file to write.
            credentials: Explicit keys resolved from a ``$secret`` reference, or ``None`` to use
                the ambient credentials of the host or the Greengrass token exchange service.

        Returns:
            The path written.

        Raises:
            BundleError: ``S3_UNAVAILABLE`` when the ``s3`` extra is not installed,
                ``URI_NOT_ALLOWED`` for a malformed URI, ``FETCH_FAILED`` on a download failure.
        """
        bucket, key = parse_s3_uri(uri)
        try:
            import boto3
        except ImportError as exc:
            raise BundleError(
                "S3_UNAVAILABLE",
                "s3:// model sources need the 's3' extra (pip install image-processor[s3])",
            ) from exc
        temp = dest.with_name(dest.name + ".part")
        try:
            client = boto3.client("s3", **_s3_client_kwargs(credentials))
            if self._max_bytes is not None:
                size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
                if size > self._max_bytes:
                    raise BundleError(
                        "FETCH_FAILED", f"the object exceeds the {self._max_bytes}-byte limit"
                    )
            client.download_file(bucket, key, str(temp))
        except BundleError:
            _discard(temp)
            raise
        except Exception as exc:
            _discard(temp)
            raise BundleError("FETCH_FAILED", f"cannot download {uri}: {exc}") from exc
        return _install(temp, dest)


def fetcher_for(
    uri: str,
    allowed_prefixes: Optional[Sequence[str]] = None,
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    max_bytes: Optional[int] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
) -> Fetcher:
    """Choose the fetcher for a bundle URI.

    Args:
        uri: The bundle location: a local path, ``file://``, ``https://``, or ``s3://``.
        allowed_prefixes: https allow-list, or ``None`` when the deployment configures none.
        timeout_secs: Socket timeout for https.
        max_bytes: Largest bundle to accept, or ``None``.
        ssl_context: TLS context for https. Defaults to the system trust store.

    Returns:
        A fetcher for the URI's scheme.

    Raises:
        BundleError: ``URI_NOT_ALLOWED`` when the scheme is not an approved one.
    """
    scheme = urllib.parse.urlsplit(uri).scheme.lower()
    if scheme in ("", "file") or (len(scheme) == 1 and uri[1:2] == ":"):
        return LocalFileFetcher()
    if scheme == "https":
        return HttpsFetcher(allowed_prefixes, timeout_secs, max_bytes, ssl_context)
    if scheme == "s3":
        return S3Fetcher(max_bytes)
    raise BundleError(
        "URI_NOT_ALLOWED", f"{scheme}:// is not an approved bundle source scheme"
    )


def _manifest_key_id(manifest_raw: bytes) -> Optional[str]:
    """Read ``keyId`` out of raw manifest bytes, before the manifest is schema-validated."""
    try:
        document = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError("MANIFEST_INVALID", f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise BundleError("MANIFEST_INVALID", "manifest.json must be a JSON object")
    key_id = document.get("keyId")
    return str(key_id) if key_id else None


def _verify_bundle_signature(
    manifest_raw: bytes,
    signature: Optional[bytes],
    signing_required: bool,
    trusted_keys: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Verify ``manifest.sig`` against the trusted key the manifest names.

    A signature that is present and verifiable against a trusted key is always checked, whether
    or not the profile requires signing: a bundle that carries a broken signature is refused
    either way. What ``signing_required`` adds is that the signature and a trusted key must be
    there at all (D-IP-10).

    Args:
        manifest_raw: The exact ``manifest.json`` bytes from the archive.
        signature: The ``manifest.sig`` bytes, or ``None`` when the bundle is unsigned.
        signing_required: Whether the profile requires a verified signature.
        trusted_keys: Public keys by ``keyId``.

    Returns:
        The ``keyId`` that verified the manifest, or ``None`` when nothing was verified.

    Raises:
        BundleError: ``SIGNATURE_MISSING``, ``UNTRUSTED_KEY``, ``BAD_SIGNATURE``, or
            ``SIGNING_KEY_INVALID``.
    """
    keys = dict(trusted_keys or {})
    key_id = _manifest_key_id(manifest_raw)
    if signing_required:
        if signature is None:
            raise BundleError(
                "SIGNATURE_MISSING", "the bundle carries no manifest.sig and signing is required"
            )
        if not key_id:
            raise BundleError("UNTRUSTED_KEY", "manifest.json names no keyId")
        if key_id not in keys:
            raise BundleError("UNTRUSTED_KEY", f"keyId {key_id!r} is not a configured trusted key")
        verify_manifest_signature(manifest_raw, signature, keys[key_id])
        return key_id
    if signature is None:
        return None
    if key_id and key_id in keys:
        verify_manifest_signature(manifest_raw, signature, keys[key_id])
        return key_id
    logger.warning(
        "bundle is signed by keyId %r, which is not configured as trusted; the signature was "
        "not verified because this profile does not require signing",
        key_id,
    )
    return None


def _check_identity(manifest, model_id: Optional[str], version: Optional[str]) -> None:
    """Refuse a bundle whose manifest does not match the model entry that asked for it."""
    if model_id is not None and manifest.model_id != model_id:
        raise BundleError(
            "MANIFEST_MISMATCH",
            f"the bundle declares modelId {manifest.model_id!r}, configuration asked for {model_id!r}",
        )
    if version is not None and manifest.version != version:
        raise BundleError(
            "MANIFEST_MISMATCH",
            f"the bundle declares version {manifest.version!r}, configuration asked for {version!r}",
        )


def _check_providers(manifest, available_providers: Optional[Sequence[str]]) -> None:
    """Refuse a bundle no available execution provider can run (DESIGN.md section 9 step 4)."""
    if available_providers is None or not manifest.providers_permitted:
        return
    if not set(manifest.providers_permitted) & set(available_providers):
        raise BundleError(
            "PROVIDER_UNSUPPORTED",
            f"the bundle permits {sorted(manifest.providers_permitted)}, this component runs "
            f"{sorted(available_providers)}",
        )


def stage_bundle(
    uri: str,
    digest: str,
    staging_root: Path,
    cache: BundleCache,
    credentials: Optional[Mapping[str, Any]] = None,
    signing_required: bool = False,
    trusted_keys: Optional[Mapping[str, Any]] = None,
    limits: ExtractLimits = ExtractLimits(),
    schema_path: Optional[Path] = None,
    allowed_prefixes: Optional[Sequence[str]] = None,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    available_providers: Optional[Sequence[str]] = None,
    validators: Sequence[Callable[[Any], None]] = (),
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    max_bytes: Optional[int] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
) -> CachedBundle:
    """Fetch, verify, and cache one bundle: DESIGN.md section 9 steps 2 to 6 without warmup.

    The order is fixed and fail-closed: download into a unique staging directory, verify the
    tarball digest, verify the signature over ``manifest.json`` when required, extract under the
    archive limits, verify the manifest schema and the per-file digests, check the declared
    identity and provider compatibility, and promote the directory into the cache. Warmup is the
    engine's job and runs after this returns, before a route's generation switches.

    Args:
        uri: The bundle location: a local path, ``file://``, ``https://``, or ``s3://``.
        digest: The pinned tarball digest, ``sha256:<hex>``.
        staging_root: Directory that holds one unique staging directory per attempt.
        cache: The content-addressed cache to promote into.
        credentials: Resolved source credentials, or ``None`` for ambient ones.
        signing_required: Whether the profile requires a verified Ed25519 signature.
        trusted_keys: Public keys by ``keyId``, resolved from configuration.
        limits: The archive extraction limits.
        schema_path: The bundle-manifest schema. Defaults to the one shipped with the component.
        allowed_prefixes: https allow-list, or ``None`` when the deployment configures none.
        model_id: The model id configuration expects, or ``None`` to skip the check.
        version: The model version configuration expects, or ``None`` to skip the check.
        available_providers: Execution providers this component can run, or ``None`` to skip the
            check.
        validators: Extra manifest checks, such as the task family's ``validate_manifest``, run
            before the bundle is promoted.
        timeout_secs: https socket timeout.
        max_bytes: Largest bundle to accept before hashing, or ``None``.
        ssl_context: TLS context for https. Defaults to the system trust store.

    Returns:
        The cached bundle, ready for warmup and activation.

    Raises:
        BundleError: With the code of the first check that failed.
    """
    normalized = normalize_digest(digest)
    try:
        cached = cache.get(normalized, verify=True)
    except BundleError as exc:
        logger.warning("re-staging %s: the cached copy no longer verifies (%s)", normalized, exc)
        cached = None
    if cached is not None:
        logger.debug("bundle %s is already cached", normalized)
        return cached

    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{normalized.split(':')[1][:12]}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        archive = staging / ARCHIVE_NAME
        fetcher = fetcher_for(uri, allowed_prefixes, timeout_secs, max_bytes, ssl_context)
        fetcher.fetch(uri, archive, credentials)
        verify_tarball_digest(archive, normalized)

        members = read_member_bytes(archive, (MANIFEST_NAME, SIGNATURE_NAME), limits)
        manifest_raw = members.get(MANIFEST_NAME)
        if manifest_raw is None:
            raise BundleError("MANIFEST_MISSING", "the bundle has no manifest.json at its root")
        _verify_bundle_signature(
            manifest_raw, members.get(SIGNATURE_NAME), signing_required, trusted_keys
        )

        extracted = staging / "bundle"
        extract_tarball(archive, extracted, limits)
        if (extracted / MANIFEST_NAME).read_bytes() != manifest_raw:
            raise BundleError(
                "MANIFEST_INVALID",
                "the extracted manifest.json differs from the one the signature covered",
            )
        manifest = load_manifest(extracted, schema_path, verify_files=True)
        _check_identity(manifest, model_id, version)
        _check_providers(manifest, available_providers)
        for validator in validators:
            validator(manifest)
        archive.unlink()
        return cache.promote(extracted, normalized)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
