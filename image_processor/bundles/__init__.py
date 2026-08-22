"""Model bundles: verification, content-addressed storage, and delivery (LLD section 4).

A bundle is an immutable tarball named by the SHA-256 of its own bytes (D-IP-11). Everything a
route needs to run a model version is inside it, and everything inside it is verified before it
is used: the tarball digest, the Ed25519 signature over ``manifest.json`` (D-IP-10), bounded and
path-safe extraction, the manifest schema, and the per-file digests. No bundle-supplied code
runs (D-IP-12).

``stage_bundle`` is the entry point: it takes a configured model source and leaves a verified
bundle in the cache, ready for warmup and activation.
"""

from .archive import (
    CHUNK_BYTES,
    MAX_IN_MEMORY_MEMBER_BYTES,
    RATIO_FLOOR_BYTES,
    BundleError,
    ExtractLimits,
    digest_hex,
    extract_tarball,
    normalize_digest,
    read_member_bytes,
    sha256_file,
    verify_tarball_digest,
)
from .cache import METADATA_SUFFIX, BundleCache
from .fetch import (
    ARCHIVE_NAME,
    DEFAULT_TIMEOUT_SECS,
    Fetcher,
    HttpsFetcher,
    LocalFileFetcher,
    S3Fetcher,
    check_https_uri,
    fetcher_for,
    local_path_for,
    normalize_https_uri,
    parse_s3_uri,
    stage_bundle,
)
from .manifest import (
    DEFAULT_SCHEMA_PATH,
    MANIFEST_NAME,
    MODEL_FILE_NAME,
    SIGNATURE_NAME,
    load_manifest,
    load_schema,
    parse_manifest,
    read_manifest_bytes,
    resolve_model_path,
    validate_document,
)
from .signature import (
    RAW_KEY_BYTES,
    SIGNATURE_BYTES,
    generate_keypair,
    load_private_key,
    load_public_key,
    private_key_pem,
    public_key_pem,
    public_key_raw,
    sign_manifest,
    verify_manifest_signature,
)

__all__ = [
    "ARCHIVE_NAME",
    "BundleCache",
    "BundleError",
    "CHUNK_BYTES",
    "DEFAULT_SCHEMA_PATH",
    "DEFAULT_TIMEOUT_SECS",
    "ExtractLimits",
    "Fetcher",
    "HttpsFetcher",
    "LocalFileFetcher",
    "MANIFEST_NAME",
    "MAX_IN_MEMORY_MEMBER_BYTES",
    "METADATA_SUFFIX",
    "MODEL_FILE_NAME",
    "RATIO_FLOOR_BYTES",
    "RAW_KEY_BYTES",
    "S3Fetcher",
    "SIGNATURE_BYTES",
    "SIGNATURE_NAME",
    "check_https_uri",
    "digest_hex",
    "extract_tarball",
    "fetcher_for",
    "generate_keypair",
    "load_manifest",
    "load_private_key",
    "load_public_key",
    "load_schema",
    "local_path_for",
    "normalize_digest",
    "normalize_https_uri",
    "parse_manifest",
    "parse_s3_uri",
    "private_key_pem",
    "public_key_pem",
    "public_key_raw",
    "read_manifest_bytes",
    "read_member_bytes",
    "resolve_model_path",
    "sha256_file",
    "sign_manifest",
    "stage_bundle",
    "validate_document",
    "verify_manifest_signature",
    "verify_tarball_digest",
]
