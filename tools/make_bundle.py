"""Build and sign a model bundle (LLD section 4, DESIGN.md section 8).

The tool packs a directory into the bundle format the component installs: a tar archive,
optionally gzip-compressed, with ``manifest.json`` at the root, a detached Ed25519 signature in
``manifest.sig``, and the model files beside them. It computes the per-file SHA-256 digests into
the manifest, merging whatever the author already wrote in the directory's own ``manifest.json``,
and prints the digest of the finished tarball. That digest is what configuration pins in
``models[].digest``.

Examples:
    Create a signing keypair::

        python tools/make_bundle.py --gen-key keys/publisher-1.pem

    Build a signed bundle::

        python tools/make_bundle.py build/line-clearance \
            --out dist/line-clearance-2026.08.20.tar.gz \
            --key keys/publisher-1.pem --key-id pharma-model-publisher-1

Archives are reproducible: member order, ownership, permissions, and timestamps are fixed, so
the same input directory and key always produce the same digest.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.bundles import (  # noqa: E402  (path bootstrap must run first)
    MANIFEST_NAME,
    SIGNATURE_NAME,
    BundleError,
    generate_keypair,
    load_private_key,
    normalize_digest,
    parse_manifest,
    sha256_file,
    sign_manifest,
    validate_document,
)

#: Fixed member metadata, so that two builds of the same directory produce the same digest.
FIXED_MODE = 0o644
FIXED_MTIME = 0


def collect_files(src_dir: Path) -> Dict[str, str]:
    """Hash every payload file in a bundle source directory.

    ``manifest.json`` and ``manifest.sig`` are not payload: the manifest cannot declare its own
    digest, and the signature is produced from the finished manifest.

    Args:
        src_dir: The directory to pack.

    Returns:
        A mapping of relative POSIX path to lowercase SHA-256 hex, ordered by path.

    Raises:
        BundleError: ``BUNDLE_EMPTY`` when the directory holds no payload files.
    """
    files: Dict[str, str] = {}
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src_dir).as_posix()
        if relative in (MANIFEST_NAME, SIGNATURE_NAME):
            continue
        files[relative] = sha256_file(path)
    if not files:
        raise BundleError("BUNDLE_EMPTY", f"{src_dir} holds no files to pack")
    return files


def build_manifest_document(
    src_dir: Path, key_id: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build the manifest document for a source directory.

    The author's own ``manifest.json`` supplies everything about the model; this fills in the
    ``files`` digests and, when given, the signing ``keyId``.

    Args:
        src_dir: The directory to pack.
        key_id: The signing key id to record, or ``None`` to keep whatever the author wrote.
        overrides: Extra top-level manifest fields to set.

    Returns:
        The manifest document.

    Raises:
        BundleError: ``MANIFEST_INVALID`` when the author's manifest is not a JSON object,
            ``BUNDLE_EMPTY`` when there is nothing to pack.
    """
    document: Dict[str, Any] = {}
    authored = src_dir / MANIFEST_NAME
    if authored.is_file():
        try:
            document = json.loads(authored.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BundleError("MANIFEST_INVALID", f"{authored} is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise BundleError("MANIFEST_INVALID", f"{authored} must hold a JSON object")
    document["files"] = collect_files(src_dir)
    if key_id is not None:
        document["keyId"] = key_id
    if overrides:
        document.update(overrides)
    return document


def serialize_manifest(document: Dict[str, Any]) -> bytes:
    """Serialize a manifest to the exact bytes the bundle carries and the signature covers.

    Args:
        document: The manifest document.

    Returns:
        UTF-8 bytes: two-space indent, sorted keys, one trailing newline.
    """
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Add an in-memory member with fixed metadata."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = FIXED_MODE
    info.mtime = FIXED_MTIME
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def _add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    """Add a file member with fixed metadata."""
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = FIXED_MODE
    info.mtime = FIXED_MTIME
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    with open(path, "rb") as handle:
        tar.addfile(info, handle)


def _write_archive(
    out_path: Path, src_dir: Path, files: List[str], manifest_bytes: bytes, signature: Optional[bytes], compress: bool
) -> None:
    """Write the tarball, manifest first, signature second, payload in sorted order.

    The gzip header carries no name and no timestamp, so compressing the same tar twice yields
    the same bytes and therefore the same bundle digest.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw:
        stream = (
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=FIXED_MTIME)
            if compress
            else raw
        )
        try:
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
                _add_bytes(tar, MANIFEST_NAME, manifest_bytes)
                if signature is not None:
                    _add_bytes(tar, SIGNATURE_NAME, signature)
                for relative in files:
                    _add_file(tar, src_dir / Path(*relative.split("/")), relative)
        finally:
            if compress:
                stream.close()


def make_bundle(
    src_dir: Path,
    out_path: Path,
    key: Optional[bytes] = None,
    key_id: Optional[str] = None,
    compress: Optional[bool] = None,
    password: Optional[bytes] = None,
    schema_path: Optional[Path] = None,
) -> str:
    """Pack a directory into a bundle tarball and return its digest.

    Args:
        src_dir: The directory to pack. Its ``manifest.json``, when present, supplies the model
            fields; the per-file digests are always recomputed.
        out_path: The tarball to write. A ``.gz`` suffix turns on compression unless
            ``compress`` says otherwise.
        key: The Ed25519 private key to sign with (PEM, DER, or raw), or ``None`` for an
            unsigned bundle.
        key_id: The signing key id to record in the manifest.
        compress: Force gzip on or off, or ``None`` to decide from the output name.
        password: Passphrase for an encrypted private key PEM.
        schema_path: When given, validate the manifest against this schema before packing.

    Returns:
        The bundle digest as ``sha256:<hex>``.

    Raises:
        BundleError: ``MANIFEST_INVALID``, ``BUNDLE_EMPTY``, ``SIGNING_KEY_INVALID``, or
            ``KEY_ID_MISSING`` when a signed bundle names no key id.
    """
    src_dir = Path(src_dir)
    out_path = Path(out_path)
    if not src_dir.is_dir():
        raise BundleError("BUNDLE_EMPTY", f"{src_dir} is not a directory")
    document = build_manifest_document(src_dir, key_id)
    if key is not None and not document.get("keyId"):
        raise BundleError(
            "KEY_ID_MISSING", "a signed bundle needs --key-id, or a keyId in the source manifest"
        )
    if schema_path is not None:
        validate_document(document, Path(schema_path))
    parse_manifest(document)
    manifest_bytes = serialize_manifest(document)
    signature = (
        sign_manifest(manifest_bytes, load_private_key(key, password)) if key is not None else None
    )
    if compress is None:
        compress = out_path.name.lower().endswith(".gz")
    _write_archive(
        out_path, src_dir, sorted(document["files"]), manifest_bytes, signature, compress
    )
    return normalize_digest(sha256_file(out_path))


def generate_key_files(path: Path, password: Optional[bytes] = None) -> Tuple[Path, Path, Path]:
    """Create an Ed25519 signing keypair on disk.

    Args:
        path: Where to write the private key PEM. The public key goes beside it as
            ``<path>.pub.pem`` and ``<path>.pub``, the raw 32 bytes a trusted-key secret holds.
        password: Passphrase to encrypt the private key PEM with.

    Returns:
        The private PEM, public PEM, and raw public key paths.

    Raises:
        BundleError: ``KEY_EXISTS`` when any of the three files is already there.
    """
    path = Path(path)
    public_pem_path = path.with_name(path.name + ".pub.pem")
    public_raw_path = path.with_name(path.name + ".pub")
    for candidate in (path, public_pem_path, public_raw_path):
        if candidate.exists():
            raise BundleError("KEY_EXISTS", f"{candidate} already exists; refusing to overwrite")
    private_pem, public_pem, public_raw = generate_keypair(password)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(private_pem)
    public_pem_path.write_bytes(public_pem)
    public_raw_path.write_bytes(public_raw)
    return path, public_pem_path, public_raw_path


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="make_bundle",
        description="Build and sign an EdgeCommons image-model bundle.",
    )
    parser.add_argument("src", nargs="?", help="directory holding the model files to pack")
    parser.add_argument("--out", help="tarball to write, for example dist/model-2026.08.20.tar.gz")
    parser.add_argument("--key", help="Ed25519 private key PEM to sign manifest.json with")
    parser.add_argument("--key-id", help="signing key id to record in the manifest")
    parser.add_argument(
        "--key-password", help="passphrase of an encrypted private key PEM", default=None
    )
    parser.add_argument(
        "--gzip", action="store_true", help="compress the tarball even if --out has no .gz suffix"
    )
    parser.add_argument(
        "--schema", help="validate the manifest against this JSON Schema before packing"
    )
    parser.add_argument(
        "--gen-key", metavar="PATH", help="generate an Ed25519 keypair at PATH and exit"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` when the bundle or the key cannot be produced.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    password = args.key_password.encode("utf-8") if args.key_password else None
    try:
        if args.gen_key:
            private, public_pem, public_raw = generate_key_files(Path(args.gen_key), password)
            print(f"private key: {private}")
            print(f"public key (PEM): {public_pem}")
            print(f"public key (raw): {public_raw}")
            return 0
        if not args.src or not args.out:
            parser.error("src and --out are required unless you pass --gen-key")
        key = Path(args.key).read_bytes() if args.key else None
        digest = make_bundle(
            src_dir=Path(args.src),
            out_path=Path(args.out),
            key=key,
            key_id=args.key_id,
            compress=True if args.gzip else None,
            password=password,
            schema_path=Path(args.schema) if args.schema else None,
        )
        print(digest)
        return 0
    except BundleError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"IO_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
