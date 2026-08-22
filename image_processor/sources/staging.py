"""Processor-owned staging, and the path, digest, and configuration primitives sources share.

Every input eventually becomes an immutable file the executor cell reads by path and expected
digest (DESIGN.md section 6.2). A spool file already is that file: the component owns the spool,
so it reads it in place. An inline or referenced trigger image is not -- the producer owns it and
may delete or rewrite it -- so it is copied into processor-owned staging under a digest-derived
name before admission.

The module also holds what every source in this package needs before it can trust a path: a
containment check that resolves reparse points and symlinks, a classifier that accepts only
regular files, and a streaming SHA-256. ``lstat`` is bound at module level so a test can
substitute a Windows reparse-point stat result without holding the privilege to create one.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat as stat_module
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Optional

#: Read size for streaming hashes and copies.
CHUNK_BYTES = 1 << 20

#: The bound the core envelope puts on a binary message body (D-IP-5). Nothing inline exceeds it.
MAX_INLINE_BYTES = 64 * 1024

#: Indirections so a test can substitute a stat result or a resolved path without holding the
#: privilege to create a symlink or a junction. Production always uses the ``os`` functions.
lstat = os.lstat
realpath = os.path.realpath


class SourceError(Exception):
    """An input-source failure carrying a stable SCREAMING_SNAKE ``code``.

    Args:
        code: The stable reason token. It reaches operators as an event reason and a metric
            dimension, so it never carries a path or a digest.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)


class StagingError(SourceError):
    """A staged copy could not be created or did not verify."""


class PathError(SourceError):
    """A path is absolute, escapes its root, or is not an acceptable regular file."""


class ConfigFieldError(SourceError):
    """A required configuration field is absent."""


_MISSING = object()
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")
_HEX = frozenset("0123456789abcdef")


def _snake(name: str) -> str:
    """Return the snake_case spelling of a camelCase configuration field name."""
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def config_field(source: Any, *names: str, default: Any = _MISSING) -> Any:
    """Read one configuration field from a mapping or an object by any of its accepted names.

    Configuration reaches this package as the dataclasses in ``image_processor/config/``, as the
    parsed JSON of ``config.schema.json``, or as a test double. The wire spelling is camelCase
    (DESIGN.md 11) and the dataclass spelling is snake_case, so both are accepted for every name.
    A field that is present but null counts as absent, which is how an optional block no schema
    default has filled in behaves.

    Args:
        source: The configuration object or mapping to read.
        *names: Accepted field names, most specific first.
        default: Value to return when no name resolves. Omit it to make the field required.

    Returns:
        The field value, or ``default``.

    Raises:
        ConfigFieldError: No name resolved and no default was given.
    """
    for name in names:
        for candidate in (name, _snake(name)):
            if isinstance(source, Mapping):
                value = source.get(candidate, _MISSING)
            else:
                value = getattr(source, candidate, _MISSING)
            if value is not _MISSING and value is not None:
                return value
    if default is _MISSING:
        raise ConfigFieldError("CONFIG_FIELD_MISSING", f"one of {names!r} is required")
    return default


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = CHUNK_BYTES) -> str:
    """Return the lowercase hex SHA-256 of the exact bytes on disk at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def plain_digest(value: str) -> str:
    """Return the bare lowercase hex of a digest written bare or as ``sha256:<hex>``.

    Raises:
        SourceError: The value is not a SHA-256 digest.
    """
    if not isinstance(value, str):
        raise SourceError("MALFORMED_DIGEST", "a digest is a string")
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise SourceError("MALFORMED_DIGEST", "a digest is 64 lowercase hex characters")
    return text


def stat_signature(path: Path) -> Optional[tuple]:
    """Return ``(size, mtime_ns)`` for ``path``, or None when it does not exist.

    Two signatures taken around a read tell you whether the bytes you hashed are still the bytes on
    disk. Hashing a file a writer is still extending yields a digest for a file that no longer
    exists, so every source re-stats after it hashes.
    """
    try:
        status = lstat(path)
    except OSError:
        return None
    return (status.st_size, status.st_mtime_ns)


def classify_path(path: Path) -> Optional[str]:
    """Return why ``path`` is unacceptable as an input image, or None when it is a regular file.

    Only regular files are accepted (DESIGN.md 4.1 and 15). Symlinks, Windows reparse points such
    as junctions, directories, devices, FIFOs, and sockets are refused without being opened,
    because opening them is what a traversal or a device-file attack needs.

    Returns:
        One of ``MISSING``, ``SYMLINK``, ``REPARSE_POINT``, ``DIRECTORY``, ``DEVICE_FILE``,
        ``NOT_REGULAR_FILE``, or None.
    """
    try:
        status = lstat(path)
    except OSError:
        return "MISSING"
    mode = status.st_mode
    if stat_module.S_ISLNK(mode):
        return "SYMLINK"
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and attributes & reparse_flag:
        return "REPARSE_POINT"
    if getattr(status, "st_reparse_tag", 0):
        return "REPARSE_POINT"
    if stat_module.S_ISDIR(mode):
        return "DIRECTORY"
    if stat_module.S_ISCHR(mode) or stat_module.S_ISBLK(mode):
        return "DEVICE_FILE"
    if not stat_module.S_ISREG(mode):
        return "NOT_REGULAR_FILE"
    return None


def normalize_relative(value: str) -> str:
    """Normalize a wire-supplied relative path to forward slashes and reject anything unsafe.

    Args:
        value: A path relative to a configured root, as a producer or a camera wrote it.

    Returns:
        The normalized relative path, forward-slashed, with no leading ``./``.

    Raises:
        PathError: The path is empty, absolute, drive-qualified, UNC, or holds a ``..`` segment.
    """
    if not isinstance(value, str) or not value.strip():
        raise PathError("MALFORMED_RELATIVE_PATH", "a relative path is a non-empty string")
    text = value.replace("\\", "/").strip()
    if text.startswith("/"):
        raise PathError("ABSOLUTE_PATH", "a relative path may not be rooted")
    if re.match(r"^[A-Za-z]:", text):
        raise PathError("ABSOLUTE_PATH", "a relative path may not be drive-qualified")
    parts = []
    for segment in text.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            raise PathError("PATH_ESCAPE", "a relative path may not contain a parent segment")
        parts.append(segment)
    if not parts:
        raise PathError("MALFORMED_RELATIVE_PATH", "a relative path resolves to nothing")
    return str(PurePosixPath(*parts))


def real_root(root: Path) -> Path:
    """Return the fully resolved form of a configured root."""
    return Path(realpath(str(Path(root).expanduser())))


def resolve_under_root(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` and prove the result stays inside it.

    The candidate is resolved with ``os.path.realpath``, so a symlink or a reparse point in any
    segment is followed before the containment test rather than after it.

    Raises:
        PathError: The path is malformed or the resolved path escapes ``root``.
    """
    normalized = normalize_relative(relative)
    base = real_root(root)
    candidate = Path(realpath(str(base / normalized)))
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise PathError("PATH_ESCAPE", "the resolved path leaves its configured root") from exc
    return candidate


def relative_to_root(root: Path, path: Path) -> str:
    """Return the forward-slashed path of ``path`` relative to ``root``."""
    return Path(path).relative_to(Path(root)).as_posix()


def staged_path_for(staging_root: Path, sha256: str, suffix: str = "") -> Path:
    """Return the deterministic staged path for a digest.

    The name is the digest, under a two-character fan-out directory so a staging root holding many
    thousands of files stays navigable. The same bytes always land on the same path, which is what
    makes staging idempotent and a retry free.
    """
    digest = plain_digest(sha256)
    return Path(staging_root) / digest[:2] / f"{digest}{suffix}"


def _sync_directory(directory: Path) -> None:
    """Flush a directory entry where the platform supports it; a no-op on Windows."""
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _temp_beside(target: Path) -> Path:
    """Return a unique hidden temporary name in the same directory as ``target``."""
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")


def _install(temp: Path, target: Path, digest: str) -> Path:
    """Verify a fully written temporary file and move it atomically onto ``target``."""
    try:
        actual = sha256_file(temp)
        if actual != digest:
            raise StagingError("DIGEST_MISMATCH", "the staged copy does not match the digest")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    _sync_directory(target.parent)
    return target


def stage_copy(src: Path, staging_root: Path, sha256: str) -> Path:
    """Copy ``src`` into staging under its digest and return the processor-owned path.

    The copy is written to a hidden temporary file, hashed, and moved into place atomically, so a
    reader never sees a partial staged file. When the digest-named target already holds the right
    bytes the copy is skipped: the same input admitted twice, or a retry after a crash, reuses the
    file already staged.

    Args:
        src: The file to copy. It is read, never modified.
        staging_root: The processor-owned staging directory for this route.
        sha256: The expected digest, bare hex or ``sha256:<hex>``.

    Returns:
        The staged path.

    Raises:
        StagingError: The copied bytes do not hash to ``sha256``.
    """
    digest = plain_digest(sha256)
    target = staged_path_for(staging_root, digest, Path(src).suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == digest:
        return target
    temp = _temp_beside(target)
    with open(src, "rb") as source, open(temp, "wb") as sink:
        while True:
            block = source.read(CHUNK_BYTES)
            if not block:
                break
            sink.write(block)
        sink.flush()
        os.fsync(sink.fileno())
    return _install(temp, target, digest)


def stage_bytes(data: bytes, staging_root: Path, sha256: str, suffix: str = "") -> Path:
    """Write ``data`` into staging under its digest and return the immutable path.

    This is the inline trigger's path onto disk: once the bytes are staged, an inline image is an
    ordinary file job (DESIGN.md 4.2).
    """
    digest = plain_digest(sha256)
    target = staged_path_for(staging_root, digest, suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == digest:
        return target
    temp = _temp_beside(target)
    with open(temp, "wb") as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())
    return _install(temp, target, digest)
