"""Digests and bounded, path-safe tarball handling for model bundles.

A bundle is a tar archive, optionally gzip-compressed, whose SHA-256 is the bundle digest
(DESIGN.md section 8, D-IP-11). Nothing here trusts the archive: every member is checked for
type and path safety before a byte is written, and member count, total size, per-member size,
and the compression ratio are enforced while the archive streams (DESIGN.md section 15).

BundleError lives in this module because it is the lowest layer of the package; the other
image_processor.bundles modules import it from here and the package re-exports it.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import ntpath
import re
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Set

logger = logging.getLogger(__name__)

#: Read size for hashing and for streaming archive members to disk.
CHUNK_BYTES = 1 << 20

#: Smallest uncompressed size the ratio guard ever allows, so that a legitimately tiny archive
#: (a few hundred compressed bytes) is not rejected for expanding by more than ``max_ratio``.
RATIO_FLOOR_BYTES = 1 << 20

#: Cap on a member read into memory by ``read_member_bytes``.
MAX_IN_MEMORY_MEMBER_BYTES = 4 << 20

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_GZIP_MAGIC = b"\x1f\x8b"


class BundleError(Exception):
    """A bundle failed to fetch, verify, extract, or promote.

    Attributes:
        code: A SCREAMING_SNAKE code that callers branch on and that reaches operators through
            events and command replies, for example ``DIGEST_MISMATCH`` or ``ARCHIVE_LIMIT``.
        message: Human-readable detail. It never carries model bytes or credentials.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Initialize the error.

        Args:
            code: The SCREAMING_SNAKE failure code.
            message: Optional human-readable detail.
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class ExtractLimits:
    """Bounds applied to every extraction (DESIGN.md section 15, LLD section 4).

    Attributes:
        max_members: Largest number of entries an archive may contain, directories included.
        max_total_bytes: Largest total uncompressed size the archive may expand to.
        max_member_bytes: Largest uncompressed size of a single member.
        max_ratio: Largest uncompressed-to-compressed expansion factor, applied against the size
            of the archive file itself and never below ``RATIO_FLOOR_BYTES``.
    """

    max_members: int = 10_000
    max_total_bytes: int = 8 * 2**30
    max_member_bytes: int = 4 * 2**30
    max_ratio: float = 100.0


def normalize_digest(digest: str) -> str:
    """Return ``digest`` in the canonical ``sha256:<lowercase hex>`` form.

    Args:
        digest: Either ``sha256:<hex>`` or a bare 64-character hex string, in any case.

    Returns:
        The canonical ``sha256:<lowercase hex>`` string.

    Raises:
        BundleError: ``DIGEST_FORMAT`` when the value is not a SHA-256 digest.
    """
    if not isinstance(digest, str):
        raise BundleError("DIGEST_FORMAT", f"digest must be a string, got {type(digest).__name__}")
    value = digest.strip().lower()
    if value.startswith("sha256:"):
        value = value[len("sha256:") :]
    elif ":" in value:
        algorithm = value.split(":", 1)[0]
        raise BundleError(
            "DIGEST_FORMAT", f"unsupported digest algorithm {algorithm!r}, expected sha256"
        )
    if not _HEX64.match(value):
        raise BundleError("DIGEST_FORMAT", "digest must be sha256:<64 hex characters>")
    return f"sha256:{value}"


def digest_hex(digest: str) -> str:
    """Return the bare lowercase hex of ``digest``.

    Args:
        digest: Either ``sha256:<hex>`` or a bare 64-character hex string.

    Returns:
        The 64-character lowercase hex string, which is also the cache directory name.

    Raises:
        BundleError: ``DIGEST_FORMAT`` when the value is not a SHA-256 digest.
    """
    return normalize_digest(digest).split(":", 1)[1]


def sha256_file(path: Path, chunk: int = CHUNK_BYTES) -> str:
    """Hash a file with SHA-256.

    Args:
        path: File to hash.
        chunk: Read size in bytes.

    Returns:
        The bare lowercase hex digest, the same form the manifest ``files`` map uses.

    Raises:
        BundleError: ``FILE_UNREADABLE`` when the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise BundleError("FILE_UNREADABLE", f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def verify_tarball_digest(path: Path, expected: str) -> None:
    """Verify that a bundle tarball hashes to the pinned digest.

    Args:
        path: The downloaded or copied tarball.
        expected: The configured digest, ``sha256:<hex>`` or bare hex.

    Raises:
        BundleError: ``DIGEST_FORMAT`` when ``expected`` is malformed, ``FILE_UNREADABLE`` when
            the tarball cannot be read, ``DIGEST_MISMATCH`` when the bytes do not match.
    """
    want = digest_hex(expected)
    got = sha256_file(path)
    if got != want:
        raise BundleError(
            "DIGEST_MISMATCH",
            f"{path.name} hashes to sha256:{got}, expected sha256:{want}",
        )


def _member_kind(member: tarfile.TarInfo) -> str:
    """Return a short word naming the member type, for error messages."""
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.ischr():
        return "character device"
    if member.isblk():
        return "block device"
    if member.isfifo():
        return "fifo"
    return "special file"


def _safe_relative_name(name: str) -> PurePosixPath:
    """Normalize an archive member name and reject anything that could escape the destination.

    Absolute paths, Windows drive letters, ``..`` components, backslashes, and NUL bytes are
    refused rather than sanitized: a bundle that carries one is not a bundle this component
    installs.

    Args:
        name: The member name exactly as the archive records it.

    Returns:
        The normalized relative path.

    Raises:
        BundleError: ``ARCHIVE_UNSAFE_MEMBER``.
    """
    if not name or name.strip() in ("", ".", "/"):
        raise BundleError("ARCHIVE_UNSAFE_MEMBER", f"empty member name {name!r}")
    if "\x00" in name:
        raise BundleError("ARCHIVE_UNSAFE_MEMBER", "member name contains a NUL byte")
    if "\\" in name:
        raise BundleError("ARCHIVE_UNSAFE_MEMBER", f"member name {name!r} contains a backslash")
    if name.startswith("/") or ntpath.isabs(name) or _WINDOWS_DRIVE.match(name):
        raise BundleError("ARCHIVE_UNSAFE_MEMBER", f"absolute member path {name!r}")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts:
        raise BundleError("ARCHIVE_UNSAFE_MEMBER", f"empty member name {name!r}")
    if ".." in parts:
        raise BundleError(
            "ARCHIVE_UNSAFE_MEMBER", f"member path {name!r} traverses out of the bundle"
        )
    return PurePosixPath(*parts)


def _resolved_target(dest_root: Path, relative: PurePosixPath) -> Path:
    """Resolve ``relative`` under ``dest_root`` and confirm it stays inside it."""
    target = (dest_root / Path(*relative.parts)).resolve()
    if target != dest_root and dest_root not in target.parents:
        raise BundleError(
            "ARCHIVE_UNSAFE_MEMBER", f"member {relative.as_posix()!r} escapes the destination"
        )
    return target


class _LimitGuard:
    """Running total of what an archive expands to, checked against the extraction limits."""

    def __init__(self, archive_bytes: int, limits: ExtractLimits) -> None:
        """Initialize the guard for one archive.

        Args:
            archive_bytes: Size of the archive file, the denominator of the ratio guard.
            limits: The bounds to enforce.
        """
        self._limits = limits
        self._members = 0
        self._total = 0
        self._allowance = max(int(archive_bytes * limits.max_ratio), RATIO_FLOOR_BYTES)

    def count_member(self, member: tarfile.TarInfo) -> None:
        """Account for one more member and its declared size.

        Args:
            member: The member the archive stream just produced.

        Raises:
            BundleError: ``ARCHIVE_LIMIT``.
        """
        self._members += 1
        if self._members > self._limits.max_members:
            raise BundleError(
                "ARCHIVE_LIMIT", f"archive holds more than {self._limits.max_members} members"
            )
        if member.size > self._limits.max_member_bytes:
            raise BundleError(
                "ARCHIVE_LIMIT",
                f"member {member.name!r} declares {member.size} bytes, over the "
                f"{self._limits.max_member_bytes}-byte per-member limit",
            )

    def count_bytes(self, count: int, name: str) -> None:
        """Account for bytes read out of a member.

        Args:
            count: Number of bytes just read.
            name: The member being read, for the error message.

        Raises:
            BundleError: ``ARCHIVE_LIMIT``.
        """
        self._total += count
        if self._total > self._limits.max_total_bytes:
            raise BundleError(
                "ARCHIVE_LIMIT",
                f"archive expands past the {self._limits.max_total_bytes}-byte total limit",
            )
        if self._total > self._allowance:
            raise BundleError(
                "ARCHIVE_LIMIT",
                f"archive expands by more than {self._limits.max_ratio}x at member {name!r}",
            )


def _open_stream(path: Path) -> tarfile.TarFile:
    """Open a bundle tarball as a forward-only stream.

    Only tar and gzip-compressed tar are accepted, which is the bundle format (D-IP-11); the
    magic bytes decide, not the file name.

    Args:
        path: The tarball to open.

    Returns:
        An open streaming ``TarFile``.

    Raises:
        BundleError: ``ARCHIVE_UNREADABLE``.
    """
    try:
        with open(path, "rb") as handle:
            magic = handle.read(2)
    except OSError as exc:
        raise BundleError("ARCHIVE_UNREADABLE", f"cannot read {path}: {exc}") from exc
    mode = "r|gz" if magic == _GZIP_MAGIC else "r|"
    try:
        return tarfile.open(path, mode=mode)
    except (tarfile.TarError, OSError) as exc:
        raise BundleError(
            "ARCHIVE_UNREADABLE", f"{path.name} is not a tar or tar.gz archive: {exc}"
        ) from exc


#: Failures a corrupt or truncated archive raises while it streams.
_STREAM_ERRORS = (tarfile.TarError, gzip.BadGzipFile, zlib.error, EOFError)


def _archive_size(path: Path) -> int:
    """Return the size of the archive file, the denominator of the ratio guard."""
    try:
        return path.stat().st_size
    except OSError as exc:
        raise BundleError("ARCHIVE_UNREADABLE", f"cannot read {path}: {exc}") from exc


def _make_directory(path: Path, member: str) -> None:
    """Create a directory for a member, reporting a conflict with an earlier member."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        raise BundleError(
            "ARCHIVE_UNSAFE_MEMBER", f"member {member!r} collides with an earlier member"
        ) from exc
    except OSError as exc:
        raise BundleError("EXTRACT_FAILED", f"cannot create {path}: {exc}") from exc


def _write_member(source, target: Path, member: str, guard: "_LimitGuard") -> None:
    """Stream one member to disk under the size and ratio limits."""
    try:
        handle = open(target, "wb")
    except (FileExistsError, NotADirectoryError, IsADirectoryError, PermissionError) as exc:
        raise BundleError(
            "ARCHIVE_UNSAFE_MEMBER", f"member {member!r} collides with an earlier member"
        ) from exc
    except OSError as exc:
        raise BundleError("EXTRACT_FAILED", f"cannot write {target}: {exc}") from exc
    with handle as out:
        while True:
            block = source.read(CHUNK_BYTES)
            if not block:
                break
            guard.count_bytes(len(block), member)
            out.write(block)


def extract_tarball(path: Path, dest: Path, limits: ExtractLimits = ExtractLimits()) -> List[Path]:
    """Extract a bundle tarball into ``dest`` under the safety and size limits.

    Members are refused, never sanitized: absolute paths, ``..`` traversal, symlinks, hardlinks,
    devices, fifos, duplicates, and anything resolving outside ``dest`` abort the extraction.
    Member count, total bytes, per-member bytes, and the expansion ratio are enforced as the
    archive streams, so a decompression bomb stops at the limit rather than after it lands.

    Extraction is not rolled back on failure: the caller extracts into a unique staging directory
    and removes it, and only a complete bundle is ever promoted into the cache.

    Args:
        path: The digest-verified tarball.
        dest: Destination directory. Created if missing.
        limits: The bounds to enforce.

    Returns:
        The absolute paths of the regular files written, in archive order.

    Raises:
        BundleError: ``ARCHIVE_UNREADABLE``, ``ARCHIVE_UNSAFE_MEMBER``, or ``ARCHIVE_LIMIT``.
    """
    guard = _LimitGuard(_archive_size(path), limits)
    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()
    written: List[Path] = []
    seen: Set[str] = set()

    stream = _open_stream(path)
    try:
        while True:
            member = stream.next()
            if member is None:
                break
            if not (member.isreg() or member.isdir()):
                raise BundleError(
                    "ARCHIVE_UNSAFE_MEMBER",
                    f"member {member.name!r} is a {_member_kind(member)}",
                )
            relative = _safe_relative_name(member.name)
            target = _resolved_target(dest_root, relative)
            guard.count_member(member)
            key = relative.as_posix()
            if member.isdir():
                _make_directory(target, key)
                continue
            if key in seen:
                raise BundleError("ARCHIVE_UNSAFE_MEMBER", f"duplicate member {key!r}")
            seen.add(key)
            _make_directory(target.parent, key)
            source = stream.extractfile(member)
            if source is None:  # pragma: no cover - every regular member has a reader
                raise BundleError("ARCHIVE_UNREADABLE", f"member {key!r} has no readable content")
            _write_member(source, target, key, guard)
            written.append(target)
    except _STREAM_ERRORS as exc:
        raise BundleError("ARCHIVE_UNREADABLE", f"{path.name} is corrupt: {exc}") from exc
    finally:
        stream.close()
    logger.debug("extracted %d files from %s into %s", len(written), path.name, dest)
    return written


def read_member_bytes(
    path: Path,
    names: Iterable[str],
    limits: ExtractLimits = ExtractLimits(),
    max_bytes: int = MAX_IN_MEMORY_MEMBER_BYTES,
) -> Mapping[str, bytes]:
    """Read named members straight out of the archive, without extracting it.

    The signature is verified over ``manifest.json`` before the payload is extracted (DESIGN.md
    section 9 step 3), so those bytes are read here under the same member-safety and size rules
    that ``extract_tarball`` applies.

    Args:
        path: The digest-verified tarball.
        names: Member names to read, as normalized relative paths such as ``manifest.json``.
        limits: The bounds to enforce while scanning.
        max_bytes: Largest single member this reads into memory.

    Returns:
        A mapping of the requested names that were present to their exact bytes. A name the
        archive does not carry is absent from the mapping.

    Raises:
        BundleError: ``ARCHIVE_UNREADABLE``, ``ARCHIVE_UNSAFE_MEMBER``, or ``ARCHIVE_LIMIT``.
    """
    wanted = set(names)
    found: Dict[str, bytes] = {}
    guard = _LimitGuard(_archive_size(path), limits)

    stream = _open_stream(path)
    try:
        while wanted:
            member = stream.next()
            if member is None:
                break
            if not (member.isreg() or member.isdir()):
                raise BundleError(
                    "ARCHIVE_UNSAFE_MEMBER",
                    f"member {member.name!r} is a {_member_kind(member)}",
                )
            relative = _safe_relative_name(member.name)
            guard.count_member(member)
            key = relative.as_posix()
            if member.isdir() or key not in wanted:
                continue
            if member.size > max_bytes:
                raise BundleError(
                    "ARCHIVE_LIMIT",
                    f"member {key!r} is {member.size} bytes, over the {max_bytes}-byte read limit",
                )
            source = stream.extractfile(member)
            if source is None:  # pragma: no cover - every regular member has a reader
                raise BundleError("ARCHIVE_UNREADABLE", f"member {key!r} has no readable content")
            payload = source.read(max_bytes + 1)
            guard.count_bytes(len(payload), key)
            if len(payload) > max_bytes:
                raise BundleError(
                    "ARCHIVE_LIMIT", f"member {key!r} is over the {max_bytes}-byte read limit"
                )
            found[key] = payload
            wanted.discard(key)
    except _STREAM_ERRORS as exc:
        raise BundleError("ARCHIVE_UNREADABLE", f"{path.name} is corrupt: {exc}") from exc
    finally:
        stream.close()
    return found
