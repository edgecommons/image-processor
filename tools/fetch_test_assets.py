"""Fetch and verify the tier-2 test assets pinned by ``tests/assets.json`` (DESIGN.md section 16.1).

Real models and real images are never committed (D-IP-19). ``tests/assets.json`` pins every one of
them by URL, SHA-256, and byte count; this tool turns that manifest into a verified local cache
under ``tests/.cache/<id>/``, which the ``EC_LIVE_MODELS`` suite reads.

The tool is idempotent: an asset already in the cache is re-hashed rather than re-downloaded, so
running it twice costs one pass over local disk. A digest that does not match the manifest is a
failure, never a repair: the tool reports the asset and exits non-zero, because a corpus that
changed silently underneath a committed golden is worse than no corpus at all.

Only the standard library is used, so the tool runs before anything is installed.

Examples:
    List the corpus and what is already cached::

        python tools/fetch_test_assets.py --list

    Fetch the default corpus, which is everything the manifest does not mark optional::

        python tools/fetch_test_assets.py

    Fetch two assets by id, or add the optional ones::

        python tools/fetch_test_assets.py --only model-yolox-nano,dataset-imagenette2-160
        python tools/fetch_test_assets.py --include-optional
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence

#: Repository root, so the defaults work from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The pinned asset manifest.
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "assets.json"

#: Where verified assets land. Gitignored: nothing here is ever committed.
DEFAULT_CACHE = REPO_ROOT / "tests" / ".cache"

#: Read size for streaming a download to disk.
CHUNK_BYTES = 1 << 20

#: Socket timeout for one request, in seconds.
DEFAULT_TIMEOUT_SECS = 120.0

#: Archive formats the tool unpacks after verification.
ARCHIVE_FORMATS = ("tar", "tar.gz", "zip")

#: Subdirectory an archive asset is unpacked into.
EXTRACT_DIR = "extracted"

#: Marker recording the digest of the archive an extraction came from.
EXTRACT_MARKER = ".extracted"


class AssetError(Exception):
    """An asset that could not be fetched, verified, or unpacked.

    Attributes:
        asset_id: The manifest id of the asset at fault.
        message: Operator-readable detail.
    """

    def __init__(self, asset_id: str, message: str) -> None:
        """Initialize the error.

        Args:
            asset_id: The manifest id of the asset at fault.
            message: Operator-readable detail.
        """
        super().__init__(f"{asset_id}: {message}")
        self.asset_id = asset_id
        self.message = message


@dataclass(frozen=True)
class Part:
    """One file of an asset.

    A model is one part. A per-image dataset slice is as many parts as it has images, each pinned
    on its own, so a single changed image is named rather than hidden inside an archive digest.

    Attributes:
        name: The path the file takes inside the asset directory, in POSIX form.
        uri: Where to fetch it.
        sha256: The pinned lowercase hex digest.
        bytes: The pinned size in bytes.
    """

    name: str
    uri: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Asset:
    """One pinned entry of ``tests/assets.json``.

    Attributes:
        id: The manifest id, which is also the cache directory name.
        kind: ``"model"`` or ``"dataset"``.
        license: The SPDX identifier or license name the upstream publishes under.
        source: Where the asset comes from, in prose.
        notes: What the suite uses it for, and anything a reader needs to know.
        parts: The files that make up the asset.
        optional: Whether the default fetch skips it.
        extract: The archive format to unpack, or ``None``.
    """

    id: str
    kind: str
    license: str
    source: str
    notes: str
    parts: List[Part] = field(default_factory=list)
    optional: bool = False
    extract: Optional[str] = None

    @property
    def total_bytes(self) -> int:
        """Total pinned size of every part.

        Returns:
            The sum of the parts' byte counts.
        """
        return sum(part.bytes for part in self.parts)


@dataclass(frozen=True)
class AssetStatus:
    """What one :func:`fetch_asset` call did.

    Attributes:
        asset: The asset acted on.
        downloaded: How many parts were fetched over the network.
        cached: How many parts were already present and verified.
        extracted: Whether the archive was unpacked in this run.
    """

    asset: Asset
    downloaded: int
    cached: int
    extracted: bool


def _require(document: Dict[str, Any], key: str, where: str) -> Any:
    """Read a required manifest key.

    Args:
        document: The manifest object to read from.
        key: The key that must be present.
        where: The manifest location, for the error message.

    Returns:
        The value.

    Raises:
        AssetError: When the key is missing.
    """
    if key not in document:
        raise AssetError(where, f"the manifest entry is missing {key!r}")
    return document[key]


def _part_from(document: Dict[str, Any], asset_id: str, default_name: Optional[str] = None) -> Part:
    """Build one :class:`Part` from a manifest object.

    Args:
        document: The asset entry or one of its ``files`` entries.
        asset_id: The owning asset id, for error messages.
        default_name: The name to use when the object carries none. Falls back to the last segment
            of the URL path.

    Returns:
        The part.

    Raises:
        AssetError: When a required field is missing, the name is unsafe, or ``sha256`` is not 64
            hexadecimal characters.
    """
    if not isinstance(document, dict):
        raise AssetError(asset_id, "every files[] entry must be an object")
    uri = str(_require(document, "uri", asset_id))
    fallback = PurePosixPath(urllib.parse.urlsplit(uri).path).name
    name = str(document.get("name") or default_name or fallback)
    unsafe = (
        not name
        or "\\" in name
        or name.startswith("/")
        or ":" in name.split("/")[0]
        or any(segment in ("", ".", "..") for segment in name.split("/"))
    )
    if unsafe:
        raise AssetError(asset_id, f"file name {name!r} is not a safe relative name")
    digest = str(_require(document, "sha256", asset_id)).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AssetError(asset_id, f"sha256 {digest!r} is not 64 hexadecimal characters")
    size = _require(document, "bytes", asset_id)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AssetError(asset_id, f"bytes must be a non-negative integer, got {size!r}")
    return Part(name=name, uri=uri, sha256=digest, bytes=size)


def parse_asset(document: Dict[str, Any]) -> Asset:
    """Build one :class:`Asset` from a manifest entry.

    An entry carries either a single ``uri``/``sha256``/``bytes`` triple or a ``files`` list of
    them. Both forms produce the same :class:`Asset`, so the rest of the tool has one shape to
    handle.

    Args:
        document: The parsed manifest entry.

    Returns:
        The asset.

    Raises:
        AssetError: When a required field is missing or holds an unusable value.
    """
    if not isinstance(document, dict):
        raise AssetError("(root)", "every entry of assets[] must be an object")
    asset_id = str(_require(document, "id", "(root)"))
    kind = str(_require(document, "kind", asset_id))
    if kind not in ("model", "dataset"):
        raise AssetError(asset_id, f"kind must be 'model' or 'dataset', got {kind!r}")
    extract = document.get("extract")
    if extract is not None and extract not in ARCHIVE_FORMATS:
        raise AssetError(asset_id, f"extract must be one of {list(ARCHIVE_FORMATS)}, got {extract!r}")
    files = document.get("files")
    if files is None:
        parts = [_part_from(document, asset_id)]
    else:
        if not isinstance(files, list) or not files:
            raise AssetError(asset_id, "files must be a non-empty list")
        parts = [_part_from(entry, asset_id) for entry in files]
    names = [part.name for part in parts]
    if len(set(names)) != len(names):
        raise AssetError(asset_id, "two files share one name")
    return Asset(
        id=asset_id,
        kind=kind,
        license=str(_require(document, "license", asset_id)),
        source=str(_require(document, "source", asset_id)),
        notes=str(document.get("notes", "")),
        parts=parts,
        optional=bool(document.get("optional", False)),
        extract=None if extract is None else str(extract),
    )


def load_assets(manifest_path: Path) -> List[Asset]:
    """Load and parse the pinned asset manifest.

    Args:
        manifest_path: The manifest to read.

    Returns:
        The assets in manifest order.

    Raises:
        AssetError: When the file cannot be read, is not valid JSON, or an entry is unusable.
    """
    try:
        raw = Path(manifest_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AssetError("(manifest)", f"cannot read {manifest_path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise AssetError("(manifest)", f"{manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("assets"), list):
        raise AssetError("(manifest)", f"{manifest_path} must hold an object with an assets list")
    assets = [parse_asset(entry) for entry in document["assets"]]
    identifiers = [asset.id for asset in assets]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise AssetError("(manifest)", f"duplicate asset ids: {', '.join(duplicates)}")
    return assets


def sha256_file(path: Path, chunk: int = CHUNK_BYTES) -> str:
    """Hash a file with SHA-256.

    Args:
        path: The file to hash.
        chunk: Read size in bytes.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(uri: str, dest: Path, timeout_secs: float = DEFAULT_TIMEOUT_SECS) -> str:
    """Stream one URL to a file and return what it hashed to.

    The body lands on a ``.part`` sibling and is renamed only once it is complete, so an
    interrupted run never leaves a short file that looks cached.

    Args:
        uri: The URL to fetch.
        dest: The file to write.
        timeout_secs: Socket timeout for the request.

    Returns:
        The lowercase SHA-256 hex digest of the bytes written.

    Raises:
        AssetError: When the request fails or the body cannot be written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(
        uri, headers={"User-Agent": "edgecommons-image-processor-tests"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            with open(temporary, "wb") as handle:
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    digest.update(block)
                    handle.write(block)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise AssetError(uri, f"download failed: {exc}") from exc
    temporary.replace(dest)
    return digest.hexdigest()


def fetch_part(
    part: Part, asset_dir: Path, asset_id: str, timeout_secs: float = DEFAULT_TIMEOUT_SECS
) -> bool:
    """Make one part present and verified in the cache.

    Args:
        part: The pinned file.
        asset_dir: The asset's cache directory.
        asset_id: The owning asset id, for error messages.
        timeout_secs: Socket timeout for the request.

    Returns:
        ``True`` when the file was downloaded, ``False`` when the cached copy already verified.

    Raises:
        AssetError: When the download fails, or the bytes do not match the pinned digest or size.
    """
    dest = asset_dir / Path(*PurePosixPath(part.name).parts)
    if dest.is_file() and dest.stat().st_size == part.bytes and sha256_file(dest) == part.sha256:
        return False
    actual = download(part.uri, dest, timeout_secs)
    size = dest.stat().st_size
    if actual != part.sha256 or size != part.bytes:
        dest.unlink(missing_ok=True)
        raise AssetError(
            asset_id,
            f"{part.name} from {part.uri} is sha256:{actual} ({size} bytes); "
            f"the manifest pins sha256:{part.sha256} ({part.bytes} bytes)",
        )
    return True


def safe_member_name(name: str, asset_id: str) -> Path:
    """Check one archive member name and return it as a relative path.

    An archive is untrusted input even when its digest is pinned, so absolute paths, drive
    letters, parent traversal, and Windows separators are refused rather than normalized away.

    Args:
        name: The member name as the archive declares it.
        asset_id: The owning asset id, for error messages.

    Returns:
        The member name as a relative :class:`~pathlib.Path`.

    Raises:
        AssetError: When the name would escape the extraction directory.
    """
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/") or ":" in cleaned.split("/")[0]:
        raise AssetError(asset_id, f"archive member {name!r} is an absolute path")
    parts = [segment for segment in cleaned.split("/") if segment not in ("", ".")]
    if any(segment == ".." for segment in parts):
        raise AssetError(asset_id, f"archive member {name!r} traverses out of the archive")
    if not parts:
        raise AssetError(asset_id, f"archive member {name!r} has no usable name")
    return Path(*parts)


def _extract_tar(archive: Path, dest: Path, asset_id: str, mode: str) -> None:
    """Unpack a tar archive under the member-safety rules.

    Args:
        archive: The verified archive.
        dest: The directory to unpack into.
        asset_id: The owning asset id, for error messages.
        mode: The ``tarfile`` open mode.

    Raises:
        AssetError: When a member is unsafe or the archive cannot be read.
    """
    try:
        with tarfile.open(archive, mode) as handle:
            for member in handle:
                if member.issym() or member.islnk():
                    raise AssetError(asset_id, f"archive member {member.name!r} is a link")
                if not (member.isfile() or member.isdir()):
                    raise AssetError(
                        asset_id,
                        f"archive member {member.name!r} is neither a file nor a directory",
                    )
                relative = safe_member_name(member.name, asset_id)
                target = dest / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:  # pragma: no cover - isfile() already guarantees a stream
                    raise AssetError(asset_id, f"archive member {member.name!r} has no content")
                with source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
    except tarfile.TarError as exc:
        raise AssetError(asset_id, f"cannot unpack {archive.name}: {exc}") from exc


def _extract_zip(archive: Path, dest: Path, asset_id: str) -> None:
    """Unpack a zip archive under the member-safety rules.

    Args:
        archive: The verified archive.
        dest: The directory to unpack into.
        asset_id: The owning asset id, for error messages.

    Raises:
        AssetError: When a member is unsafe or the archive cannot be read.
    """
    try:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise AssetError(asset_id, f"archive member {info.filename!r} is a link")
                relative = safe_member_name(info.filename, asset_id)
                target = dest / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
    except zipfile.BadZipFile as exc:
        raise AssetError(asset_id, f"cannot unpack {archive.name}: {exc}") from exc


def extract_archive(archive: Path, dest: Path, archive_format: str, asset_id: str) -> None:
    """Unpack a verified archive into a directory.

    Args:
        archive: The verified archive.
        dest: The directory to unpack into. An existing directory is replaced.
        archive_format: One of :data:`ARCHIVE_FORMATS`.
        asset_id: The owning asset id, for error messages.

    Raises:
        AssetError: When the format is unknown or a member is unsafe.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if archive_format == "zip":
        _extract_zip(archive, dest, asset_id)
    elif archive_format == "tar":
        _extract_tar(archive, dest, asset_id, "r:")
    elif archive_format == "tar.gz":
        _extract_tar(archive, dest, asset_id, "r:gz")
    else:  # pragma: no cover - parse_asset rejects any other value
        raise AssetError(asset_id, f"unknown archive format {archive_format!r}")


def fetch_asset(
    asset: Asset, cache_root: Path, timeout_secs: float = DEFAULT_TIMEOUT_SECS
) -> AssetStatus:
    """Make one asset present, verified, and unpacked in the cache.

    Args:
        asset: The asset to fetch.
        cache_root: The cache root, normally ``tests/.cache``.
        timeout_secs: Socket timeout for each request.

    Returns:
        What the call did.

    Raises:
        AssetError: When a part cannot be fetched or verified, or an archive cannot be unpacked.
    """
    asset_dir = Path(cache_root) / asset.id
    asset_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for part in asset.parts:
        if fetch_part(part, asset_dir, asset.id, timeout_secs):
            downloaded += 1

    extracted = False
    if asset.extract is not None:
        archive = asset_dir / Path(*PurePosixPath(asset.parts[0].name).parts)
        target = asset_dir / EXTRACT_DIR
        marker = asset_dir / EXTRACT_MARKER
        stamp = asset.parts[0].sha256
        current = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if current != stamp or not target.is_dir():
            extract_archive(archive, target, asset.extract, asset.id)
            marker.write_text(stamp + "\n", encoding="utf-8")
            extracted = True
    return AssetStatus(
        asset=asset,
        downloaded=downloaded,
        cached=len(asset.parts) - downloaded,
        extracted=extracted,
    )


def select(assets: Sequence[Asset], only: Optional[str], include_optional: bool) -> List[Asset]:
    """Choose which assets a run acts on.

    Args:
        assets: Every asset in the manifest.
        only: A comma-separated id list, or ``None`` for the whole manifest. Naming an asset
            explicitly selects it even when it is optional.
        include_optional: Whether the default selection includes the optional assets.

    Returns:
        The selected assets, in manifest order.

    Raises:
        AssetError: When ``only`` names an id the manifest does not carry.
    """
    if only:
        wanted = [entry.strip() for entry in only.split(",") if entry.strip()]
        known = {asset.id for asset in assets}
        missing = [name for name in wanted if name not in known]
        if missing:
            raise AssetError("(selection)", f"unknown asset id(s): {', '.join(missing)}")
        return [asset for asset in assets if asset.id in set(wanted)]
    return [asset for asset in assets if include_optional or not asset.optional]


def cached_state(asset: Asset, cache_root: Path) -> str:
    """Describe how much of an asset is already in the cache.

    The check is by presence and size rather than by digest: ``--list`` is a glance at the cache,
    and hashing a two-gigabyte archive to print a table is not.

    Args:
        asset: The asset to inspect.
        cache_root: The cache root.

    Returns:
        ``"cached"``, ``"missing"``, or ``"partial (n/m)"``.
    """
    asset_dir = Path(cache_root) / asset.id
    present = 0
    for part in asset.parts:
        path = asset_dir / Path(*PurePosixPath(part.name).parts)
        if path.is_file() and path.stat().st_size == part.bytes:
            present += 1
    if present == len(asset.parts):
        return "cached"
    if present == 0:
        return "missing"
    return f"partial ({present}/{len(asset.parts)})"


def human_size(size: int) -> str:
    """Format a byte count for the listing.

    Args:
        size: The count in bytes.

    Returns:
        A short human-readable size.
    """
    value = float(size)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def list_assets(assets: Sequence[Asset], cache_root: Path, stream) -> None:
    """Print the corpus, one line per asset.

    Args:
        assets: The assets to print.
        cache_root: The cache root, so the listing can say what is present.
        stream: Where to write.
    """
    columns = ("id", "kind", "files", "size", "state", "license")
    header = "{:<34} {:<8} {:>5} {:>10} {:<16} {}".format(*columns)
    print(header, file=stream)
    print("-" * len(header), file=stream)
    for asset in assets:
        flag = " (optional)" if asset.optional else ""
        print(
            "{:<34} {:<8} {:>5} {:>10} {:<16} {}{}".format(
                asset.id,
                asset.kind,
                len(asset.parts),
                human_size(asset.total_bytes),
                cached_state(asset, cache_root),
                asset.license,
                flag,
            ),
            file=stream,
        )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_test_assets",
        description="Fetch and verify the tier-2 test assets pinned by tests/assets.json.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="the pinned asset manifest")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="where verified assets land")
    parser.add_argument("--only", help="comma-separated asset ids to fetch")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also fetch the assets the manifest marks optional",
    )
    parser.add_argument("--list", action="store_true", help="print the corpus and exit")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECS, help="socket timeout per request"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every selected asset is present and verified, ``1`` otherwise.
    """
    args = _parser().parse_args(argv)
    cache_root = Path(args.cache)
    try:
        assets = load_assets(Path(args.manifest))
        selected = select(assets, args.only, args.include_optional)
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        list_assets(selected, cache_root, sys.stdout)
        return 0

    failures = 0
    for asset in selected:
        try:
            status = fetch_asset(asset, cache_root, args.timeout)
        except AssetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue
        detail = f"{status.downloaded} downloaded, {status.cached} cached"
        if status.extracted:
            detail += ", unpacked"
        print(f"ok: {asset.id} ({detail})")
    if failures:
        print(f"{failures} asset(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
