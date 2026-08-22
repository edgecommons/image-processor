"""The content-addressed bundle cache (DESIGN.md section 9, LLD section 4).

A verified bundle lives at ``<root>/<sha256hex>/`` with a small ``<root>/<sha256hex>.json``
metadata file beside it. The digest is the whole identity: a model version is immutable, so a
bundle is either present under its digest or it is not, and promotion is a rename rather than a
copy. The metadata file is written after the rename, so a directory without metadata is an
interrupted promotion and reads as absent.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from ..types import BundleManifest, CachedBundle
from .archive import BundleError, digest_hex, normalize_digest
from .manifest import load_manifest, resolve_model_path

logger = logging.getLogger(__name__)

#: Metadata file suffix, written beside the bundle directory.
METADATA_SUFFIX = ".json"

#: Prefix of the cache-local temporary directories promotion uses.
TEMP_PREFIX = ".tmp-"

#: How old a leftover temporary directory must be before garbage collection removes it.
TEMP_MAX_AGE_SECS = 3600.0

_HEX_DIR = re.compile(r"^[0-9a-f]{64}$")


def _force_rmtree(path: Path) -> None:
    """Remove a directory tree, clearing read-only files that Windows would otherwise keep."""

    def _clear_readonly(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            logger.warning("could not remove %s from the bundle cache", target)

    shutil.rmtree(path, onexc=_clear_readonly)


class BundleCache:
    """Content-addressed storage for verified model bundles.

    Attributes:
        root: The cache directory, created on first use.
    """

    def __init__(self, root: Path, schema_path: Optional[Path] = None) -> None:
        """Open, and create if needed, a bundle cache.

        Args:
            root: The cache directory, usually ``paths.modelCache`` from configuration.
            schema_path: The bundle-manifest schema used when reading a cached manifest.
                Defaults to the schema shipped with the component.
        """
        self.root = Path(root)
        self._schema_path = schema_path
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        """Return the directory a digest occupies, whether or not it exists.

        Args:
            digest: ``sha256:<hex>`` or bare hex.

        Returns:
            The bundle directory path.
        """
        return self.root / digest_hex(digest)

    def metadata_path(self, digest: str) -> Path:
        """Return the metadata file path for a digest.

        Args:
            digest: ``sha256:<hex>`` or bare hex.

        Returns:
            The ``<hex>.json`` path beside the bundle directory.
        """
        return self.root / (digest_hex(digest) + METADATA_SUFFIX)

    def get(self, digest: str, verify: bool = False) -> Optional[CachedBundle]:
        """Read a cached bundle.

        Args:
            digest: ``sha256:<hex>`` or bare hex.
            verify: Whether to re-hash every declared file. Staging does this before it trusts a
                cache hit; cheap reads such as ``get-models`` do not.

        Returns:
            The cached bundle, or ``None`` when the digest is not cached or its promotion was
            interrupted.

        Raises:
            BundleError: When the cached bundle is present but no longer verifies.
        """
        normalized = normalize_digest(digest)
        directory = self.path_for(normalized)
        if not directory.is_dir() or not self.metadata_path(normalized).is_file():
            return None
        manifest = load_manifest(directory, self._schema_path, verify_files=verify)
        return CachedBundle(
            digest=normalized,
            root=directory,
            manifest=manifest,
            model_path=resolve_model_path(directory, manifest),
        )

    def _write_metadata(self, digest: str, manifest: BundleManifest, model_path: Path) -> None:
        """Write the metadata file for a promoted bundle, atomically."""
        payload = {
            "digest": normalize_digest(digest),
            "schemaVersion": manifest.schema_version,
            "modelId": manifest.model_id,
            "version": manifest.version,
            "family": manifest.family.value,
            "transformVersion": manifest.transform_version,
            "keyId": manifest.key_id,
            "declaredFiles": len(manifest.files),
            "modelPath": model_path.relative_to(self.path_for(digest)).as_posix(),
            "promotedAtMs": int(time.time() * 1000),
        }
        target = self.metadata_path(digest)
        temp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, target)

    def _read(self, digest: str, verify: bool) -> CachedBundle:
        """Load a promoted bundle directory into a ``CachedBundle``."""
        directory = self.path_for(digest)
        manifest = load_manifest(directory, self._schema_path, verify_files=verify)
        return CachedBundle(
            digest=normalize_digest(digest),
            root=directory,
            manifest=manifest,
            model_path=resolve_model_path(directory, manifest),
        )

    def _move_into_place(self, extracted_dir: Path, target: Path) -> None:
        """Rename an extracted directory into the cache.

        A rename is atomic within one filesystem, which is the normal case because staging and
        the cache sit under the same component state directory. When they do not, the bundle is
        copied into a cache-local temporary directory first and that copy is renamed, so the
        directory still appears at its digest in one step.
        """
        try:
            os.replace(extracted_dir, target)
            return
        except OSError as exc:
            cross_device = exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17
            if not cross_device:
                raise BundleError(
                    "PROMOTE_FAILED", f"cannot move the staged bundle into the cache: {exc}"
                ) from exc
        staging = self.root / (TEMP_PREFIX + uuid.uuid4().hex)
        try:
            shutil.copytree(extracted_dir, staging)
            os.replace(staging, target)
        except OSError as exc:
            if staging.exists():
                _force_rmtree(staging)
            raise BundleError(
                "PROMOTE_FAILED", f"cannot copy the staged bundle into the cache: {exc}"
            ) from exc
        _force_rmtree(extracted_dir)

    def promote(self, extracted_dir: Path, digest: str) -> CachedBundle:
        """Move a verified, extracted bundle into the cache under its digest.

        The move is a rename, so a reader either sees no directory or sees the whole bundle.
        Promotion is idempotent: when the digest is already cached, that copy is verified and
        returned and the extracted one is dropped. A cached copy that no longer verifies, and a
        directory left behind by an interrupted promotion, are replaced by the extracted bundle.

        Args:
            extracted_dir: The staging directory holding the verified bundle. It no longer
                exists when this returns.
            digest: The bundle digest, ``sha256:<hex>`` or bare hex.

        Returns:
            The cached bundle.

        Raises:
            BundleError: ``PROMOTE_FAILED`` when the bundle cannot be moved into the cache, or
                any manifest error when the extracted directory is not a valid bundle.
        """
        normalized = normalize_digest(digest)
        target = self.path_for(normalized)
        source = Path(extracted_dir)
        manifest = load_manifest(source, self._schema_path, verify_files=False)
        model_relative = resolve_model_path(source, manifest).relative_to(source)

        if target.is_dir():
            try:
                cached = self._read(normalized, verify=True)
            except BundleError as exc:
                logger.warning(
                    "replacing the cached bundle %s, which no longer verifies: %s", normalized, exc
                )
                _force_rmtree(target)
            else:
                _force_rmtree(source)
                if not self.metadata_path(normalized).is_file():
                    self._write_metadata(normalized, cached.manifest, cached.model_path)
                return cached

        self._move_into_place(source, target)
        self._write_metadata(normalized, manifest, target / model_relative)
        logger.info(
            "promoted bundle %s (%s %s) into the cache",
            normalized,
            manifest.model_id,
            manifest.version,
        )
        return self._read(normalized, verify=False)

    def list(self) -> List[CachedBundle]:
        """List every promoted bundle in the cache.

        A directory that no longer loads is logged and skipped rather than raised, so an
        operator listing the cache still sees the healthy bundles.

        Returns:
            The cached bundles, ordered by digest.
        """
        bundles: List[CachedBundle] = []
        for digest in sorted(self._promoted_digests()):
            try:
                bundles.append(self._read(digest, verify=False))
            except BundleError as exc:
                logger.warning("skipping unreadable cached bundle %s: %s", digest, exc)
        return bundles

    def _promoted_digests(self) -> Set[str]:
        """Return the hex digests that have both a directory and a metadata file."""
        found: Set[str] = set()
        for entry in self.root.iterdir():
            if entry.is_dir() and _HEX_DIR.match(entry.name):
                if (self.root / (entry.name + METADATA_SUFFIX)).is_file():
                    found.add(entry.name)
        return found

    def gc(self, pinned: Iterable[str]) -> List[str]:
        """Remove every cached bundle that is not pinned.

        The caller supplies the pins, which are the generations DESIGN.md section 9 protects:
        the active generation of every route, any generation an in-flight job is pinned to, the
        rollback generation, and any bundle a pending job references.

        Interrupted promotions and stale temporary directories are removed too, but only after
        ``TEMP_MAX_AGE_SECS`` so that a promotion running concurrently is left alone.

        Args:
            pinned: Digests to keep, as ``sha256:<hex>`` or bare hex.

        Returns:
            The digests removed, canonical and sorted.
        """
        keep: Set[str] = {digest_hex(item) for item in pinned}
        removed: List[str] = []
        now = time.time()
        for entry in sorted(self.root.iterdir()):
            name = entry.name
            if entry.is_dir() and name.startswith(TEMP_PREFIX):
                if now - entry.stat().st_mtime > TEMP_MAX_AGE_SECS:
                    _force_rmtree(entry)
                continue
            if entry.is_dir() and _HEX_DIR.match(name):
                if name in keep:
                    continue
                _force_rmtree(entry)
                metadata = self.root / (name + METADATA_SUFFIX)
                if metadata.is_file():
                    metadata.unlink()
                removed.append(f"sha256:{name}")
                continue
            if entry.is_file() and name.endswith(METADATA_SUFFIX):
                stem = name[: -len(METADATA_SUFFIX)]
                if _HEX_DIR.match(stem) and stem not in keep and not (self.root / stem).is_dir():
                    entry.unlink()
                    removed.append(f"sha256:{stem}")
        if removed:
            logger.info("garbage-collected %d bundles from %s", len(removed), self.root)
        return sorted(set(removed))

    def metadata(self, digest: str) -> Optional[Dict[str, object]]:
        """Read the metadata file written when a digest was promoted.

        Args:
            digest: ``sha256:<hex>`` or bare hex.

        Returns:
            The metadata document, or ``None`` when the digest is not cached.
        """
        path = self.metadata_path(digest)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable cache metadata at %s: %s", path, exc)
            return None
