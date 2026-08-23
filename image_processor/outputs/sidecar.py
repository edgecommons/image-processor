"""The evidence sidecar and the ordered installation it requires (DESIGN.md 7, 12.4, LLD 8).

``<image>.inference.json`` is the local evidence record for one inference: the capture identity,
the verified source, the route and its configuration generation, the exact model generation, the
full result and its timings, and the identities a downstream evidence chain joins on. It is
immutable once installed.

The order is the contract. The document is written to a unique temporary file, flushed, and only
then installed at its deterministic path with an atomic rename; the directory is flushed after
the rename where the platform supports it. The ledger transaction that records the result, the
sidecar digest, and the outbox rows runs after all of that, so a crash at any point leaves either
no sidecar and no committed result, or a sidecar recovery can verify and adopt. Nothing here
touches the ledger: the caller owns the transaction, and this module owns the bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: The suffix appended to an image path to form its evidence sidecar.
SIDECAR_SUFFIX = ".inference.json"

#: The evidence document contract version.
SIDECAR_SCHEMA_VERSION = "1.0"

#: Seams. A test substitutes them to fail exactly between two steps of the install.
fsync = os.fsync
replace = os.replace


class SidecarError(Exception):
    """The evidence sidecar could not be installed.

    Attributes:
        code: Stable SCREAMING_SNAKE code.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class InstalledSidecar:
    """One installed evidence sidecar.

    Attributes:
        path: Where it was installed.
        sha256: The digest of the installed bytes, which the ledger binds to the job.
        bytes: How large it is.
    """

    path: Path
    sha256: str
    bytes: int


def sidecar_path_for(image_path: Any) -> Path:
    """Return the deterministic evidence path for one input.

    Args:
        image_path: The input image path.

    Returns:
        ``<image>.inference.json`` beside the image.
    """
    path = Path(image_path)
    return path.with_name(path.name + SIDECAR_SUFFIX)


def sidecar_document(
    job: Any,
    body: dict,
    *,
    evidence_id: str,
    config_generation: int = 0,
    manifest: Any = None,
    written_at_ms: Optional[int] = None,
) -> dict:
    """Build the evidence document for one inference (DESIGN.md 12.4).

    The document binds what the published message carries to what only the device knows: which
    configuration generation the route ran under, which provider policy the manifest demanded,
    and when the record was written.

    Args:
        job: The durable job.
        body: The full, unbounded result body.
        evidence_id: The identity a downstream evidence chain joins on.
        config_generation: The configuration generation the route ran under.
        manifest: The bundle manifest of the pinned generation, when it is known.
        written_at_ms: The wall clock in milliseconds, or ``None`` to read it now.

    Returns:
        The document to install.
    """
    document: dict = {
        "schemaVersion": SIDECAR_SCHEMA_VERSION,
        "evidenceId": evidence_id,
        "inferenceId": job.inference_id,
        "routeId": job.route_id,
        "configGeneration": int(config_generation),
        "writtenAtMs": int(written_at_ms if written_at_ms is not None else _now_ms()),
        "result": body,
    }
    policy = getattr(manifest, "provider_policy", None)
    if policy:
        document["providerPolicy"] = str(policy)
    minimum = getattr(manifest, "min_onnxruntime", None)
    if minimum:
        document["minOnnxruntime"] = str(minimum)
    transform_version = job.transform_version or getattr(manifest, "transform_version", None)
    if transform_version:
        document["transformVersion"] = str(transform_version)
    if job.staged_path:
        document["stagedPath"] = str(job.staged_path)
    return document


def _now_ms() -> int:
    """Return the current wall clock in milliseconds."""
    import time

    return int(time.time() * 1000)


def encode_document(document: dict) -> bytes:
    """Serialize an evidence document to the exact bytes that get installed.

    Args:
        document: The evidence document.

    Returns:
        UTF-8 JSON with a trailing newline, so the file reads well in a terminal.

    Raises:
        SidecarError: ``SIDECAR_NOT_SERIALIZABLE`` when the document cannot be expressed as JSON.
    """
    try:
        return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SidecarError("SIDECAR_NOT_SERIALIZABLE", str(exc)) from exc


def write_sidecar(path: Any, document: dict, *, durable: bool = True) -> InstalledSidecar:
    """Install one evidence sidecar: temporary file, flush, atomic install, directory flush.

    An existing sidecar is never overwritten. A byte-identical one is adopted, which makes a
    retried commit idempotent; one holding different bytes is a collision, because an evidence
    record that changes after the fact is not evidence.

    Args:
        path: Where to install it.
        document: The evidence document.
        durable: Whether to flush the file and its directory. Only a test that is measuring the
            ordering rather than the durability turns it off.

    Returns:
        The installed sidecar, its digest, and its size.

    Raises:
        SidecarError: The document cannot be serialized, an incompatible sidecar is already
            installed, or the filesystem refused the write.
    """
    target = Path(path)
    data = encode_document(document)
    digest = hashlib.sha256(data).hexdigest()
    if target.exists():
        return _adopt(target, digest, len(data))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SidecarError("SIDECAR_DIR_UNWRITABLE", str(exc)) from exc
    temporary = target.with_name(f"{target.name}.{os.getpid()}.{digest[:8]}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            if durable:
                fsync(handle.fileno())
        replace(str(temporary), str(target))
    except OSError as exc:
        _discard(temporary)
        raise SidecarError("SIDECAR_WRITE_FAILED", str(exc)) from exc
    except BaseException:
        _discard(temporary)
        raise
    if durable:
        _fsync_dir(target.parent)
    logger.debug("installed evidence sidecar %s (%d bytes)", target, len(data))
    return InstalledSidecar(path=target, sha256=digest, bytes=len(data))


def _adopt(target: Path, digest: str, size: int) -> InstalledSidecar:
    """Accept an already-installed sidecar, or refuse one that differs."""
    try:
        existing = target.read_bytes()
    except OSError as exc:
        raise SidecarError("SIDECAR_UNREADABLE", str(exc)) from exc
    installed = hashlib.sha256(existing).hexdigest()
    if installed != digest:
        raise SidecarError(
            "SIDECAR_COLLISION",
            f"{target} already holds a different evidence record",
        )
    logger.debug("evidence sidecar %s is already installed", target)
    return InstalledSidecar(path=target, sha256=installed, bytes=len(existing))


def _discard(path: Path) -> None:
    """Remove a temporary file, ignoring a failure to do so."""
    try:
        path.unlink()
    except OSError:  # pragma: no cover - the temporary is already gone
        pass


def _fsync_dir(directory: Path) -> None:
    """Flush a directory entry where the platform supports it.

    Windows has no directory handle to flush, which is the same accommodation camera-adapter
    makes; the rename is still atomic there.
    """
    if os.name == "nt":
        return
    try:  # pragma: no cover - POSIX only
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - POSIX only
        return
    try:  # pragma: no cover - POSIX only
        fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)
