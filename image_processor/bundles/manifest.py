"""Reading and verifying a bundle ``manifest.json`` (DESIGN.md section 8, LLD section 4).

The manifest is the bundle contract: it declares the model identity, the per-file SHA-256
digests, the runtime and provider requirements, the tensor shapes, the task family and its
parameters, the preprocessing and decision rules, the result bounds, the warmup samples and
tolerances, the engine compatibility keys, the provenance, and the signing key id.

Loading a manifest validates it against the bundle-manifest JSON Schema and then verifies every
file the manifest declares. A file the manifest does not declare is refused as well: the
signature covers ``manifest.json`` alone, so undeclared content in a bundle would be unverified
content ("bundles are verified, never trusted", AGENTS.md).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from ..types import BundleManifest, Family, TensorSpec
from .archive import BundleError, _safe_relative_name, sha256_file

logger = logging.getLogger(__name__)

#: The manifest is always at the root of the bundle.
MANIFEST_NAME = "manifest.json"

#: The detached signature over ``manifest.json``, when the publisher signed the bundle.
SIGNATURE_NAME = "manifest.sig"

#: Conventional name of the ONNX graph inside a bundle (DESIGN.md section 8).
MODEL_FILE_NAME = "model.onnx"

#: The bundle-manifest contract, owned by WP1. ``load_manifest`` reads it from here unless the
#: caller passes another path.
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "model-bundle-manifest.schema.json"
)

_SCHEMA_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the bundle-manifest JSON Schema.

    Args:
        schema_path: The schema to load. Defaults to ``DEFAULT_SCHEMA_PATH``.

    Returns:
        The parsed schema document.

    Raises:
        BundleError: ``SCHEMA_UNAVAILABLE`` when the schema is missing or is not valid JSON.
    """
    path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = _SCHEMA_CACHE.get(key)
        if cached is None:
            cached = json.loads(path.read_text(encoding="utf-8"))
            _SCHEMA_CACHE[key] = cached
        return cached
    except OSError as exc:
        raise BundleError(
            "SCHEMA_UNAVAILABLE", f"cannot read the bundle manifest schema at {path}: {exc}"
        ) from exc
    except ValueError as exc:
        raise BundleError(
            "SCHEMA_UNAVAILABLE", f"the bundle manifest schema at {path} is not valid JSON: {exc}"
        ) from exc


def validate_document(document: Any, schema_path: Optional[Path] = None) -> None:
    """Validate a manifest document against the bundle-manifest schema.

    Args:
        document: The parsed ``manifest.json``.
        schema_path: The schema to validate against. Defaults to ``DEFAULT_SCHEMA_PATH``.

    Raises:
        BundleError: ``MANIFEST_INVALID`` when the document does not satisfy the schema,
            ``SCHEMA_UNAVAILABLE`` when the schema itself cannot be read.
    """
    schema = load_schema(schema_path)
    try:
        jsonschema.validate(instance=document, schema=schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "(root)"
        raise BundleError("MANIFEST_INVALID", f"{location}: {exc.message}") from exc
    except jsonschema.SchemaError as exc:
        raise BundleError("SCHEMA_UNAVAILABLE", f"the manifest schema is invalid: {exc}") from exc


def _tensor_specs(entries: Any, where: str) -> List[TensorSpec]:
    """Build ``TensorSpec`` values from the manifest's ``inputs`` or ``outputs`` list."""
    specs: List[TensorSpec] = []
    if not isinstance(entries, list):
        raise BundleError("MANIFEST_INVALID", f"{where} must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise BundleError("MANIFEST_INVALID", f"{where}[{index}] must be an object")
        try:
            name = entry["name"]
            dtype = entry["dtype"]
            shape = entry["shape"]
        except KeyError as exc:
            raise BundleError(
                "MANIFEST_INVALID", f"{where}[{index}] is missing {exc.args[0]!r}"
            ) from exc
        if not isinstance(shape, list) or not all(isinstance(dim, (int, str)) for dim in shape):
            raise BundleError(
                "MANIFEST_INVALID",
                f"{where}[{index}].shape must be a list of integers and dynamic-axis names",
            )
        specs.append(TensorSpec(name=str(name), dtype=str(dtype), shape=tuple(shape)))
    return specs


def _declared_files(document: Dict[str, Any]) -> Dict[str, str]:
    """Return the manifest's ``files`` map with every key checked for path safety."""
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("MANIFEST_INVALID", "files must be a non-empty object of path to sha256")
    checked: Dict[str, str] = {}
    for raw_path, digest in files.items():
        try:
            relative = _safe_relative_name(str(raw_path))
        except BundleError as exc:
            raise BundleError("MANIFEST_INVALID", f"files: {exc.message}") from exc
        key = relative.as_posix()
        if key == MANIFEST_NAME:
            raise BundleError("MANIFEST_INVALID", "files cannot declare the digest of manifest.json")
        if not isinstance(digest, str):
            raise BundleError("MANIFEST_INVALID", f"files[{key!r}] must be a sha256 hex string")
        checked[key] = digest.strip().lower()
    return checked


def parse_manifest(document: Dict[str, Any]) -> BundleManifest:
    """Build a ``BundleManifest`` from an already-validated manifest document.

    Fields the schema does not require fall back to empty values, so the schema stays the single
    authority on what a bundle must declare.

    Args:
        document: The parsed ``manifest.json``.

    Returns:
        The parsed manifest.

    Raises:
        BundleError: ``MANIFEST_INVALID`` when a declared value has the wrong shape.
    """
    if not isinstance(document, dict):
        raise BundleError("MANIFEST_INVALID", "manifest.json must be a JSON object")
    try:
        family = Family(document["family"])
    except KeyError as exc:
        raise BundleError("MANIFEST_INVALID", "manifest is missing 'family'") from exc
    except ValueError as exc:
        supported = ", ".join(item.value for item in Family)
        raise BundleError(
            "MANIFEST_INVALID",
            f"family {document['family']!r} is not a supported task family ({supported})",
        ) from exc
    missing = [key for key in ("schemaVersion", "modelId", "version") if key not in document]
    if missing:
        raise BundleError("MANIFEST_INVALID", f"manifest is missing {', '.join(missing)}")
    return BundleManifest(
        schema_version=int(document["schemaVersion"]),
        model_id=str(document["modelId"]),
        version=str(document["version"]),
        files=_declared_files(document),
        min_onnxruntime=str(document.get("minOnnxRuntime", "")),
        providers_permitted=list(document.get("providersPermitted", [])),
        provider_policy=str(document.get("providerPolicy", "")),
        inputs=_tensor_specs(document.get("inputs", []), "inputs"),
        outputs=_tensor_specs(document.get("outputs", []), "outputs"),
        dynamic_batch=bool(document.get("dynamicBatch", False)),
        family=family,
        family_params=dict(document.get("familyParams", {})),
        preprocess=dict(document.get("preprocess", {})),
        decision_rules=dict(document.get("decisionRules", {})),
        max_result_items=int(document.get("maxResultItems", 0)),
        estimated_device_mib=int(document.get("estimatedDeviceMiB", 0)),
        warmup=list(document.get("warmup", [])),
        tolerances=dict(document.get("tolerances", {})),
        compatibility_keys=dict(document.get("compatibilityKeys", {})),
        provenance=dict(document.get("provenance", {})),
        key_id=document.get("keyId"),
        transform_version=str(document.get("transformVersion", "")),
    )


def read_manifest_bytes(bundle_dir: Path) -> bytes:
    """Read the exact bytes of ``manifest.json`` from an extracted bundle.

    The signature covers these bytes, so they are never re-serialized.

    Args:
        bundle_dir: The root of the extracted bundle.

    Returns:
        The manifest bytes.

    Raises:
        BundleError: ``MANIFEST_MISSING`` when the bundle has no ``manifest.json`` at its root.
    """
    path = bundle_dir / MANIFEST_NAME
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BundleError(
            "MANIFEST_MISSING", f"the bundle has no {MANIFEST_NAME} at its root: {exc}"
        ) from exc


def _verify_declared_files(bundle_dir: Path, manifest: BundleManifest, verify_files: bool) -> None:
    """Check that every declared file is present and, when asked, that it hashes as declared."""
    for relative, expected in manifest.files.items():
        path = bundle_dir / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise BundleError("FILE_MISSING", f"the bundle does not contain {relative!r}")
        if not verify_files:
            continue
        actual = sha256_file(path)
        if actual != expected:
            raise BundleError(
                "FILE_DIGEST_MISMATCH",
                f"{relative} hashes to sha256:{actual}, manifest declares sha256:{expected}",
            )


def _reject_undeclared_files(bundle_dir: Path, manifest: BundleManifest) -> None:
    """Refuse a bundle that carries a file the manifest does not declare."""
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle_dir).as_posix()
        if relative in (MANIFEST_NAME, SIGNATURE_NAME) or relative in manifest.files:
            continue
        raise BundleError(
            "FILE_UNDECLARED",
            f"the bundle carries {relative!r}, which manifest.json does not declare",
        )


def resolve_model_path(bundle_dir: Path, manifest: BundleManifest) -> Path:
    """Locate the ONNX graph inside an extracted bundle.

    The conventional name is ``model.onnx``; a bundle that names its graph differently is
    accepted when exactly one declared file has the ``.onnx`` suffix.

    Args:
        bundle_dir: The root of the extracted bundle.
        manifest: The parsed manifest.

    Returns:
        The path of the ONNX graph.

    Raises:
        BundleError: ``MODEL_FILE_MISSING`` when no single ONNX graph can be identified.
    """
    if MODEL_FILE_NAME in manifest.files:
        return bundle_dir / MODEL_FILE_NAME
    candidates = [name for name in manifest.files if name.lower().endswith(".onnx")]
    if len(candidates) == 1:
        return bundle_dir / Path(*PurePosixPath(candidates[0]).parts)
    raise BundleError(
        "MODEL_FILE_MISSING",
        f"the bundle declares {len(candidates)} .onnx files and no {MODEL_FILE_NAME}",
    )


def load_manifest(
    bundle_dir: Path,
    schema_path: Optional[Path] = None,
    verify_files: bool = True,
) -> BundleManifest:
    """Load, validate, and verify the manifest of an extracted bundle.

    The order is the one DESIGN.md section 9 step 3 fixes: schema validation first, then the
    per-file digests. Files the manifest does not declare are refused, and ``manifest.json`` must
    be at the root of the bundle.

    Args:
        bundle_dir: The root of the extracted bundle.
        schema_path: The bundle-manifest schema. Defaults to ``DEFAULT_SCHEMA_PATH``.
        verify_files: Whether to re-hash every declared file. Staging always does; a cache read
            of an already-verified bundle checks presence only.

    Returns:
        The parsed, verified manifest.

    Raises:
        BundleError: ``MANIFEST_MISSING``, ``MANIFEST_INVALID``, ``SCHEMA_UNAVAILABLE``,
            ``FILE_MISSING``, ``FILE_DIGEST_MISMATCH``, or ``FILE_UNDECLARED``.
    """
    raw = read_manifest_bytes(bundle_dir)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError("MANIFEST_INVALID", f"manifest.json is not valid JSON: {exc}") from exc
    validate_document(document, schema_path)
    manifest = parse_manifest(document)
    _verify_declared_files(bundle_dir, manifest, verify_files)
    _reject_undeclared_files(bundle_dir, manifest)
    logger.debug(
        "loaded manifest for %s %s with %d declared files",
        manifest.model_id,
        manifest.version,
        len(manifest.files),
    )
    return manifest
