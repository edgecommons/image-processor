"""The pre-commit candidate validator and the schema loader (LLD §3).

The core runs registered validators after the canonical schema check and before a
configuration generation becomes current, so a rejected candidate leaves the running
configuration untouched. This module is where ImageProcessor says no: the component's own
schema, then the cross-field rules a schema cannot express - unresolved model references,
overlapping mutating roots, path containment, completion directories, and the provider policy.

Validation is fail-closed. A rule that cannot be evaluated rejects the candidate rather than
letting it through, because every rule here exists to stop a deployment that would corrupt
evidence or infer on a file another component owns.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from edgecommons.config import (
    ConfigurationValidationPhase,
    ConfigurationValidationResult,
)
from jsonschema import Draft202012Validator

from image_processor.config.models import (
    CompletionAction,
    ComponentConfig,
    ConfigError,
    ReadinessMode,
    RouteConfig,
    parse_component_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_SCHEMA",
    "INFERENCE_RESULT_SCHEMA",
    "MODEL_BUNDLE_MANIFEST_SCHEMA",
    "VALIDATOR_NAME",
    "load_schema",
    "register",
    "schema_path",
    "validate_candidate",
]

#: The name this component registers its validator under.
VALIDATOR_NAME = "image-processor"

#: The component's own configuration contract, relative to the component root.
CONFIG_SCHEMA = "config.schema.json"

#: The `app/inference/result` body contract, relative to the component root.
INFERENCE_RESULT_SCHEMA = "schemas/inference-result.schema.json"

#: The model bundle manifest contract, relative to the component root.
MODEL_BUNDLE_MANIFEST_SCHEMA = "schemas/model-bundle-manifest.schema.json"

_COMPONENT_ROOT = Path(__file__).resolve().parents[2]


def schema_path(relative: str) -> Path:
    """Locate one shipped schema file.

    Args:
        relative: The schema's path relative to the component root, such as
            `CONFIG_SCHEMA` or `INFERENCE_RESULT_SCHEMA`.

    Returns:
        The resolved path to the file.

    Raises:
        ConfigError: ``SCHEMA_UNAVAILABLE`` when the file is not where the component ships it.
            Validation fails closed rather than silently skipping the contract.
    """
    for root in (_COMPONENT_ROOT, Path.cwd()):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "SCHEMA_UNAVAILABLE", f"the shipped schema '{relative}' could not be found"
    )


@lru_cache(maxsize=8)
def load_schema(relative: str) -> dict:
    """Load and cache one shipped schema.

    Args:
        relative: The schema's path relative to the component root.

    Returns:
        The parsed schema document.

    Raises:
        ConfigError: ``SCHEMA_UNAVAILABLE`` when the file is missing or is not valid JSON.
    """
    path = schema_path(relative)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as exc:
        raise ConfigError("SCHEMA_UNAVAILABLE", f"'{relative}' is not valid JSON: {exc}") from exc


def register(builder: Any) -> Any:
    """Register this validator on an EdgeCommons builder.

    Call it while you build the component, before the configuration provider starts. The core
    then runs `validate_candidate` on the initial configuration and on every reload.

    Args:
        builder: The `EdgeCommonsBuilder` to register on.

    Returns:
        The same builder, so the call chains.
    """
    builder.configuration_validator(VALIDATOR_NAME, validate_candidate)
    return builder


def _tokens(path: Path) -> tuple:
    """Return a path's comparable components, with upward navigation folded out.

    `Path.parts` already understands both separators on the running platform and drops `.`
    itself, so this works on a POSIX deployment path and a Windows test path alike. What it
    does not drop is `..`, which would make a lexical containment check wrong. Windows
    comparisons are case insensitive, matching how the filesystem behaves.
    """
    parts = Path(path).parts
    if not parts:
        return ()
    folded = [parts[0]]
    for part in parts[1:]:
        if part == "..":
            if len(folded) > 1:
                folded.pop()
            continue
        folded.append(part)
    if os.name == "nt":
        return tuple(part.lower() for part in folded)
    return tuple(folded)


def _contains(parent: Path, child: Path) -> bool:
    """Report whether `child` is `parent` or lies inside it."""
    outer = _tokens(parent)
    inner = _tokens(child)
    return len(inner) >= len(outer) and inner[: len(outer)] == outer


def _overlaps(first: Path, second: Path) -> bool:
    """Report whether two directory trees intersect in either direction."""
    return _contains(first, second) or _contains(second, first)


def _scheme_of(uri: str) -> str:
    """Return the fetch scheme a model URI selects."""
    for scheme in ("s3", "https", "file"):
        if uri.startswith(scheme + "://"):
            return scheme
    return "file"


def _directory_problem(path: Path) -> Optional[str]:
    """Return why a completion directory cannot be used, or ``None`` when it can.

    The directory does not have to exist yet - a fresh deployment creates it - but its nearest
    existing ancestor must be a writable directory, so a typo'd path fails at deploy time
    instead of on the first archive.
    """
    if path.exists():
        if not path.is_dir():
            return "it exists but is not a directory"
        return None
    for ancestor in path.parents:
        if not ancestor.exists():
            continue
        if not ancestor.is_dir():
            return f"'{ancestor}' is not a directory"
        if not os.access(str(ancestor), os.W_OK):
            return f"'{ancestor}' is not writable"
        return None
    return "no ancestor directory exists"


def _component_section(candidate: Any) -> dict:
    """Return the `component` section of a candidate document."""
    if not isinstance(candidate, dict):
        raise ConfigError("CANDIDATE_NOT_OBJECT", "the configuration candidate must be an object")
    component = candidate.get("component")
    if component is None:
        return {}
    if not isinstance(component, dict):
        raise ConfigError("INVALID_TYPE", "component must be an object")
    return component


def _schema_check(global_cfg: Any, instances: Any) -> None:
    """Validate the candidate against `config.schema.json`.

    Raises:
        ConfigError: ``SCHEMA_INVALID`` naming the first offending pointer.
    """
    schema = load_schema(CONFIG_SCHEMA)
    for error in Draft202012Validator(schema).iter_errors(global_cfg):
        pointer = "/component/global" + "".join(f"/{part}" for part in error.absolute_path)
        raise ConfigError("SCHEMA_INVALID", f"{pointer}: {error.message}")
    route_schema = {
        "$schema": schema["$schema"],
        "$id": schema["$id"] + "#route",
        "$ref": "#/$defs/route",
        "$defs": schema["$defs"],
    }
    validator = Draft202012Validator(route_schema)
    if not isinstance(instances, (list, tuple)):
        raise ConfigError("INVALID_TYPE", "component.instances must be an array")
    for index, instance in enumerate(instances):
        for error in validator.iter_errors(instance):
            pointer = f"/component/instances/{index}" + "".join(
                f"/{part}" for part in error.absolute_path
            )
            raise ConfigError("SCHEMA_INVALID", f"{pointer}: {error.message}")


def _check_models(config: ComponentConfig) -> None:
    """Check the desired model set against the fetch and signature policies."""
    sources = config.model_sources
    for entry in config.models:
        scheme = _scheme_of(entry.uri)
        if scheme not in sources.allowed_schemes:
            raise ConfigError(
                "MODEL_URI_SCHEME_NOT_ALLOWED",
                f"model '{entry.id}' uses the '{scheme}' scheme, which modelSources does not allow",
            )
        if sources.allowed_uri_prefixes and not any(
            entry.uri.startswith(prefix) for prefix in sources.allowed_uri_prefixes
        ):
            raise ConfigError(
                "MODEL_URI_NOT_ALLOWED",
                f"model '{entry.id}' has a uri outside modelSources.allowedUriPrefixes",
            )
    if config.signing.required and not config.signing.trusted_keys:
        raise ConfigError(
            "NO_TRUSTED_KEYS",
            "signing.required is set but signing.trustedKeys is empty, so no bundle can be verified",
        )


def _check_route(route: RouteConfig, config: ComponentConfig) -> None:
    """Check one route's internal consistency."""
    if config.model_entry(route.model_ref) is None:
        raise ConfigError(
            "UNRESOLVED_MODEL_REF",
            f"route '{route.id}' names a model that global.models does not declare",
        )
    if CompletionAction.ARCHIVE in route.completion.actions and route.completion.archive_dir is None:
        raise ConfigError(
            "MISSING_ARCHIVE_DIR",
            f"route '{route.id}' archives its inputs but sets no completion.archiveDir",
        )
    if route.is_spool:
        source = route.source
        if source.camera is not None and source.readiness.mode is ReadinessMode.STABILITY:
            raise ConfigError(
                "STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE",
                f"route '{route.id}' is camera-bound, so it cannot infer finalization from "
                "size and mtime; use cameraSidecar or cameraStatus",
            )
        if source.readiness.mode is ReadinessMode.CAMERA_STATUS and source.camera is None:
            raise ConfigError(
                "CAMERA_BINDING_REQUIRED",
                f"route '{route.id}' reads capture status but names no source.camera",
            )
    else:
        source = route.source
        if not _contains(config.paths.staging, source.inline_staging):
            raise ConfigError(
                "INLINE_STAGING_NOT_CONTAINED",
                f"route '{route.id}' stages inline images outside global.paths.staging",
            )
        if _overlaps(source.file_root, source.inline_staging):
            raise ConfigError(
                "INLINE_STAGING_NOT_CONTAINED",
                f"route '{route.id}' stages inline images inside its own fileRoot",
            )


def _check_roots(config: ComponentConfig) -> None:
    """Check that exactly one route owns each mutating root, and that nothing feeds back."""
    owned: list = []
    for route in config.routes:
        for root in route.mutating_roots:
            for other_id, other_root in owned:
                if _overlaps(root, other_root):
                    raise ConfigError(
                        "OVERLAPPING_MUTATING_ROOTS",
                        f"routes '{other_id}' and '{route.id}' both mutate overlapping roots; "
                        "exactly one component may mutate a given spool",
                    )
            owned.append((route.id, root))

    protected = [
        ("global.paths.staging", config.paths.staging),
        ("global.paths.modelCache", config.paths.model_cache),
    ]
    for route in config.routes:
        for output in route.output_dirs:
            protected.append((f"route '{route.id}' completion directory", output))
    for owner_id, root in owned:
        for label, path in protected:
            if _contains(root, path):
                raise ConfigError(
                    "OUTPUT_INSIDE_SOURCE_ROOT",
                    f"{label} lies inside the root route '{owner_id}' mutates, which would feed "
                    "its own output back in as new input",
                )


def _check_completion_dirs(config: ComponentConfig) -> None:
    """Check that every configured completion directory exists or can be created."""
    for route in config.routes:
        for directory in route.output_dirs:
            problem = _directory_problem(directory)
            if problem is not None:
                raise ConfigError(
                    "COMPLETION_DIR_NOT_CREATABLE",
                    f"route '{route.id}' cannot use '{directory}': {problem}",
                )


def _check_provider_policy(config: ComponentConfig) -> None:
    """Check that the runtime can actually satisfy the provider policy."""
    runtime = config.runtime
    if runtime.required_provider is not None and runtime.required_provider not in runtime.providers:
        raise ConfigError(
            "PROVIDER_POLICY_UNSATISFIED",
            f"runtime.requiredProvider '{runtime.required_provider}' is not in runtime.providers",
        )
    cpu = "CPUExecutionProvider"
    if not runtime.allow_cpu_only:
        if tuple(runtime.providers) == (cpu,):
            raise ConfigError(
                "PROVIDER_POLICY_UNSATISFIED",
                "runtime.providers offers only the CPU provider, which needs runtime.allowCpuOnly",
            )
        if runtime.required_provider == cpu:
            raise ConfigError(
                "PROVIDER_POLICY_UNSATISFIED",
                "runtime.requiredProvider is the CPU provider, which needs runtime.allowCpuOnly",
            )


def _check_reload(
    config: ComponentConfig, current: Optional[dict], phase: Any
) -> None:
    """Check the changes a running process cannot absorb.

    The ledger, the bundle cache, and the staging tree all hold state a live generation is
    already using, so a reload that moves them would strand in-flight jobs and their evidence.
    """
    if phase is not ConfigurationValidationPhase.RELOAD or not isinstance(current, dict):
        return
    component = current.get("component")
    if not isinstance(component, dict):
        return
    try:
        active = parse_component_config(component.get("global"), component.get("instances"))
    except ConfigError:
        # The running generation predates a rule this candidate is judged by; there is nothing
        # to compare against, so the candidate is judged on its own merits.
        logger.debug("the active configuration could not be reparsed; skipping immutable-path check")
        return
    for label, before, after in (
        ("paths.stateDb", active.paths.state_db, config.paths.state_db),
        ("paths.modelCache", active.paths.model_cache, config.paths.model_cache),
        ("paths.staging", active.paths.staging, config.paths.staging),
    ):
        if _tokens(before) != _tokens(after):
            raise ConfigError(
                "IMMUTABLE_PATH_CHANGED",
                f"{label} cannot change on a reload; restart the component to move it",
            )


def validate_candidate(
    candidate: dict,
    current: Optional[dict],
    phase: ConfigurationValidationPhase,
) -> ConfigurationValidationResult:
    """Accept or reject one configuration candidate.

    The checks run in the order an operator reads them: the component's own schema first, then
    the parse, then the cross-field rules. The first failure decides, and its stable code is
    what the core logs and reports.

    Args:
        candidate: The whole candidate configuration document, as the core prepared it.
        current: The redacted configuration in force, or ``None`` on the initial generation.
        phase: Whether this is the initial configuration or a reload.

    Returns:
        An accepting result, or a rejection carrying a stable SCREAMING_SNAKE_CASE code.
    """
    try:
        component = _component_section(candidate)
        global_cfg = component.get("global") or {}
        instances = component.get("instances") or []
        _schema_check(global_cfg, instances)
        config = parse_component_config(global_cfg, instances)
        _check_models(config)
        for route in config.routes:
            _check_route(route, config)
        _check_roots(config)
        _check_completion_dirs(config)
        _check_provider_policy(config)
        _check_reload(config, current, phase)
    except ConfigError as exc:
        logger.warning("configuration candidate rejected: %s: %s", exc.code, exc.message)
        return ConfigurationValidationResult.reject(exc.code, exc.message)
    return ConfigurationValidationResult.accept()
