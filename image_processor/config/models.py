"""Configuration models and the parser that builds them (LLD §3).

`config.schema.json` is the contract for what a document may contain; this module is the
contract for what the rest of the component sees. Parsing turns one `component.global` object
and one `component.instances[]` list into a frozen `ComponentConfig`: defaults are resolved,
`completionDefaults` are folded into every route, and every `$secret` reference becomes a
`SecretRef` value that is still unresolved. Nothing here touches the filesystem, the vault, or
the network, so a candidate can be parsed before it is committed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from image_processor.types import CompletionAction, Family, ModelRef, SecretRef, SourceKind

__all__ = [
    "CollisionPolicy",
    "CompletionPolicy",
    "ComponentConfig",
    "ConfigError",
    "DecisionSignal",
    "DiscoveryConfig",
    "GpuConfig",
    "ModelActivation",
    "ModelEntry",
    "ModelSourcesConfig",
    "OutputsConfig",
    "Paths",
    "PublishConfig",
    "PublishFailureAction",
    "ReadinessConfig",
    "ReadinessMode",
    "RouteConfig",
    "RuntimeConfig",
    "SchedulerConfig",
    "SigningConfig",
    "SpoolSourceConfig",
    "TriggerSourceConfig",
    "TrustedKey",
    "CameraBinding",
    "MUTATING_ACTIONS",
    "MAX_INLINE_BYTES",
    "parse_component_config",
]

#: The core envelope's binary-body cap. An inline trigger image never exceeds it (D-IP-5).
MAX_INLINE_BYTES = 65536

_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_UNS_TOKEN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MODEL_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_SIGNAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ABSOLUTE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\)")

_UNSET = object()


class ConfigError(ValueError):
    """A configuration document the component refuses to accept.

    The code is stable SCREAMING_SNAKE_CASE so it can be reported verbatim as a candidate
    validator's rejection code, which the core requires to match that shape.

    Attributes:
        code: The stable rejection code.
        message: An operator-safe explanation naming the offending field.
    """

    def __init__(self, code: str, message: str) -> None:
        if not _CODE.fullmatch(code or ""):
            raise ValueError("a ConfigError code must be stable SCREAMING_SNAKE_CASE")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ReadinessMode(str, Enum):
    """How a spool route proves that a file is finalized (DESIGN.md §4.1).

    ``str()`` returns the configured spelling, not the member name, so a consumer that
    normalizes a configuration field with ``str()`` sees the same value whether it was handed a
    parsed route or the raw JSON. ``sources/readiness.py`` does exactly that.
    """

    CAMERA_SIDECAR = "cameraSidecar"
    CAMERA_STATUS = "cameraStatus"
    MARKER = "marker"
    STABILITY = "stability"

    __str__ = str.__str__


class CollisionPolicy(str, Enum):
    """What a completion action does when its target already holds a different object.

    ``FAIL`` records ``CLEANUP_FAILED`` and leaves both files intact. ``SUFFIX`` installs the
    input beside the occupant under a deterministic name derived from its own digest, so the
    move completes without overwriting evidence and the name is reproducible from the ledger
    (DESIGN.md §7). Neither policy ever replaces an existing object.
    """

    FAIL = "fail"
    SUFFIX = "suffix"

    __str__ = str.__str__


class PublishFailureAction(str, Enum):
    """What a route does with its input when publication exhausts its attempts.

    The enum has one member because only one behavior is reachable (D-IP-20). A publish failure
    leaves the result committed and its outbox row pending, so the input has to stay where it is
    until an operator re-drives the publication with ``retry-publication``; moving or removing it
    would take the evidence out from under a publication that is still expected to happen. The
    key stays configurable so a regulated profile can state the behavior rather than inherit it.
    """

    RETAIN_IN_PLACE = "retainInPlace"

    __str__ = str.__str__


#: The configured spelling of each completion action, as `config.schema.json` writes it.
COMPLETION_ACTION_NAMES = {
    "archive": CompletionAction.ARCHIVE,
    "delete": CompletionAction.DELETE,
    "retainInPlace": CompletionAction.RETAIN,
    "quarantine": CompletionAction.QUARANTINE,
}

#: The configured spellings of the collision policies.
_COLLISION_POLICIES = tuple(item.value for item in CollisionPolicy)

#: The configured spellings a publish failure accepts. There is exactly one (D-IP-20).
_PUBLISH_FAILURE_ACTIONS = tuple(item.value for item in PublishFailureAction)

#: The actions that move or remove the input, and therefore claim ownership of its root.
MUTATING_ACTIONS = frozenset(
    {CompletionAction.ARCHIVE, CompletionAction.DELETE, CompletionAction.QUARANTINE}
)


def _object(value: Any, where: str, allowed: "tuple[str, ...]") -> dict:
    """Return `value` as an object whose keys are all known.

    Args:
        value: The parsed JSON value.
        where: A dotted path used in the diagnostic.
        allowed: The keys this object accepts.

    Returns:
        The object itself.

    Raises:
        ConfigError: ``INVALID_TYPE`` when the value is not an object, ``UNKNOWN_KEY`` when it
            carries a key the schema does not declare.
    """
    if not isinstance(value, dict):
        raise ConfigError("INVALID_TYPE", f"{where} must be an object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError("UNKNOWN_KEY", f"{where} has unknown keys: {', '.join(unknown)}")
    return value


def _child(node: dict, key: str, where: str, allowed: "tuple[str, ...]") -> dict:
    """Return one nested object, or an empty object when the key is absent."""
    if key not in node:
        return {}
    return _object(node[key], f"{where}.{key}", allowed)


def _string(
    node: dict,
    key: str,
    where: str,
    *,
    default: Any = _UNSET,
    pattern: "Optional[re.Pattern]" = None,
    choices: "Optional[tuple[str, ...]]" = None,
) -> Any:
    """Return one string field, applying its default, pattern, and value set.

    Raises:
        ConfigError: ``MISSING_FIELD`` when a required field is absent, ``INVALID_TYPE`` when
            the value is not a string, ``INVALID_VALUE`` when it fails the pattern or is not
            one of the permitted values.
    """
    if key not in node or node[key] is None:
        if default is _UNSET:
            raise ConfigError("MISSING_FIELD", f"{where}.{key} is required")
        return default
    value = node[key]
    if not isinstance(value, str) or not value:
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be a non-empty string")
    if pattern is not None and not pattern.fullmatch(value):
        raise ConfigError("INVALID_VALUE", f"{where}.{key} is not a well-formed value")
    if choices is not None and value not in choices:
        raise ConfigError(
            "INVALID_VALUE", f"{where}.{key} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def _bool(node: dict, key: str, where: str, *, default: bool) -> bool:
    """Return one boolean field, applying its default."""
    if key not in node or node[key] is None:
        return default
    value = node[key]
    if not isinstance(value, bool):
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be true or false")
    return value


def _number(
    node: dict,
    key: str,
    where: str,
    *,
    default: Any,
    integer: bool = False,
    minimum: "Optional[float]" = None,
    maximum: "Optional[float]" = None,
) -> Any:
    """Return one numeric field, applying its default and bounds.

    Raises:
        ConfigError: ``INVALID_TYPE`` when the value is not a number of the required kind, and
            ``INVALID_VALUE`` when it falls outside the declared bounds.
    """
    if key not in node or node[key] is None:
        return default
    value = node[key]
    if isinstance(value, bool):
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be a number")
    if integer:
        if not isinstance(value, int):
            raise ConfigError("INVALID_TYPE", f"{where}.{key} must be an integer")
    elif not isinstance(value, (int, float)):
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be a number")
    if minimum is not None and value < minimum:
        raise ConfigError("INVALID_VALUE", f"{where}.{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError("INVALID_VALUE", f"{where}.{key} must be at most {maximum}")
    return value


def _path(node: dict, key: str, where: str, *, default: Any = _UNSET) -> Any:
    """Return one absolute-path field as a `Path`.

    Raises:
        ConfigError: ``MISSING_FIELD`` when a required path is absent, ``INVALID_TYPE`` when it
            is not a string, and ``PATH_NOT_ABSOLUTE`` when it is relative. Containment between
            paths is a cross-field rule and lives in the candidate validator.
    """
    if key not in node or node[key] is None:
        if default is _UNSET:
            raise ConfigError("MISSING_FIELD", f"{where}.{key} is required")
        return None if default is None else Path(default)
    value = node[key]
    if not isinstance(value, str) or not value:
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be a non-empty string")
    if not _ABSOLUTE.match(value):
        raise ConfigError("PATH_NOT_ABSOLUTE", f"{where}.{key} must be an absolute path")
    return Path(value)


def _strings(node: dict, key: str, where: str, *, default: Any, min_items: int = 0) -> tuple:
    """Return one array-of-strings field as a tuple, applying its default.

    Raises:
        ConfigError: ``INVALID_TYPE`` when the value is not an array of non-empty strings, and
            ``INVALID_VALUE`` when it holds fewer entries than the schema requires.
    """
    if key not in node or node[key] is None:
        value = list(default)
    else:
        value = node[key]
    if not isinstance(value, (list, tuple)):
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be an array of strings")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ConfigError("INVALID_TYPE", f"{where}.{key} must be an array of strings")
    if len(value) < min_items:
        raise ConfigError("INVALID_VALUE", f"{where}.{key} needs at least {min_items} entries")
    return tuple(value)


def _array(node: dict, key: str, where: str) -> list:
    """Return one array field, or an empty list when the key is absent."""
    if key not in node or node[key] is None:
        return []
    value = node[key]
    if not isinstance(value, list):
        raise ConfigError("INVALID_TYPE", f"{where}.{key} must be an array")
    return value


def _secret_ref(value: Any, where: str) -> SecretRef:
    """Return one `$secret` reference without resolving it.

    Raises:
        ConfigError: ``INVALID_SECRET_REF`` when the value is not a well-formed reference.
    """
    try:
        return SecretRef.parse(value)
    except ValueError as exc:
        raise ConfigError("INVALID_SECRET_REF", f"{where}: {exc}") from exc


def _completion_action(node: dict, key: str, where: str, *, default: Any) -> CompletionAction:
    """Return one completion action, mapping the configured spelling onto the enum."""
    name = _string(node, key, where, default=None, choices=tuple(COMPLETION_ACTION_NAMES))
    if name is None:
        return default
    return COMPLETION_ACTION_NAMES[name]


def _publish_failure_action(node: dict, key: str, where: str) -> CompletionAction:
    """Return the publish-failure completion, refusing any value that cannot take effect.

    Args:
        node: The completion block being parsed.
        key: The key to read, always ``onPublishFailure``.
        where: A dotted path used in the diagnostic.

    Returns:
        :data:`~image_processor.types.CompletionAction.RETAIN`, whether the key was written out
        or left to its default.

    Raises:
        ConfigError: ``ON_PUBLISH_FAILURE_NOT_SUPPORTED`` when the value is anything but
            ``retainInPlace``. The ledger has no cleanup edge out of ``PUBLISH_EXHAUSTED``, so a
            mutating value would be configuration the component silently ignores (D-IP-20).
    """
    value = node.get(key)
    if value is None:
        return CompletionAction.RETAIN
    if value not in _PUBLISH_FAILURE_ACTIONS:
        raise ConfigError(
            "ON_PUBLISH_FAILURE_NOT_SUPPORTED",
            f"{where}.{key} is {value!r}; only "
            f"{PublishFailureAction.RETAIN_IN_PLACE.value!r} takes effect, because a publish "
            "failure keeps the result and its outbox row and waits for retry-publication",
        )
    return CompletionAction.RETAIN


@dataclass(frozen=True)
class Paths:
    """Where the component keeps the state it owns."""

    state_db: Path
    model_cache: Path
    staging: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Which execution providers sessions may use, and how much loading runs at once."""

    providers: tuple
    required_provider: Optional[str]
    allow_cpu_only: bool
    executor_cells_per_gpu: int
    load_concurrency_per_gpu: int


@dataclass(frozen=True)
class GpuConfig:
    """Device selection and the device-memory budget."""

    devices: tuple
    resident_memory_budget_percent: int
    reserve_mib: int


@dataclass(frozen=True)
class SchedulerConfig:
    """Queueing, batching, residency, and the retry budget."""

    max_batch_size: int
    max_batch_latency_ms: int
    hot_ttl_secs: int
    min_residency_secs: int
    max_attempts: int
    retry_backoff_secs: float
    max_retry_backoff_secs: float
    queue_age_warning_secs: int


@dataclass(frozen=True)
class PublishConfig:
    """Confirmed publication of the cleanup-gating result."""

    confirmation_timeout_secs: float
    max_attempts: int
    outbox_capacity: int
    outbox_reserve_budget_mib: int

    @property
    def outbox_reserve_budget_bytes(self) -> int:
        """The reservation budget in bytes, as the ledger takes it."""
        return self.outbox_reserve_budget_mib * 1024 * 1024


@dataclass(frozen=True)
class DiscoveryConfig:
    """How often a spool route walks its root, and how long it debounces notifications."""

    rescan_secs: int
    debounce_ms: int

    @property
    def debounce_secs(self) -> float:
        """The debounce window in seconds, as the spool source takes it."""
        return self.debounce_ms / 1000.0


@dataclass(frozen=True)
class TrustedKey:
    """One Ed25519 public key the component accepts signatures from."""

    key_id: str
    public_key: Union[str, SecretRef]


@dataclass(frozen=True)
class SigningConfig:
    """The bundle signature policy."""

    required: bool
    trusted_keys: tuple

    def key(self, key_id: str) -> Optional[TrustedKey]:
        """Return the trusted key with this id, or ``None`` when it is not trusted."""
        for entry in self.trusted_keys:
            if entry.key_id == key_id:
                return entry
        return None


@dataclass(frozen=True)
class ModelSourcesConfig:
    """The transport controls that bound where bundles may be fetched from."""

    allowed_schemes: tuple
    allowed_uri_prefixes: tuple
    verify_tls: bool


@dataclass(frozen=True)
class ModelActivation:
    """When a staged bundle may take over a route."""

    require_warmup: bool
    retain_for_rollback: bool


@dataclass(frozen=True)
class ModelEntry:
    """One desired model bundle: an immutable reference plus how to fetch it."""

    id: str
    version: str
    digest: str
    uri: str
    credentials_ref: Optional[SecretRef]
    activation: ModelActivation

    @property
    def ref(self) -> ModelRef:
        """Return this entry's immutable identity."""
        return ModelRef(id=self.id, version=self.version, digest=self.digest)


@dataclass(frozen=True)
class CompletionPolicy:
    """What happens to the input after a job reaches a terminal outcome.

    A route's resolved policy satisfies the structural protocol
    ``image_processor.completion.actions.CompletionPolicy`` that WP3's ``Completer`` consumes:
    ``source_root`` is the directory the job's relative path resolves under, and the four action
    fields plus ``on_collision`` and the two directories say what to do with it. Every use site
    in ``completion/`` wraps a path in ``Path(...)``, so the ``Path`` values here satisfy a
    protocol that annotates them as ``str``.

    ``source_root`` is ``None`` on ``ComponentConfig.completion_defaults``, which is a template
    every route inherits from rather than a policy any route runs under.
    """

    on_success: CompletionAction
    on_invalid_input: CompletionAction
    on_operational_failure: CompletionAction
    #: Always ``RETAIN``: the only publish-failure completion that takes effect (D-IP-20).
    on_publish_failure: CompletionAction
    on_collision: CollisionPolicy
    source_root: Optional[Path] = None
    archive_dir: Optional[Path] = None
    failed_dir: Optional[Path] = None

    def with_source_root(self, root: Path) -> "CompletionPolicy":
        """Return this policy resolved against another root.

        Args:
            root: The directory a job's relative path resolves under.

        Returns:
            A copy carrying that root.
        """
        return replace(self, source_root=root)

    @property
    def actions(self) -> tuple:
        """Return every action this policy can take."""
        return (
            self.on_success,
            self.on_invalid_input,
            self.on_operational_failure,
            self.on_publish_failure,
        )

    @property
    def mutates(self) -> bool:
        """Report whether any action moves or removes the input.

        A route that mutates claims ownership of its root, and no other route may claim an
        overlapping one.
        """
        return any(action in MUTATING_ACTIONS for action in self.actions)


@dataclass(frozen=True)
class ReadinessConfig:
    """How a spool route proves that a file is finalized."""

    mode: ReadinessMode
    quiet_secs: float
    marker_suffix: str


@dataclass(frozen=True)
class CameraBinding:
    """The camera whose spool a route reads."""

    component: str
    instance: str
    subscribe_announcements: bool
    reconcile_capture_status_secs: int


@dataclass(frozen=True)
class SpoolSourceConfig:
    """A directory of finalized image files the component owns."""

    root: Path
    include: tuple
    exclude: tuple
    readiness: ReadinessConfig
    camera: Optional[CameraBinding] = None

    #: The discriminator this source is selected by.
    kind: str = "spool"


@dataclass(frozen=True)
class TriggerSourceConfig:
    """Topic filters whose messages each carry one image."""

    subscribe: tuple
    file_root: Path
    inline_staging: Path
    max_inline_bytes: int

    #: The discriminator this source is selected by.
    kind: str = "trigger"


@dataclass(frozen=True)
class DecisionSignal:
    """One normalized reading mirrored onto `data/<signalId>`."""

    id: str
    value: str


@dataclass(frozen=True)
class OutputsConfig:
    """What a route emits besides the authoritative result."""

    write_result_sidecar: bool
    decision_signals: tuple


@dataclass(frozen=True)
class RouteConfig:
    """One route: one input source bound to one immutable model generation."""

    id: str
    enabled: bool
    priority: int
    source: Union[SpoolSourceConfig, TriggerSourceConfig]
    model_ref: ModelRef
    outputs: OutputsConfig
    completion: CompletionPolicy
    reprocess_existing_on_model_change: bool

    @property
    def is_spool(self) -> bool:
        """Report whether this route reads a watched directory."""
        return isinstance(self.source, SpoolSourceConfig)

    @property
    def is_trigger(self) -> bool:
        """Report whether this route reads subscription messages."""
        return isinstance(self.source, TriggerSourceConfig)

    @property
    def input_root(self) -> Path:
        """Return the directory this route's inputs live under."""
        if isinstance(self.source, SpoolSourceConfig):
            return self.source.root
        return self.source.file_root

    @property
    def mutating_roots(self) -> tuple:
        """Return the roots this route claims exclusive ownership of.

        A route that only retains its inputs claims nothing, so two such routes may read the
        same directory.
        """
        if not self.completion.mutates:
            return ()
        return (self.input_root,)

    def completion_for(self, kind: "SourceKind") -> CompletionPolicy:
        """Return the completion policy resolved for one input's source kind.

        A spool input, and an inline trigger image staged by the component, both resolve under
        the route's own root. A trigger input that arrived as a file reference resolves under
        ``fileRoot`` instead, which is a different directory, so the policy handed to the
        completer carries the root that input actually came from.

        Args:
            kind: The source kind of the input being completed.

        Returns:
            The policy, with ``source_root`` set for that kind.
        """
        if kind is SourceKind.REFERENCE and isinstance(self.source, TriggerSourceConfig):
            return self.completion.with_source_root(self.source.file_root)
        return self.completion

    @property
    def output_dirs(self) -> tuple:
        """Return the directories this route writes completed inputs into."""
        dirs = []
        if self.completion.archive_dir is not None:
            dirs.append(self.completion.archive_dir)
        if self.completion.failed_dir is not None:
            dirs.append(self.completion.failed_dir)
        return tuple(dirs)


@dataclass(frozen=True)
class ComponentConfig:
    """The whole of `component.global` plus every parsed route."""

    paths: Paths
    runtime: RuntimeConfig
    gpu: GpuConfig
    scheduler: SchedulerConfig
    discovery: DiscoveryConfig
    publish: PublishConfig
    signing: SigningConfig
    model_sources: ModelSourcesConfig
    models: tuple
    completion_defaults: CompletionPolicy
    routes: tuple

    def route(self, route_id: str) -> Optional[RouteConfig]:
        """Return the route with this id, or ``None`` when there is none."""
        for route in self.routes:
            if route.id == route_id:
                return route
        return None

    def model_entry(self, ref: ModelRef) -> Optional[ModelEntry]:
        """Return the model entry a reference names, or ``None`` when it is unresolved."""
        for entry in self.models:
            if entry.id == ref.id and entry.version == ref.version and entry.digest == ref.digest:
                return entry
        return None

    @property
    def enabled_routes(self) -> tuple:
        """Return the routes that claim new work."""
        return tuple(route for route in self.routes if route.enabled)


#: The execution providers `config.schema.json` accepts.
EXECUTION_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

_GLOBAL_KEYS = (
    "paths",
    "runtime",
    "gpu",
    "scheduler",
    "discovery",
    "publish",
    "signing",
    "modelSources",
    "models",
    "completionDefaults",
)
_ROUTE_KEYS = (
    "id",
    "enabled",
    "priority",
    "source",
    "modelRef",
    "outputs",
    "completion",
    "reprocessExistingOnModelChange",
)


def _parse_paths(global_cfg: dict) -> Paths:
    """Build the durable-state locations, applying the schema's defaults."""
    node = _child(global_cfg, "paths", "global", ("stateDb", "modelCache", "staging"))
    return Paths(
        state_db=_path(
            node, "stateDb", "global.paths",
            default="/var/lib/edgecommons/image-processor/state.db",
        ),
        model_cache=_path(
            node, "modelCache", "global.paths",
            default="/var/lib/edgecommons/image-processor/models",
        ),
        staging=_path(
            node, "staging", "global.paths",
            default="/var/lib/edgecommons/image-processor/staging",
        ),
    )


def _parse_runtime(global_cfg: dict) -> RuntimeConfig:
    """Build the execution profile, applying the schema's defaults."""
    node = _child(
        global_cfg, "runtime", "global",
        ("providers", "requiredProvider", "allowCpuOnly", "executorCellsPerGpu",
         "loadConcurrencyPerGpu"),
    )
    providers = _strings(
        node, "providers", "global.runtime",
        default=("CUDAExecutionProvider",), min_items=1,
    )
    for name in providers:
        if name not in EXECUTION_PROVIDERS:
            raise ConfigError(
                "INVALID_VALUE",
                f"global.runtime.providers names an unknown execution provider: {name}",
            )
    required = _string(
        node, "requiredProvider", "global.runtime",
        default="CUDAExecutionProvider", choices=EXECUTION_PROVIDERS,
    )
    if "requiredProvider" in node and node["requiredProvider"] is None:
        required = None
    return RuntimeConfig(
        providers=providers,
        required_provider=required,
        allow_cpu_only=_bool(node, "allowCpuOnly", "global.runtime", default=False),
        executor_cells_per_gpu=_number(
            node, "executorCellsPerGpu", "global.runtime",
            default=1, integer=True, minimum=1,
        ),
        load_concurrency_per_gpu=_number(
            node, "loadConcurrencyPerGpu", "global.runtime",
            default=1, integer=True, minimum=1,
        ),
    )


def _parse_gpu(global_cfg: dict) -> GpuConfig:
    """Build the device selection and memory budget, applying the schema's defaults."""
    node = _child(
        global_cfg, "gpu", "global",
        ("devices", "residentMemoryBudgetPercent", "reserveMiB"),
    )
    return GpuConfig(
        # An explicit empty list is the CPU-only development path (WP6): the supervisor runs a
        # single CPU cell, and `runtime.allowCpuOnly` is what makes that legal. An omitted
        # `devices` still defaults to the single-GPU deployment.
        devices=_strings(node, "devices", "global.gpu", default=("0",), min_items=0),
        resident_memory_budget_percent=_number(
            node, "residentMemoryBudgetPercent", "global.gpu",
            default=80, integer=True, minimum=1, maximum=100,
        ),
        reserve_mib=_number(
            node, "reserveMiB", "global.gpu", default=2048, integer=True, minimum=0
        ),
    )


def _parse_scheduler(global_cfg: dict) -> SchedulerConfig:
    """Build the scheduling profile, applying the schema's defaults."""
    where = "global.scheduler"
    node = _child(
        global_cfg, "scheduler", "global",
        ("maxBatchSize", "maxBatchLatencyMs", "hotTtlSecs", "minResidencySecs", "maxAttempts",
         "retryBackoffSecs", "maxRetryBackoffSecs", "queueAgeWarningSecs"),
    )
    return SchedulerConfig(
        max_batch_size=_number(node, "maxBatchSize", where, default=1, integer=True, minimum=1),
        max_batch_latency_ms=_number(
            node, "maxBatchLatencyMs", where, default=20, integer=True, minimum=0
        ),
        hot_ttl_secs=_number(node, "hotTtlSecs", where, default=120, integer=True, minimum=0),
        min_residency_secs=_number(
            node, "minResidencySecs", where, default=15, integer=True, minimum=0
        ),
        max_attempts=_number(node, "maxAttempts", where, default=5, integer=True, minimum=1),
        retry_backoff_secs=_number(node, "retryBackoffSecs", where, default=2.0, minimum=0),
        max_retry_backoff_secs=_number(
            node, "maxRetryBackoffSecs", where, default=300.0, minimum=0
        ),
        queue_age_warning_secs=_number(
            node, "queueAgeWarningSecs", where, default=300, integer=True, minimum=1
        ),
    )


def _parse_publish(global_cfg: dict) -> PublishConfig:
    """Build the publication policy, applying the schema's defaults."""
    where = "global.publish"
    node = _child(
        global_cfg, "publish", "global",
        ("confirmationTimeoutSecs", "maxAttempts", "outboxCapacity", "outboxReserveBudgetMiB"),
    )
    return PublishConfig(
        confirmation_timeout_secs=_number(
            node, "confirmationTimeoutSecs", where, default=10.0, minimum=0
        ),
        max_attempts=_number(node, "maxAttempts", where, default=100, integer=True, minimum=1),
        outbox_capacity=_number(
            node, "outboxCapacity", where, default=100000, integer=True, minimum=1
        ),
        outbox_reserve_budget_mib=_number(
            node, "outboxReserveBudgetMiB", where, default=256, integer=True, minimum=1
        ),
    )


def _parse_discovery(global_cfg: dict) -> DiscoveryConfig:
    """Build the spool-walk cadence, applying the schema's defaults."""
    where = "global.discovery"
    node = _child(global_cfg, "discovery", "global", ("rescanSecs", "debounceMs"))
    return DiscoveryConfig(
        rescan_secs=_number(node, "rescanSecs", where, default=60, integer=True, minimum=1),
        debounce_ms=_number(node, "debounceMs", where, default=500, integer=True, minimum=0),
    )


def _parse_signing(global_cfg: dict) -> SigningConfig:
    """Build the signature policy and resolve every key reference into a `SecretRef`."""
    node = _child(global_cfg, "signing", "global", ("required", "trustedKeys"))
    keys = []
    seen = set()
    for index, raw in enumerate(_array(node, "trustedKeys", "global.signing")):
        where = f"global.signing.trustedKeys[{index}]"
        entry = _object(raw, where, ("keyId", "publicKey"))
        key_id = _string(entry, "keyId", where, pattern=_KEY_ID)
        if key_id in seen:
            raise ConfigError(
                "DUPLICATE_KEY_ID", f"global.signing.trustedKeys names '{key_id}' twice"
            )
        seen.add(key_id)
        material = entry.get("publicKey")
        if SecretRef.is_ref(material):
            public_key: Union[str, SecretRef] = _secret_ref(material, f"{where}.publicKey")
        elif isinstance(material, str) and material:
            public_key = material
        else:
            raise ConfigError(
                "INVALID_TYPE",
                f"{where}.publicKey must be a non-empty string or a $secret reference",
            )
        keys.append(TrustedKey(key_id=key_id, public_key=public_key))
    return SigningConfig(
        required=_bool(node, "required", "global.signing", default=False),
        trusted_keys=tuple(keys),
    )


def _parse_model_sources(global_cfg: dict) -> ModelSourcesConfig:
    """Build the fetch controls, applying the schema's defaults."""
    where = "global.modelSources"
    node = _child(
        global_cfg, "modelSources", "global",
        ("allowedSchemes", "allowedUriPrefixes", "verifyTls"),
    )
    schemes = _strings(
        node, "allowedSchemes", where, default=("s3", "https", "file"), min_items=1
    )
    for scheme in schemes:
        if scheme not in ("s3", "https", "file"):
            raise ConfigError("INVALID_VALUE", f"{where}.allowedSchemes names an unknown scheme: {scheme}")
    return ModelSourcesConfig(
        allowed_schemes=schemes,
        allowed_uri_prefixes=_strings(node, "allowedUriPrefixes", where, default=()),
        verify_tls=_bool(node, "verifyTls", where, default=True),
    )


def _parse_models(global_cfg: dict) -> tuple:
    """Build the desired model set, resolving credential references without opening them."""
    entries = []
    seen = set()
    for index, raw in enumerate(_array(global_cfg, "models", "global")):
        where = f"global.models[{index}]"
        node = _object(raw, where, ("id", "version", "digest", "uri", "credentials", "activation"))
        model_id = _string(node, "id", where, pattern=_MODEL_ID)
        version = _string(node, "version", where, pattern=_MODEL_VERSION)
        if (model_id, version) in seen:
            raise ConfigError(
                "DUPLICATE_MODEL_ID",
                f"global.models declares '{model_id}' version '{version}' twice",
            )
        seen.add((model_id, version))
        credentials = node.get("credentials")
        activation_node = _child(node, "activation", where, ("requireWarmup", "retainForRollback"))
        entries.append(
            ModelEntry(
                id=model_id,
                version=version,
                digest=_string(node, "digest", where, pattern=_DIGEST),
                uri=_string(node, "uri", where),
                credentials_ref=(
                    None if credentials is None else _secret_ref(credentials, f"{where}.credentials")
                ),
                activation=ModelActivation(
                    require_warmup=_bool(
                        activation_node, "requireWarmup", f"{where}.activation", default=True
                    ),
                    retain_for_rollback=_bool(
                        activation_node, "retainForRollback", f"{where}.activation", default=True
                    ),
                ),
            )
        )
    return tuple(entries)


def _parse_completion_defaults(global_cfg: dict) -> CompletionPolicy:
    """Build the inherited completion policy, applying the schema's defaults."""
    where = "global.completionDefaults"
    node = _child(
        global_cfg, "completionDefaults", "global",
        ("onSuccess", "onInvalidInput", "onOperationalFailure", "onPublishFailure",
         "onCollision"),
    )
    collision = CollisionPolicy(
        _string(
            node, "onCollision", where,
            default=CollisionPolicy.FAIL.value, choices=_COLLISION_POLICIES,
        )
    )
    return CompletionPolicy(
        on_success=_completion_action(
            node, "onSuccess", where, default=CompletionAction.ARCHIVE
        ),
        on_invalid_input=_completion_action(
            node, "onInvalidInput", where, default=CompletionAction.QUARANTINE
        ),
        on_operational_failure=_completion_action(
            node, "onOperationalFailure", where, default=CompletionAction.RETAIN
        ),
        on_publish_failure=_publish_failure_action(node, "onPublishFailure", where),
        on_collision=collision,
    )


def _parse_completion(node: dict, where: str, defaults: CompletionPolicy) -> CompletionPolicy:
    """Build one route's completion policy on top of `completionDefaults`."""
    block = _child(
        node, "completion", where,
        ("onSuccess", "onInvalidInput", "onOperationalFailure", "onPublishFailure",
         "onCollision", "archiveDir", "failedDir"),
    )
    place = f"{where}.completion"
    collision = CollisionPolicy(
        _string(
            block, "onCollision", place,
            default=defaults.on_collision.value, choices=_COLLISION_POLICIES,
        )
    )
    return CompletionPolicy(
        on_success=_completion_action(block, "onSuccess", place, default=defaults.on_success),
        on_invalid_input=_completion_action(
            block, "onInvalidInput", place, default=defaults.on_invalid_input
        ),
        on_operational_failure=_completion_action(
            block, "onOperationalFailure", place, default=defaults.on_operational_failure
        ),
        on_publish_failure=_publish_failure_action(block, "onPublishFailure", place),
        on_collision=collision,
        archive_dir=_path(block, "archiveDir", place, default=None),
        failed_dir=_path(block, "failedDir", place, default=None),
    )


def _parse_source(node: dict, where: str) -> Union[SpoolSourceConfig, TriggerSourceConfig]:
    """Build one route's input source, discriminated on `kind`.

    Raises:
        ConfigError: ``MISSING_FIELD`` when the route declares no source, and
            ``INVALID_SOURCE_KIND`` when `kind` is neither ``spool`` nor ``trigger``.
    """
    if "source" not in node or node["source"] is None:
        raise ConfigError("MISSING_FIELD", f"{where}.source is required")
    raw = node["source"]
    if not isinstance(raw, dict):
        raise ConfigError("INVALID_TYPE", f"{where}.source must be an object")
    kind = raw.get("kind")
    place = f"{where}.source"
    if kind == "spool":
        return _parse_spool_source(raw, place)
    if kind == "trigger":
        return _parse_trigger_source(raw, place)
    raise ConfigError("INVALID_SOURCE_KIND", f"{place}.kind must be 'spool' or 'trigger'")


def _parse_spool_source(raw: dict, where: str) -> SpoolSourceConfig:
    """Build a watched-directory source."""
    node = _object(raw, where, ("kind", "root", "include", "exclude", "readiness", "camera"))
    readiness_node = _child(node, "readiness", where, ("mode", "quietSecs", "markerSuffix"))
    if not readiness_node:
        raise ConfigError("MISSING_FIELD", f"{where}.readiness is required")
    mode = ReadinessMode(
        _string(
            readiness_node, "mode", f"{where}.readiness",
            choices=tuple(item.value for item in ReadinessMode),
        )
    )
    camera = None
    if node.get("camera") is not None:
        place = f"{where}.camera"
        camera_node = _object(
            node["camera"], place,
            ("component", "instance", "subscribeAnnouncements", "reconcileCaptureStatusSecs"),
        )
        camera = CameraBinding(
            component=_string(
                camera_node, "component", place, default="camera-adapter", pattern=_UNS_TOKEN
            ),
            instance=_string(camera_node, "instance", place, pattern=_UNS_TOKEN),
            subscribe_announcements=_bool(
                camera_node, "subscribeAnnouncements", place, default=True
            ),
            reconcile_capture_status_secs=_number(
                camera_node, "reconcileCaptureStatusSecs", place,
                default=30, integer=True, minimum=0,
            ),
        )
    return SpoolSourceConfig(
        root=_path(node, "root", where),
        include=_strings(
            node, "include", where,
            default=("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.tif", "**/*.tiff"),
        ),
        exclude=_strings(node, "exclude", where, default=()),
        readiness=ReadinessConfig(
            mode=mode,
            quiet_secs=_number(
                readiness_node, "quietSecs", f"{where}.readiness", default=5.0, minimum=0
            ),
            marker_suffix=_string(
                readiness_node, "markerSuffix", f"{where}.readiness", default=".done"
            ),
        ),
        camera=camera,
    )


def _parse_trigger_source(raw: dict, where: str) -> TriggerSourceConfig:
    """Build a subscription source.

    Raises:
        ConfigError: ``INLINE_LIMIT_EXCEEDED`` when `maxInlineBytes` is above the core
            envelope's 64 KiB binary-body cap, which the wire contract does not raise.
    """
    node = _object(
        raw, where, ("kind", "subscribe", "fileRoot", "inlineStaging", "maxInlineBytes")
    )
    max_inline = _number(
        node, "maxInlineBytes", where, default=MAX_INLINE_BYTES, integer=True, minimum=1
    )
    if max_inline > MAX_INLINE_BYTES:
        raise ConfigError(
            "INLINE_LIMIT_EXCEEDED",
            f"{where}.maxInlineBytes may not exceed the {MAX_INLINE_BYTES}-byte binary-body cap",
        )
    return TriggerSourceConfig(
        subscribe=_strings(node, "subscribe", where, default=(), min_items=1),
        file_root=_path(node, "fileRoot", where),
        inline_staging=_path(node, "inlineStaging", where),
        max_inline_bytes=max_inline,
    )


def _parse_outputs(node: dict, where: str) -> OutputsConfig:
    """Build one route's outputs, checking every decision signal's shape."""
    block = _child(node, "outputs", where, ("writeResultSidecar", "decisionSignals"))
    place = f"{where}.outputs"
    signals = []
    seen = set()
    for index, raw in enumerate(_array(block, "decisionSignals", place)):
        spot = f"{place}.decisionSignals[{index}]"
        entry = _object(raw, spot, ("id", "value"))
        signal_id = _string(entry, "id", spot, pattern=_SIGNAL_ID)
        if signal_id in seen:
            raise ConfigError(
                "INVALID_DECISION_SIGNAL", f"{place}.decisionSignals names '{signal_id}' twice"
            )
        seen.add(signal_id)
        expression = _string(entry, "value", spot)
        if not expression.startswith("$"):
            raise ConfigError(
                "INVALID_DECISION_SIGNAL",
                f"{spot}.value must be a JSONPath expression starting at the root '$'",
            )
        signals.append(DecisionSignal(id=signal_id, value=expression))
    return OutputsConfig(
        write_result_sidecar=_bool(block, "writeResultSidecar", place, default=True),
        decision_signals=tuple(signals),
    )


def _parse_route(raw: Any, index: int, defaults: CompletionPolicy) -> RouteConfig:
    """Build one route from one `component.instances[]` entry."""
    where = f"instances[{index}]"
    node = _object(raw, where, _ROUTE_KEYS)
    route_id = _string(node, "id", where, pattern=_UNS_TOKEN)
    where = f"instances[{route_id}]"
    model_node = _object(
        node.get("modelRef") if node.get("modelRef") is not None else {},
        f"{where}.modelRef", ("id", "version", "digest"),
    )
    if not model_node:
        raise ConfigError("MISSING_FIELD", f"{where}.modelRef is required")
    source = _parse_source(node, where)
    primary_root = (
        source.root if isinstance(source, SpoolSourceConfig) else source.inline_staging
    )
    return RouteConfig(
        id=route_id,
        enabled=_bool(node, "enabled", where, default=True),
        priority=_number(node, "priority", where, default=0, integer=True, minimum=0, maximum=1000),
        source=source,
        model_ref=ModelRef(
            id=_string(model_node, "id", f"{where}.modelRef", pattern=_MODEL_ID),
            version=_string(model_node, "version", f"{where}.modelRef", pattern=_MODEL_VERSION),
            digest=_string(model_node, "digest", f"{where}.modelRef", pattern=_DIGEST),
        ),
        outputs=_parse_outputs(node, where),
        completion=_parse_completion(node, where, defaults).with_source_root(primary_root),
        reprocess_existing_on_model_change=_bool(
            node, "reprocessExistingOnModelChange", where, default=False
        ),
    )


def parse_component_config(global_cfg: Any, instances: Any) -> ComponentConfig:
    """Parse `component.global` and `component.instances[]` into a frozen configuration.

    Defaults come from `config.schema.json` and are applied here, `completionDefaults` are
    folded into every route's completion policy, and every `$secret` reference becomes an
    unresolved `SecretRef`. Nothing is read from the filesystem or the vault, so this is safe
    to call on a candidate the component has not committed to.

    Args:
        global_cfg: The `component.global` object, or ``None`` when the deployment sets none.
        instances: The `component.instances[]` list, or ``None`` when there are no routes.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: With a stable SCREAMING_SNAKE_CASE code naming the offending field.
    """
    if global_cfg is None:
        global_cfg = {}
    global_cfg = _object(global_cfg, "global", _GLOBAL_KEYS)
    if instances is None:
        instances = []
    if not isinstance(instances, (list, tuple)):
        raise ConfigError("INVALID_TYPE", "instances must be an array")

    completion_defaults = _parse_completion_defaults(global_cfg)
    routes = []
    seen = set()
    for index, raw in enumerate(instances):
        route = _parse_route(raw, index, completion_defaults)
        if route.id in seen:
            raise ConfigError("DUPLICATE_ROUTE_ID", f"instances declares route '{route.id}' twice")
        seen.add(route.id)
        routes.append(route)

    return ComponentConfig(
        paths=_parse_paths(global_cfg),
        runtime=_parse_runtime(global_cfg),
        gpu=_parse_gpu(global_cfg),
        scheduler=_parse_scheduler(global_cfg),
        discovery=_parse_discovery(global_cfg),
        publish=_parse_publish(global_cfg),
        signing=_parse_signing(global_cfg),
        model_sources=_parse_model_sources(global_cfg),
        models=_parse_models(global_cfg),
        completion_defaults=completion_defaults,
        routes=tuple(routes),
    )
