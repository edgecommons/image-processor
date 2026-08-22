# LLD — ImageProcessor module structure and interfaces

This document specifies the Python package that realizes `DESIGN.md`. Each module has one owner
work package (§9); the interfaces in §3–§8 are the contract between packages, so they are fixed
before implementation and changed only by editing this document in the same change.

Conventions: Python 3.12, type hints everywhere, `dataclasses` (frozen where the value is
immutable), `logging.getLogger(__name__)`, no module-level side effects, Windows and Linux both
supported (directory fsync is best-effort on Windows, as in camera-adapter). Tests are pytest;
the 90% line gate applies to `image_processor/`; `# pragma: no cover` is allowed only on a live
seam (subprocess entry, real transport) with an inline reason.

## 1. Package layout

```text
image_processor/
  __init__.py
  ImageProcessor.py        app wiring: builds every subsystem from config, runs, stops (WP6)
  types.py                 shared value types and enums (§2, WP1)
  config/                  schema-backed config models + candidate validator (WP1)
    __init__.py  models.py  validate.py
  bundles/                 tarball verification, manifest, signature, cache, fetchers (WP2)
    __init__.py  manifest.py  archive.py  signature.py  cache.py  fetch.py
  ledger/                  SQLite job ledger, outbox, cleanup intents, recovery (WP3)
    __init__.py  schema.py  ledger.py  recovery.py
  engine/                  decode, task families, decision rules, executor cells, scheduler (WP4a, WP4b)
    __init__.py  decode.py  families/{__init__,classification,detection,segmentation,anomaly}.py
    decision.py  protocol.py  cell.py  cell_main.py  supervisor.py  scheduler.py  residency.py
  sources/                 spool discovery, readiness, camera hint/status, trigger subscription (WP5)
    __init__.py  spool.py  readiness.py  camera.py  trigger.py  staging.py
  outputs/                 result building, outbox publisher, decision mirror, sidecar, events (WP6)
    __init__.py  result.py  publisher.py  mirror.py  sidecar.py  events.py
  completion/              archive/delete/retain/quarantine with intents (WP3)
    __init__.py  actions.py
  commands.py  metrics.py  health.py  connectivity.py          (WP6)
schemas/
  inference-result.schema.json      (WP1)
  model-bundle-manifest.schema.json (WP1)
config.schema.json                  (WP1; the component.global + instances[] contract)
tools/
  make_bundle.py           build + sign a bundle tarball from a directory (WP2)
  fetch_test_assets.py     download/verify tests/assets.json into tests/.cache (WP7)
tests/
  fixtures/build.py        synthetic ONNX graphs + images + camera-shaped spool fixtures (WP7)
  assets.json              pinned URL + SHA-256 manifest for tier-2 assets (WP7)
  goldens/*.json           committed tier-2 expected results (WP7)
  unit + integration tests per package
```

Dependencies (`pyproject.toml`): runtime `numpy`, `pillow`, `onnxruntime` (CPU), `jsonpath-ng`,
`cryptography` (Ed25519), `jsonschema` (declared by WP1), `watchdog`; extras `gpu = ["onnxruntime-gpu"]`,
`s3 = ["boto3"]`, `nvml = ["nvidia-ml-py"]`, `test = ["pytest", "pytest-cov", "onnx"]`. `onnx` is
a test-only dependency (fixture generation); the runtime never imports it.

## 2. Shared types (`image_processor/types.py`)

```python
class JobState(str, Enum):
    DISCOVERED, READY, INPUT_INVALID, QUARANTINED, CLAIMED, WAITING_MODEL, INFERENCING,
    BLOCKED_CONFIGURATION, RETRY_WAIT, PROCESSING_EXHAUSTED, RETAINED_FAILED, RESULT_COMMITTED,
    PUBLISH_PENDING, PUBLISHED, PUBLISH_EXHAUSTED, CLEANUP_PENDING, CLEANUP_FAILED, COMPLETED

class SourceKind(str, Enum): SPOOL = "spool"; INLINE = "inline"; REFERENCE = "reference"
class Family(str, Enum): CLASSIFICATION, DETECTION, SEGMENTATION, ANOMALY
class Outcome(str, Enum): CLEAR, HOLD, FAIL
class CompletionAction(str, Enum): ARCHIVE, DELETE, RETAIN, QUARANTINE

@dataclass(frozen=True)
class ModelRef: id: str; version: str; digest: str            # digest = "sha256:<hex>"

@dataclass(frozen=True)
class SecretRef: name: str; field: str | None = None          # an unresolved {"$secret": ...} ref
    # is_ref(value) / parse(value) / to_config(); parsing never opens the vault, because the core
    # auto-resolves $secret only under `streaming` and this component resolves its own (DESIGN §9)

@dataclass(frozen=True)
class SourceIdentity:
    kind: SourceKind; route_id: str; relative_path: str; bytes: int; sha256: str
    capture_id: str | None = None; camera_id: str | None = None
    correlation_id: str | None = None; captured_at_ms: int | None = None

@dataclass(frozen=True)
class Job:
    inference_id: str            # ULID-like, derived once per logical key (§4.3 of DESIGN.md)
    route_id: str; source: SourceIdentity; model: ModelRef; transform_version: str
    state: JobState; attempts: int = 0; next_attempt_at_ms: int | None = None
    staged_path: str | None = None   # the immutable processor-owned copy the cell reads
    config_generation: int = 0

@dataclass(frozen=True)
class TensorSpec: name: str; dtype: str; shape: tuple[int | str, ...]   # "N" marks a dynamic batch axis

@dataclass(frozen=True)
class BundleManifest:                     # parsed, validated manifest.json
    schema_version: int; model_id: str; version: str; files: dict[str, str]  # path -> sha256
    min_onnxruntime: str; providers_permitted: list[str]; provider_policy: str
    inputs: list[TensorSpec]; outputs: list[TensorSpec]; dynamic_batch: bool
    family: Family; family_params: dict; preprocess: dict; decision_rules: dict
    max_result_items: int; estimated_device_mib: int; warmup: list[dict]; tolerances: dict
    compatibility_keys: dict; provenance: dict; key_id: str | None; transform_version: str

@dataclass(frozen=True)
class CachedBundle: digest: str; root: Path; manifest: BundleManifest; model_path: Path

@dataclass(frozen=True)
class Detection: label: str; index: int; score: float; box: tuple[float, float, float, float]  # x,y,w,h normalized
@dataclass(frozen=True)
class ClassScore: label: str; index: int; score: float

@dataclass(frozen=True)
class NormalizedOutput:
    family: Family
    classes: list[ClassScore] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    segments: dict = field(default_factory=dict)     # {label: {"pixels": int, "bbox": [x,y,w,h]}}
    anomaly: dict = field(default_factory=dict)      # {"score": float, "threshold": float, "summary": {...}}
    raw_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

@dataclass(frozen=True)
class Decision: outcome: Outcome; passed: bool; confidence: float | None; threshold: float | None; rule: str

@dataclass(frozen=True)
class Timings: queue_ms: float; model_load_ms: float; preprocess_ms: float; inference_ms: float; postprocess_ms: float; total_ms: float

@dataclass(frozen=True)
class InferenceResult:                    # the cell's answer for one job
    inference_id: str; status: str         # SUCCEEDED | FAILED
    normalized: NormalizedOutput | None; decision: Decision | None
    providers: list[str]; gpu_device: str | None; gpu_class: str | None
    timings: Timings; memory_high_water_mib: int | None; error: str | None = None
    error_class: str | None = None         # transient | permanent | contaminating

@dataclass(frozen=True)
class CleanupIntent:
    inference_id: str; action: CompletionAction; source_path: str; source_sha256: str
    target_path: str | None; members: tuple[str, ...]    # sidecar/companion paths moved with the image
```

Identity: `derive_inference_id(route_id, capture_id | None, source_sha256, normalized_source_id,
model_digest) -> str` lives in `types.py`; deterministic (BLAKE2b of the logical key, base32), so
a re-discovery of the same input under the same model yields the same id.

## 3. `config/` (WP1)

`config.schema.json` (JSON Schema 2020-12, `additionalProperties: false` throughout) describes
`component.global` and each `component.instances[]` entry exactly as DESIGN.md §11: `paths`,
`runtime`, `gpu`, `scheduler`, `publish`, `signing`, `modelSources`, `models[]`,
`completionDefaults`; per instance `id`, `enabled`, `priority`, `source` (`kind: spool | trigger`,
discriminated), `modelRef`, `outputs`, `completion`, `reprocessExistingOnModelChange`.
`schemas/inference-result.schema.json` and `schemas/model-bundle-manifest.schema.json` are the
wire/bundle contracts (DESIGN.md §12.1, §8).

`modelSources` (`allowedSchemes`, `allowedUriPrefixes`, `verifyTls`), `discovery` (`rescanSecs`,
`debounceMs`) and the numeric policy fields `scheduler.{maxAttempts, retryBackoffSecs,
maxRetryBackoffSecs, queueAgeWarningSecs}` and `publish.{maxAttempts, outboxCapacity,
outboxReserveBudgetMiB}` are not in DESIGN.md §11's illustrative example; they carry the numbers
§5.1, §7, §12.3 and §15 require, and they are the constructor arguments WP3 and WP5 take:
`discovery` feeds `SpoolSource(rescan_interval_secs=, debounce_secs=)`, and
`publish.outboxReserveBudgetMiB` feeds `Ledger(reserve_budget_bytes=)`. `outboxCapacity` bounds
the number of pending rows, `outboxReserveBudgetMiB` bounds their bytes; they are different
limits and both are kept.

Config-level completion actions use §11's spelling (`archive | delete | retainInPlace |
quarantine`) and map onto `CompletionAction`, where `retainInPlace` is `RETAIN`; `onCollision` is
`fail | suffix`, matching `completion.actions.COLLISION_FAIL` / `COLLISION_SUFFIX`. A route with
no `failedDir` quarantines in place.

`ReadinessMode` and `CollisionPolicy` define `__str__ = str.__str__`, so `str(mode)` yields the
configured spelling. `sources/readiness.py` normalizes that field with `str()`, and without this a
parsed route would present `'ReadinessMode.STABILITY'` and be refused as an unknown mode.

`schemas/model-bundle-manifest.schema.json` is the authority on what a bundle must declare, and
its `preprocess`, `familyParams` and `decisionRules` accept exactly the vocabulary
`engine/families/*.py` and `engine/decision.py` consume (documented in
`docs/reference/data-types.md`). Every bundle `tests/fixtures/build.py` writes validates against
it, which is the gate that keeps the two from drifting.

```python
# config/models.py
class ConfigError(ValueError): code: str; message: str      # stable SCREAMING_SNAKE codes
class ReadinessMode(str, Enum): CAMERA_SIDECAR, CAMERA_STATUS, MARKER, STABILITY
class CollisionPolicy(str, Enum): FAIL = "fail"
MAX_INLINE_BYTES = 65536; MUTATING_ACTIONS = {ARCHIVE, DELETE, QUARANTINE}
COMPLETION_ACTION_NAMES: dict[str, CompletionAction]; EXECUTION_PROVIDERS: tuple[str, ...]

@dataclass(frozen=True) class Paths: state_db, model_cache, staging                      # Path
@dataclass(frozen=True) class RuntimeConfig: providers, required_provider, allow_cpu_only,
    executor_cells_per_gpu, load_concurrency_per_gpu
@dataclass(frozen=True) class GpuConfig: devices, resident_memory_budget_percent, reserve_mib
@dataclass(frozen=True) class SchedulerConfig: max_batch_size, max_batch_latency_ms, hot_ttl_secs,
    min_residency_secs, max_attempts, retry_backoff_secs, max_retry_backoff_secs,
    queue_age_warning_secs
@dataclass(frozen=True) class PublishConfig: confirmation_timeout_secs,
    require_confirmation_before_cleanup, max_attempts, outbox_capacity,
    outbox_reserve_budget_mib; .outbox_reserve_budget_bytes
@dataclass(frozen=True) class DiscoveryConfig: rescan_secs, debounce_ms; .debounce_secs
@dataclass(frozen=True) class TrustedKey: key_id; public_key: str | SecretRef
@dataclass(frozen=True) class SigningConfig: required; trusted_keys; key(key_id) -> TrustedKey|None
@dataclass(frozen=True) class ModelSourcesConfig: allowed_schemes, allowed_uri_prefixes, verify_tls
@dataclass(frozen=True) class ModelActivation: require_warmup, retain_for_rollback
@dataclass(frozen=True) class ModelEntry: id, version, digest, uri, credentials_ref: SecretRef|None,
    activation: ModelActivation; .ref -> ModelRef
@dataclass(frozen=True) class CompletionPolicy: on_success, on_invalid_input,
    on_operational_failure, on_publish_failure (CompletionAction), on_collision (CollisionPolicy),
    source_root: Path|None, archive_dir: Path|None, failed_dir: Path|None;
    .actions, .mutates, .with_source_root(root)
    # satisfies completion.actions.CompletionPolicy (a runtime_checkable Protocol). It annotates
    # its paths as `str`; every use site wraps them in `Path(...)`, so `Path` values satisfy it.
    # source_root is None on ComponentConfig.completion_defaults, which is a template, not a policy.
@dataclass(frozen=True) class ReadinessConfig: mode: ReadinessMode; quiet_secs; marker_suffix
@dataclass(frozen=True) class CameraBinding: component, instance, subscribe_announcements,
    reconcile_capture_status_secs
@dataclass(frozen=True) class SpoolSourceConfig: root: Path; include; exclude;
    readiness: ReadinessConfig; camera: CameraBinding|None; kind = "spool"
@dataclass(frozen=True) class TriggerSourceConfig: subscribe; file_root: Path;
    inline_staging: Path; max_inline_bytes; kind = "trigger"
@dataclass(frozen=True) class DecisionSignal: id; value                      # value is a JSONPath
@dataclass(frozen=True) class OutputsConfig: write_result_sidecar; decision_signals
@dataclass(frozen=True) class RouteConfig: id, enabled, priority, source, model_ref: ModelRef,
    outputs, completion, reprocess_existing_on_model_change;
    .is_spool, .is_trigger, .input_root, .mutating_roots, .output_dirs
    .completion_for(kind: SourceKind) -> CompletionPolicy
    # `completion` carries the route's own root: the spool root, or a trigger route's
    # inlineStaging. A REFERENCE input on a trigger route resolves under fileRoot instead, so WP6
    # hands the completer `route.completion_for(job.source.kind)` rather than `route.completion`.
@dataclass(frozen=True) class ComponentConfig: paths, runtime, gpu, scheduler, discovery, publish,
    signing, model_sources, models, completion_defaults, routes;
    .route(id), .model_entry(ref), .enabled_routes

def parse_component_config(global_cfg: dict|None, instances: list[dict]|None) -> ComponentConfig
    # raises ConfigError(code, message) with SCREAMING_SNAKE codes; applies the schema's defaults,
    # folds completionDefaults into each route, turns every $secret into an unresolved SecretRef
# config/validate.py
VALIDATOR_NAME = "image-processor"
CONFIG_SCHEMA / INFERENCE_RESULT_SCHEMA / MODEL_BUNDLE_MANIFEST_SCHEMA   # paths from the repo root
def schema_path(relative: str) -> Path ; def load_schema(relative: str) -> dict   # cached, fail-closed
def register(builder) -> builder     # builder.configuration_validator(VALIDATOR_NAME, validate_candidate)
def validate_candidate(candidate: dict, current: dict | None, phase) -> ConfigurationValidationResult
    # checks: schema (SCHEMA_INVALID), parse codes, UNRESOLVED_MODEL_REF,
    # OVERLAPPING_MUTATING_ROOTS, OUTPUT_INSIDE_SOURCE_ROOT, MISSING_ARCHIVE_DIR,
    # COMPLETION_DIR_NOT_CREATABLE, INLINE_STAGING_NOT_CONTAINED,
    # STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE (the code sources/readiness.py raises),
    # CAMERA_BINDING_REQUIRED, PROVIDER_POLICY_UNSATISFIED, MODEL_URI_{SCHEME_,}NOT_ALLOWED,
    # NO_TRUSTED_KEYS, and on RELOAD IMMUTABLE_PATH_CHANGED for stateDb/modelCache/staging
```

Decision-signal expressions are checked structurally here (non-empty, rooted at `$`); compiling
them is `engine/decision.py`'s job (WP4a), which owns the `jsonpath-ng` dependency.

## 4. `bundles/` (WP2)

```python
# archive.py  (BundleError is defined here, the lowest layer; the package re-exports it)
class BundleError(Exception): code: str; message: str        # SCREAMING_SNAKE codes, e.g. DIGEST_MISMATCH
@dataclass(frozen=True) class ExtractLimits: max_members: int = 10_000; max_total_bytes: int = 8 * 2**30; max_member_bytes: int = 4 * 2**30; max_ratio: float = 100.0
def normalize_digest(digest: str) -> str ; def digest_hex(digest: str) -> str   # accept "sha256:<hex>" or bare hex
def sha256_file(path: Path, chunk: int = 1 << 20) -> str
def verify_tarball_digest(path: Path, expected: str) -> None          # raises BundleError("DIGEST_MISMATCH")
def extract_tarball(path: Path, dest: Path, limits: ExtractLimits = ExtractLimits()) -> list[Path]
    # rejects absolute paths, "..", backslashes, symlinks/hardlinks/devices, duplicates, members outside dest,
    # and members colliding with an earlier one; streams with member/total/per-member/ratio guards; tar and tar.gz by magic
def read_member_bytes(path: Path, names: Iterable[str], limits: ExtractLimits = ExtractLimits(), max_bytes: int = 4 << 20) -> Mapping[str, bytes]
    # manifest.json and manifest.sig are read under the same rules before extraction, so the signature gates the payload (DESIGN.md §9 step 3)
# manifest.py
DEFAULT_SCHEMA_PATH = <repo>/schemas/model-bundle-manifest.schema.json          # the WP1 contract
def load_manifest(bundle_dir: Path, schema_path: Path | None = None, verify_files: bool = True) -> BundleManifest
    # manifest.json at the root -> schema -> per-file digests; a file the manifest does not declare is refused (FILE_UNDECLARED)
def parse_manifest(document: dict) -> BundleManifest ; def validate_document(document: dict, schema_path: Path | None = None) -> None
def read_manifest_bytes(bundle_dir: Path) -> bytes ; def resolve_model_path(bundle_dir: Path, m: BundleManifest) -> Path
# signature.py
def verify_manifest_signature(manifest_bytes: bytes, signature: bytes, public_key_pem_or_raw: bytes) -> None
    # Ed25519 via cryptography; PEM, DER, or raw 32 bytes, as bytes or text; raises BundleError("BAD_SIGNATURE")
def sign_manifest(manifest_bytes: bytes, private_key: bytes) -> bytes           # used by tools/make_bundle.py and tests
def generate_keypair(password: bytes | None = None) -> tuple[bytes, bytes, bytes]   # private PEM, public PEM, raw public key
def load_public_key / load_private_key / public_key_pem / public_key_raw / private_key_pem
# cache.py
class BundleCache:
    def __init__(self, root: Path, schema_path: Path | None = None): ...   # <root>/<sha256hex>/{manifest.json, model.onnx, ...} + <root>/<sha256hex>.json metadata
    def get(self, digest: str, verify: bool = False) -> CachedBundle | None    # None when absent or half-promoted; raises when a cached bundle no longer verifies
    def promote(self, extracted_dir: Path, digest: str) -> CachedBundle        # rename into place (copy first across filesystems); idempotent: an existing copy is verified and kept, one that fails is replaced
    def list(self) -> list[CachedBundle] ; def metadata(self, digest: str) -> dict | None
    def gc(self, pinned: Iterable[str]) -> list[str]                       # removes unpinned digests, orphan metadata, and stale staging temporaries
# fetch.py
class Fetcher(Protocol): def fetch(self, uri: str, dest: Path, credentials: dict | None) -> Path
def check_https_uri(uri: str, allowed_prefixes: Sequence[str] | None) -> str   # https only, no credentials in the URL, prefix allow-list over the normalized URL (None = no allow-list configured)
def fetcher_for(uri: str, allowed_prefixes: Sequence[str] | None = None, timeout_secs: float = 60.0, max_bytes: int | None = None, ssl_context: ssl.SSLContext | None = None) -> Fetcher
    # plain path or file://, https:// (TLS verified, the policy re-applied to every redirect), s3:// (boto3 optional extra; ImportError -> BundleError("S3_UNAVAILABLE"))
def stage_bundle(uri: str, digest: str, staging_root: Path, cache: BundleCache, credentials: Mapping | None = None,
                 signing_required: bool = False, trusted_keys: Mapping[str, bytes] | None = None,
                 limits: ExtractLimits = ExtractLimits(), schema_path: Path | None = None,
                 allowed_prefixes: Sequence[str] | None = None, model_id: str | None = None, version: str | None = None,
                 available_providers: Sequence[str] | None = None, validators: Sequence[Callable[[BundleManifest], None]] = (),
                 timeout_secs: float = 60.0, max_bytes: int | None = None, ssl_context: ssl.SSLContext | None = None) -> CachedBundle
    # DESIGN.md §9 steps 2-6 except warmup: unique staging dir -> fetch -> tarball digest -> signature over manifest.json ->
    # bounded extract -> schema + per-file digests -> identity, provider, and injected validators -> cache.promote
```

`stage_bundle` takes plain arguments rather than `ModelEntry`/`SigningConfig` so that `bundles/`
does not depend on `config/`; WP6 adapts one onto the other when it wires the artifact manager.
Provider compatibility is checked here when `available_providers` is given; task-family support
(DESIGN.md §9 step 4) reaches it as a `validators` entry, because the families live in
`engine/` (WP4a). A signature is verified whenever it is present and its `keyId` is trusted, and
`signing_required` additionally makes the signature and a trusted key mandatory.

`tools/make_bundle.py`: `python tools/make_bundle.py <dir> --out <file.tar[.gz]> [--key
<ed25519-private.pem>] [--key-id <id>] [--key-password <pass>] [--gzip] [--schema <path>]`
computes per-file digests into `manifest.json` (merging an author-supplied `manifest.json`),
validates it, writes `manifest.sig`, packs the archive with fixed member metadata so the digest
is reproducible, and prints the tarball digest. `--gen-key <path>` creates an Ed25519 keypair:
the private PEM, `<path>.pub.pem`, and the raw 32-byte `<path>.pub`. Importable functions
(`make_bundle`, `build_manifest_document`, `collect_files`, `generate_key_files`, `main`) back
the CLI so tests and WP7 call it directly.

## 5. `ledger/` and `completion/` (WP3)

```python
class Ledger:
    def __init__(self, path: Path, synchronous: str = "FULL", busy_timeout_ms: int = 5000,
                 reserve_budget_bytes: int = 256 * 2**20, clock=_now_ms): ...
        # WAL; one writer thread + queue; reads on a separate connection; foreign_keys ON
    def close(self) -> None                                            # also a context manager
    # jobs
    def admit(self, job: Job, reserve_bytes: int) -> bool              # False when already present (same inference_id) or capacity reserve fails
    def reserved_bytes(self) -> int
    def get(self, inference_id: str) -> Job | None
    def last_error(self, inference_id: str) -> str | None
    def transition(self, inference_id: str, expected: JobState, new: JobState, **fields) -> Job
        # raises LedgerConflict if state != expected, IllegalTransition if the edge is not in TRANSITIONS;
        # **fields is limited to attempts, next_attempt_at_ms, staged_path, config_generation, last_error;
        # a self-edge (new == expected) is allowed so a field-only update reuses the same gate
    def claimable(self, route_id: str | None, limit: int) -> list[Job]
    def by_state(self, states: Iterable[JobState], route_id: str | None = None, cursor: str | None = None, limit: int = 100) -> tuple[list[Job], str | None]
    # results + outbox (one transaction)
    def commit_result(self, inference_id: str, result_json: bytes, sidecar: tuple[str, str] | None, outbox: list[OutboxRow]) -> None
        # INFERENCING -> RESULT_COMMITTED -> PUBLISH_PENDING (the second edge only when outbox rows exist)
    def result_bytes(self, inference_id: str) -> bytes | None
    def outbox_for(self, inference_id: str) -> list[OutboxRow]
    def pending_outbox(self, limit: int) -> list[OutboxRow]            # OutboxRow(id, inference_id, topic, encoded_bytes, attempts, gating: bool, last_error); eligible only while the job is PUBLISH_PENDING
    def mark_published(self, outbox_id: int) -> None                   # when every gating row for the job is published -> job PUBLISHED
    def mark_publish_attempt(self, outbox_id: int, error: str) -> None
    def exhaust_publish(self, inference_id: str) -> Job                # PUBLISH_PENDING -> PUBLISH_EXHAUSTED
    def retry_publication(self, inference_id: str) -> Job              # PUBLISH_EXHAUSTED -> PUBLISH_PENDING; clears unpublished row attempts
    # cleanup
    def record_cleanup_intent(self, intent: CleanupIntent) -> Job      # PUBLISHED|CLEANUP_FAILED -> CLEANUP_PENDING; INPUT_INVALID and PROCESSING_EXHAUSTED keep their state
    def cleanup_intent(self, inference_id: str) -> CleanupIntent | None
    def cleanup_observed(self, inference_id: str) -> str | None
    def complete_cleanup(self, inference_id: str, observed: str) -> Job  # CLEANUP_PENDING -> COMPLETED, INPUT_INVALID -> QUARANTINED, PROCESSING_EXHAUSTED -> RETAINED_FAILED
    def fail_cleanup(self, inference_id: str, error: str) -> Job       # CLEANUP_PENDING -> CLEANUP_FAILED; otherwise the state is kept with the error, never success
    def pending_cleanup(self, limit: int) -> list[CleanupIntent]       # intents with no observed outcome
    # model generations
    def set_route_generation(self, route_id: str, desired: str, active: str | None) -> None
    def route_generation(self, route_id: str) -> tuple[str | None, str | None]
    # key/value (WP5 reconciliation watermarks)
    def kv_get(self, key: str) -> str | None
    def kv_set(self, key: str, value: str | None) -> None
    # recovery
    def recover(self) -> RecoveryReport                                # INFERENCING -> READY (same attempt), CLAIMED/WAITING_MODEL -> READY, due RETRY_WAIT -> READY, PUBLISH_PENDING retained, CLEANUP_PENDING -> reconcile list
    def requeue_for_reinference(self, inference_id: str, reason: str) -> Job
        # DESIGN.md §7: a committed record whose sidecar is absent returns to re-inference;
        # drops the result bytes, the result digest, the sidecar binding, and the outbox rows
# completion/actions.py
class Completer:
    def __init__(self, ledger: Ledger, fs: FsOps = RealFs(), on_collision: str = "fail", clock=_now_ms): ...
    def plan(self, job: Job, policy: CompletionPolicy, members: list[Path]) -> CleanupIntent
    def apply(self, intent: CleanupIntent, on_collision: str | None = None) -> None    # intent persisted first; temp+rename; cross-fs copy+verify+remove; collision -> CLEANUP_FAILED
    def reconcile(self, intent: CleanupIntent, on_collision: str | None = None) -> JobState   # DESIGN.md §7 observed-state rules
```

`TRANSITIONS` (`ledger/schema.py`) is the DESIGN.md §7 state diagram, edge for edge, and is the
only place an edge is legal; `transition()` refuses anything else with `IllegalTransition`.
`RECOVERY_EDGES` (`ledger/recovery.py`) is the separate, smaller table a restart uses, because
recovery moves a job backwards along edges the forward lifecycle never takes: `INFERENCING`,
`CLAIMED`, `WAITING_MODEL`, and a due `RETRY_WAIT` to `READY`, plus `RESULT_COMMITTED` and
`PUBLISH_PENDING` to `READY` for `requeue_for_reinference`. `RecoveryReport` carries the count per
edge, the open cleanup intents, the inference ids still eligible for the publisher, and the
sidecar bindings of committed-but-uncleaned jobs so the caller can run the DESIGN.md §7 filesystem
reconciliation.

`CompletionPolicy` is a structural `Protocol` in `completion/actions.py` carrying `source_root`,
`archive_dir`, `failed_dir`, `on_success`, `on_invalid_input`, `on_operational_failure`,
`on_publish_failure`, and `on_collision`; WP1's `config.models.CompletionPolicy` dataclass
satisfies it. `source_root` is the route's `source.root`, which completion needs to resolve an
input's absolute path from `Job.source.relative_path`. Action values accept both the durable enum
and the config spellings, including `retainInPlace`.

Schema (`ledger/schema.py`): `jobs(inference_id PK, route_id, state, source_json, model_json,
transform_version, attempts, next_attempt_at_ms, staged_path, config_generation, result_json,
result_sha256, sidecar_path, sidecar_sha256, last_error, created_at_ms, updated_at_ms)`,
`outbox(id PK, inference_id, topic, payload BLOB, gating, attempts, last_error, published_at_ms)`,
`cleanup_intents(inference_id PK, action, source_path, source_sha256, target_path, members_json,
observed, created_at_ms)`, `route_generations(route_id PK, desired, active, updated_at_ms)`,
`reservations(inference_id PK, bytes)`, `kv(key PK, value)`. Migrations:
`CREATE TABLE IF NOT EXISTS` + a `meta(schema_version)` row.

## 6. `engine/` (WP4a: decode, families, decision; WP4b: protocol, cell, supervisor, scheduler, residency)

```python
# decode.py
@dataclass(frozen=True) class DecodeLimits: max_bytes: int = 64 * 2**20; max_pixels: int = 50_000_000; max_dim: int = 16_384
def decode_image(data: bytes, limits: DecodeLimits) -> np.ndarray      # HWC uint8 RGB (or uint16 for 16-bit TIFF); Pillow with MAX_IMAGE_PIXELS set from limits; raises DecodeError(permanent)
# families/__init__.py
class TaskFamily(Protocol):
    family: Family
    def validate_manifest(self, m: BundleManifest) -> None              # refuse unsupported heads at staging: raises FamilyError
    def preprocess(self, image: np.ndarray, m: BundleManifest) -> dict[str, np.ndarray]   # resize/letterbox/normalize/layout per m.preprocess
    def postprocess(self, outputs: dict[str, np.ndarray], m: BundleManifest, image_hw: tuple[int, int]) -> NormalizedOutput
FAMILIES: dict[Family, TaskFamily]
# decision.py
def decide(normalized: NormalizedOutput, rules: dict) -> Decision
    # rules: {"pass": <expr>, "confidence": "<jsonpath>", "threshold": number|"<jsonpath>",
    #         "outcomeOnPass": "CLEAR", "outcomeOnFail": "HOLD"|"FAIL", "failOnEmpty": bool}
    # <expr> = {"path","op","value"} | {"all": [...]} | {"any": [...]}; ops >= > <= < == != exists absent count>=
    # anything unevaluable, and any exception, -> HOLD (never CLEAR); grammar in docs/reference/data-types.md
# protocol.py  (parent <-> cell messages; plain dataclasses, pickled over multiprocessing pipes)
LoadModel(digest, bundle_root, providers, provider_policy, warmup: bool) -> Loaded(digest, providers_assigned, load_ms, device_mib) | LoadFailed(digest, error, error_class)
Infer(inference_id, staged_path, sha256, digest, transform_version) -> InferenceResult
Unload(digest) -> Unloaded(digest, freed_mib)
Stats() -> CellStats(resident: list[str], device_free_mib, device_total_mib, uptime_s)
Shutdown()
# cell_main.py  (subprocess entry; the only place onnxruntime sessions live) — pragma: no cover, covered by an in-process harness that calls the same handler functions
# cell.py
class ExecutorCell:                       # parent-side handle; spawn context; one per GPU (or CPU for dev)
    def __init__(self, cell_id: str, device: str | None, providers: list[str]): ...
    def start(self) / stop(self, timeout_s) / is_alive() / call(msg, timeout_s) -> reply
# supervisor.py
class Supervisor:                          # owns cells; restarts on contaminating errors or death; exposes healthy()
# residency.py
class ResidencyPolicy:                     # cost-aware score per DESIGN.md §10.3; admission check per §10.2 (manifest estimate + measured + free + reserve)
    def admit(self, digest: str, estimate_mib: int, free_mib: int) -> bool
    def victims(self, needed_mib: int, resident: dict[str, ResidencyStats], leased: set[str]) -> list[str]
# scheduler.py
class Scheduler:
    def __init__(self, ledger: Ledger, supervisor: Supervisor, cache: BundleCache, policy: ResidencyPolicy, cfg: SchedulerConfig): ...
    def submit(self, job: Job) -> None                                  # READY/CLAIMED -> lanes per digest
    def run_once(self) -> int                                           # one scheduling pass; returns jobs dispatched (deterministic, testable)
    def on_result(self, job: Job, result: InferenceResult) -> None     # callback to WP6 (result pipeline)
```

Phase 1 scheduler scope: single cell, per-digest lanes, min residency + hysteresis, cost-aware
eviction, priority + age weighting, single-flight load, no micro-batching (Phase 2).

## 7. `sources/` (WP5)

```python
# __init__.py    the events protocol + the configuration shapes the sources read
class SourceEvents(Protocol):             # what a source reports to the app (WP6 implements)
    def discovered(self, route_id: str, source: SourceIdentity, staged_path: Path | None) -> None
    def invalid(self, route_id: str, relative_path: str, reason: str) -> None
# Structural configuration protocols, field names exactly as DESIGN.md §11 spells them, so the
# WP1 dataclasses satisfy them without importing anything from this package:
#   RouteConfig(id, source); SpoolSourceConfig(root, include, exclude, readiness, camera)
#   ReadinessConfig(mode, quietSecs, markerSuffix)
#   CameraConfig(component, instance, subscribeAnnouncements, reconcileCaptureStatusSecs)
#   TriggerSourceConfig(subscribe, fileRoot, inlineStaging, maxInlineBytes)
# Read through staging.config_field(obj, *names, default=...): mapping or attribute access, camelCase
# or snake_case, a null value counts as absent.

# staging.py     processor-owned staging + the path/digest/config primitives the sources share
lstat = os.lstat ; realpath = os.path.realpath        # seams: a test injects a reparse-point stat
class SourceError(Exception): code: str               # StagingError | PathError | ConfigFieldError
def config_field(source, *names, default=...) -> Any
def sha256_bytes(data) -> str ; def sha256_file(path, chunk=1<<20) -> str
def plain_digest(value) -> str                        # accepts "<hex>" or "sha256:<hex>"
def stat_signature(path) -> tuple | None              # (size, mtime_ns); re-stat around every hash
def classify_path(path) -> str | None                 # MISSING|SYMLINK|REPARSE_POINT|DIRECTORY|DEVICE_FILE|NOT_REGULAR_FILE
def normalize_relative(value) -> str                  # forward slashes; rejects absolute/drive/".."
def real_root(root) -> Path ; def relative_to_root(root, path) -> str
def resolve_under_root(root, relative) -> Path        # realpath first, containment second
def staged_path_for(staging_root, sha256, suffix="") -> Path      # <root>/<hex[:2]>/<hex><suffix>
def stage_copy(src, staging_root, sha256) -> Path     # temp + fsync + verify + atomic install; idempotent
def stage_bytes(data, staging_root, sha256, suffix="") -> Path    # the inline trigger's path onto disk

# readiness.py
@dataclass(frozen=True) class ReadyVerdict: ready: bool; identity: SourceIdentity | None; reason: str
class ReadinessStrategy(Protocol): mode: str; def ready(self, path, relative_path) -> ReadyVerdict
class CameraSidecarReadiness  # parses <image>.json, verifies image.bytes/sha256, returns the identity
class CameraStatusReadiness   # ready only on a verified SUCCEEDED record; re-stats, does not re-hash
class MarkerReadiness         # <path><suffix> exists; identity is None, the caller hashes
class StabilityReadiness      # size+mtime still for quietSecs, injected clock; prune(keep)
class Readiness:                                      # the rule one route uses
    @staticmethod
    def for_route(route, *, status_lookup=None, clock=time.monotonic) -> Readiness
        # default mode: cameraSidecar when camera-bound, else stability
        # raises ReadinessError: UNKNOWN_READINESS_MODE | STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE
        #                      | CAMERA_STATUS_REQUIRES_RECONCILER | MARKER_SUFFIX_REQUIRED
    mode: str ; strategy: ReadinessStrategy
    def ready(self, path, relative_path) -> ReadyVerdict
    def companion_suffixes(self) -> tuple                 # what the walk skips beside an image
    def prune(self, keep) -> None
def parse_timestamp_ms(value) -> int | None               # RFC 3339 as chrono writes it
def read_sidecar(path) -> dict | None
def verify_declared_image(path, image) -> tuple[int | None, str | None, str | None]
def identity_from_capture(route_id, relative_path, document, size, sha256, kind=SPOOL) -> SourceIdentity

# camera.py
def capture_status_topic(device, component, instance=None) -> str   # .../cmd/sb/capture-status
def image_captured_topic(device, component, instance) -> str        # .../app/image/captured
@dataclass(frozen=True) class CaptureRecord: capture_id; relative_path; identity; signature; terminal_at_ms
class CaptureStatusReconciler:
    def __init__(self, *, route_id, root, topic, request, kv_get, kv_set, instance=None,
                 interval_secs=30.0, page_limit=100, request_timeout_secs=10.0,
                 on_verified=None, kv_key=None, clock=time.monotonic)
        # request(topic, body, timeout_secs) -> reply ; kv_get/kv_set are the WP3 ledger KV
    def poll_once(self) -> int      # List mode {"states":["SUCCEEDED"],...}, follows every nextCursor,
                                    # dedupes by captureId, verifies file+digest, advances the watermark
    def lookup(self, relative_path) -> CaptureRecord | None      # what CameraStatusReadiness calls
    def lookup_capture(self, capture_id) -> dict | None          # Capture mode; None on CAPTURE_NOT_FOUND
    def records(self) -> dict ; def start(self) ; def stop(self, timeout_s=5.0)

# spool.py
def compile_glob(pattern) -> re.Pattern   # ** crosses separators, "**/" also matches zero directories
class SpoolSource:
    def __init__(self, route, events, clock=time.monotonic, *, status_lookup=None,
                 observer_factory=None, debounce_secs=0.5, rescan_interval_secs=60.0)
    def start(self) ; def stop(self, timeout_s=5.0) ; def nudge(self) -> None
    def rescan(self) -> int         # authoritative walk; also the trigger-rescan command's path
    def on_hint(self, body: dict) -> None       # ImageCaptured: relativePath under the root, never
                                                # absolutePath; verifies bytes/sha256 = ready by proof
    def prime(self, pairs) -> None  # seed the announced (relative_path, sha256) set from the ledger
    def seen(self) -> set
    # counters WP6 reports as ImageProcessorDiscovery: discovered_count, rejected_count, rescans,
    # nudges, hints_accepted, hints_rejected, hints_unmapped

# trigger.py
class TriggerSource:
    def __init__(self, route, events, staging: Path | None = None, *, file_root=None,
                 max_inline_bytes=None)          # both default from the route; the cap is clamped to 64 KiB
    subscribe: tuple[str, ...]
    def on_message(self, message) -> None
        # "relativePath" in the body → reference: containment + bytes/sha256 verified + stage_copy
        # otherwise → inline: opaque binary body, body["image"] bytes, or the core binary marker
        # SourceIdentity carries correlation_id and reply_to so WP6 can answer the requester
    accepted: int ; rejected: int
def suffix_for(data) -> str ; def message_body(message) ; def request_correlation(message) -> tuple
```

Spool discovery uses `watchdog` observers feeding a debounced scan queue; `rescan()` is the
authoritative path and the watchdog only nudges it. A periodic interval nudges it as well, so a
dropped notification costs latency and never a job.

## 8. `outputs/`, wiring, commands, metrics, health (WP6)

```python
# outputs/result.py
def build_result_body(job: Job, result: InferenceResult, manifest: BundleManifest, artifacts: dict | None, limits) -> dict   # validates against schemas/inference-result.schema.json
# outputs/publisher.py
class OutboxPublisher:                     # drains ledger.pending_outbox via gg.instance(route).app().publish_confirmed(PreparedAppMessage(topic, bytes)) with timeout; marks published/attempt
# outputs/mirror.py     DecisionMirror.publish(route_id, decision_signals, result_body) via gg.instance(route).data()
# outputs/sidecar.py    write_sidecar(path, body) -> sha256 ; temp + fsync + atomic install
# outputs/events.py     typed helpers over gg.instance(route).evt()
# ImageProcessor.py     class ImageProcessor(gg): run()/stop(); builds config -> ledger -> cache -> supervisor -> scheduler -> sources -> publisher -> completer; the result pipeline: on_result -> sidecar -> ledger.commit_result (+ outbox rows) -> publisher -> completer; artifact manager thread (stage models[] -> warmup via cell -> route generation switch)
# commands.py           registers DESIGN.md §13 verbs with CommandInbox.register_outcome
# metrics.py            MetricBuilder groups per DESIGN.md §12.3, flushed on an interval
# health.py / connectivity.py   readiness rules §14; InstanceConnectivity per route for gg.set_instance_connectivity_provider
```

## 9. Work packages and ownership

| WP | Branch | Owns | Depends on |
|---|---|---|---|
| WP1 contracts | `feat/wp1-contracts` | `types.py`, `config/`, `config.schema.json`, `schemas/*.json` | — (this LLD) |
| WP2 bundles | `feat/wp2-bundles` | `bundles/`, `tools/make_bundle.py` | `types.py` (copy from this LLD; reconciled at merge) |
| WP3 ledger | `feat/wp3-ledger` | `ledger/`, `completion/` | `types.py` |
| WP4a families | `feat/wp4a-families` | `engine/decode.py`, `engine/families/`, `engine/decision.py`, `tests/fixtures/build.py` (synthetic models + images) | `types.py` |
| WP5 sources | `feat/wp5-sources` | `sources/` | `types.py` |
| WP7 assets | `feat/wp7-assets` | `tests/assets.json`, `tools/fetch_test_assets.py`, `tests/goldens/`, nightly workflow | WP4a fixtures |
| WP4b engine | `feat/wp4b-engine` | `engine/protocol.py`, `cell*.py`, `supervisor.py`, `scheduler.py`, `residency.py` | WP1, WP2, WP4a merged |
| WP6 wiring | `feat/wp6-wiring` | `ImageProcessor.py`, `outputs/`, `commands.py`, `metrics.py`, `health.py`, `connectivity.py`, docs | all above merged |
| WP8 camera playlist | camera-adapter `feat/sim-playlist` | `src/backend/sim.rs`, config, docs | — |
| WP9 replicator cleanup | file-replicator `fix/cleanup-failure-not-success` | `src/instance/worker.rs`, state, docs | — |

Every WP ships its own tests at ≥ 90% for the files it owns, updates this LLD if an interface
must change (and says so in the PR), and never edits files owned by another WP. Shared files
(`pyproject.toml` dependencies, `requirements.txt`) may be appended under a `# WPn` comment.
