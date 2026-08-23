# Reference — Configuration

Every option ImageProcessor understands. The contract is `config.schema.json`; this page explains
what each key means. For the shape of what the component publishes, see
[messaging-interface.md](messaging-interface.md).

## Config source

The component reads one JSON document from `-c/--config`, defaulting by platform: `HOST` reads
`FILE`, `GREENGRASS` reads `GG_CONFIG`, and `KUBERNETES` reads `CONFIGMAP`. ImageProcessor's own
settings live under `component`; the sibling sections (`tags`, `hierarchy`, `identity`, `topic`,
`messaging`, `logging`, `metricEmission`, `heartbeat`, `credentials`) are standard EdgeCommons
sections owned by the canonical schema and are not redeclared in `config.schema.json`.

Validation is fail-closed and runs before a configuration generation becomes current. An unknown
key, a route that names a model you did not declare, two routes that claim the same spool, or a
provider policy the device cannot satisfy rejects the candidate; the running configuration is left
untouched and the rejection code appears in the log. The codes are listed under
[Validation](#validation).

## `component.token`

The UNS component token: the `{component}` segment of every topic this component publishes on, and
the `identity.component` field of every message envelope. Set it to `image-processor`. The
Greengrass component name is the reverse-DNS `com.mbreissi.edgecommons.ImageProcessor` and never
reaches the wire.

## `component.global.paths`

Where the component keeps the state it owns. Each path is absolute, and each stays outside every
spool root the component mutates.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stateDb` | string | `/var/lib/edgecommons/image-processor/state.db` | The SQLite job ledger and outbox. The database runs in WAL mode, so the directory also holds the `-wal` and `-shm` files. This path cannot change on a reload — the ledger cannot move under a running process. |
| `modelCache` | string | `/var/lib/edgecommons/image-processor/models` | The content-addressed bundle cache. Each verified bundle lives under its own tarball digest, so two routes that name the same digest share one copy on disk and one session on the GPU. |
| `staging` | string | `/var/lib/edgecommons/image-processor/staging` | Processor-owned scratch space: downloaded bundles before verification, and the immutable copies of trigger images that executor cells read. Every route's `source.inlineStaging` sits under this directory. |

## `component.global.runtime`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `providers` | string[] | `["CUDAExecutionProvider"]` | The execution providers a session may use, in preference order. Allowed values are `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`. A bundle declares the providers it permits; a route runs only where the two agree. |
| `requiredProvider` | string \| null | `"CUDAExecutionProvider"` | The provider a session must actually be assigned. A route whose session lands on anything else fails closed rather than degrading silently. Set it to `null` to accept any provider listed in `providers`. |
| `allowCpuOnly` | boolean | `false` | Whether the component may run inference on `CPUExecutionProvider` alone. This is a development setting: on a production route it turns a missing GPU into a silent throughput collapse instead of a reported failure. |
| `executorCellsPerGpu` | integer ≥ 1 | `1` | How many executor subprocesses serve one GPU. One cell per GPU owns one CUDA context and one session cache; more cells raise device-memory overhead and widen the blast radius of a cell restart. |
| `loadConcurrencyPerGpu` | integer ≥ 1 | `1` | How many model loads run concurrently on one GPU. A load allocates its peak device memory before it reaches steady state, so raising this raises the chance that two peaks collide. |

## `component.global.gpu`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `devices` | string[] | `["0"]` | The devices the component may place sessions on, as CUDA ordinals (`"0"`) or device UUIDs. The component starts one executor cell per listed device. An empty list is the CPU-only development path and requires `runtime.allowCpuOnly`; a deployment that serves decisions names its devices. |
| `residentMemoryBudgetPercent` | integer 1–100 | `80` | The share of each device's total memory that resident model sessions may occupy. Each executor cell's CUDA context comes off this share once, because it is the cell's overhead rather than any model's. The remainder absorbs activation peaks, allocator fragmentation, and any co-located process. |
| `reserveMiB` | integer ≥ 0 | `2048` | Device memory held back from the residency budget in absolute terms. Admission subtracts this reserve from free memory before it compares against a model's estimate, so a load never consumes the last free megabyte. |

## `component.global.scheduler`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `maxBatchSize` | integer ≥ 1 | `1` | The largest micro-batch the scheduler forms from same-model, compatible-shape jobs. Batching applies only when the bundle manifest declares a dynamic batch axis; `1` disables it. |
| `maxBatchLatencyMs` | integer ≥ 0 | `20` | How long a partially filled batch waits for a sibling job before it runs as is. This is the latency a batch may add, not a queueing timeout. |
| `hotTtlSecs` | integer ≥ 0 | `120` | How long an idle session stays resident before it becomes an eviction candidate. Reloading a bundle costs seconds, so a short value on a busy device trades latency for capacity. |
| `minResidencySecs` | integer ≥ 0 | `15` | The minimum time a freshly loaded session is protected from eviction. It stops two competing models from evicting each other on every arrival. |
| `maxAttempts` | integer ≥ 1 | `5` | How many times a job may be attempted before its retry budget is exhausted and it moves to `PROCESSING_EXHAUSTED`. Attempts count transient execution and model failures; a permanent input failure never retries. |
| `retryBackoffSecs` | number > 0 | `2` | The delay before the first retry. Later retries back off exponentially from this value. |
| `maxRetryBackoffSecs` | number > 0 | `300` | The ceiling on the exponential retry delay. |
| `queueAgeWarningSecs` | integer ≥ 1 | `300` | The age of the oldest queued job at which a route reports a degraded condition on `evt`. Backlog never discards admitted work, so this threshold is how you learn that evidence delivery is falling behind. |

## `component.global.discovery`

How a spool route finds work. Filesystem state is authoritative: OS notifications only nudge a
walk, and the periodic walk runs whether or not a notification ever arrives, so a missed event
delays a job rather than losing it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rescanSecs` | integer ≥ 1 | `60` | How often a route walks its root with no notification at all. This is the interval within which a file whose event was dropped is still found. |
| `debounceMs` | integer ≥ 0 | `500` | How long notifications must stop before a nudged walk runs. A camera writing a burst of files produces one walk rather than one per file; `0` walks on every nudge. |

## `component.global.publish`

The result on `app/inference/result` is the only cleanup-gating output. It is prepared once, stored
byte for byte in the ledger, and published with positive transport confirmation.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `confirmationTimeoutSecs` | number > 0 | `10` | How long one publish attempt waits for a PUBACK at QoS 1, or for the Greengrass IPC publish operation to complete, before it counts as a failed attempt. The stored bytes are retried unchanged. |
| `maxAttempts` | integer ≥ 1 | `100` | How many publish attempts one outbox row makes before the job moves to `PUBLISH_EXHAUSTED` and waits for the `retry-publication` command. The row and its bytes are kept. |
| `outboxCapacity` | integer ≥ 1 | `100000` | How many rows the outbox holds. This bounds the count of pending publications; `outboxReserveBudgetMiB` bounds their size. |
| `outboxReserveBudgetMiB` | integer ≥ 1 | `256` | How much outbox and evidence capacity admission may hold in total. A job is admitted only after it reserves room for the largest result its route can produce, so a finished job is never stranded by a full outbox; when the reserve cannot be met the component stops claiming new work and reports the pressure. |

## `component.global.signing`

Bundle digests are verified always. Signatures are Ed25519 detached signatures over the bundle's
`manifest.json`, and trust is a list of public keys named by `keyId`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `required` | boolean | `false` | Whether a bundle without a valid signature from a trusted key is refused at staging. A regulated deployment sets it to `true`; with `false` a signature that is present is still verified. |
| `trustedKeys` | object[] | `[]` | The public keys whose signatures the component accepts. A manifest names the key that signed it by `keyId`. |
| `trustedKeys[].keyId` | string | — | The identifier a manifest uses to name the key that signed it, such as `pharma-model-publisher-1`. |
| `trustedKeys[].publicKey` | string \| secret ref | — | The Ed25519 public key, either inline as a PEM or base64 string, or as a [secret reference](#secret-references). |

Setting `required` to `true` with an empty `trustedKeys` rejects the candidate: no bundle could be
verified.

## `component.global.modelSources`

The transport controls that apply to every `models[].uri`. They bound what a configuration change
can make the component download.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `allowedSchemes` | string[] | `["s3", "https", "file"]` | The URI schemes a model source may use. `file` also covers a plain absolute path. |
| `allowedUriPrefixes` | string[] | `[]` | An allowlist of URI prefixes. When it is non-empty, every `models[].uri` starts with one of them. An empty list allows any URI whose scheme is permitted. |
| `verifyTls` | boolean | `true` | Whether `https://` sources verify the server certificate chain and hostname. Turning it off is a development setting. |

## `component.global.models[]`

The bundles this device makes available. The component stages each entry into the content-addressed
cache, verifies it, warms it, and only then lets a route switch to it. A route names the entry it
uses through `modelRef`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | — | The model's published id, matched by a route's `modelRef.id`. |
| `version` | string | — | The model's published version. Identity is the digest, so a version is a label for people. |
| `digest` | string | — | The expected SHA-256 of the bundle tarball, as `sha256:<64 lowercase hex>`. The component verifies it before it extracts anything. |
| `uri` | string | — | Where to fetch the tarball: an `s3://` object, an `https://` URL, a `file://` URL, or a local absolute path. It must satisfy `modelSources`. |
| `credentials` | secret ref | *(ambient)* | The credentials for the source, resolved through the credentials vault at fetch time. Omit it to use ambient or token-exchange credentials. |
| `activation.requireWarmup` | boolean | `true` | Whether the bundle's golden warmup samples reproduce their expected outputs on the target provider before a route may switch to it. A model that fails warmup degrades its routes instead of serving wrong answers. |
| `activation.retainForRollback` | boolean | `true` | Whether the previous generation stays in the cache after the switch. Garbage collection never removes a retained rollback generation. |

Two entries may share an `id` with different versions. Two entries with the same `id` and `version`
reject the candidate.

## `component.global.completionDefaults`

The completion actions every route inherits. A route overrides any of them in its own `completion`
block, which is also where the archive and quarantine directories live.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `onSuccess` | action | `archive` | What happens to the input after the result is published and confirmed. |
| `onInvalidInput` | action | `quarantine` | What happens to an input that can never succeed: a corrupt or undecodable image, a digest that does not match its sidecar, an input over the byte bound, an input that is no longer there, or a reference that escapes its root. |
| `onOperationalFailure` | action | `retainInPlace` | What happens when inference exhausts its retry budget, and what happens to a permanent failure the image did not cause — a model, bundle, provider, GPU, runtime, or postprocess-schema failure. The input is intact, so retaining it keeps a later reprocess possible once you repair the deployment. |
| `onPublishFailure` | `retainInPlace` | `retainInPlace` | What happens when publication exhausts its attempts. `retainInPlace` is the only value: the result is already committed and the outbox row is kept, so the input stays where it is until you run `retry-publication`. Any other value is rejected with `ON_PUBLISH_FAILURE_NOT_SUPPORTED`. |
| `onCollision` | `fail` \| `suffix` | `fail` | What happens when an action's target path already holds a different object. `fail`: the job records `CLEANUP_FAILED` and both files are left intact, so it waits for `retry-cleanup` or an operator. `suffix`: the input is installed beside the occupant under a deterministic name derived from its own digest, so the move completes and the name is reproducible from the ledger. Neither policy overwrites the object already there. |

An **action** is one of:

| Action | Meaning |
|---|---|
| `archive` | Move the input and its companion files to `completion.archiveDir`. A route that resolves to `archive` without an `archiveDir` rejects the candidate. |
| `delete` | Remove the input and its companion files. |
| `retainInPlace` | Leave them where they are and record the outcome in the ledger. |
| `quarantine` | Move them to `completion.failedDir`. With no `failedDir`, quarantine happens in place: the ledger records the terminal state and the file stays put, which is the right answer when the input directory is itself the evidence store. |

## `component.instances[]` (a route)

One instance is one **route**: it binds one input source to one immutable model version, one set of
outputs, and one completion policy. Routes are independent, so a backlog on one never stalls
another.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | — | Unique route id, lower-kebab (`^[a-z0-9]+(?:-[a-z0-9]+)*$`). It is the `{instance}` token of this route's UNS topics and of the identity stamped on everything it publishes. |
| `enabled` | boolean | `true` | Whether the route claims new work. A disabled route keeps its ledger rows and its durable state, and in-flight jobs finish. Use it to retire a camera without losing its history. |
| `priority` | integer 0–1000 | `0` | The route's weight when the scheduler picks among ready jobs. Higher wins, and job age is weighted alongside it so a low-priority route never starves. |
| `source` | object | — | Where the images come from. `kind` discriminates: [`spool`](#a-spool-source) or [`trigger`](#a-trigger-source). |
| `modelRef` | object | — | The model version this route runs, as `{id, version, digest}`. It names an entry of `component.global.models[]`; a route is pinned to one immutable model generation at a time. |
| `outputs` | object | *(see below)* | What the route emits besides the authoritative result. |
| `completion` | object | *(inherited)* | What happens to the input after a job reaches a terminal outcome. |
| `reprocessExistingOnModelChange` | boolean | `false` | Whether activating a new model digest replays inputs that already reached a terminal state under the previous digest. It is off by default, because a model upgrade on a busy spool would otherwise re-infer the entire archive; you can still request a replay with a command. |

### A spool source

A directory of finalized image files that this component owns. Filesystem state is authoritative:
OS notifications only nudge a scan, and a periodic authoritative rescan recovers anything a
notification missed. Only regular files under the root are accepted, and symlinks and path escapes
are rejected.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"spool"` | — | Selects this source. |
| `root` | string | — | The absolute directory tree this route reads. Exactly one component may mutate a given spool, so no two routes with a mutating completion policy name overlapping roots. |
| `include` | string[] | `["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.tif", "**/*.tiff"]` | Glob patterns, matched against the path relative to `root` with `/` separators. A file matches at least one pattern to be considered. |
| `exclude` | string[] | `[]` | Glob patterns that veto a match from `include`. Use it to skip a camera's own working subdirectory. |
| `readiness` | object | — | How the route decides that a file is finalized. |
| `camera` | object | *(none)* | The camera whose spool this route reads. |

`readiness`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | enum | — | The evidence the route accepts as proof of finalization; see the table below. |
| `quietSecs` | number > 0 | `5` | How long size and mtime stay unchanged before `stability` mode calls a file finalized. The other modes ignore it. |
| `markerSuffix` | string | `.done` | The suffix appended to the image path to form the companion marker file in `marker` mode. The other modes ignore it. |

| `mode` | Meaning |
|---|---|
| `cameraSidecar` | The camera metadata sidecar `<image>.json` exists, parses, and its `image.bytes` and `image.sha256` match the file. camera-adapter writes the sidecar before the image becomes visible, so a visible image with a sidecar is complete by construction. A regulated route uses this mode. |
| `cameraStatus` | A `SUCCEEDED` record from the camera's paged `sb/capture-status` names the file and its digest verifies. It needs a `camera` binding. |
| `marker` | A companion file `<path><markerSuffix>` exists. |
| `stability` | Size and mtime are unchanged for `quietSecs`. This is the weakest rule and the only one that infers finalization instead of observing it, so it is not permitted on a camera-bound route. |

`camera`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `component` | string | `camera-adapter` | The UNS `{component}` token of the camera that writes this spool. |
| `instance` | string | — | The UNS `{instance}` token of the camera route that writes this spool. |
| `subscribeAnnouncements` | boolean | `true` | Whether to subscribe to the camera's `app/image/captured` announcements as a discovery hint. The processor maps `image.relativePath` under its own `root`, never trusts `image.absolutePath`, and verifies `image.bytes` and `image.sha256` against the file. |
| `reconcileCaptureStatusSecs` | integer ≥ 0 | `30` | How often to page the camera's `sb/capture-status` for captures the hints missed. Reconciliation follows every `nextCursor`, deduplicates by `captureId`, and records its watermark. `0` disables reconciliation. |

A hint is not a queue: a lost hint delays a job, it never loses one, because the authoritative
rescan finds the file anyway.

### A trigger source

Topic filters whose messages each carry one image. The body is either an inline binary image, or a
file reference `{relativePath, sha256, bytes}` resolved under `fileRoot`. When the envelope carries
`reply_to`, the route publishes the bounded result summary as the correlated reply as well as on
its normal outputs.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"trigger"` | — | Selects this source. |
| `subscribe` | string[] | — | The topic filters this route consumes. Wildcards are allowed. |
| `fileRoot` | string | — | The absolute directory a file reference resolves under. `relativePath` is resolved inside this root and rejected if it escapes; `sha256` and `bytes` are verified against the file before the job is admitted. |
| `inlineStaging` | string | — | Where inline image bytes are written before inference. It sits under `component.global.paths.staging`, because the staged copy is processor-owned and immutable for the life of the job. |
| `maxInlineBytes` | integer 1–65536 | `65536` | The largest inline image this route accepts. The ceiling is the core envelope's binary-body cap of 65536 bytes; a larger image arrives as a file reference. |

### `outputs`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `writeResultSidecar` | boolean | `true` | Whether to install the evidence sidecar `<image>.inference.json` beside the input. The sidecar carries the full result even when the published message is a bounded summary, and it is immutable once installed. |
| `decisionSignals` | object[] | `[]` | Up to 32 normalized readings mirrored onto `data/<signalId>`. |
| `decisionSignals[].id` | string | — | The signal id. It becomes the channel of `ecv1/{device}/image-processor/{routeId}/data/<id>`, so it may contain `/` to nest but never a wildcard character. |
| `decisionSignals[].value` | string | — | A JSONPath expression evaluated against the committed result body. It starts at the document root `$`. |

The mirror is best effort and never cleanup-gating. A consumer that enforces a safety gate
subscribes to `app/inference/result` instead, and treats a missing, failed, stale, or unverified
inference as not clear.

### `completion`

The five action keys are the same as `component.global.completionDefaults`; unset keys inherit from
there. Two more keys are route-local:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `archiveDir` | string | *(none)* | Where `archive` moves the input and its companion files. It is required when any action resolves to `archive`, and it sits outside every spool root the component mutates so archived evidence is never rediscovered as new input. |
| `failedDir` | string | *(none)* | Where `quarantine` moves the input and its companion files. Leave it unset to quarantine in place. |

Every action is preceded by a persisted cleanup intent, so a crash mid-move is reconciled from
observed state rather than guessed at. A failed action is recorded as `CLEANUP_FAILED` and retried,
never reported as success.

## Secret references

Anywhere the tables above say **secret ref**, you may write a pointer to a credentials-vault entry
instead of an inline value:

```json
{ "$secret": "model-signing/publisher-1" }
```

Add `"field": "<key>"` to take one field of the secret's JSON document instead of its whole string
value. ImageProcessor resolves its own references at the moment it uses them, so the secret never
reaches the logged or published configuration snapshot.

## Validation

Every rejection carries a stable code. The most common ones:

| Code | Meaning |
|---|---|
| `SCHEMA_INVALID` | The document does not match `config.schema.json`. The message names the JSON pointer. |
| `UNKNOWN_KEY` | An object carries a key the schema does not declare. A typo is a mistake, not a no-op. |
| `MISSING_FIELD` | A required field is absent. |
| `INVALID_TYPE`, `INVALID_VALUE` | A field has the wrong type, or a value outside its permitted set or bounds. |
| `PATH_NOT_ABSOLUTE` | A path field is relative. POSIX, Windows drive, and UNC forms are all absolute. |
| `DUPLICATE_ROUTE_ID` | Two instances share an `id`. |
| `DUPLICATE_MODEL_ID` | Two `models[]` entries share an `id` and `version`. |
| `DUPLICATE_KEY_ID` | Two `signing.trustedKeys[]` entries share a `keyId`. |
| `UNRESOLVED_MODEL_REF` | A route's `modelRef` names no entry of `models[]` by id, version, and digest. |
| `OVERLAPPING_MUTATING_ROOTS` | Two routes with a mutating completion policy claim intersecting roots. Exactly one component may mutate a given spool. |
| `OUTPUT_INSIDE_SOURCE_ROOT` | A completion directory, the staging tree, or the model cache lies inside a root the component mutates, which would feed its own output back in as new input. |
| `MISSING_ARCHIVE_DIR` | A route archives its inputs but sets no `completion.archiveDir`. |
| `COMPLETION_DIR_NOT_CREATABLE` | A completion directory exists as a file, or its nearest existing ancestor is missing or not writable. |
| `INLINE_STAGING_NOT_CONTAINED` | A trigger route stages inline images outside `paths.staging`, or inside its own `fileRoot`. |
| `INLINE_LIMIT_EXCEEDED` | `maxInlineBytes` is above the core envelope's 64 KiB binary-body cap. |
| `STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE` | A camera-bound route uses `stability` readiness. |
| `CAMERA_BINDING_REQUIRED` | A route reads `cameraStatus` but names no `source.camera`. |
| `PROVIDER_POLICY_UNSATISFIED` | `requiredProvider` is not among `providers`, or CPU-only execution is configured without `allowCpuOnly`. |
| `MODEL_URI_SCHEME_NOT_ALLOWED`, `MODEL_URI_NOT_ALLOWED` | A `models[].uri` uses a scheme or a prefix `modelSources` does not allow. |
| `NO_TRUSTED_KEYS` | `signing.required` is set but `signing.trustedKeys` is empty. |
| `INVALID_SECRET_REF` | A `$secret` reference is malformed. |
| `INVALID_DECISION_SIGNAL` | A decision signal repeats an id, or its `value` is not a JSONPath starting at `$`. |
| `IMMUTABLE_PATH_CHANGED` | A reload moves `paths.stateDb`, `paths.modelCache`, or `paths.staging`. Restart the component to move them. |
| `CANDIDATE_NOT_OBJECT` | The candidate document is not a JSON object. |

## Complete example

```jsonc
{
  "hierarchy": { "levels": ["site", "device"] },
  "identity": { "site": "dallas" },
  "metricEmission": { "target": "messaging" },
  "component": {
    "token": "image-processor",
    "global": {
      "paths": {
        "stateDb": "/var/lib/edgecommons/image-processor/state.db",
        "modelCache": "/var/lib/edgecommons/image-processor/models",
        "staging": "/var/lib/edgecommons/image-processor/staging"
      },
      "runtime": {
        "providers": ["CUDAExecutionProvider"],
        "requiredProvider": "CUDAExecutionProvider",
        "allowCpuOnly": false,
        "executorCellsPerGpu": 1,
        "loadConcurrencyPerGpu": 1
      },
      "gpu": { "devices": ["0"], "residentMemoryBudgetPercent": 80, "reserveMiB": 2048 },
      "scheduler": { "maxBatchLatencyMs": 20, "hotTtlSecs": 120, "minResidencySecs": 15 },
      "discovery": { "rescanSecs": 60, "debounceMs": 500 },
      "publish": { "confirmationTimeoutSecs": 10, "outboxReserveBudgetMiB": 256 },
      "signing": {
        "required": true,
        "trustedKeys": [
          { "keyId": "pharma-model-publisher-1", "publicKey": { "$secret": "model-signing/publisher-1" } }
        ]
      },
      "modelSources": { "allowedUriPrefixes": ["s3://approved-models/"] },
      "models": [
        {
          "id": "line-clearance-cam-01",
          "version": "2026.08.20",
          "digest": "sha256:1f0c9a2b3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8",
          "uri": "s3://approved-models/line-clearance-cam-01/2026.08.20.tar.gz",
          "credentials": { "$secret": "model-source/approved-models" },
          "activation": { "requireWarmup": true, "retainForRollback": true }
        }
      ],
      "completionDefaults": {
        "onSuccess": "archive",
        "onInvalidInput": "quarantine",
        "onOperationalFailure": "retainInPlace",
        "onPublishFailure": "retainInPlace",
        "onCollision": "fail"
      }
    },
    "instances": [
      {
        "id": "clearance-cam-01",
        "priority": 100,
        "source": {
          "kind": "spool",
          "root": "/var/spool/camera-adapter/cam-01",
          "include": ["**/*.jpg", "**/*.png", "**/*.tiff"],
          "readiness": { "mode": "cameraSidecar" },
          "camera": { "component": "camera-adapter", "instance": "cam-01", "subscribeAnnouncements": true, "reconcileCaptureStatusSecs": 30 }
        },
        "modelRef": {
          "id": "line-clearance-cam-01",
          "version": "2026.08.20",
          "digest": "sha256:1f0c9a2b3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8"
        },
        "outputs": {
          "writeResultSidecar": true,
          "decisionSignals": [
            { "id": "line-clearance/pass", "value": "$.decision.pass" },
            { "id": "line-clearance/confidence", "value": "$.decision.confidence" },
            { "id": "line-clearance/status", "value": "$.status" }
          ]
        },
        "completion": {
          "onSuccess": "archive",
          "archiveDir": "/var/spool/image-processor/processed/cam-01",
          "failedDir": "/var/spool/image-processor/failed/cam-01"
        }
      },
      {
        "id": "adhoc-inspect",
        "source": {
          "kind": "trigger",
          "subscribe": ["ecv1/+/inspection-ui/+/app/inspect/request"],
          "fileRoot": "/var/spool/inspection",
          "inlineStaging": "/var/lib/edgecommons/image-processor/staging/adhoc"
        },
        "modelRef": {
          "id": "line-clearance-cam-01",
          "version": "2026.08.20",
          "digest": "sha256:1f0c9a2b3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8"
        },
        "outputs": { "writeResultSidecar": false, "decisionSignals": [] },
        "completion": { "onSuccess": "delete" }
      }
    ]
  }
}
```
