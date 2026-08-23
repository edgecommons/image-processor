# Reference — Messaging interface and CLI

Every topic and message this component publishes or accepts, and the CLI flags it runs under.
Addressing follows the Unified Namespace: `ecv1/{device}/{component}/{instance}/{class}[/channel]`.

- `{device}` — the resolved Thing name (the last `hierarchy` level, or `-t` directly).
- `{component}` — `image-processor`, set by `component.token`. It is a separate identifier from the
  Greengrass component name (`com.mbreissi.edgecommons.ImageProcessor`), which never appears on the
  wire.
- `{instance}` — the route id for everything a route publishes; absent on the component-scope
  surfaces (the `state` keepalive, `metric`, and the component command inbox).

## Envelope

Every message is the EdgeCommons protobuf envelope `{header, identity, tags, body}`. A subscriber
that prints raw payloads sees bytes; decode them with the library
(`edgecommons.messaging.message.Message.from_bytes`) or use `ec-uns-cmd` for commands.

## Topics

| Class | Message | Direction | Topic | Reply |
|-------|---------|-----------|-------|-------|
| `app` | `ImageInferenceResult` | component → bus | `ecv1/{device}/image-processor/{routeId}/app/inference/result` | — |
| `app` | `ImageInferenceResult` | component → requester | the request's `reply_to`, correlated | — |
| `app` | trigger request | bus → component | whatever `source.subscribe` names | the bounded result, when the request carries `reply_to` |
| `app` | `ImageCaptured` | bus → component | `ecv1/{device}/camera-adapter/{instance}/app/image/captured` | — |
| `data` | `SouthboundSignalUpdate` | component → bus | `ecv1/{device}/image-processor/{routeId}/data/{signalId}` | — |
| `evt` | operator conditions | component → bus | `ecv1/{device}/image-processor/{routeId}/evt/{severity}/{type}` | — |
| `cmd` | the verbs below | bus → component | `ecv1/{device}/image-processor/{instance}/cmd/{verb}` | `{ok, result}` or `{ok, error}` |
| `cmd` | `sb/capture-status` | component → camera | `ecv1/{device}/camera-adapter/{instance}/cmd/sb/capture-status` | the camera's paged answer |
| `metric` | the eight groups | component → bus (automatic) | `ecv1/{device}/image-processor/metric/{group}` | — |
| `state` | keepalive | component → bus (automatic) | `ecv1/{device}/image-processor/state` | — |

`state`, `metric`, `cfg`, and `log` are library-owned reserved classes: a component publish
directly to them is rejected.

## The inference result (`app/inference/result`)

Header name `ImageInferenceResult`, version `1.0`.
`schemas/inference-result.schema.json` is the contract; the component validates every body against
it before the message is prepared.

This is the authoritative, cleanup-gating output. It is prepared once, its exact bytes stored in
the ledger, and published with positive transport confirmation; the input is archived, deleted, or
quarantined only after that. A crash between transport confirmation and the local acknowledgement
may publish a duplicate, so consumers deduplicate on `inferenceId`.

```json
{
  "schemaVersion": "1.0",
  "inferenceId": "blvuf6ru3i6jzmmksre6hkgccm",
  "routeId": "clearance-cam-01",
  "status": "SUCCEEDED",
  "source": {
    "kind": "spool",
    "captureId": "cap_0001",
    "cameraId": "cam-01",
    "relativePath": "2026/08/23/cap-0001.png",
    "bytes": 112,
    "sha256": "1ce61926…",
    "correlationId": "corr_cap_0001",
    "capturedAtMs": 1787486400512
  },
  "model": {
    "id": "synthetic-anomaly-scalar",
    "version": "1.0.0",
    "digest": "sha256:7dfefce8…",
    "runtime": "onnxruntime",
    "providers": ["CPUExecutionProvider"],
    "gpu": null,
    "transformVersion": "1"
  },
  "decision": { "outcome": "CLEAR", "pass": true, "confidence": 0.0, "threshold": 0.05, "rule": "pass" },
  "outputs": {
    "family": "anomaly",
    "anomaly": { "score": 0.0, "threshold": 0.05, "anomalous": false, "direction": "higherIsAnomalous" },
    "truncated": false
  },
  "timingsMs": { "queue": 58.0, "modelLoad": 51.2, "preprocess": 11.7, "inference": 0.1, "postprocess": 23.5, "total": 93.5 },
  "artifacts": {
    "evidenceId": "blvuf6ru3i6jzmmksre6hkgccm",
    "localRelativePath": "cap-0001.png.inference.json",
    "sha256": "ad659011…",
    "bytes": 1805
  }
}
```

| Field | Meaning |
|---|---|
| `inferenceId` | The stable identity of this inference, derived from the route, the capture or source digest, and the model digest. A retry reuses it; reprocessing under a different model digest produces a different one. |
| `status` | `SUCCEEDED` carries `decision` and `outputs`; `FAILED` carries `error`, and its decision is never `CLEAR`. |
| `source.kind` | `spool` (discovered in a watched root), `inline` (carried in a trigger message), or `reference` (named by a trigger message and resolved under `fileRoot`). |
| `source.sha256` | The verified digest of the bytes the executor cell checked before it decoded anything. |
| `model.providers` | The session's actual provider assignment, in the order the runtime reports it — never the configured preference. |
| `model.gpu` | `{deviceId, class}`, or `null` on a CPU session. |
| `decision.outcome` | `CLEAR` (the rules evaluated and passed), `HOLD` (they did not pass, or could not be evaluated), `FAIL` (the inference itself failed). |
| `outputs` | Only the declared task family's collection is populated. Masks and tensors are never published; segmentation reports pixel counts and regions. |
| `outputs.truncated` | Whether collections were bounded to fit the message budget. When it is `true`, `artifacts` names the sidecar holding the full result. |
| `artifacts` | The evidence sidecar this result belongs to, present whenever the route writes one. |
| `error` | `{code, message, class}` on a failure. `class` is `transient`, `permanent`, or `contaminating`. A `permanent` failure repeats on every attempt; `code` says whether the image or the deployment caused it, and that is what decides the completion action. |

A failed result is still published, because a consumer that hears nothing cannot tell a held image
from a component that stopped:

```json
{
  "schemaVersion": "1.0",
  "inferenceId": "…",
  "status": "FAILED",
  "decision": { "outcome": "HOLD", "pass": false, "confidence": null, "threshold": null, "rule": "DECODE_FAILED" },
  "error": { "code": "DECODE_FAILED", "message": "the image is truncated", "class": "permanent" },
  "…": "…"
}
```

## Trigger requests

A trigger route accepts one image per message, in one of two forms.

| Body | Handling |
|---|---|
| An inline image: an opaque binary body, or a structured body whose `image` field is bytes | Bounded by the core envelope's 64 KiB binary-body cap. The bytes are hashed, written into processor-owned staging under a digest-derived name, and from then on the job is an ordinary file job. |
| A file reference: `{"relativePath": "…", "sha256": "…", "bytes": n}` | `relativePath` resolves under the route's `fileRoot` with containment enforced, and `sha256` and `bytes` are verified against the file before the job is admitted. |

Anything that is neither is refused with a stable reason rather than guessed at:
`MALFORMED_BODY`, `EMPTY_BODY`, `INLINE_TOO_LARGE`, `NO_FILE_ROOT`, `SIZE_MISMATCH`,
`DIGEST_MISMATCH`, `CHANGED_DURING_READ`, or a path-containment code.

When the request envelope carries `reply_to`, the route publishes the bounded result summary there
under the request's correlation id, in addition to the normal outputs.

## The decision mirror (`data` class)

Each configured `outputs.decisionSignals` entry publishes a `SouthboundSignalUpdate` on
`ecv1/{device}/image-processor/{routeId}/data/{signalId}`, with `value` read out of the committed
result body by JSONPath:

```json
{
  "signal": { "id": "line-clearance/pass" },
  "samples": [{ "value": true, "quality": "GOOD", "serverTs": "2026-08-23T12:13:05.710038Z" }]
}
```

The mirror is best effort and never cleanup-gating. A path that resolves nothing publishes nothing,
a path that resolves a document rather than a scalar publishes nothing, and a failed publication is
counted rather than retried. A reading derived from a `FAILED` result carries `quality: BAD`.

## Events (`evt` class)

Severity derives the channel, so the topic and the body can never disagree. A condition that is a
state rather than an occurrence is raised and cleared as an alarm, and only on a transition.

| Type | Severity | When |
|---|---|---|
| `model-staging-failed` | critical | A bundle could not be fetched, verified, extracted, or interpreted. |
| `model-warmup-failed` | critical | A staged bundle failed its golden warmup, so no route switched to it. |
| `model-activated` | info | A route switched to a new model generation. |
| `executor-unavailable` | critical (alarm) | No healthy executor cell can serve the routes that need one. |
| `executor-recycled` | warning | A cell was drained and restarted. |
| `route-degraded` | critical (alarm) | An enabled, unpaused route cannot execute. |
| `queue-age-exceeded` | warning (alarm) | The oldest queued job passed `scheduler.queueAgeWarningSecs`. |
| `publish-backlog` | warning (alarm) | The outbox is approaching `publish.outboxCapacity`. |
| `publish-exhausted` | critical | A result spent `publish.maxAttempts`. The input is retained for an operator retry. |
| `evidence-failed` | critical | The evidence sidecar could not be installed, so nothing was committed. |
| `cleanup-failed` | critical | A completion action failed. The job is `CLEANUP_FAILED`, never success. |
| `input-rejected` | warning | An input can never be admitted as it stands. |
| `inference-failed` | critical | An inference ended without a result. Its decision is `HOLD`. |
| `disk-pressure` | warning (alarm) | The state or cache filesystem is low on free space. |
| `gpu-pressure` | warning (alarm) | A device cannot admit a model a route needs. |

Success is not an event. Every context value is a bounded scalar: no image bytes, no tensors, no
credentials, and no unbounded model output reach this class.

```jsonc
"body": {
  "severity": "critical", "type": "cleanup-failed",
  "message": "the archive of blvuf6ru3i6jzmmksre6hkgccm failed",
  "timestamp": "2026-08-23T12:14:13Z",
  "context": { "inferenceId": "blvuf6ru3i6jzmmksre6hkgccm", "action": "archive", "error": "COLLISION: …" }
}
```

## Commands (`cmd` class)

A request is a `cmd` envelope on `ecv1/{device}/image-processor/{instance}/cmd/{verb}` whose
`header.name` equals the verb. A request carrying `header.reply_to` gets a reply there under its
correlation id: `{"ok": true, "result": {…}}` or
`{"ok": false, "error": {"code": …, "message": …}}`. A request without `reply_to` is
fire-and-forget.

`ec-uns-cmd` builds the envelope for you:

```bash
ec-uns-cmd --device smoke-device --component image-processor get-queue --body '{"max": 20}'
ec-uns-cmd --device smoke-device --component image-processor --instance clearance-cam-01 pause
```

### Library built-ins

`ping`, `describe`, `reload-config`, `get-configuration`, and `status`. `status` answers with this
component's per-route connectivity — the same sample the `state` keepalive pushes, so a pulled
answer can never disagree with a pushed one:

```json
{
  "status": "RUNNING",
  "uptimeSecs": 82,
  "instances": [
    {
      "instance": "clearance-cam-01",
      "connected": true,
      "state": "ONLINE",
      "detail": "/var/spool/camera-adapter/cam-01",
      "attributes": {
        "desiredGeneration": "sha256:7dfefce8…",
        "activeGeneration": "sha256:7dfefce8…",
        "sourceReachable": true,
        "executorHealthy": true,
        "queued": 0,
        "oldestAgeSecs": 0.0,
        "paused": false
      }
    }
  ]
}
```

`state` is `ONLINE`, `STAGING` (configuration is ahead of the running generation; the route keeps
serving the last known good model), `DEGRADED` (it cannot decide right now), or `DISABLED`.

### Component verbs

| Verb | Scope | Request body | Reply |
|---|---|---|---|
| `get-models` | component | `{cursor?, max?}` | `{models: [{id, version, digest, uri, staged, warmed, warmupSamples, loadMs, deviceMiB, activeRoutes, stagingRoutes, rollback, error}], nextCursor, total}` |
| `get-queue` | both | `{route?, states?, cursor?, max?}` | `{route, jobs: [{inferenceId, route, state, attempts, source, model, lastError}], nextCursor, counts, scheduler}` |
| `trigger-rescan` | both | `{route?}` | `{route, discovered}` |
| `preload-model` | component | `{id?, digest?}` | deferred: `{id, version, digest, staged, warmed, routesSwitched}` |
| `evict-model` | component | `{digest}` | `{evicted, digest, cells, reason}`; a leased generation is `CONFLICT` |
| `reload-model-catalog` | component | `{}` | deferred: `{routesSwitched, collected, models}` |
| `set-route-activation-override` | instance | `{enabled}` — `true`, `false`, or `null` to clear | `{route, configured, override, effective}` |
| `retry-publication` | both | `{route?, inferenceId?}` | deferred: `{returned, published}` |
| `retry-cleanup` | both | `{route?, inferenceId?}` | deferred: `{repaired, stillFailed}` |
| `reconcile` | both | `{route?}` | deferred: `{reconciled, counts}` |
| `pause` | both | `{route?}` | `{paused: true, routes}` |
| `resume` | both | `{route?}` | `{paused: false, routes}` |

A component-scope verb addressed to an instance is refused before the handler runs, and so is an
instance-scope verb that names no route on a component with several. A slow verb takes a deferred
reply rather than blocking the inbox: the request is accepted immediately and the reply arrives
when the work finishes, within the core's 31-minute bound.

Pagination is an opaque `cursor` and a `max` (default 100, ceiling 500). A reply never grows with
how busy the component has been.

### Error codes

| Code | Meaning |
|---|---|
| `BAD_ARGS` | The arguments are unusable: a `max` that is not a positive integer, a cursor that is not a string, a state name that is not one, an `enabled` that is not a boolean or null, or a verb addressed to the wrong scope. |
| `NOT_FOUND` | The request names a route or a model this component does not have. |
| `CONFLICT` | The request is well formed but cannot be honoured now — an eviction of a generation still leased by draining work. |
| `OPERATION_FAILED` | A deferred operation ran and failed. The message carries the stable code from the subsystem, such as `DIGEST_MISMATCH` or `WARMUP_FAILED`. |
| `UNKNOWN_VERB`, `HANDLER_ERROR`, `RELOAD_FAILED`, `NO_CONFIG` | The library's own codes. |

## Metrics and state

`metric` and `state` are automatic and library-owned. See
[Reference — Metrics](metrics.md) for the eight groups and their measures; the `state` keepalive
carries the same `instances[]` array `status` returns.

## CLI

| Flag | Values | Notes |
|------|--------|-------|
| `--platform` | `GREENGRASS` \| `HOST` \| `KUBERNETES` \| `auto` | Default `auto`. |
| `--transport` | `MQTT [path]` \| `IPC` | HOST and Kubernetes use MQTT; the path is the messaging config. |
| `-c/--config` | `FILE <path>` \| `ENV` \| `GG_CONFIG` \| `SHADOW` \| `CONFIG_COMPONENT` \| `CONFIGMAP` | Default from the platform profile. |
| `-t/--thing` | `<name>` | IoT Thing name; the `{device}` token of every UNS topic. |
