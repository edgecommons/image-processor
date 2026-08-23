# Sample configurations

Five shapes, each one a whole `component` block you can paste into a deployment. For the
exhaustive option list see [reference/configuration.md](reference/configuration.md); for how a
route decides an image is finished, see [explanation.md](explanation.md).

Two files make a HOST run: the component configuration (`-c FILE <path>`) and the messaging
configuration (`--transport MQTT <path>`). They are separate because the broker a device talks to
is a property of the device, not of the component.

```json
{ "messaging": { "local": { "host": "localhost", "port": 1883, "clientId": "image-processor-local" } } }
```

## The development profile: CPU, a local bundle, no network

This is `test-configs/config.json`. `gpu.devices` is empty and `runtime.allowCpuOnly` is set, which
is the only way to run inference on the CPU; a deployment that serves decisions names its devices
and drops the flag, so a route whose provider policy cannot be satisfied fails closed rather than
answering from the CPU. `modelSources.allowedSchemes` is `["file"]`, so nothing can make this
component fetch over the network.

```jsonc
{
  "component": {
    "token": "image-processor",
    "global": {
      "paths": {
        "stateDb": "/var/lib/edgecommons/image-processor/state.db",
        "modelCache": "/var/lib/edgecommons/image-processor/models",
        "staging": "/var/lib/edgecommons/image-processor/staging"
      },
      "runtime": {
        "providers": ["CPUExecutionProvider"],
        "requiredProvider": "CPUExecutionProvider",
        "allowCpuOnly": true
      },
      "gpu": { "devices": [], "reserveMiB": 512 },
      "discovery": { "rescanSecs": 15, "debounceMs": 250 },
      "publish": { "confirmationTimeoutSecs": 10, "maxAttempts": 20, "outboxCapacity": 10000 },
      "modelSources": { "allowedSchemes": ["file"] },
      "models": [
        {
          "id": "synthetic-anomaly-scalar",
          "version": "1.0.0",
          "digest": "sha256:4a87394f34ab6c1b1f17c9d048ebed277e5fdae8f76823a8d2371cfd57178ee1",
          "uri": "/var/lib/edgecommons/image-processor/bundles/synthetic-anomaly-scalar-1.0.0.tar"
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
    "instances": [ /* the camera route and the trigger route below */ ]
  }
}
```

## A camera route

The regulated shape: `cameraSidecar` readiness, a camera binding so the route hears announcements
and reconciles the camera's capture status, three mirrored decision signals, and an archive
directory outside the spool so archived evidence is never rediscovered as new input.

```jsonc
{
  "id": "clearance-cam-01",
  "priority": 100,
  "source": {
    "kind": "spool",
    "root": "/var/spool/camera-adapter/cam-01",
    "include": ["**/*.jpg", "**/*.jpeg", "**/*.png"],
    "readiness": { "mode": "cameraSidecar" },
    "camera": {
      "component": "camera-adapter",
      "instance": "cam-01",
      "subscribeAnnouncements": true,
      "reconcileCaptureStatusSecs": 30
    }
  },
  "modelRef": { "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:…" },
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
}
```

Leave `failedDir` out and a quarantine happens in place: the ledger records the terminal state and
the file stays where it is, which is the right answer when the input directory is itself the
evidence store.

## A trigger route

Images that arrive as messages. `fileRoot` is where a `{relativePath, sha256, bytes}` reference
resolves, with containment enforced and both declarations verified before admission;
`inlineStaging` is where an inline image inside the envelope's 64 KiB binary cap is written, and it
sits under `paths.staging` because the staged copy is processor-owned and immutable for the life of
the job.

```jsonc
{
  "id": "adhoc-inspect",
  "priority": 50,
  "source": {
    "kind": "trigger",
    "subscribe": ["ecv1/+/inspection-ui/+/app/inspect/request"],
    "fileRoot": "/var/spool/inspection",
    "inlineStaging": "/var/lib/edgecommons/image-processor/staging/adhoc"
  },
  "modelRef": { "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:…" },
  "outputs": { "writeResultSidecar": false, "decisionSignals": [] },
  "completion": { "onSuccess": "delete" }
}
```

## A GPU device with several cameras

One process, one GPU, three routes sharing one model generation. Routes bound to the same digest
share one resident session, so a third camera costs queue depth rather than device memory.
`priority` is the weight the scheduler gives a route when it picks among ready jobs; job age is
weighted alongside it, so a low-priority route never starves.

```jsonc
{
  "runtime": {
    "providers": ["CUDAExecutionProvider"],
    "requiredProvider": "CUDAExecutionProvider",
    "allowCpuOnly": false,
    "executorCellsPerGpu": 1,
    "loadConcurrencyPerGpu": 1
  },
  "gpu": { "devices": ["0"], "residentMemoryBudgetPercent": 80, "reserveMiB": 2048 },
  "scheduler": { "hotTtlSecs": 120, "minResidencySecs": 15, "maxAttempts": 5, "queueAgeWarningSecs": 300 },
  "modelSources": {
    "allowedSchemes": ["s3"],
    "allowedUriPrefixes": ["s3://approved-models/"],
    "verifyTls": true
  },
  "signing": {
    "required": true,
    "trustedKeys": [
      { "keyId": "pharma-model-publisher-1", "publicKey": { "$secret": "model-signing/publisher-1" } }
    ]
  },
  "models": [
    {
      "id": "line-clearance",
      "version": "2026.08.20",
      "digest": "sha256:…",
      "uri": "s3://approved-models/line-clearance/2026.08.20.tar.gz",
      "credentials": { "$secret": "model-source/approved-models" },
      "activation": { "requireWarmup": true, "retainForRollback": true }
    }
  ]
}
```

`signing.required` makes a signature and a trusted key mandatory for every bundle. The `$secret`
references are resolved by this component through the credentials vault at the moment they are
used, so neither the key nor the source credentials appear in a logged or published configuration.

`modelSources` is what bounds a configuration change: `allowedSchemes` and `allowedUriPrefixes`
decide what a new `models[]` entry can make the component download at all.

## Several models on one device

Each route names its own generation. The component stages and warms each one before any route
switches to it, and while a route's desired and active generations differ it reports `STAGING` and
keeps serving the model it already has.

```jsonc
{
  "models": [
    { "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:aaa…", "uri": "s3://approved-models/line-clearance/2026.08.20.tar.gz" },
    { "id": "cap-inspection", "version": "2026.07.02", "digest": "sha256:bbb…", "uri": "s3://approved-models/cap-inspection/2026.07.02.tar.gz" }
  ],
  "instances": [
    { "id": "clearance-cam-01", "modelRef": { "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:aaa…" }, "…": "…" },
    { "id": "caps-cam-02", "modelRef": { "id": "cap-inspection", "version": "2026.07.02", "digest": "sha256:bbb…" }, "…": "…" }
  ]
}
```

Upgrading a model is a configuration change: publish the new bundle, change the route's `modelRef`
digest (and the `models[]` entry), and deploy. Inputs that already reached a terminal state are not
replayed unless the route sets `reprocessExistingOnModelChange`, because a model upgrade on a busy
spool would otherwise re-infer the entire archive.

## What validation refuses

A candidate configuration is checked before it becomes current, and a rejected one leaves the
running configuration untouched. The rules that catch real deployment mistakes:

| Code | What it means |
|---|---|
| `UNRESOLVED_MODEL_REF` | A route names a model `global.models[]` does not declare. |
| `OVERLAPPING_MUTATING_ROOTS` | Two routes that move or delete their inputs claim overlapping roots. Exactly one component may mutate a spool. |
| `OUTPUT_INSIDE_SOURCE_ROOT` | An archive or quarantine directory sits inside a spool the component reads, so archived evidence would be rediscovered as new input. |
| `MISSING_ARCHIVE_DIR` | A route archives its inputs but sets no `completion.archiveDir`. |
| `INLINE_STAGING_NOT_CONTAINED` | A trigger route stages inline images outside `paths.staging`, or inside its own `fileRoot`. |
| `STABILITY_NOT_PERMITTED_ON_CAMERA_ROUTE` | A camera-bound route uses the `stability` readiness rule, which infers finalization instead of observing it. |
| `CAMERA_BINDING_REQUIRED` | A route reads capture status but names no `source.camera`. |
| `PROVIDER_POLICY_UNSATISFIED` | `requiredProvider` is not in `providers`, or the configuration asks for CPU-only inference without `allowCpuOnly`. |
| `MODEL_URI_SCHEME_NOT_ALLOWED`, `MODEL_URI_NOT_ALLOWED` | A model source is outside what `modelSources` permits. |
| `NO_TRUSTED_KEYS` | `signing.required` is set but no trusted key is configured, so no bundle could be verified. |
| `IMMUTABLE_PATH_CHANGED` | A reload moves `stateDb`, `modelCache`, or `staging`, which would orphan the durable state. |
