# ImageProcessor

`com.mbreissi.edgecommons.ImageProcessor` is the EdgeCommons Python inference component for image
models. It loads signed, content-addressed ONNX model bundles, runs inference on finalized image
files from a spool it owns or on subscription-triggered images, and publishes one durably confirmed
inference result per image.

```text
  camera spool ─┐                        ┌─► app/inference/result   (confirmed, cleanup-gating)
                ├─► readiness ─► ledger ─┼─► data/<signal>          (decision mirror, best effort)
  trigger topic ┘        │        │      ├─► <image>.inference.json (evidence sidecar)
                         │        │      └─► processed/ | failed/   (the input, after confirmation)
                         │        └─ executor cell (ONNX Runtime, CUDA or CPU)
                         └─ model bundle cache (verified, warmed, atomically activated)
```

One process hosts many routes. A route binds one input source to one immutable model version, one
set of outputs, and one completion policy, and routes are independent: a backlog on one never
stalls another.

## Quick start

You need a local MQTT broker (`docker run -d -p 1883:1883 --name emqx emqx/emqx:latest`), or run
`docker compose up --build`, which starts one for you.

1. Install the component and its test tooling.

   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   pip install -e . -r requirements-test.txt
   ```

2. Build the test corpus and pack the model bundle `test-configs/config.json` pins. The corpus is
   generated rather than downloaded, so the digest is reproducible on the same ONNX Runtime;
   `make_bundle.py` prints it.

   ```bash
   python3 tests/fixtures/build.py --out tests/fixtures/out
   mkdir -p /var/lib/edgecommons/image-processor/bundles
   python3 tools/make_bundle.py tests/fixtures/out/bundles/synthetic-anomaly-scalar-1.0.0 \
     --out /var/lib/edgecommons/image-processor/bundles/synthetic-anomaly-scalar-1.0.0.tar
   ```

3. Create the directories the shipped configuration names, and run the component.

   ```bash
   mkdir -p /var/spool/camera-adapter/cam-01 /var/spool/image-processor/processed/cam-01 \
     /var/spool/image-processor/failed/cam-01 /var/spool/inspection
   python3 main.py --platform HOST --transport MQTT ./test-configs/standalone-messaging.json \
     -c FILE ./test-configs/config.json -t smoke-device
   ```

   It stages the bundle, warms it, activates it on both routes, and reports ready.

4. Watch what it publishes.

   ```bash
   mosquitto_sub -h localhost -p 1883 -v \
     -t 'ecv1/+/image-processor/+/app/inference/result' \
     -t 'ecv1/+/image-processor/+/data/#' \
     -t 'ecv1/+/image-processor/+/evt/#' \
     -t 'ecv1/+/image-processor/state'
   ```

5. Drop a capture into the spool the way a camera does, metadata sidecar first and then the image.
   `tests/fixtures/out/images/anomaly-good.png` decides `CLEAR`, `anomaly-bad.png` decides `HOLD`.
   The [tutorial](docs/tutorial.md) walks the whole run, including the sidecar you write.

6. Ask it about itself with `ec-uns-cmd`:

   ```bash
   ec-uns-cmd --device smoke-device --component image-processor status
   ec-uns-cmd --device smoke-device --component image-processor get-models
   ec-uns-cmd --device smoke-device --component image-processor get-queue
   ```

## Layout

| Path | What it is |
|------|-----------|
| `main.py` | Entry point: builds `EdgeCommons`, registers the configuration validator and the command verbs, runs the app. |
| `image_processor/ImageProcessor.py` | The wiring and the result pipeline: sidecar, ledger transaction, confirmed publish, mirror, completion. |
| `image_processor/config/` | The schema-backed configuration model and the pre-commit candidate validator. |
| `image_processor/bundles/` | Bundle fetch, digest and signature verification, bounded extraction, the content-addressed cache. |
| `image_processor/ledger/`, `image_processor/completion/` | The SQLite job ledger and outbox, and the archive, delete, retain, and quarantine actions under write-ahead intents. |
| `image_processor/engine/` | Decode, task families, decision rules, the executor cells, the supervisor, and the model-aware scheduler. |
| `image_processor/sources/` | Spool discovery and readiness, the camera hint and capture-status reconciler, the trigger subscription. |
| `image_processor/outputs/` | The result body, the evidence sidecar, the confirmed outbox publisher, the decision mirror, the events. |
| `image_processor/artifacts.py` | Model staging, warmup, and atomic route-generation activation. |
| `config.schema.json`, `schemas/` | The configuration contract, the result body contract, and the bundle manifest contract. |
| `test-configs/` | A working `config.json` and the MQTT `standalone-messaging.json` for local HOST runs. |
| `tools/` | `make_bundle.py` builds and signs a bundle; the rest fetch and verify the tier-2 test assets. |

## Configuration

`config.schema.json` is the contract and
[docs/reference/configuration.md](docs/reference/configuration.md) documents every field.
`component.global` sets the paths, the runtime and GPU profile, the scheduler and publication
policy, the model sources and signing trust, the model set, and the completion defaults;
`component.instances[]` is one route each.

```jsonc
{
  "component": {
    "token": "image-processor",
    "global": {
      "paths": { "stateDb": "…/state.db", "modelCache": "…/models", "staging": "…/staging" },
      "runtime": { "providers": ["CUDAExecutionProvider"], "requiredProvider": "CUDAExecutionProvider" },
      "gpu": { "devices": ["0"] },
      "models": [{ "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:…", "uri": "s3://approved-models/…" }]
    },
    "instances": [
      {
        "id": "clearance-cam-01",
        "source": {
          "kind": "spool",
          "root": "/var/spool/camera-adapter/cam-01",
          "readiness": { "mode": "cameraSidecar" },
          "camera": { "component": "camera-adapter", "instance": "cam-01" }
        },
        "modelRef": { "id": "line-clearance", "version": "2026.08.20", "digest": "sha256:…" },
        "outputs": { "decisionSignals": [{ "id": "line-clearance/pass", "value": "$.decision.pass" }] },
        "completion": { "onSuccess": "archive", "archiveDir": "/var/spool/image-processor/processed/cam-01" }
      }
    ]
  }
}
```

Validation is fail-closed and runs before a configuration becomes current: unknown fields,
duplicate route ids, unresolved model references, overlapping mutating roots, a missing archive
directory, an unsatisfiable provider policy, and a trigger route staging outside `paths.staging`
all reject the candidate and leave the running configuration untouched.
[docs/sample-configurations.md](docs/sample-configurations.md) has the shapes for the common
deployments.

## What it publishes, and what it accepts

| Surface | Topic | Notes |
|---|---|---|
| Inference result | `ecv1/{device}/image-processor/{routeId}/app/inference/result` | The authoritative output. Prepared once, stored byte for byte, published with positive transport confirmation. |
| Decision mirror | `ecv1/{device}/image-processor/{routeId}/data/{signalId}` | Best effort, derived from the committed result, never cleanup-gating. |
| Events | `ecv1/{device}/image-processor/{routeId}/evt/{severity}/{type}` | Bounded operator conditions. Success is not an event. |
| Metrics | `ecv1/{device}/image-processor/metric/{group}` | Eight groups; nothing that identifies one image is a dimension. |
| Commands | `ecv1/{device}/image-processor/{instance}/cmd/{verb}` | The library built-ins plus `get-models`, `get-queue`, `trigger-rescan`, `preload-model`, `evict-model`, `reload-model-catalog`, `set-route-activation-override`, `retry-publication`, `retry-cleanup`, `reconcile`, `pause`, and `resume`. |
| Trigger input | whatever `source.subscribe` names | An inline image within the envelope's 64 KiB binary cap, or `{relativePath, sha256, bytes}` resolved under `fileRoot`. A request carrying `reply_to` gets the bounded result as its correlated reply. |

[docs/reference/messaging-interface.md](docs/reference/messaging-interface.md) has the bodies, the
request and reply shapes, and the error codes;
[docs/reference/metrics.md](docs/reference/metrics.md) has the measures.

## The invariants — do not remove these

* **Exactly one mutating owner per spool.** The component archives or deletes only roots it owns,
  and validation rejects overlapping mutating roots.
* **The `app/inference/result` message is the only cleanup-gating output.** It is prepared once,
  its exact bytes stored in the ledger, and published with `publish_confirmed()` at QoS 1; the
  input is archived, deleted, or quarantined only after that confirmation. The `data` decision
  signals are a mirror, and a consumer enforcing a safety gate reads the result instead.
* **Ordered durability.** Sidecar temp, flush, atomic install, then one SQLite transaction (result,
  gating outbox rows, sidecar digest, `RESULT_COMMITTED`), and only then is the outbox eligible.
  Cleanup intents are persisted before any file mutation, and a cleanup failure is
  `CLEANUP_FAILED` — retried by policy, repairable by command, never success.
* **No silent CPU fallback.** A route whose provider policy is not satisfied fails closed;
  `allowCpuOnly` with an empty `gpu.devices` is the development profile. The `providers` on every
  result are the actual session assignment.
* **Bundles are verified, never trusted.** Tarball digest, then the signature when one is required,
  then bounded extraction, per-file digests, the manifest schema, and task-family support, before
  staging completes. No bundle-supplied code executes.
* **A failed, missing, stale, or unverified inference is `HOLD`, never `CLEAR`.**
* **Pixels never enter the parent process.** Decode and inference happen in executor cells that
  read the immutable staged file by path and expected digest.
* **Accepted jobs are never dropped.** Backpressure stops claiming new work and reports
  degradation; it never discards an admitted image.

## Tests

```bash
python3 -m pytest                     # tier 1: no broker, no GPU, no network
EC_LIVE_MODELS=1 python3 -m pytest    # tier 2: real models, assets fetched and cached
EC_NVIDIA=1 python3 -m pytest         # tier 3: the NVIDIA residency and burst suite
```

The org gate is 90% line coverage over `image_processor/` (`ci.yml`). Run
`edgecommons component validate -p .` before a PR.

## Documentation

[docs/](docs/) follows Diátaxis: a [tutorial](docs/tutorial.md),
[how-to guides](docs/how-to-guides.md), [reference](docs/reference/), and an
[explanation](docs/explanation.md). `DESIGN.md` is the design-fidelity contract and `LLD.md` is the
module structure.
