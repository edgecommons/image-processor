# ImageProcessor — agent guidance

`com.mbreissi.edgecommons.ImageProcessor` (`image-processor`) is the EdgeCommons Python inference
component for image models: it loads signed, content-addressed ONNX model bundles, runs NVIDIA GPU
inference on finalized image files from an owned spool or on subscription-triggered images, and
publishes a durably confirmed inference result per image. It follows the org-wide conventions in
the parent EdgeCommons workspace's `AGENTS.md`; what follows is scoped to this repo.

## Design documents

- `DESIGN.md` — the design-fidelity contract: identity, decision register **D-IP-1…20**, input
  sources, topology, architecture, job lifecycle, bundle contract, model delivery, NVIDIA
  execution, configuration, message contracts, commands, validation tiers, phases.
- `LLD.md` — the module structure and the interfaces between packages, plus the work-package
  ownership table. Implement against it; change an interface only by editing `LLD.md` in the
  same change.

## Name tokens

| Token | Meaning |
|---|---|
| `ImageProcessor` | PascalCase class/file name |
| `com.mbreissi.edgecommons.ImageProcessor` | Reverse-DNS Greengrass component name |
| `image-processor` | Kebab-case UNS `component.token`, artifact, and repo name |
| `image_processor` | Python package |

## Layout

See `LLD.md` §1. In short: `config/` (schema-backed config + candidate validator), `bundles/`
(tarball verification, manifest, Ed25519 signature, content-addressed cache, fetchers), `ledger/`
and `completion/` (SQLite job ledger, outbox, cleanup intents, recovery, archive/delete/retain/
quarantine), `engine/` (decode, task families, decision rules, executor cells, supervisor,
scheduler, residency), `sources/` (spool discovery, readiness, camera hint/status, trigger
subscription), `outputs/` (result, confirmed outbox publisher, decision mirror, sidecar, events),
`artifacts.py` (model staging, warmup, route-generation activation), `ImageProcessor.py` (the
wiring and the result pipeline), `commands.py` / `metrics.py` / `health.py` / `connectivity.py`
(the operator surfaces), `schemas/` (wire and bundle contracts), `tools/` (bundle authoring,
test-asset fetch), `tests/` (fixtures builder, goldens, suites).

## Non-negotiable invariants (do not remove)

- **Exactly one mutating owner per spool.** The component archives/deletes only roots it owns;
  validation rejects overlapping mutating roots.
- **The `app/inference/result` message is the only cleanup-gating output.** It is prepared once
  (`app().prepare()`), its exact bytes stored in the ledger, and published with
  `publish_confirmed()` at QoS 1; archive/delete happens only after confirmation. `data` decision
  signals are a best-effort mirror.
- **Ordered durability.** Sidecar temp → flush → atomic install → one SQLite transaction
  (result, gating outbox rows, sidecar digest, `RESULT_COMMITTED`) → outbox eligible. Cleanup
  intents are persisted before any file mutation; cleanup failure is `CLEANUP_FAILED`, never success.
- **No silent CPU fallback.** A route whose provider policy is not satisfied fails closed;
  `allowCpuOnly` is development-only. The recorded `providers` on every result are the actual
  session assignment.
- **Bundles are verified, never trusted.** Tarball digest → signature (when required) → bounded
  extraction → per-file digests → manifest schema → family support, before staging completes.
  No bundle-supplied code executes.
- **A failed, missing, stale, or unverified inference is `HOLD`, never `CLEAR`.**
- **Pixels never enter the parent process.** Decode and inference happen in executor cells that
  read the immutable staged file by path and expected digest.
- **Inline trigger images respect the core 64 KiB binary-body cap**; larger images arrive by file
  reference with path containment and digest verification.
- **Accepted jobs are never dropped.** Backpressure stops claiming new work and reports
  degradation; it never discards admitted jobs.

## Validation expectations

- `python -m pytest` passes with no broker, no GPU, no network (tier 1, `CPUExecutionProvider`);
  the org gate is **90% line coverage** over `image_processor/` (`ci.yml` `coverage` job).
  `# pragma: no cover` only on a live seam (subprocess entry, real transport) with an inline reason.
- `EC_LIVE_MODELS=1` enables the tier-2 real-model suite (nightly + on demand; assets fetched by
  `tools/fetch_test_assets.py` from `tests/assets.json`, goldens in `tests/goldens/`).
- `EC_NVIDIA=1` enables the tier-3 NVIDIA suite (RTX 5080 WSL2, `lab-5950x` RTX 2080 Super).
- Run `edgecommons component validate -p .` before a PR; `config.schema.json` and
  `schemas/*.json` are contracts — a config or message field added in code needs the schema
  change in the same PR.

## Docs stay in sync with code

Any change to config, commands, metrics, events, the result body, or the bundle manifest must
update `config.schema.json`/`schemas/`, `DESIGN.md`, `LLD.md` (if an interface moves), and the
relevant page under `docs/` in the same change. Stale docs are a defect.

## Work-package discipline

Implementation is split into the work packages in `LLD.md` §9, one branch each. A package edits
only the files it owns; shared files (`pyproject.toml`, `requirements.txt`) are appended under a
`# WPn` comment. Every package ships tests for what it owns and says in its PR which LLD
interfaces it relied on and whether any changed.
