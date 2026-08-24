# How-to Guides

Recipes for specific tasks. For concepts see [explanation.md](explanation.md); for exhaustive
options see [reference/](reference/).

---

## Build and sign a model bundle

A model bundle is the unit the component installs: one tar archive, optionally gzip-compressed,
that carries `manifest.json`, the detached signature `manifest.sig`, and the model files. The
archive's SHA-256 is the bundle digest, and that digest is what configuration pins. Build a
bundle with `tools/make_bundle.py`.

1. Create a signing keypair. Do this once per publisher, and keep the private key off the
   devices that consume bundles.

   ```bash
   python tools/make_bundle.py --gen-key keys/pharma-model-publisher-1.pem
   ```

   The command writes three files: the private key PEM, the public key PEM
   (`...pem.pub.pem`), and the raw 32-byte public key (`...pem.pub`). To protect the private key
   with a passphrase, add `--key-password`.

2. Lay out the bundle directory. Put the ONNX graph at `model.onnx`, and add the files the task
   family and the operators need beside it:

   ```text
   line-clearance-cam-01/
     manifest.json
     model.onnx
     labels.json
     transforms.json
     result.schema.json
     model-card.json
     warmup/input-01.bin
     warmup/expected-01.json
   ```

3. Write `manifest.json`. Declare the model identity, the runtime and provider requirements, the
   tensor shapes, the task family and its parameters, the preprocessing, the decision rules, the
   result bounds, the warmup samples and tolerances, the engine compatibility keys, and the
   provenance. Leave `files` out: the tool computes every per-file digest.

   ```json
   {
     "schemaVersion": 1,
     "modelId": "line-clearance-cam-01",
     "version": "2026.08.20",
     "minOnnxRuntime": "1.18.0",
     "providersPermitted": ["CUDAExecutionProvider"],
     "providerPolicy": "required",
     "inputs": [{ "name": "images", "dtype": "float32", "shape": ["N", 3, 224, 224] }],
     "outputs": [{ "name": "logits", "dtype": "float32", "shape": ["N", 2] }],
     "dynamicBatch": true,
     "family": "classification",
     "familyParams": { "topK": 2, "activation": "softmax" },
     "preprocess": { "resize": [224, 224], "layout": "NCHW" },
     "decisionRules": { "pass": "$.classes[0].label == 'clear'", "threshold": 0.8 },
     "maxResultItems": 10,
     "estimatedDeviceMiB": 512,
     "warmup": [{ "input": "warmup/input-01.bin", "expected": "warmup/expected-01.json" }],
     "tolerances": { "score": 0.001 },
     "compatibilityKeys": { "gpuClass": "sm_86" },
     "provenance": { "publisher": "pharma-mlops" },
     "transformVersion": "2026.08.20-1"
   }
   ```

4. Pack and sign the bundle. `--key-id` is the name the device configuration maps to the public
   key, and it is recorded in the manifest.

   ```bash
   python tools/make_bundle.py line-clearance-cam-01 \
       --out dist/line-clearance-cam-01-2026.08.20.tar.gz \
       --key keys/pharma-model-publisher-1.pem \
       --key-id pharma-model-publisher-1 \
       --schema schemas/model-bundle-manifest.schema.json
   ```

   The tool merges your manifest with the computed `files` digests, validates it against the
   schema, signs the exact manifest bytes, packs the archive, and prints the bundle digest:

   ```text
   sha256:9f2c1e...c41d
   ```

   Builds are reproducible. The same directory and key always produce the same digest, so you
   can rebuild a bundle to confirm the digest you shipped.

5. Publish the archive to a local path, an https origin, or an S3 bucket, and pin it in
   `component.global.models[]` with the digest the tool printed:

   ```jsonc
   {
     "id": "line-clearance-cam-01",
     "version": "2026.08.20",
     "digest": "sha256:9f2c1e...c41d",
     "uri": "s3://approved-models/line-clearance-cam-01/2026.08.20.tar.gz"
   }
   ```

6. Trust the signing key on the device. Add the public key under
   `component.global.signing.trustedKeys[]` against the same `keyId`, and set
   `signing.required` to `true` where a verified signature is mandatory. The raw public key
   (`...pem.pub`) is the value to store; a `$secret` reference keeps it out of the deployment
   document.

   ```jsonc
   "signing": {
     "required": true,
     "trustedKeys": [
       {
         "keyId": "pharma-model-publisher-1",
         "publicKey": { "$secret": "model-signing/publisher-1" }
       }
     ]
   }
   ```

The component verifies the tarball digest, then the signature, then extracts under path, count,
size, and ratio limits, then verifies the per-file digests and the manifest schema, and only then
promotes the bundle into its content-addressed cache. Any bundle that fails a check is refused
and the route stays on its last-known-good model.

## Serve bundles from S3

`s3://` sources need the `s3` extra, which is not installed by default:

```bash
pip install ".[s3]"
```

Without it, an `s3://` URI fails with `S3_UNAVAILABLE` and the rest of the component keeps
running. Give the model entry a `credentials` `$secret` reference to use explicit keys, or leave
it out to use the ambient credentials of the host or the Greengrass token exchange service.

## Restrict where bundles come from

An `https://` source is downloaded with certificate and hostname verification, and a URL that
carries credentials is refused. Where the deployment configures allow-listed prefixes, both the
URL and every redirect target must start with one of them, so a redirect cannot walk a download
off the approved origin.

## Deploy to a platform

**HOST:** `python3 main.py --platform HOST --transport MQTT ./messaging.json -c FILE ./config.json -t my-thing`
(or `docker compose up --build`).

**Greengrass:** `gdk component build && gdk component publish`. The recipe's default configuration
ships two working routes and the CPU development profile, so point the routes at the spool the
device's camera writes into, name the device's GPU ordinals in `gpu.devices`, drop
`runtime.allowCpuOnly`, and point `models[].uri` at the object your deployment publishes. Set a real
S3 bucket in `gdk-config.json` first — a repository with no bucket configured carries a visible
sentinel that `component validate` treats as an error.

**Kubernetes:** build `./Dockerfile`, push or `kind load` it, set `image:` in `k8s/deployment.yaml`,
then `kubectl apply -f k8s/`. With `--platform auto` the library detects KUBERNETES, reads config
from the mounted ConfigMap (hot-reloaded on `kubectl apply`), and resolves identity from the Downward
API — no CLI args needed.

## Add a route

A route is one entry of `component.instances[]`, and routes are independent, so adding one is a
configuration change rather than a code change. It needs four things: an `id` that is a valid UNS
token, a `source`, a `modelRef` naming an entry of `global.models[]` by id, version, and digest, and
a `completion` policy.

1. Publish the model the route runs and add it to `global.models[]`, or reuse a model another route
   already names — routes bound to the same digest share one resident session.
2. Add the instance. [Sample configurations](sample-configurations.md) has a camera route and a
   trigger route you can copy.
3. Deploy. The candidate validator runs before the configuration becomes current, so a route that
   would claim another route's spool, archive into a directory it reads, or name a model nobody
   published is refused and the running configuration is left alone.

The component reconciles routes without dropping admitted work: a route that appeared starts, a
route that disappeared stops, and a route whose source or model changed is rebuilt, while jobs
already in the ledger keep their pinned model generation and finish under it.

## Repair a job a crash or an outage left behind

Every durable state a job can be stuck in has an operator verb, and each one reports what the
ledger and the filesystem actually show afterwards rather than assuming it worked.

**A result that could not be published.** After `publish.maxAttempts` the job is
`PUBLISH_EXHAUSTED`, the component emits `publish-exhausted`, and the input stays where it is —
archiving it would move the evidence out from under a publication that is still expected to happen.
When the broker is back:

```bash
ec-uns-cmd --device my-thing --component image-processor retry-publication
ec-uns-cmd --device my-thing --component image-processor retry-publication --body '{"inferenceId": "blvuf6ru…"}'
```

**A completion that failed.** A move that collided with an existing object, or a directory that was
not writable, leaves the job `CLEANUP_FAILED` with the intent still on record. The supervision pass
retries it every few seconds; to retry now, or to see what is still failing:

```bash
ec-uns-cmd --device my-thing --component image-processor retry-cleanup
```

**Anything else that looks stuck.** `reconcile` re-decides every open cleanup intent against
observed filesystem state — source present and target absent retries the move, source absent and
target present with a matching digest completes, and a target holding different bytes is a
collision rather than an overwrite:

```bash
ec-uns-cmd --device my-thing --component image-processor reconcile
ec-uns-cmd --device my-thing --component image-processor get-queue --body '{"states": ["CLEANUP_FAILED", "PUBLISH_EXHAUSTED"]}'
```

**A model that will not stage.** `get-models` reports the error against the generation. Fix the
source or the digest and re-evaluate the catalog without a redeployment:

```bash
ec-uns-cmd --device my-thing --component image-processor get-models
ec-uns-cmd --device my-thing --component image-processor reload-model-catalog
```

**A route you need to stop feeding.** `pause` stops it claiming new work and lets in-flight jobs
finish; `set-route-activation-override` persists the decision across restarts and is reported beside
the configured value, so a deployment stays the source of truth for what the route is.

```bash
ec-uns-cmd --device my-thing --component image-processor --instance clearance-cam-01 pause
ec-uns-cmd --device my-thing --component image-processor --instance clearance-cam-01   set-route-activation-override --body '{"enabled": false}'
```

## Wire up CI

`.github/workflows/ci.yml` calls the org's reusable component-CI workflow plus a `coverage` job
enforcing the 90% line-coverage gate; `.github/workflows/deploy-docs.yml` refreshes the docs site on
doc-only pushes once this component is registered in `registry/components.json`. Both are inert until
pushed to GitHub with the org secrets configured.

## Run the real-model suite

The real-model suite runs seven real ONNX exports — two ImageNet classifiers, three COCO
detectors, a Pascal VOC segmentation model, and a PatchCore anomaly model — over real images and
compares the answers to the JSON goldens in `tests/goldens/`. It runs nightly and on demand, not on
every pull request, so you turn it on with `EC_LIVE_MODELS=1`.

Models and images are never committed. `tests/assets.json` pins each one by URL, SHA-256, and byte
count, and `tools/fetch_test_assets.py` verifies them into `tests/.cache/`, which is gitignored.

### Fetch the corpus

1. See what the corpus holds and what is already cached:

   ```bash
   python tools/fetch_test_assets.py --list
   ```

1. Fetch it. The default selection is about 440 MB and skips the assets marked optional:

   ```bash
   python tools/fetch_test_assets.py
   ```

   The command is idempotent: an asset already in the cache is re-hashed rather than downloaded
   again. A file whose digest does not match the manifest is reported and the command exits
   non-zero.

1. To fetch one asset, or to add the optional ones, name them:

   ```bash
   python tools/fetch_test_assets.py --only model-yolox-nano
   python tools/fetch_test_assets.py --include-optional
   ```

### Run the suite

1. Run it with the switch set:

   ```bash
   EC_LIVE_MODELS=1 python -m pytest tests/live_models -o addopts="" -q
   ```

   Each model is packed into a signed bundle, staged through the content-addressed cache, and run
   on `CPUExecutionProvider`, so the suite exercises the same install path a deployment uses.

1. A model whose assets are missing skips with the command that fetches them, so a partial corpus
   runs what it can.

### Add the CUDA leg

On a machine with the CUDA provider, set `EC_NVIDIA` as well and the suite runs twice: once on
`CPUExecutionProvider` and once on `CUDAExecutionProvider`, with the same images, the same bundles,
and the same goldens.

```bash
EC_LIVE_MODELS=1 EC_NVIDIA=1 python -m pytest tests/live_models -o addopts="" -q
```

The goldens are produced on CPU, so the CUDA parametrization asserts against them rather than
against a second baseline: a model whose CUDA answer leaves the golden's top-5 overlap, box IoU,
pixel fraction, or anomaly score is a parity failure. The bundle is packed and staged once and both
legs open a session on it, so the second leg costs only the sessions and the inference. The run
prints the graph time of every image per model and provider when it ends.

`--update-goldens` is refused on any provider other than CPU, so a CUDA run cannot overwrite the
baseline it is being compared to.

### Build the anomaly model

The PatchCore model is built rather than downloaded, because PatchCore needs no training epochs and
rebuilding it from the pinned dataset is cheaper than hosting a binary. It needs `anomalib` and
`torch`, which are not dependencies of this component, so install them into a scratch environment:

1. Fetch VisA, which is optional and about 1.8 GB:

   ```bash
   python tools/fetch_test_assets.py --only dataset-visa
   ```

1. Install the build dependencies and build the model:

   ```bash
   python -m venv .venv-anomalib
   .venv-anomalib/bin/pip install anomalib onnx jsonpath-ng jsonschema cryptography
   .venv-anomalib/bin/python tools/build_anomaly_model.py
   ```

   The build writes `model.onnx` and a `build.json` record into
   `tests/.cache/model-patchcore-visa-capsules/`. The record carries the threshold the bundle
   declares and the separation the memory bank achieves on held-out images. The build is
   deterministic: the same VisA archive and the same arguments produce the same graph.

1. Run the suite again. The anomaly tests skip with these instructions whenever the model is absent.

### Update the goldens

Regenerate a golden after a deliberate change to preprocessing, postprocessing, or the decision
rules — never to make a failing comparison pass.

1. Regenerate every golden from a real run:

   ```bash
   python tools/update_goldens.py
   ```

1. Regenerate one model's golden:

   ```bash
   python tools/update_goldens.py --only mobilenetv2-12
   ```

1. Read the diff before committing it. A changed top-1 label, a moved box, or a flipped decision
   outcome is a behavior change, and the diff is where you see it.

The same thing happens inside pytest with `--update-goldens`, which is what the tool passes through.

## Run on an NVIDIA GPU

The `gpu` and `nvml` extras install `onnxruntime-gpu` and the NVML bindings. `onnxruntime-gpu`
links against a specific CUDA major version (1.29 links against CUDA 13), and the matching runtime
libraries install from PyPI, so no system CUDA toolkit is required. The executor cell preloads
these libraries before it creates a session.

1. Create a Linux (or WSL2) virtual environment with Python 3.12:
   `uv venv --python 3.12 ~/ip-gpu-venv && source ~/ip-gpu-venv/bin/activate`
2. Install the component with the GPU extras and the CUDA 13 runtime (never together with the
   CPU wheel -- `onnxruntime` and `onnxruntime-gpu` ship the same package, so do not install
   `requirements.txt` or the `cpu` extra in this environment):
   `pip install -e '.[gpu,nvml]' 'nvidia-cuda-runtime==13.*' 'nvidia-cublas==13.*' 'nvidia-cuda-nvrtc==13.*' 'nvidia-cufft==12.*' 'nvidia-curand==10.*' nvidia-cudnn-cu13 'nvidia-nvjitlink==13.*'`
3. Confirm the provider: `python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"`
   lists `CUDAExecutionProvider`.
4. Run the NVIDIA suite: `EC_NVIDIA=1 python -m pytest tests/engine/test_nvidia.py -o addopts="" -q`.

A route whose `runtime.requiredProvider` is `CUDAExecutionProvider` refuses to run when the session
lands on CPU (`PROVIDER_CPU_ONLY`); the result's `model.providers` always names the actual assignment.

## Run the tier-3 residency benchmark

The tier-3 suite measures what a GPU host does when the configured model set is several times
larger than device memory: cold load, warm inference, eviction, reload, executor recycles, and the
queue and inference latency of four arrival patterns. It needs a synthesized corpus, because the
real models of the tier-2 corpus all fit on any card at once and nothing is ever evicted.

Every run writes `tests/nvidia/results/<gpu-class>-<date>.json`, and those numbers are the SLO
baseline recorded in `DESIGN.md` section 17.

### Build the corpus

`tools/synth_corpus.py` grows N distinct signed bundles from the MobileNetV2 and YOLOX-S
architectures in the tier-2 cache. Each one gets seeded weight perturbations, so no two share a
digest or a byte of weight data, and a padded initializer that sizes it to a target tier. The pad
costs device memory and one host-to-device copy per load; it costs no meaningful compute, so read
the inference latencies as the base architecture's, measured under residency pressure.

1. Fetch the tier-2 corpus the bases come from, if you have not already:
   `python tools/fetch_test_assets.py`
2. Build the corpus on a local disk. On WSL2 use a Linux path, not a Windows mount: the build
   writes about 47 GB and the mount is several times slower.

   ```bash
   python tools/synth_corpus.py --out ~/ip-corpus
   ```

   The defaults are 40 bundles across the 50 MB, 200 MB, 600 MB, and 1.5 GB tiers, which is about
   23 GB of `model.onnx` and takes about five minutes on an NVMe disk. Use `--count`, `--tiers`,
   and `--seed` to change the shape.

   The seed fixes the models: `model.onnx`, `labels.json`, `transforms.json`, the warmup input
   tensor, and the signing key are byte-identical for a given seed on any machine. It does not fix
   the golden warmup answers, which record what a CPU ONNX Runtime session produced, and CPU
   kernels differ between processors. Two runs on one machine reproduce every bundle digest; the
   same seed on a different processor can give a different digest for a model whose golden is
   large. Build the corpus on the machine that will run the suite.

3. Check `~/ip-corpus/corpus.json`. It records every bundle's digest, tier, pad size, base
   architecture, and estimated device memory, and it is what the suite reads.

### Run the suite

1. Point the suite at the corpus and run it:

   ```bash
   EC_NVIDIA=1 EC_NVIDIA_CORPUS=~/ip-corpus \
     python -m pytest tests/nvidia -o addopts="" -q
   ```

2. Read the results file the run prints. The table in `DESIGN.md` section 17 is built from it.

The knobs are environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `EC_NVIDIA` | Runs the suite. Without it every test is skipped. | unset |
| `EC_NVIDIA_CORPUS` | The corpus root. | `~/ip-corpus` |
| `EC_NVIDIA_DEVICE` | The CUDA device ordinal. | `0` |
| `EC_NVIDIA_ROUTES` | How many corpus bundles to bind routes to. | every bundle |
| `EC_NVIDIA_ARRIVALS` | Images per arrival pattern. | `96` |
| `EC_NVIDIA_RATE` | Arrivals per second for the paced patterns. | `8` |
| `EC_NVIDIA_PATTERNS` | A subset of `uniform,zipf,burst,prefetch`. | all four |
| `EC_NVIDIA_WORK` | Scratch space for the ledger and the route spools. | `~/ip-tier3-work` |
| `EC_NVIDIA_RESULTS` | Where the results file is written. | `tests/nvidia/results` |

Keep the work directory on a local disk as well. The rig discovers arrivals through the
component's own spool source, which watches the directory, and a Windows mount is outside
inotify's reach.

### What the patterns do

| Pattern | Arrivals |
|---|---|
| `uniform` | One capture per slot, round-robin over every route. |
| `zipf` | The same rate, with routes drawn from a Zipf distribution, so a few routes take most of the traffic and the tail still has to be served. |
| `burst` | Every route fires at once, in waves, with a quiet gap between them. |
| `prefetch` | The next segment of routes is staged and warmed with the `preload-model` path while the queue is quiet, and only then do its arrivals reach the spool. |

### On a smaller card

The suite refuses to run when the routed corpus fits inside the residency budget, because then
nothing is ever evicted and the measurement means nothing. On an 8 GB card the default corpus is
already several times the budget; to shorten a run instead, lower `EC_NVIDIA_ROUTES` and
`EC_NVIDIA_ARRIVALS` together, and keep enough of the 1.5 GB tier in the routed set to stay over
the budget.

## Browse the inference results

The test suites report pass or fail. The results explorer reports what the models actually did:
every image of both corpora, the whole `app/inference/result` body the component publishes for it,
and the detections, segments, or anomaly regions drawn on a copy of the picture.

`tools/build_results_site.py` builds a static site into `tests/.site/`, which is gitignored. The
site is four files plus the pictures, it fetches nothing, and it opens from disk.

### Build the site

1. Build both suites on CPU:

   ```bash
   python tools/build_results_site.py --out tests/.site --suites synthetic,live
   ```

   The `synthetic` suite generates the tier-1 corpus into a temporary directory first, so it needs
   no network. The `live` suite reads `tests/.cache`, so fetch the corpus first with
   `python tools/fetch_test_assets.py`; see [Run the real-model suite](#run-the-real-model-suite).

1. Build only what you want to look at:

   ```bash
   python tools/build_results_site.py --out tests/.site --suites synthetic
   python tools/build_results_site.py --out tests/.site --models yolox-s,fcn-resnet50-12 --limit 5
   ```

1. Open `tests/.site/index.html`. The command prints the `file://` URL when it finishes.

### Add a CUDA run beside the CPU one

A run is one execution provider on one device. `--merge` folds a new run into the site already in
`--out` instead of replacing it, so a CPU build and a CUDA build sit side by side and the summary
view gains a strip comparing median graph time per model.

Run the CUDA leg where the CUDA provider is installed. On this workstation that is WSL, with the
`gpu` and `nvml` extras:

```bash
cd /mnt/c/Users/you/source/edgecommons/image-processor
PYTHONPATH=$PWD ~/ip-gpu-venv/bin/python tools/build_results_site.py \
    --out tests/.site --suites live --provider CUDAExecutionProvider --merge
```

The session is refused unless ONNX Runtime actually assigns `CUDAExecutionProvider`, so a leg that
silently fell back to CPU fails the build rather than mislabelling its numbers. Rebuilding a run
replaces that run and leaves the other one alone.

### Serve it

The site works from `file://`. Serve it when you want to look at a build from another machine, or
when a browser is configured to treat local files strictly:

```bash
python tools/build_results_site.py --out tests/.site --suites synthetic --serve 8000
```

### What the views hold

| View | Holds |
|---|---|
| Summary | One card per model: task family, corpus, image count, decision outcome counts, and the minimum, median, 95th percentile, and maximum graph time. Clicking a card opens the gallery filtered to that model. Below the cards, the tier-1 bad-input set with the decode code each fixture raised. |
| Gallery | Every entry as a thumbnail, with facets for family, model, suite, outcome, and run, and a substring filter on the image name. The filters are in the URL fragment, so a filtered view is a link. |
| Detail | The input and the overlay side by side, each opening full size in a new tab; a classification shows its top-k table where the overlay would be. Below them the model card, the decision, the timings, and the full result body as collapsible JSON with a copy button. `ArrowLeft` and `ArrowRight` move within the current filter, and `Escape` returns to the gallery. |

The timings are the two the builder measures: `session` is the graph alone, and `wall total` covers
reading the file, decoding, preprocessing, the graph, postprocessing, and the decision rules. The
per-stage fields of the result body read zero, because the builder does not time those stages
separately.
