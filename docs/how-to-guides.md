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
ships one working route — **edit its `publishTopic`** so the device token matches the thing you
deploy to, because a processor with no valid routes has nothing to run and refuses to start. Set a
real S3 bucket in `gdk-config.json` first — a scaffold with no bucket configured carries a visible
sentinel that `component validate` treats as an error.

**Kubernetes:** build `./Dockerfile`, push or `kind load` it, set `image:` in `k8s/deployment.yaml`,
then `kubectl apply -f k8s/`. With `--platform auto` the library detects KUBERNETES, reads config
from the mounted ConfigMap (hot-reloaded on `kubectl apply`), and resolves identity from the Downward
API — no CLI args needed.

## Wire up CI

`.github/workflows/ci.yml` calls the org's reusable component-CI workflow plus a `coverage` job
enforcing the 90% line-coverage gate; `.github/workflows/deploy-docs.yml` refreshes the docs site on
doc-only pushes once this component is registered in `registry/components.json`. Both are inert until
pushed to GitHub with the org secrets configured.
