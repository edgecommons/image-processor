# ImageProcessor — Documentation

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

| Doc | Start here when you want to… |
|-----|------------------------------|
| **[Tutorial](tutorial.md)** | learn by doing — run one image through the component and read every output it produced |
| **[How-to guides](how-to-guides.md)** | accomplish a task — build and sign a bundle, add a route, deploy, repair a stuck job |
| **[Reference](reference/)** | look up an exact config option, topic, command, or metric |
| **[Explanation](explanation.md)** | understand how an image becomes a job, and why the walk is authoritative |

## Quick routing

- **"I'm new here."** → [Tutorial](tutorial.md).
- **"What config option does X?"** → [Reference — Configuration](reference/configuration.md).
- **"What message on which topic, and what does a command reply look like?"** →
  [Reference — Messaging interface](reference/messaging-interface.md).
- **"What does this metric mean?"** → [Reference — Metrics](reference/metrics.md).
- **"What does a task family produce, and how do decision rules read it?"** →
  [Reference — Normalized output and decision rules](reference/data-types.md).
- **"What do I put in a config for a camera route, a trigger route, or a GPU device?"** →
  [Sample configurations](sample-configurations.md).
- **"Why does a lost camera announcement not lose an image?"** → [Explanation](explanation.md).

## Audience

These pages are for the integrator wiring a camera to a model and the operator running the result.
They describe what the component does; `DESIGN.md` and `LLD.md` in the repository root record why
it does it that way and how the modules fit together.
