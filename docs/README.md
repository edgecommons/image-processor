# ImageProcessor — Documentation

*This documents the generated scaffold; rewrite it as you build the component out.*

`com.mbreissi.edgecommons.ImageProcessor` is a Python **processing component** built on the `edgecommons` library: it
subscribes to messages, transforms them through a pipeline of stages, and forwards the result. It
runs as a Greengrass v2 component, a standalone HOST process, or a Kubernetes pod.

```text
  subscribe(filter) ──► bounded queue ──► one thread per route ──► publish
                                             (Pipeline)           local | northbound
```

| Doc | Start here when you want to… |
|-----|------------------------------|
| **[Tutorial](tutorial.md)** | learn by doing — run the scaffold end to end and watch it republish |
| **[How-to guides](how-to-guides.md)** | accomplish a task — write a stage, add a route, deploy |
| **[Reference](reference/)** | look up an exact config option, topic, or metric |
| **[Explanation](explanation.md)** | understand the processor archetype and its non-negotiable guards |

## Quick routing

- **"I'm new here."** → [Tutorial](tutorial.md).
- **"What config option does X?"** → [Reference — Configuration](reference/configuration.md).
- **"What message on which topic?"** → [Reference — Messaging Interface](reference/messaging-interface.md).
- **"What does this metric mean?"** → [Reference — Metrics](reference/metrics.md).
- **"Why the self-echo guard? Why `get_messaging()` and not `data()`?"** → [Explanation](explanation.md).

## Audience

These docs are for whoever picks up this scaffold next — the integrator wiring a real route, and
the operator deploying it. They describe the component **as generated**; once you add stages to
`image_processor/pipeline.py` or routes to config, update the pages that describe them (see `AGENTS.md`).
