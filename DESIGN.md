# DESIGN — ImageProcessor

> Treat this document as the design-fidelity contract for this component: before changing behavior,
> update the relevant section here in the same change: what changed, why, and what it means for
> config/commands/metrics/validation. A build that compiles but drifts from what this document says
> is not done.

## What it is

ImageProcessor is a `com.mbreissi.edgecommons.ImageProcessor` processing component built on the `edgecommons`
Python library: it subscribes to a set of topics, runs each message through a pipeline of stages,
and republishes the result. Describe here, once you build it out: what it consumes, what
transformation it performs, and where the result goes (local bus, northbound, or both).

## Decisions

Record each significant design decision as it's made, numbered so later sessions can cite it:

- **D-1.** _(example)_ — replace with your first real decision (e.g. why a given stage lives in this
  component rather than upstream).

## Config

What each route (`component.instances[]` entry) means for this component: its subscription filters,
its pipeline, its target, and any stage-specific arguments. Keep this section's claims verified
against `config.schema.json` — if they disagree, the schema is the source of truth and this section
is stale.

## Command surface

This scaffold registers no custom command verbs beyond the library built-ins
(`ping`/`reload-config`/`get-configuration`/`status`). Document any you add here: request/reply
shape and error codes.

## Metrics

`processorThroughput` (`received`/`published`/`dropped`/`errors`) ships by default. Document any
metric you add: name, dimensions (keep them low-cardinality), measures, and what each one means
operationally.

## Validation

What "this component works" means in practice for this repo: which tests must pass
(`python -m pytest`, no broker/device needed), the coverage gate (90% line coverage — see
`.github/workflows/ci.yml`), and any platform-specific smoke test (HOST/Greengrass/Kubernetes) this
component needs before a change ships.
