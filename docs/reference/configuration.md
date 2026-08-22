# Reference — Configuration

*This documents the generated scaffold; rewrite it as you build the component out.*

Every configuration option this scaffold itself understands. For *why*, see
[explanation.md](../explanation.md); for tasks, see the [how-to guides](../how-to-guides.md).

## Config source

The component reads one JSON document from `-c/--config`, defaulting by platform: `HOST` → `FILE`,
`GREENGRASS` → `GG_CONFIG`, `KUBERNETES` → `CONFIGMAP`. This scaffold's own settings live under
`component`; the sibling sections (`tags`, `hierarchy`, `identity`, `topic`, `messaging`, `logging`,
`metricEmission`, `heartbeat`) are standard edgecommons sections owned by the canonical schema and
are **not** redeclared in `config.schema.json`.

## `component.token`

The UNS component token: the `{component}` segment of every topic this component publishes on, and
the `identity.component` field of every message envelope. UNS tokens are lower-kebab
(`image-processor`); the Greengrass component name is the reverse-DNS `com.mbreissi.edgecommons.ImageProcessor` and never
reaches the wire. Every shipped configuration and the recipe set it. Leave it set — without it the
library falls back to the short form of the component name, `ImageProcessor`, and the topics in
this documentation stop matching what the component publishes.

## `component.global.defaults`

| Key | Type | Default | Definition |
|-----|------|---------|-----------|
| `tickMs` | integer | `10000` | How often stateful stages are ticked. A stage that emits on time rather than on arrival (a window, a batch, a debounce) produces its output on this cadence. |
| `maxQueue` | integer | `256` | How many messages may be queued for a route before new ones are dropped and counted. Bounded on purpose: an unbounded queue does not remove backpressure, it relocates the failure to the heap. |

## `component.instances[]` (a route)

A processor models **one route per instance** — a set of topic filters, a pipeline of stages, and a
target. Routes are independent: one thread each, so a slow route cannot stall another, and per-key
state inside a stage needs no lock.

| Key | Type | Required | Definition |
|-----|------|----------|-----------|
| `id` | string | yes | Unique route id, lower-kebab (`^[a-z0-9]+(?:-[a-z0-9]+)*$`). The `{instance}` token of this route's UNS topics and of everything it publishes. |
| `subscribe` | string[] | no | Topic filters this route consumes. Wildcards allowed (e.g. `ecv1/+/+/+/data/#`). |
| `publishTopic` | string | yes | The topic the transformed result is published on. Named by config, not minted from a signal id — the payload-agnostic archetype. |
| `target` | `local` \| `northbound` | `local` | Where the result goes: the device-local bus, or straight to the northbound broker. |
| `pipeline` | array of stage | `[]` | The stages, in order. An empty pipeline is a pass-through republisher. |
| `maxQueue` | integer | instance/global default | Per-route override of the queue bound. |
| `tickMs` | integer | instance/global default | Per-route override of the stage tick cadence. |

## Stages (`pipeline[]` entries)

A stage is a single-key object naming the stage and its arguments. Two ship with the scaffold:

| Stage | Arguments | Behavior |
|---|---|---|
| `fieldEquals` | `path` (string, dotted), `value` (any JSON type) | Keeps only messages whose dotted body path equals `value`; drops the rest. |
| `countPerTick` | *(none)* | Accumulates arrivals and emits one `{count, last}` rollup per tick. Emits nothing on arrival — output happens in `on_tick`. |

Add your own to `image_processor/pipeline.py`'s `_STAGES` table **and** to this schema's `stage` definition — the
two are one contract. An unknown or misspelt stage name is rejected when the route is parsed, at
config time, not on the first message.

## Precedence

`tickMs` / `maxQueue` resolve: **route value ▸ `component.global.defaults`**.

## Complete example

```jsonc
{
  "hierarchy": { "levels": ["site", "device"] },
  "identity": { "site": "factory-1" },
  "metricEmission": { "target": "messaging" },
  "component": {
    "token": "image-processor",
    "global": { "defaults": { "tickMs": 10000, "maxQueue": 256 } },
    "instances": [
      {
        "id": "rollup",
        "subscribe": ["ecv1/+/+/+/data/#"],
        "publishTopic": "ecv1/gw-01/image-processor/rollup/data/summary",
        "target": "local",
        "pipeline": [
          { "fieldEquals": { "path": "signal.id", "value": "temperature-1" } },
          { "countPerTick": {} }
        ]
      }
    ]
  }
}
```

## Limitations

- Unknown keys anywhere in a route or a stage are **rejected, not ignored** — a typo'd key is a
  mistake, not a no-op.
- A processor with **zero valid routes** refuses to start rather than idle silently.
