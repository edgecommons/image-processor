# Sample Configurations

*This documents the generated scaffold; rewrite it as you build the component out.*

The scaffold ships one working config, `test-configs/config.json`, plus the MQTT
`standalone-messaging.json` for local HOST runs. For the exhaustive option list see
[reference/configuration.md](reference/configuration.md); for the pipeline model see
[explanation.md](explanation.md).

## `test-configs/config.json` — one route, a filter + a rollup

```jsonc
{
  "logging": { "level": "INFO", "python_format": "..." },
  "hierarchy": { "levels": ["site", "device"] },
  "identity": { "site": "factory-1" },
  "heartbeat": { "enabled": true, "intervalSecs": 5, "measures": { "cpu": true, "memory": true }, "destination": "local" },
  "metricEmission": { "target": "log", "namespace": "edgecommons", "targetConfig": { "logFileName": "{ComponentFullName}.metric.log" } },
  "tags": { "site": "factory-1" },
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
        ],
        "tickMs": 10000
      }
    ]
  }
}
```

**What each option does at runtime**

| Option | Effect |
|--------|--------|
| `component.global.defaults.tickMs` | Fallback tick cadence for a route that doesn't override it — how often stateful stages (like `countPerTick`) emit. |
| `component.global.defaults.maxQueue` | Fallback queue bound — how many messages may wait for this route's thread before new ones are dropped and counted. |
| `instances[].id` | The route's UNS instance token (`rollup`) and the identity stamped on everything it publishes. Must be lower-kebab. |
| `subscribe` | Topic filters this route consumes; wildcards allowed. Here, the whole `data` class. |
| `publishTopic` | Where the transformed result lands. Named by config — a processor is payload-agnostic and does not mint this from a signal id. |
| `target: local` | Publishes to the device-local bus (the common case). `northbound` sends straight to the northbound broker instead. |
| `pipeline` | Two stages: `fieldEquals` keeps only `signal.id == "temperature-1"`, dropping everything else; `countPerTick` accumulates survivors and emits `{count, last}` once per `tickMs`. |
| `tickMs` (per-route) | Overrides the global default for this route only. |

## `test-configs/standalone-messaging.json` — the HOST/MQTT broker

```json
{ "messaging": { "local": { "host": "localhost", "port": 1883, "clientId": "image-processor-local" } } }
```

Passed as the `--transport MQTT <path>` argument, independent of the `-c FILE ...` component config.

## Adding a second route

Routes are independent — add another entry to `component.instances[]` with its own `id`,
`subscribe`, `publishTopic`, and `pipeline`. A slow or misconfigured route never stalls another (each
gets its own thread and its own bounded queue); a malformed route is skipped at startup with a
warning rather than crashing the whole component.
