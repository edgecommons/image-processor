# Reference — Metrics

*This documents the generated scaffold; rewrite it as you build the component out.*

The scaffold emits one metric through the EdgeCommons metric service. With
`metricEmission.target: messaging`, it is published on the reserved UNS `metric` class:

```text
ecv1/{device}/image-processor/main/metric/{metricName}
```

The component never writes reserved `metric` topics directly — it defines the metric through
`MetricBuilder`, so the same name, measures, and dimensions are used whether the target is
`messaging`, `log`, `cloudwatch`, or `prometheus`.

## `processorThroughput`

One metric across every route, flushed every 60 seconds (`METRIC_INTERVAL_SECS`).

Dimensions: none (aggregate across all routes; add a `route` dimension if you need per-route
breakdown, and keep it bounded to your configured route ids).

| Measure | Unit | Purpose |
|---|---:|---|
| `received` | Count | Messages accepted off the bus across every route's subscriptions. |
| `published` | Count | Messages successfully published after running the pipeline. |
| `dropped` | Count | Messages dropped because a route's bounded queue was full. Never silent — a processor that discards messages quietly is worse than one that crashes. |
| `errors` | Count | Publish attempts that raised (see the `publish-failed` event for the reason). |

## Adding your own metric

Follow the same shape: define once via `MetricBuilder.create(name).with_config(self._cm)` in
`__init__`, emit via `self._metrics.emit_metric(name, {...})` wherever the value changes. Keep
dimensions low-cardinality — a route id or a result code, never a signal name, a topic, or raw error
text; those belong in events, logs, or command replies.
