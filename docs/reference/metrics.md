# Reference — Metrics

Eight metric groups, defined once through `MetricBuilder` and flushed every 60 seconds. The same
names, measures, and dimensions reach a log file, CloudWatch, Prometheus, or the reserved `metric`
class depending only on `metricEmission.target`. With `target: messaging` they are published on:

```text
ecv1/{device}/image-processor/metric/{groupName}
```

Three kinds of measure feed a group and are flushed together: counters the component increments as
work happens (reported as the interval's total and reset), averages it observes per job (reported
as the interval's mean and omitted when nothing was observed), and gauges sampled from the live
subsystems at flush time.

## Cardinality

No file name, capture id, inference id, model version, or topic is ever a dimension. Each of those
identifies one image or one deployment artifact, and a dimension that identifies one image turns a
metric backend into an unbounded index of them. That detail belongs in the result, the evidence
sidecar, the events, and the command replies, all of which are bounded by something other than
traffic.

## `ImageProcessorDiscovery`

What the input sources saw.

| Measure | Unit | Meaning |
|---|---|---|
| `discovered` | Count | Inputs admitted to the ledger. |
| `rejected` | Count | Inputs refused because they can never be admitted as they stand. |
| `rescans` | Count | Authoritative walks of a spool root. |
| `nudges` | Count | Notifications and hints that asked for a walk sooner. |
| `hintsAccepted` | Count | Camera announcements whose declared size and digest verified against the file. |
| `hintsRejected` | Count | Announcements that did not verify. |
| `hintsUnmapped` | Count | Announcements naming a path outside this route's root. |
| `captureRecords` | Count | `SUCCEEDED` capture-status records verified by reconciliation. |
| `triggersAccepted` | Count | Trigger messages admitted. |
| `triggersRejected` | Count | Trigger messages refused. |

## `ImageProcessorQueue`

How much work exists and how long it has waited.

| Measure | Unit | Meaning |
|---|---|---|
| `admitted` | Count | Jobs newly admitted this interval. |
| `queued` | Count | Jobs in flight: discovered, ready, claimed, waiting for a model, inferencing, or in retry. |
| `retryWaiting` | Count | Jobs whose retry timer has not elapsed. |
| `dispatched` | Count | Jobs the scheduler sent to a cell. |
| `oldestAgeSecs` | Seconds | How long the oldest in-flight job has waited since admission. |
| `pausedRoutes` | Count | Routes an operator has paused. |

## `ImageProcessorModelCache`

Model delivery and activation.

| Measure | Unit | Meaning |
|---|---|---|
| `staged` | Count | Bundles fetched, verified, and promoted into the cache. |
| `activated` | Count | Route generation switches. |
| `stagingFailures` | Count | Bundles that could not be fetched, verified, extracted, or interpreted. |
| `warmupFailures` | Count | Bundles that staged but failed golden warmup, so no route switched to them. |
| `rollbacks` | Count | Returns to a retained previous generation. |
| `cachedBundles` | Count | Verified bundles in the content-addressed cache. |
| `routesStaging` | Count | Routes whose desired generation is ahead of their active one. |

## `ImageProcessorGpu`

The executor boundary.

| Measure | Unit | Meaning |
|---|---|---|
| `residentModels` | Count | Model generations resident across the cells. |
| `residentMiB` | Megabytes | What those sessions hold. |
| `loads` | Count | Sessions loaded. |
| `evictions` | Count | Sessions released to make room or on operator request. |
| `recycles` | Count | Cells drained and restarted, which evicts everything that cell held. |
| `healthyCells` | Count | Cells alive and able to serve. |

## `ImageProcessorInference`

Outcomes and stage timings, averaged over the interval.

| Measure | Unit | Meaning |
|---|---|---|
| `succeeded` | Count | Inferences that produced a result. |
| `failed` | Count | Inferences that ended without one. Their decision is `HOLD`. |
| `retried` | Count | Transient failures returned to the retry timer. |
| `exhausted` | Count | Jobs that spent their retry budget. |
| `blocked` | Count | Jobs blocked by a permanent model or provider failure. |
| `queueMs` | Milliseconds | Mean time from admission to dispatch. |
| `inferenceMs` | Milliseconds | Mean session run time. |
| `totalMs` | Milliseconds | Mean end-to-end time per attempt. |

## `ImageProcessorOutbox`

Confirmed publication.

| Measure | Unit | Meaning |
|---|---|---|
| `pending` | Count | Committed results waiting for transport confirmation. |
| `published` | Count | Results confirmed this interval. |
| `attempted` | Count | Publication attempts. |
| `failed` | Count | Attempts the transport refused or did not confirm in time. |
| `exhausted` | Count | Results that spent `publish.maxAttempts`. Their inputs are retained. |
| `reservedBytes` | Bytes | Outbox and evidence capacity held by admitted jobs. |

## `ImageProcessorCompletion`

What happened to the inputs.

| Measure | Unit | Meaning |
|---|---|---|
| `completed` | Count | Completion actions that succeeded. |
| `archived` | Count | Inputs moved to `archiveDir`. |
| `deleted` | Count | Inputs removed. |
| `quarantined` | Count | Inputs moved to `failedDir`. |
| `retained` | Count | Inputs left where they are. |
| `failed` | Count | Completion actions that failed. Those jobs are `CLEANUP_FAILED`, never success. |
| `mirrored` | Count | Decision-signal readings published. |

## `ImageProcessorDisk`

What the component is using, and what is left.

| Measure | Unit | Meaning |
|---|---|---|
| `stateDbBytes` | Bytes | The ledger file. |
| `modelCacheBytes` | Bytes | The content-addressed bundle cache. |
| `stagingBytes` | Bytes | Staged inputs and in-progress bundle extractions. |
| `freeMiB` | Megabytes | Free space on the filesystem holding the durable state. |
