# Reference — normalized output and decision rules

Every model this component serves belongs to one of four **task families**. A family decides what
the model is fed and what its output tensors mean, and it produces one **normalized output**: a
family-neutral shape that the result body carries and that the **decision rules** evaluate.

There is no bundle-supplied code. A model whose head no family can interpret is refused when the
bundle is staged, so an unreadable head becomes a staging error rather than a wrong answer.

## Normalized output

The normalized output has one field per family plus `rawShapes`, which records the shape of each
output tensor the session returned. A family populates its own field and leaves the rest empty.

| Field | Populated by | Meaning |
|---|---|---|
| `family` | every family | `classification`, `detection`, `segmentation`, or `anomaly`. |
| `classes` | classification | Scored classes, highest first. |
| `detections` | detection | Boxes surviving suppression, highest score first. |
| `segments` | segmentation | Per-class pixel counts and regions. |
| `anomaly` | anomaly | One score against one threshold. |
| `rawShapes` | every family | Output tensor name to shape. Diagnostic; a decision rule never needs it. |

Coordinates are always normalized to the **source image**, in `[0, 1]`, as `[x, y, w, h]` with the
origin at the top left. A box decoded on a letterboxed model canvas is reported against the picture
the camera took, so a consumer needs nothing but the result to draw it.

### `classes`

| Field | Meaning |
|---|---|
| `label` | The class name from the manifest's label set, or `class-<index>` when the manifest names a count rather than labels. |
| `index` | The class index the model produced. |
| `score` | The activated score. With `activation: softmax` the scores of the full class set sum to 1; with `sigmoid` each class is scored independently; with `none` the raw value is reported. |

The list holds at most `topK` entries, further bounded by the manifest's `maxResultItems`, and drops
anything below `scoreThreshold`.

### `detections`

| Field | Meaning |
|---|---|
| `label` | The class name from the manifest's label set. |
| `index` | The class index, after `classIndexOffset` is subtracted. |
| `score` | Objectness times class score for a grid head; the head's own confidence for a decoded head. |
| `box` | `[x, y, w, h]` normalized to the source image and clipped to it. |

Suppression is class-aware: a high-scoring bolt never suppresses a washer occupying the same pixels,
because two different things can be in one place. The list holds at most `maxDetections` entries,
further bounded by `maxResultItems`.

### `segments`

A mapping of label to `{pixels, fraction, bbox}`. Masks are never published.

| Field | Meaning |
|---|---|
| `pixels` | How many pixels of the class map the class claims. |
| `fraction` | That count divided by the size of the class map, so a rule can be written against the image rather than against a resolution. |
| `bbox` | The region those pixels occupy, normalized to the source image, or `null` when the class claims none. |

In `argmax` mode every label gets an entry, including the ones with no pixels, so a rule such as "no
defect pixels" evaluates on a clean image instead of failing to resolve its path. In `threshold`
mode the entry is the one named by `positiveLabel`. `minPixels` drops classes below that count.

### `anomaly`

| Field | Meaning |
|---|---|
| `score` | The activated, normalized score. In `[0, 1]` when the manifest declares a `normalization`. |
| `threshold` | The manifest's `threshold`, in the same units as `score`. |
| `anomalous` | Whether the score crossed the threshold in the direction `direction` names. |
| `direction` | `higherIsAnomalous` or `lowerIsAnomalous`. |
| `summary` | Present for a map head only. |

The `summary` of a map head carries `min`, `max`, `mean`, `aboveThresholdPixels`, `fraction`, and
`bbox` — the region of the crossing pixels, normalized to the source image, or `null` when none
crossed. The map itself stays in the executor cell.

## Decision rules

The manifest's `decisionRules` produce the result's `decision`. The rules are JSONPath expressions
over the normalized output, so a rule can name any field any family produces.

```json
{
  "pass": { "path": "$.classes[0].score", "op": ">=", "value": 0.9 },
  "confidence": "$.classes[0].score",
  "threshold": 0.9,
  "outcomeOnPass": "CLEAR",
  "outcomeOnFail": "HOLD",
  "failOnEmpty": false
}
```

| Key | Meaning |
|---|---|
| `pass` | The expression that decides the outcome. Required. |
| `confidence` | A JSONPath resolving to the number reported as `decision.confidence`. Optional. |
| `threshold` | A number, or a JSONPath resolving to one, reported as `decision.threshold`. Optional. |
| `outcomeOnPass` | The outcome when `pass` holds. Defaults to `CLEAR`. |
| `outcomeOnFail` | The outcome when `pass` does not hold: `HOLD` or `FAIL`. Defaults to `HOLD`. `CLEAR` is refused. |
| `failOnEmpty` | Whether a path that matches nothing is a plain failure rather than a broken rule. Defaults to `false`. |

### Expressions

An expression is a leaf or a group.

```json
{ "all": [ {"path": "$.detections[*].score", "op": ">=", "value": 0.5},
           {"path": "$.detections[*].label", "op": "!=", "value": "washer"} ] }
```

| Form | Meaning |
|---|---|
| `{"path": …, "op": …, "value": …}` | A leaf comparison. |
| `{"all": [ … ]}` | Holds when every child holds. |
| `{"any": [ … ]}` | Holds when at least one child holds. |

Groups nest to any depth.

### Operators

| Operator | Meaning |
|---|---|
| `>=`, `>`, `<=`, `<` | Numeric comparison. Both sides must be numbers; a flag or a string is a broken rule. |
| `==`, `!=` | Equality against the literal in `value`. Works for numbers, strings, and booleans. |
| `exists` | The path matches at least one value. Takes no `value`. |
| `absent` | The path matches nothing. Takes no `value`. |
| `count>=` | The number of matched values is at least `value`. |

A leaf whose path matches several values is a claim about **every** match:
`$.detections[*].label != "washer"` means "no detection is a washer". `exists`, `absent`, and
`count>=` describe the match set itself rather than its contents.

### Outcomes

| Outcome | Meaning |
|---|---|
| `CLEAR` | The rule evaluated and passed. |
| `HOLD` | The rule evaluated and failed, or could not be evaluated at all. |
| `FAIL` | The rule evaluated and failed, and the manifest classes that failure as a defect rather than a doubt. |

**A rule that cannot be evaluated yields `HOLD`, never `CLEAR`.** A missing path, a malformed
expression, a confidence that resolves to nothing or to a non-number, a threshold that resolves to
nothing, an `outcomeOnFail` of `CLEAR`: all of them hold the image. `decision.rule` says which rule
decided, so an operator can tell a failed image from a failed rule set:

| `decision.rule` | Meaning |
|---|---|
| `pass` | The rules evaluated and passed. |
| `pass.all[1]: $.classes[0].score >= 0.99` | This leaf is the one that failed. |
| `pass.any: none of 2 matched` | An `any` group had no child that held. |
| `UNEVALUABLE:confidence: path '$.nowhere' matched nothing` | The rule set is broken, not the image. |

`decision.confidence` and `decision.threshold` are `null` whenever the outcome came from an
unevaluable rule, because a number that could not be read is not a number to report.
