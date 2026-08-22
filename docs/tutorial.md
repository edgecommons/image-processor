# Tutorial — From zero to a live route

*This documents the generated scaffold; rewrite it as you build the component out.*

By the end you'll have `ImageProcessor` running one route that filters and rolls up data messages
and republishes the result, end to end over a local MQTT broker.

## 1. Prerequisites

- Python 3.9+, and a local MQTT broker on `localhost:1883`
  (`docker run -d -p 1883:1883 emqx/emqx:latest`, or `docker compose up --build`, which starts one
  for you alongside the processor).
- From this directory: `pip install -r requirements.txt`.

## 2. Run it

```bash
python3 main.py --platform HOST --transport MQTT ./test-configs/standalone-messaging.json \
  -c FILE ./test-configs/config.json -t my-thing
```

`test-configs/config.json` declares one route, `rollup`: it subscribes to `ecv1/+/+/+/data/#`, keeps
only messages whose `signal.id` equals `temperature-1` (`fieldEquals`), counts them, and emits a
rollup every 10 seconds (`countPerTick`) onto `ecv1/gw-01/image-processor/rollup/data/summary`.

## 3. Feed it something and watch it republish

```bash
mosquitto_sub -h localhost -p 1883 -t 'ecv1/+/+/+/data/#' -v
mosquitto_pub -h localhost -p 1883 -t 'ecv1/gw-01/sim/data/temperature-1' \
  -m '{"header":{"name":"SouthboundSignalUpdate","version":"1.0"},"body":{"signal":{"id":"temperature-1"},"samples":[{"value":21.5}]}}'
```

Publish a few of these within a 10-second window; on the next tick you'll see one rollup message
land on `ecv1/gw-01/image-processor/rollup/data/summary` carrying `{"count": N, "last": {...}}`. Publish
a message with a different `signal.id` and confirm it does **not** show up in the count — the
`fieldEquals` stage dropped it.

## 4. Confirm the self-echo guard holds

The route also subscribes to `ecv1/+/+/+/data/#`, which matches its own published rollup. Watch the
count stay bounded to what you actually published — the processor never reprocesses its own output
(see [explanation.md](explanation.md) for why this matters).

## 5. Watch the reported surface

```bash
mosquitto_sub -h localhost -t 'ecv1/+/+/state' -t 'ecv1/+/+/metric/#' -t 'ecv1/+/+/+/evt/#' -v
```

- **`state`** — the automatic keepalive. No `instances[]` section — a processor's routes are
  subscriptions, not device connections, so it reports none (see
  [explanation.md](explanation.md#instance-connectivity-a-processor-reports-none)).
- **`metric/processorThroughput`** — `received`/`published`/`dropped`/`errors`, once a minute.
- **`evt/warning/publish-failed`** (on the route's instance token) — only if a publish actually
  fails; you won't see this in the happy path above.

## 6. Run the tests

```bash
python -m pytest
```

No broker needed. `tests/test_pipeline.py` exercises the stages, the tick, and the self-echo guard as
pure logic; `tests/test_connectivity.py` exercises the one seam the app wires into the library.

Next: the [how-to guides](how-to-guides.md) to add your own stage and deploy; the
[reference](reference/) for every option and topic; the [explanation](explanation.md) for why the
processor is shaped this way.
