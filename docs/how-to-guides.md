# How-to Guides

*This documents the generated scaffold; rewrite it as you build the component out.*

Recipes for specific tasks. Each assumes the scaffold runs (see the [tutorial](tutorial.md)). For
concepts see [explanation.md](explanation.md); for exhaustive options see [reference/](reference/).

---

## Write a new stage

A stage is a `Processor` subclass: `process(msg) -> List[ProcMsg]` (0..N out — filter/map/fan-out are
all this shape), and optionally `on_tick(now_ms) -> List[ProcMsg]` for a *stateful* stage (a window, a
debounce, a batch) that emits on a timer rather than on arrival:

```python
class ThresholdCross(Processor):
    def __init__(self, path: str, above: float):
        self.path, self.above = path, above

    def process(self, m: ProcMsg) -> List[ProcMsg]:
        v = pluck(m.msg.body, self.path)
        return [m] if isinstance(v, (int, float)) and v > self.above else []
```

Register it in `image_processor/pipeline.py`'s `_STAGES` table **and** add the matching variant to
`config.schema.json`'s `stage` definition — the two are one contract, and an unknown or misspelt
stage name is rejected when the route is parsed, at config time, not on the first message.

## Add a route

Each entry of `component.instances[]` is one route — its own thread, its own bounded queue, its own
pipeline:

```jsonc
{
  "id": "alarms",
  "subscribe": ["ecv1/+/+/+/data/#"],
  "publishTopic": "ecv1/gw-01/image-processor/alarms/data/summary",
  "target": "local",
  "pipeline": [ { "fieldEquals": { "path": "signal.id", "value": "pressure-1" } } ],
  "tickMs": 5000
}
```

`id` becomes the route's UNS instance token — it must be lower-kebab. A slow or malformed route never
stalls another: a bad one is skipped at startup with a warning (unless *every* route is malformed, in
which case the component refuses to start rather than idle silently).

## Send a route's output northbound

Set `"target": "northbound"` on the route instead of `"local"`. The dispatcher then calls
`get_messaging().publish_northbound(...)` with `Qos.AT_LEAST_ONCE` instead of the default local
publish — useful for a rollup that should reach IoT Core directly rather than staying on the
device-local bus.

## Keep the self-echo guard intact

If your route's `subscribe` filter could ever match its own `publishTopic` (a very common case —
`ecv1/+/+/+/data/#` matches almost everything), do **not** remove the `is_self_echo` check in
`_handler`. `main.py`'s `receive_own_messages(False)` only holds on Greengrass IPC; an MQTT broker
redelivers your own publishes to your own wildcard subscription regardless, so the guard in
`image_processor/pipeline.py` is what actually stops the loop.

## Report a real connection

A processor's routes are subscriptions on a bus the library already reports on — not links to a
device — so `instance_connectivity()` returns an empty list by default. Once a stage calls out to
something with its own liveness (a database, an upstream API), report it:

```python
from edgecommons.heartbeat.instance_connectivity import InstanceConnectivity

def instance_connectivity(self):
    return [
        InstanceConnectivity.of("enrich", self._db.is_connected(), "postgres://plant-db")
        .with_state("ONLINE")
        .with_attributes({"pool": self._db.pool_size()})
    ]
```

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
