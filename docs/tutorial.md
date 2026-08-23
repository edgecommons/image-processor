# Tutorial — one image, end to end

By the end of this you have run the component against a local broker, put one image through it, and
read every output it produced: the confirmed result on the bus, the decision mirror, the evidence
sidecar on disk, and the archived image. It takes about ten minutes and needs no GPU, no network,
and no camera.

## Before you start

- Python 3.12 or later.
- An MQTT broker on `localhost:1883`. `docker run -d -p 1883:1883 --name emqx emqx/emqx:latest`
  gives you one.
- `mosquitto_sub`, or any MQTT client that can print what arrives.

## 1. Install the component

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[cpu]" -r requirements-test.txt
```

## 2. Build a model bundle

The component only runs models that arrive as verified bundles, so you need one before it has
anything to do. The test corpus generates its models rather than downloading them, which makes the
bundle digest reproducible — the same commands on the same ONNX Runtime produce the same bytes,
which is what `test-configs/config.json` pins.

```bash
python3 tests/fixtures/build.py --out tests/fixtures/out
mkdir -p /var/lib/edgecommons/image-processor/bundles
python3 tools/make_bundle.py tests/fixtures/out/bundles/synthetic-anomaly-scalar-1.0.0 \
  --out /var/lib/edgecommons/image-processor/bundles/synthetic-anomaly-scalar-1.0.0.tar
```

The last command prints the bundle digest:

```text
sha256:4a87394f34ab6c1b1f17c9d048ebed277e5fdae8f76823a8d2371cfd57178ee1
```

That is the digest `models[0].digest` and `instances[].modelRef.digest` already name in
`test-configs/config.json`. If your build prints a different one, put it in both places: a
reference that does not match the bytes on disk is refused, which is the point of naming it.

The model is an anomaly detector. It scores an image between 0 and 1 and the bundle's decision
rules pass anything below 0.05, so `tests/fixtures/out/images/anomaly-good.png` decides `CLEAR` and
`anomaly-bad.png` decides `HOLD`.

## 3. Create the directories the configuration names

The component owns its durable state and the spool it reads, so both have to exist and be
writable.

```bash
mkdir -p /var/spool/camera-adapter/cam-01 \
         /var/spool/image-processor/processed/cam-01 \
         /var/spool/image-processor/failed/cam-01 \
         /var/spool/inspection
```

## 4. Run it

```bash
python3 main.py --platform HOST --transport MQTT ./test-configs/standalone-messaging.json \
  -c FILE ./test-configs/config.json -t smoke-device
```

Watch the log. In order, it stages the bundle, warms it on the executor cell, activates it on both
routes, subscribes the camera announcement channel and the trigger topic, and reports ready:

```text
staged synthetic-anomaly-scalar 1.0.0 (sha256:4a87394f…)
warmed synthetic-anomaly-scalar on cpu-0 in 51 ms (1 golden sample(s), providers CPUExecutionProvider)
route clearance-cam-01 is now running synthetic-anomaly-scalar (sha256:4a87394f…)
route clearance-cam-01 subscribed camera hints on ecv1/smoke-device/camera-adapter/cam-01/app/image/captured
route adhoc-inspect subscribed ecv1/+/inspection-ui/+/app/inspect/request
ImageProcessor is running: 2 route(s), 1 model(s)
```

Nothing was ready before the model was: a route with no active generation reports itself degraded
rather than accepting an image it cannot decide.

## 5. Watch the bus

In a second terminal:

```bash
mosquitto_sub -h localhost -p 1883 -v \
  -t 'ecv1/+/image-processor/+/app/inference/result' \
  -t 'ecv1/+/image-processor/+/data/#' \
  -t 'ecv1/+/image-processor/+/evt/#' \
  -t 'ecv1/+/image-processor/state'
```

The envelope is protobuf, so a plain subscriber prints bytes. To read the bodies, use a client
built on the library — `ec-uns-cmd` for commands, or a short paho script that parses each payload
with `edgecommons.messaging.message.Message.from_bytes`.

## 6. Drop a capture into the spool

A camera writes the metadata sidecar first and the image second, which is what makes
`readiness.mode: cameraSidecar` sound: a visible image with a sidecar is complete by construction.
Do the same. Save this as `drop.py` and run it:

```python
import hashlib, json, shutil
from pathlib import Path

root = Path("/var/spool/camera-adapter/cam-01")
relative = "2026/08/23/cap-0001.png"
data = Path("tests/fixtures/out/images/anomaly-good.png").read_bytes()
target = root / relative
target.parent.mkdir(parents=True, exist_ok=True)
body = {
    "schemaVersion": 1,
    "eventId": "evt_cap_0001",
    "captureId": "cap_0001",
    "cameraId": "cam-01",
    "correlationId": "corr_cap_0001",
    "timestamps": {"persistedAt": "2026-08-23T12:00:00.512Z"},
    "image": {
        "absolutePath": str(target),
        "relativePath": relative,
        "contentType": "image/png",
        "encoding": "png",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "metadataSidecarRelativePath": relative + ".json",
    },
    "camera": {"backend": "sim"},
}
target.with_name(target.name + ".json").write_text(json.dumps(body, indent=2))
target.write_bytes(data)
```

The component notices the write, verifies the declared size and digest against the file, admits the
job, runs it, and completes it:

```text
committed blvuf6ru3i6jzmmksre6hkgccm: CLEAR on ecv1/smoke-device/image-processor/clearance-cam-01/app/inference/result
completed blvuf6ru3i6jzmmksre6hkgccm: archive
```

## 7. Read what came out

**The result**, on `ecv1/smoke-device/image-processor/clearance-cam-01/app/inference/result`, is
the authoritative output — the one a safety gate reads:

```json
{
  "schemaVersion": "1.0",
  "inferenceId": "blvuf6ru3i6jzmmksre6hkgccm",
  "routeId": "clearance-cam-01",
  "status": "SUCCEEDED",
  "source": {
    "kind": "spool",
    "captureId": "cap_0001",
    "cameraId": "cam-01",
    "relativePath": "2026/08/23/cap-0001.png",
    "bytes": 112,
    "sha256": "1ce61926d4bc2c0f9a42452e86c205d26f8f1e2a1f0fcf94cd760a70cf2fc49f"
  },
  "model": {
    "id": "synthetic-anomaly-scalar",
    "version": "1.0.0",
    "digest": "sha256:4a87394f34ab6c1b1f17c9d048ebed277e5fdae8f76823a8d2371cfd57178ee1",
    "runtime": "onnxruntime",
    "providers": ["CPUExecutionProvider"],
    "gpu": null
  },
  "decision": { "outcome": "CLEAR", "pass": true, "confidence": 0.0, "threshold": 0.05, "rule": "pass" },
  "outputs": { "family": "anomaly", "anomaly": { "score": 0.0, "threshold": 0.05, "anomalous": false, "direction": "higherIsAnomalous" }, "truncated": false },
  "artifacts": { "evidenceId": "blvuf6ru3i6jzmmksre6hkgccm", "localRelativePath": "cap-0001.png.inference.json", "sha256": "ad659011…", "bytes": 1805 }
}
```

`providers` is what the session actually ran on, not what configuration asked for.

**The decision mirror**, three `SouthboundSignalUpdate` readings on
`…/clearance-cam-01/data/line-clearance/{pass,confidence,status}`, is the same verdict in the shape
a historian or an HMI tag already understands. It is best effort: a consumer enforcing a gate reads
the result above instead, because a mirror that silently stopped updating looks exactly like one
that keeps reporting the last value.

**The evidence sidecar and the archived image** are in `processed/`, because the route archives on
success and the move happens only after the result is confirmed:

```bash
$ find /var/spool/image-processor -type f
/var/spool/image-processor/processed/cam-01/2026/08/23/cap-0001.png
/var/spool/image-processor/processed/cam-01/2026/08/23/cap-0001.png.bundle.json
/var/spool/image-processor/processed/cam-01/2026/08/23/cap-0001.png.inference.json
/var/spool/image-processor/processed/cam-01/2026/08/23/cap-0001.png.json
```

The image travelled with its camera sidecar (`.json`), its evidence record (`.inference.json`), and
the manifest of the move (`.bundle.json`). The evidence record holds the full result plus what only
this device knows: the configuration generation the route ran under, the provider policy the
manifest demanded, and when the record was written.

## 8. Ask it a question

The component answers the library built-ins and its own verbs on its command inbox. `ec-uns-cmd`
builds the request envelope for you:

```bash
$ ec-uns-cmd --device smoke-device --component image-processor status
{
  "instances": [
    {
      "attributes": {
        "activeGeneration": "sha256:4a87394f34ab6c1b1f17c9d048ebed277e5fdae8f76823a8d2371cfd57178ee1",
        "desiredGeneration": "sha256:4a87394f34ab6c1b1f17c9d048ebed277e5fdae8f76823a8d2371cfd57178ee1",
        "executorHealthy": true,
        "oldestAgeSecs": 0.0,
        "paused": false,
        "queued": 0,
        "sourceReachable": true
      },
      "connected": true,
      "detail": "/var/spool/camera-adapter/cam-01",
      "instance": "clearance-cam-01",
      "state": "ONLINE"
    }
  ],
  "status": "RUNNING",
  "uptimeSecs": 82
}
```

`get-models` reports what is staged and which routes run it, and `get-queue` reports the jobs by
state with the scheduler's own view of its lanes and cells.

Pause a route and the capture you drop stays in the spool; resume it and the next walk picks the
capture up:

```bash
ec-uns-cmd --device smoke-device --component image-processor pause
ec-uns-cmd --device smoke-device --component image-processor resume
```

## 9. Trigger an inference over the bus

The second route in the shipped configuration takes images as messages. Publish a file reference
and read the correlated reply:

```python
import hashlib, shutil
from pathlib import Path
import paho.mqtt.client as mqtt
from edgecommons.messaging.message import Message, MessageHeader

target = Path("/var/spool/inspection/batch/part-01.png")
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile("tests/fixtures/out/images/anomaly-good.png", target)
data = target.read_bytes()

header = MessageHeader("InspectRequest", "1.0")
header.reply_to = "ecv1/smoke-device/inspection-ui/panel/app/inspect/reply"
message = Message(header=header, body={
    "relativePath": "batch/part-01.png",
    "sha256": hashlib.sha256(data).hexdigest(),
    "bytes": len(data),
})

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)
client.loop_start()
client.publish("ecv1/smoke-device/inspection-ui/panel/app/inspect/request",
               message.to_bytes(), qos=1).wait_for_publish()
```

The route verifies the declared size and digest against the file, copies it into processor-owned
staging, infers, publishes the result on
`…/adhoc-inspect/app/inference/result`, and answers your `reply_to` with the same body under your
correlation id. Its completion policy is `delete`, so the referenced file is gone afterwards.

## 10. Stop it

Press Ctrl-C. The component stops claiming work, lets the executor drain, publishes what is
already committed, and closes its ledger:

```text
Received signal 2; beginning graceful shutdown
stopping ImageProcessor
scheduler loop stopped
ImageProcessor stopped
Clean disconnection from local broker
```

Nothing is lost if you kill it instead: a job that was mid-inference is durable and restarts as
`READY`, and a result that was committed but not yet published is published on the next start.
Try it — kill the process between the "committed" and "completed" lines, start it again, and watch
the result go out.

## What next

- The [how-to guides](how-to-guides.md) cover signing a bundle, adding a route, and repairing a job
  a crash left behind.
- [Sample configurations](sample-configurations.md) has the shapes for a GPU device, several
  cameras, and the trigger-only deployment.
- The [messaging interface](reference/messaging-interface.md) documents every topic, body, and
  command reply.
