# Explanation - how input sources work, and why

This page is the mental model for how an image becomes a job. For exact options see
[reference/](reference/); for tasks, the [how-to guides](how-to-guides.md).

An `image-processor` route binds one input source to one immutable model version. Two kinds of
source feed it, and they answer the same question in two different ways.

## Filesystem state is authoritative

A spool route watches a directory the component owns. What is in that directory is the truth about
what work exists, and a periodic deterministic walk is the only thing that admits work.

That is a deliberate inversion of the usual design, where an event stream is the queue and the
directory is a side effect. Notification streams lose events. `inotify` queues overflow under
burst; a watch is not established until after the directory is opened, so anything written in that
window is never announced; network and container filesystems deliver notifications unevenly or not
at all; and a process that restarts learns nothing about what happened while it was down. Every one
of those failures is silent. If the notification stream were the queue, each one would be a lost
image, and for a line-clearance decision a lost image is a decision that never happened.

So the walk is authoritative and everything else only makes it happen sooner:

- A `watchdog` observer coalesces filesystem notifications into debounced scan nudges.
- A periodic interval walks anyway, whether or not a notification arrived.
- The camera's capture announcement is a third nudge, and it carries proof.
- The `trigger-rescan` command walks on demand.

A missed notification therefore costs latency, never a job.

## Readiness proves a file is finished

A file appearing in a directory is not an input. A writer creates the entry before it writes the
bytes, so an image discovered the instant it appears is usually a truncated one. An input is a file
that something has proved is complete, and the proof depends on who wrote it.

Configure the proof with `readiness.mode`:

| Mode | What it proves | When to use it |
|---|---|---|
| `cameraSidecar` | The camera metadata sidecar `<image>.json` parses, and its `image.bytes` and `image.sha256` match the file exactly. | Any route fed by `camera-adapter`. Regulated routes require it. |
| `cameraStatus` | The camera's durable catalog reports the capture `SUCCEEDED`, and the record's declared size and digest verify against the file. | A camera route where the catalog, not the file, is the gate. |
| `marker` | A companion file `<path><suffix>` exists. | A writer you control that creates a marker last. |
| `stability` | Size and mtime have not moved for `quietSecs`. | A writer that offers no completion signal at all. |

The first three verify. `stability` infers, and inference is weaker: a slow or stalled writer looks
exactly like a finished one. That is why `stability` is refused on a camera-bound route. When a
route has proof available, settling for a guess is a downgrade, so the component rejects the
configuration rather than accepting a quieter contract than the camera already offers.

`cameraSidecar` is the strongest of the four, and it costs nothing extra, because of how
`camera-adapter` installs a capture. It writes the sidecar, flushes it, and only then renames the
image into place under its final name; while the image is being written it carries a hidden
`.camera-adapter-<token>.image.partial` name that no route matches. An image visible under its
final name beside a matching sidecar is therefore a complete capture by construction, not by
timing. The sidecar also carries the capture provenance - the `captureId`, `cameraId`,
`correlationId`, and persistence timestamp - that the result message and the evidence sidecar
record.

## The camera hint arrives with proof

A camera-bound route also subscribes to
`ecv1/{device}/camera-adapter/{instance}/app/image/captured`. The announcement declares the image's
root-relative path, its exact byte count, and its SHA-256, which is proof enough on its own: a hint
whose declared size and digest match the file is admitted immediately, with no sidecar read and no
wait for the next walk.

The hint is not a queue. It shortens latency and nothing else, and a lost, duplicated, or
out-of-order hint changes only when a job starts, never whether it runs. A duplicate announces an
input the component has already announced, and the identity rules below collapse it into the same
job.

The announcement's `image.absolutePath` is never followed. It is the camera's view of its own
filesystem, and honoring it would let a message on the bus choose which file the component reads.
The path is always `image.relativePath` resolved under the route's configured root, and a path that
would leave the root is refused.

## Capture-status reconciliation is the authority

`camera-adapter` documents its own position: a consumer that must not miss an outcome polls
`sb/capture-status` rather than relying on the announcement. A camera-bound route does exactly
that, every `camera.reconcileCaptureStatusSecs`.

One sweep pages the camera's `SUCCEEDED` list and follows every `nextCursor` to the end. Stopping
at a page boundary would leave captures unreconciled until a later sweep happened to page past
them, which is a backlog that grows rather than drains. Each record is deduplicated by `captureId`
and verified against the file on disk before it counts, and the sweep records a watermark - the
newest terminal time it has absorbed - so a restart does not re-announce work it already did.

The camera retains results for a bounded window: `state.resultRetentionHours` of history or
`state.maxResultRecords` records, whichever binds first. Past that a lookup answers
`CAPTURE_NOT_FOUND`. That is an answer, not a fault: the catalog record is gone, the file and its
sidecar remain, and the route learns about the image from the walk instead.

## Trigger routes accept images on the bus

A trigger route subscribes to configured topic filters and accepts two body forms.

An **inline image** is an opaque binary body, or a structured body whose `image` field is bytes. The
EdgeCommons message envelope caps a binary body at 64 KiB in all four language libraries, so an
inline image is small by construction; `maxInlineBytes` can lower that ceiling but never raise it.
The bytes are hashed and written into processor-owned staging under a digest-derived name, and from
that point the job is an ordinary file job.

A **file reference** is `{"relativePath": ..., "sha256": ..., "bytes": n}`. The path resolves under
the route's `fileRoot` with containment enforced, and the declared size and digest are verified
against the file before the job is admitted. The verified file is then copied into staging, because
the producer still owns the original and may delete or rewrite it while the job waits for a GPU.

A body that is neither form is refused with a stable reason rather than guessed at.

When the request envelope names a `reply_to`, the correlation travels with the input, so the result
publisher answers the requester with the bounded result summary in addition to publishing the
normal outputs.

## What makes two discoveries the same job

A spool route announces each `(relative path, sha256)` pair once. The digest is part of the key on
purpose: the same path rewritten with different bytes is a different image and a new job, while the
same bytes rediscovered by a walk, a hint, and a reconciled record are one job announced once. The
set of announced pairs is primed from the job ledger at startup, so a restart does not rediscover
finished work.

Above that, `inferenceId` is derived from the logical key rather than minted, so a retry reuses it
and a new model digest produces a new inference. The preferred key is
`routeId + captureId + modelDigest`; without a durable capture identity it is
`routeId + normalizedSourceId + sourceSha256 + modelDigest`.

## What a source refuses to touch

An input root is a trust boundary. A source accepts only plain regular files inside the root it was
configured with, and it decides that from `lstat`, without opening the candidate, because opening
it is what a traversal or a device-file attack needs:

- Symlinks and Windows reparse points, including junctions, in a file position or a directory
  position. A symlinked directory is not descended.
- Any path that resolves outside the root. Resolution happens first and the containment test
  second, so a link in any path segment is followed before the check rather than after it.
- Directories, devices, FIFOs, and sockets.
- Wire-supplied paths that are absolute, drive-qualified, or contain a parent segment.

A refusal is reported once per path with a stable reason, not on every walk, so a permanently
unusable file does not become a repeating alarm.

The walk also skips what is metadata rather than an input: the camera's hidden in-progress names,
its `<image>.json` sidecars, this component's own `<image>.inference.json` evidence sidecars, and a
`marker` route's completion markers. Reading back an evidence sidecar would be an output feedback
loop.

## Digests come from the bytes, and are re-checked

Every admitted input carries a SHA-256 the component computed itself from the exact bytes on disk.
A digest that arrives in a message or a sidecar is a claim to check, never a value to record.

Hashing is bracketed by a `stat`: size and mtime are read before the hash and again after it, and a
file that moved in between is not admitted on that pass. Otherwise the digest would describe a file
that no longer exists, and every later verification - the cell's check against the staged file, the
evidence sidecar, the result message - would carry a digest for bytes nobody has.

## Staging is what the executor reads

An executor cell reads its input by path and expected digest, and it must be reading something no
one else can change.

A spool file already is that, because exactly one component may mutate a given spool and the
process-first topology gives that ownership to `image-processor`. A spool job therefore stages
nothing and the cell reads the file in place.

An inline or referenced trigger image is not, because the producer owns it. Those are copied into
staging under a name derived from the digest, written to a hidden temporary file, verified, and
moved into place atomically, so a reader never sees a partial staged file. Naming by digest is what
makes staging idempotent: the same bytes always land on the same path, so the same input admitted
twice, or a retry after a crash, reuses the file already there.

## Test tiers

Testing an inference component pulls in two directions. A test that proves the arithmetic wants a
model whose answer is known in advance; a test that proves the plumbing wants a model somebody
really exported. `DESIGN.md` §16.1 splits them into four tiers so neither compromises the other.

**Tier 1** builds its own ONNX graphs with `onnx.helper` and fixed weights, one per task family,
and its own images with Pillow. The expected answer is computed arithmetically rather than recorded
from a previous run, so a passing test means the preprocessing, the head decoding, the suppression,
and the decision rules are right, not merely unchanged. Nothing is downloaded and nothing binary is
committed, which is what lets the tier run on every pull request under the 90% coverage gate.

**Tier 2** runs real exports: MobileNetV2 and ResNet-50, YOLOX-Nano and YOLOX-S, SSD-MobileNetV1,
FCN-ResNet50, and a PatchCore built on VisA capsules. Real exports carry the conventions a synthetic
graph never does — YOLOX's BGR letterbox onto a grey canvas, SSD's one-based category ids and
already-suppressed boxes, an auxiliary segmentation head the family must not read. The answers are
compared to committed JSON goldens under tolerances rather than for equality, because the same graph
gives slightly different last digits on a different runtime build. Models and images are pinned by
URL and SHA-256 in `tests/assets.json` and cached under `tests/.cache/`; the licenses are permissive,
which is why YOLOv8 and YOLO11 (AGPL-3.0) and MVTec AD (CC BY-NC) are not in the corpus.

**Tier 3** measures residency and burst behavior on real GPUs, against a synthesized corpus of
bundles sized past what the card holds. Real models cannot supply it: the point is to overcommit the
device, and that takes more bundles at controlled sizes than any public collection offers.

**Tier 4** runs the whole system — camera-adapter, the processor, file-replicator, uns-bridge, and
edge-console — on procedurally rendered line-clearance scenes and on VisA, replayed through the real
camera path by the `sim` backend's playlist pattern.

Each tier answers a question the others cannot. Tier 1 says the arithmetic is right, tier 2 says
real models are read correctly, tier 3 says the device stays inside its memory, and tier 4 says the
components agree with each other.
