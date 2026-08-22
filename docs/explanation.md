# Explanation — How the processor archetype works, and why

*This documents the generated scaffold; rewrite it as you build the component out.*

This page is the mental model for the **processor** archetype. For exact options see
[reference/](reference/); for tasks, the [how-to guides](how-to-guides.md).

## What a processor is

A processor **subscribes**, **transforms**, and **forwards**. That is the whole archetype, and it
lives in three types (`image_processor/pipeline.py`):

- `ProcMsg` — the unit flowing through the pipeline: a message plus the topic it arrived on.
- `Processor` — one stage. It takes a message and returns **zero or more** messages, so a stage can
  filter (return nothing), map (return one), or fan out (return several).
- `Pipeline` — an ordered chain of stages; each stage's output is the next stage's input.

Why `0..N` rather than `Optional[ProcMsg]`: a filter drops, a projection maps, an aggregator emits on
a timer rather than on arrival. `0..N` covers all three without a special case, and it is what lets
`Processor.on_tick` exist — a *stateful* stage (a window, a debounce, a batch) accumulates in
`process` and emits in `on_tick`, so time-driven output is not a different mechanism from
data-driven output. A tick flows through the rest of the pipeline on the same pass, so a window
closing in stage 1 is still projected by stage 2 without waiting for the next message.

## One thread per route, so state needs no lock

Each entry of `component.instances[]` is **one route**, and each route owns its `Pipeline` in a
single worker thread. That is deliberate: per-key state inside a stage is a plain attribute with no
`Lock` anywhere — which is what makes a stateful stage cheap to write correctly, and what keeps one
slow or stuck route from stalling another.

## Why `get_messaging()` and not `data()`

The mistake this archetype invites, and worth reading twice. The `data()` facade is for a component
that *produces* readings: it mints its own topic from a signal id and imposes the
`SouthboundSignalUpdate` body. A processor is **payload-agnostic** — it republishes what it was
handed, on a topic its route names in config. Routing that through `data()` would rewrite both the
topic and the body, which is exactly what a republisher must not do. So: raw `gg.get_messaging()`,
and topics from config, not minted in code.

## Two guards that are not optional

- **Self-echo.** A processor that publishes onto a class it also subscribes to will consume its own
  output, reprocess it, republish it, and saturate the device. `is_self_echo` drops anything carrying
  our own identity (`{path, component}`). `main.py` also asks the transport not to echo
  (`receive_own_messages(False)`), but only Greengrass IPC can honor that — an MQTT broker redelivers
  our own publishes to our own wildcard subscription regardless. The guard is what actually holds.
- **Identity restamp.** What we publish is *ours*, not the producer's. `_dispatch` rebuilds every
  outbound message through `gg.instance(route.id).new_message(...)`, which stamps this component's
  config-resolved identity with the route's instance token. Without it the fleet cannot tell who
  emitted a message — and the self-echo guard downstream cannot work either.

## The queue is bounded, and a full queue drops and *counts*

An unbounded queue does not remove backpressure; it relocates the failure to the heap, and by the
time you notice you have lost the ability to report it. `_handler` uses `put_nowait`, never `put`: a
full queue increments the `dropped` measure rather than blocking the transport's dispatch thread. A
processor that silently discards messages is worse than one that crashes.

## Instance connectivity — a processor reports none

`instance_connectivity()` is registered once and read from two places: the `state` keepalive pushes
it into `instances[]` on every tick, and the built-in `status` verb returns the same sample when
asked. A processor's routes are **subscriptions on a bus the library already reports on**, not links
to a device — so this scaffold reports an empty list, a real answer, not a missing one. The seam is
registered anyway so it's visible the day a stage of yours does own a connection (an enrichment
lookup against a database, say).

## UNS addressing

Topics follow `ecv1/{device}/{component}/{instance}/{class}[/channel]`, built and validated by the
library. A processor's `publishTopic` is named by config rather than minted in code — that is the
archetype — but everything the library publishes on your behalf (`state`, `metric`, `evt`) is minted
through `gg.uns()`, and the reserved classes (`state`/`metric`/`cfg`/`log`) are library-owned and
rejected on direct publish.
