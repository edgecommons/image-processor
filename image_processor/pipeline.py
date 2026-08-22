"""The pipeline: what a *processor* is.

A processor **subscribes**, **transforms**, and **forwards**. That is the whole archetype, and it
lives in three types:

* :class:`ProcMsg` -- the unit that flows through the pipeline: a message plus the topic it arrived
  on.
* :class:`Processor` -- one stage. It takes a message and returns **zero or more** messages, so a
  stage can filter (return nothing), map (return one), or fan out (return several).
* :class:`Pipeline` -- an ordered chain of stages. The output of each stage is the input of the
  next.

Why stages return a list and not an ``Optional[ProcMsg]``
--------------------------------------------------------
A filter drops, a projection maps, an aggregator emits on a timer rather than on arrival. ``0..N``
covers all three without a special case, and it is what lets :meth:`Processor.on_tick` exist: a
*stateful* stage (a window, a debounce, a batch) accumulates in :meth:`Processor.process` and emits
in :meth:`Processor.on_tick`, so time-driven output is not a different mechanism from data-driven
output.

One thread per route, so state needs no lock
--------------------------------------------
Each route owns its :class:`Pipeline` in a single worker thread. That is deliberate: per-key state
inside a stage is a plain attribute with no ``Lock`` anywhere, which is what makes a stateful stage
cheap to write correctly.

**This module deliberately does not import ``edgecommons``.** It is the payload-agnostic core of the
component -- pure logic over a duck-typed message (``msg.body``, ``msg.header``,
``msg.get_identity()``) -- so it can be unit-tested on its own, with no broker, no transport and no
library import. The library-facing wiring lives in ``image_processor/ImageProcessor.py``.
"""
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ProcMsg:
    """A message in flight, and the topic it arrived on.

    The topic is carried because a stage may want to route on it, and because the dispatcher needs
    it to decide where the result goes.
    """

    __slots__ = ("topic", "msg")

    def __init__(self, topic: str, msg: Any):
        self.topic = topic
        self.msg = msg

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ProcMsg(topic={self.topic!r}, msg={self.msg!r})"


class Processor(ABC):
    """One stage of the pipeline. **This is the class you implement.**"""

    @abstractmethod
    def process(self, m: ProcMsg) -> List[ProcMsg]:
        """Handle one inbound message. Return what should continue downstream (0..N messages)."""

    def on_tick(self, now_ms: int) -> List[ProcMsg]:
        """Called periodically, for stages that emit on time rather than on arrival (a window, a
        batch, a debounce). The default is to emit nothing -- a stateless stage ignores time."""
        return []


class Pipeline:
    """An ordered chain of stages."""

    def __init__(self, stages: List[Processor]):
        self._stages = list(stages)

    def run(self, inputs: List[ProcMsg], now_ms: Optional[int] = None) -> List[ProcMsg]:
        """Run a batch through every stage in order.

        When ``now_ms`` is not ``None``, each stage additionally gets an :meth:`Processor.on_tick`
        after its data pass, and whatever it emits joins the batch flowing downstream -- so a window
        closing in stage 1 is still projected by stage 2 on the same pass, rather than waiting for
        the next message to shake it loose.
        """
        carried = list(inputs)
        for stage in self._stages:
            nxt: List[ProcMsg] = []
            for m in carried:
                nxt.extend(stage.process(m))
            if now_ms is not None:
                nxt.extend(stage.on_tick(now_ms))
            carried = nxt
        return carried


# --- Demo stages ---------------------------------------------------------------------------------
#
# Two stages, enough to show both halves of the abstraction. Replace them with your own; nothing
# below is required by the library.


class FieldEquals(Processor):
    """Drops any message whose dotted body path does not equal an expected value.

    A filter is the simplest useful stage: it returns nothing, and the message stops there.
    """

    def __init__(self, path: str, value: Any):
        self.path = path
        self.value = value

    def process(self, m: ProcMsg) -> List[ProcMsg]:
        found = pluck(m.msg.body, self.path)
        return [m] if found is not None and found == self.value else []


class CountPerTick(Processor):
    """Counts messages and emits a rollup on each tick.

    This is the stateful half of the abstraction: it accumulates in :meth:`process` (emitting
    nothing) and produces its output in :meth:`on_tick`. Windows, batches and debounces are all this
    shape.
    """

    def __init__(self):
        self.seen = 0
        self.last: Optional[ProcMsg] = None

    def process(self, m: ProcMsg) -> List[ProcMsg]:
        self.seen += 1
        self.last = m
        return []  # nothing goes downstream on arrival -- see on_tick

    def on_tick(self, now_ms: int) -> List[ProcMsg]:
        m, n = self.last, self.seen
        self.last, self.seen = None, 0
        if m is None or n == 0:
            return []
        # A shallow copy of the carrier: the accumulated message is mutated into the rollup, and the
        # original object may still be referenced by a stage upstream.
        out = ProcMsg(m.topic, copy.copy(m.msg))
        out.msg.body = {"count": n, "last": m.msg.body}
        return [out]


#: The stage registry: the config key -> how to build it. Add yours here as you write it, and add
#: the matching property to `config.schema.json` -- the two are one contract.
_STAGES = {
    "fieldEquals": lambda a: FieldEquals(_require(a, "path", str), _require_present(a, "value")),
    "countPerTick": lambda a: CountPerTick(),
}


def build_stage(cfg: Dict[str, Any]) -> Processor:
    """Build one stage from its single-key config object (``{"fieldEquals": {...}}``).

    :raises ValueError: on a malformed stage object or an unknown stage name -- a typo'd stage is a
        mistake, not a no-op.
    """
    if not isinstance(cfg, dict) or len(cfg) != 1:
        raise ValueError(f"a stage is a single-key object naming the stage, got: {cfg!r}")
    name, args = next(iter(cfg.items()))
    if name not in _STAGES:
        raise ValueError(f"unknown stage `{name}`; known stages: {sorted(_STAGES)}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f"stage `{name}` takes an object of arguments, got: {args!r}")
    return _STAGES[name](args)


def pluck(body: Any, path: str) -> Any:
    """Resolve a dotted path (``signal.id``) inside a JSON body. ``None`` when it does not resolve."""
    cur = body
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def is_self_echo(msg: Any, my_path: str, my_component: str) -> bool:
    """Would consuming this message mean consuming our own output?

    **The guard that is not optional.** A processor that publishes onto a class it also subscribes to
    will consume its own output, reprocess it, republish it, and saturate the device. An MQTT broker
    redelivers our own publishes to our own wildcard subscription as a matter of course, so this is
    the common case, not a corner case.
    """
    identity = msg.get_identity()
    if identity is None:
        return False
    return identity.path == my_path and identity.component == my_component


# --- Route configuration -------------------------------------------------------------------------

#: The queue bound, when config names none. Bounded on purpose: an unbounded queue does not remove
#: backpressure, it relocates the failure to the heap -- and by the time you notice, you have lost
#: the ability to report it.
DEFAULT_MAX_QUEUE = 256
#: How often stateful stages are ticked, when config names no cadence.
DEFAULT_TICK_MS = 10_000

_TARGETS = ("local", "northbound")
_ROUTE_KEYS = {"id", "subscribe", "publishTopic", "target", "pipeline", "maxQueue", "tickMs"}


class RouteConfig:
    """One route == one entry of ``component.instances[]``.

    Routes are independent -- one thread each -- so a slow route cannot stall another, and per-key
    state inside a stage needs no lock.
    """

    __slots__ = ("id", "subscribe", "publish_topic", "target", "pipeline", "max_queue", "tick_ms")

    def __init__(self, id: str, subscribe: List[str], publish_topic: str, target: str,
                 pipeline: List[Dict[str, Any]], max_queue: int, tick_ms: int):
        self.id = id
        self.subscribe = subscribe
        self.publish_topic = publish_topic
        self.target = target
        self.pipeline = pipeline
        self.max_queue = max_queue
        self.tick_ms = tick_ms

    def build_pipeline(self) -> Pipeline:
        """Materialize this route's stages. An empty pipeline is a pass-through republisher."""
        return Pipeline([build_stage(s) for s in self.pipeline])


def parse_route(inst: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> RouteConfig:
    """Parse one ``component.instances[]`` entry, applying ``component.global.defaults``.

    Unknown keys are **rejected, not ignored** -- a typo'd route key is a mistake, and a config knob
    that silently does nothing is the worst kind of bug to find in the field.

    :raises ValueError: on a missing/ill-typed/unknown key, or a stage that does not build
    """
    defaults = defaults or {}
    if not isinstance(inst, dict):
        raise ValueError(f"an instance must be an object, got: {inst!r}")

    unknown = set(inst) - _ROUTE_KEYS
    if unknown:
        raise ValueError(f"unknown route key(s): {sorted(unknown)}")

    route_id = _require(inst, "id", str)
    publish_topic = _require(inst, "publishTopic", str)

    subscribe = inst.get("subscribe", [])
    if not isinstance(subscribe, list) or any(not isinstance(s, str) or not s for s in subscribe):
        raise ValueError("`subscribe` must be a list of non-empty topic filters")

    target = inst.get("target", "local")
    if target not in _TARGETS:
        raise ValueError(f"`target` must be one of {list(_TARGETS)}, got {target!r}")

    stages = inst.get("pipeline", [])
    if not isinstance(stages, list):
        raise ValueError("`pipeline` must be a list of stages")
    for s in stages:
        build_stage(s)  # fail here, at config time -- not on the first message

    max_queue = _positive_int(inst, "maxQueue", defaults.get("maxQueue", DEFAULT_MAX_QUEUE))
    tick_ms = _positive_int(inst, "tickMs", defaults.get("tickMs", DEFAULT_TICK_MS))

    return RouteConfig(route_id, subscribe, publish_topic, target, stages, max_queue, tick_ms)


def _require(cfg: Dict[str, Any], key: str, typ: type) -> Any:
    value = cfg.get(key)
    if not isinstance(value, typ) or (typ is str and not value):
        raise ValueError(f"`{key}` is required and must be a non-empty {typ.__name__}")
    return value


def _require_present(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"`{key}` is required")
    return cfg[key]


def _positive_int(cfg: Dict[str, Any], key: str, fallback: int) -> int:
    value = cfg.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{key}` must be a positive integer, got {value!r}")
    return value
