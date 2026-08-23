"""The tier-3 rig: real ledger, cache, supervisor, residency policy, and scheduler on one device.

Nothing here is a stand-in. The harness builds the same objects ``ImageProcessor`` builds -- a
SQLite ``Ledger``, the content-addressed ``BundleCache`` the corpus was staged into, a
``Supervisor`` that spawns a real executor cell holding a real CUDA context, the cost-aware
``ResidencyPolicy``, and the ``Scheduler`` -- and drives them with a spool writer that lays down
camera-format captures at rate. Discovery is the component's own ``SpoolSource`` in
``cameraSidecar`` readiness mode, so an arrival travels the path a camera-adapter capture travels:
sidecar first, then the image, then a walk, then a verified admission, then a lane.

Two things are added for measurement only, and neither changes a decision:

* every executor cell is wrapped so the wall clock of each ``LoadModel``, ``Unload``, ``Stats``,
  and ``Infer`` exchange is recorded at the process boundary; and
* the result callback finishes each job the way the application does -- commit, confirm, clean up
  -- so "every admitted job reaches a terminal state" is a claim the ledger can answer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from image_processor.bundles import BundleCache
from image_processor.engine.cell import CellError, ExecutorCell
from image_processor.engine.protocol import (
    CUDA_PROVIDER,
    LoadFailed,
    LoadModel,
    Loaded,
    Unload,
)
from image_processor.engine.residency import NvmlProbe, ResidencyPolicy
from image_processor.engine.scheduler import Scheduler
from image_processor.engine.supervisor import Supervisor
from image_processor.ledger.ledger import Ledger, OutboxRow
from image_processor.sources.spool import SpoolSource
from image_processor.types import (
    TERMINAL_STATES,
    CleanupIntent,
    CompletionAction,
    Job,
    JobState,
    ModelRef,
    derive_inference_id,
)
from tests.nvidia import REPO_ROOT

logger = logging.getLogger("tier3")

#: The bundle-manifest schema the cache validates against.
SCHEMA_PATH = REPO_ROOT / "schemas" / "model-bundle-manifest.schema.json"

#: The topic the harness records on its outbox rows. Nothing is published; the row exists so the
#: job takes the real PUBLISH_PENDING to PUBLISHED edge.
RESULT_TOPIC = "ecv1/tier3/image-processor/{route}/app/inference/result"

#: The Zipf exponent the skewed pattern draws routes with. At 1.2 the busiest route takes about a
#: fifth of the arrivals and the tail is still served, which is what a camera fleet with a few
#: high-rate lines looks like.
ZIPF_EXPONENT = 1.2

#: How long the burst pattern waits between synchronized waves.
BURST_INTERVAL_SECS = 20.0

#: How many routes one scheduled-prefetch segment covers.
PREFETCH_SEGMENT = 8

#: How long a scheduled prefetch waits for the queue to go quiet before warming the next segment.
PREFETCH_QUIET_SECS = 30.0

#: How long the drain waits with nothing moving before it reports the queue as stuck. It has to
#: outlast the slowest single load in the corpus, which is a 1.5 GiB bundle.
DRAIN_STALL_SECS = 120.0


@dataclass
class CellExchange:
    """One request and its reply across the executor boundary.

    Attributes:
        kind: The request class name.
        digest: The bundle digest the request names, or the empty string.
        ms: Wall-clock milliseconds the exchange took, measured in the parent.
        at: When it finished, on the monotonic clock.
        ok: Whether the reply was the successful one for that request.
        device_mib: Device memory the reply reported, for a load or an unload.
        detail: The failure code, when there was one.
        pressure: Whether the runtime reported device-memory exhaustion.
        cold: For a load, whether this digest had never been resident on this cell before.
        issued_by: ``scheduler`` for an exchange a scheduling pass made, ``preload`` for one a
            scheduled prefetch made. The two are kept apart because a prefetch's load and release
            are not residency decisions, and counting them as reloads and evictions would inflate
            both.
    """

    kind: str
    digest: str
    ms: float
    at: float
    ok: bool = True
    device_mib: int = 0
    detail: str = ""
    pressure: bool = False
    cold: bool = False
    issued_by: str = "scheduler"


@dataclass
class JobRecord:
    """What one job cost, end to end.

    Attributes:
        inference_id: The job identity.
        route_id: The route that discovered it.
        digest: The bundle generation it was pinned to.
        status: SUCCEEDED or FAILED.
        queue_ms: How long it waited in its lane before dispatch.
        inference_ms: The session run.
        total_ms: Decode, preprocess, inference, and postprocess in the cell.
        e2e_ms: From the image becoming visible in the spool to the result reaching the callback.
        providers: The session assignment the cell actually got, comma separated.
        gpu_device: The device ordinal the cell ran on.
        error: The failure, when there was one.
    """

    inference_id: str
    route_id: str
    digest: str
    status: str
    queue_ms: float
    inference_ms: float
    total_ms: float
    e2e_ms: float
    providers: str = ""
    gpu_device: str = ""
    error: str = ""


@dataclass(frozen=True)
class Route:
    """One route of the rig: a spool directory bound to one bundle generation.

    Attributes:
        id: The route id.
        model_id: The bundle's model id.
        digest: The bundle digest the route is pinned to.
        root: The spool directory the writer lays captures into.
        priority: The route priority the scheduler weights queue age by.
        images: The corpus images this route replays, in order.
        tier_mib: The bundle's target model.onnx size.
        base: The architecture the bundle was grown from.
        family: The task family.
    """

    id: str
    model_id: str
    digest: str
    root: Path
    priority: int
    images: Tuple[Path, ...]
    tier_mib: int
    base: str
    family: str

    @property
    def model_ref(self) -> ModelRef:
        """Return the model reference a job of this route pins."""
        return ModelRef(id=self.model_id, version="1.0.0", digest=self.digest)


class MeasuringCell:
    """An executor-cell handle that records the wall clock of every exchange.

    The wrapper delegates everything it does not time, so the supervisor and the scheduler see the
    handle they expect. It makes no decision: it observes a boundary the parent already crosses.

    Args:
        inner: The real executor cell.
        sink: Called with one CellExchange per completed exchange.
    """

    def __init__(self, inner: ExecutorCell, sink: Callable[[CellExchange], None]) -> None:
        """Wrap one cell."""
        self._inner = inner
        self._sink = sink
        self._sent: Any = None
        self._sent_at: Optional[float] = None

    def __getattr__(self, name: str):
        """Delegate anything the wrapper does not define.

        Args:
            name: The attribute name.

        Returns:
            The inner cell's attribute.
        """
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        """Return a short identification for logs."""
        return f"<MeasuringCell {self._inner!r}>"

    def _record(self, message: Any, reply: Any, elapsed_ms: float) -> None:
        """Record one exchange.

        Args:
            message: The request that was sent.
            reply: The reply that came back.
            elapsed_ms: How long the exchange took.
        """
        ok, device_mib, detail = True, 0, ""
        pressure = bool(getattr(reply, "memory_pressure", False))
        if isinstance(reply, Loaded):
            device_mib = int(reply.device_mib or 0)
        elif isinstance(reply, LoadFailed):
            ok, detail = False, str(reply.code or "LOAD_FAILED")
        elif isinstance(message, Unload):
            device_mib = int(getattr(reply, "freed_mib", 0) or 0)
        elif getattr(reply, "status", None) == "FAILED":
            ok, detail = False, str(getattr(reply, "error", ""))[:200]
        self._sink(
            CellExchange(
                kind=type(message).__name__,
                digest=str(getattr(message, "digest", "") or ""),
                ms=elapsed_ms,
                at=time.monotonic(),
                ok=ok,
                device_mib=device_mib,
                detail=detail,
                pressure=pressure,
            )
        )

    def call(self, message, timeout_s=None):
        """Send one request, wait for its reply, and record the exchange.

        Args:
            message: The request.
            timeout_s: The deadline in seconds.

        Returns:
            The reply.
        """
        started = time.perf_counter()
        reply = self._inner.call(message, timeout_s)
        self._record(message, reply, (time.perf_counter() - started) * 1000.0)
        return reply

    def send(self, message) -> None:
        """Send one request without waiting for its reply.

        Args:
            message: The request.
        """
        self._inner.send(message)
        self._sent, self._sent_at = message, time.perf_counter()

    def receive(self, timeout_s=None):
        """Wait for the reply to the request in flight and record the exchange.

        Args:
            timeout_s: The deadline in seconds.

        Returns:
            The reply.
        """
        message, started = self._sent, self._sent_at
        self._sent, self._sent_at = None, None
        reply = self._inner.receive(timeout_s)
        if message is not None and started is not None:
            self._record(message, reply, (time.perf_counter() - started) * 1000.0)
        return reply


def write_capture(root: Path, camera_id: str, ordinal: int, image: bytes) -> Dict[str, Any]:
    """Lay one camera-format capture into a spool, sidecar first.

    The order is the point. camera-adapter installs the metadata sidecar before the image becomes
    visible, which is what makes cameraSidecar readiness sound: a visible image beside a matching
    sidecar is a finished capture (DESIGN.md section 4.1).

    Args:
        root: The route's spool root.
        camera_id: The camera the sidecar claims.
        ordinal: The capture ordinal, which names the file and seeds the identifiers.
        image: The encoded image bytes to install.

    Returns:
        The capture record: its ``captureId``, ``relativePath``, ``bytes``, ``sha256``, and the
        monotonic clock reading at the moment the image became visible.
    """
    import hashlib

    capture_id = f"{camera_id}-{ordinal:08d}"
    relative = f"2026/08/23/{capture_id}.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(image).hexdigest()
    second = ordinal % 60
    minute = (ordinal // 60) % 60
    stamp = f"2026-08-23T12:{minute:02d}:{second:02d}Z"
    body = {
        "schemaVersion": 1,
        "eventId": f"{capture_id}-evt",
        "captureId": capture_id,
        "cameraId": camera_id,
        "correlationId": f"00000000-0000-4000-8000-{ordinal:012d}",
        "trigger": {"type": "schedule", "scheduleId": "tier3", "intendedFireTime": stamp},
        "captureProfile": "tier3",
        "captureMode": "simulated",
        "timestamps": {
            "requestedAt": stamp,
            "persistedAt": f"2026-08-23T12:{minute:02d}:{second:02d}.031Z",
            "cameraFrameTimestampQuality": "adapter-receive",
        },
        "durationsMs": {"total": 31},
        "image": {
            "absolutePath": path.resolve().as_posix(),
            "relativePath": relative,
            "fileUri": path.resolve().as_uri(),
            "contentType": "image/jpeg",
            "encoding": "jpeg",
            "bytes": len(image),
            "sha256": digest,
            "metadataSidecarRelativePath": f"{relative}.json",
        },
        "camera": {"backend": "sim", "vendor": "EdgeCommons", "model": "tier3", "serial": camera_id},
        "metadata": {},
    }
    sidecar = path.with_name(path.name + ".json")
    sidecar.write_bytes((json.dumps(body) + "\n").encode("utf-8"))
    path.write_bytes(image)
    return {
        "captureId": capture_id,
        "relativePath": relative,
        "bytes": len(image),
        "sha256": digest,
        "at": time.monotonic(),
    }


def route_config(route: Route) -> SimpleNamespace:
    """Build the configuration object a spool source reads for one route.

    The source protocol is structural (LLD section 7): any object carrying the DESIGN.md section 11
    field names works, so the rig states the same shape configuration would.

    Args:
        route: The route.

    Returns:
        The configuration object.
    """
    return SimpleNamespace(
        id=route.id,
        source=SimpleNamespace(
            root=str(route.root),
            include=("**/*.jpg", "**/*.jpeg", "**/*.png"),
            exclude=(),
            readiness=SimpleNamespace(mode="cameraSidecar", quietSecs=1.0, markerSuffix=""),
            camera=SimpleNamespace(
                component="camera-adapter",
                instance=route.id,
                subscribeAnnouncements=False,
                reconcileCaptureStatusSecs=0,
            ),
        ),
    )


class Tier3Harness:
    """The rig: real subsystems on one device, driven by a spool writer.

    Args:
        corpus_root: The corpus ``tools/synth_corpus.py`` built and staged.
        image_pools: The images each task family replays, keyed by family name.
        workdir: Scratch space for the ledger and the route spools.
        device: The CUDA device ordinal.
        route_count: How many corpus bundles to bind routes to, or None for all of them.
        budget_percent: ``gpu.residentMemoryBudgetPercent``.
        reserve_mib: ``gpu.reserveMiB``.
    """

    def __init__(
        self,
        corpus_root: Path,
        image_pools: Dict[str, Tuple[Path, ...]],
        workdir: Path,
        device: int = 0,
        route_count: Optional[int] = None,
        budget_percent: int = 80,
        reserve_mib: int = 2048,
    ) -> None:
        """Build every subsystem. Nothing runs until start()."""
        self.corpus_root = Path(corpus_root)
        self.index = json.loads((self.corpus_root / "corpus.json").read_text(encoding="utf-8"))
        self.workdir = Path(workdir)
        self.device = int(device)
        self.image_pools = {key: tuple(value) for key, value in image_pools.items()}

        # DESIGN.md section 11, with the tier-3 device budget the task fixes.
        self.runtime = SimpleNamespace(
            providers=[CUDA_PROVIDER],
            requiredProvider=CUDA_PROVIDER,
            allowCpuOnly=False,
            executorCellsPerGpu=1,
            loadConcurrencyPerGpu=1,
        )
        self.gpu = SimpleNamespace(
            devices=[str(self.device)],
            residentMemoryBudgetPercent=budget_percent,
            reserveMiB=reserve_mib,
        )
        self.scheduler_cfg = SimpleNamespace(
            maxBatchSize=1,
            maxBatchLatencyMs=20,
            hotTtlSecs=120,
            minResidencySecs=15,
            maxAttempts=5,
            retryBackoffSecs=2,
            maxRetryBackoffSecs=60,
        )

        self.routes = self._build_routes(route_count)
        self.by_digest = {route.digest: route for route in self.routes}

        self.exchanges: List[CellExchange] = []
        self.jobs: List[JobRecord] = []
        self.invalids: List[Tuple[str, str, str]] = []
        self.admission_failures: List[str] = []
        self.completion_failures: List[str] = []
        self.preload_refusals = 0
        self.preloads = 0
        self._lock = threading.Lock()
        self._arrivals: Dict[str, float] = {}
        self._ordinal = 0
        self._emitted = 0
        self._ever_loaded: set = set()
        self._counters_before: Dict[str, int] = {}
        self._recycles_before = 0
        self._admitted_before = 0
        self._peak_resident_mib = 0
        self._min_free_mib = None
        self._preloading = False
        self._probe = NvmlProbe()

        self.ledger = Ledger(self.workdir / "state.db", reserve_budget_bytes=64 * 2**20)
        self.cache = BundleCache(self.corpus_root / "cache", schema_path=SCHEMA_PATH)
        self.supervisor = Supervisor(
            runtime=self.runtime, gpu=self.gpu, cell_factory=self._cell_factory
        )
        self.policy = ResidencyPolicy(gpu=self.gpu, scheduler=self.scheduler_cfg)
        self.scheduler = Scheduler(
            self.ledger,
            self.supervisor,
            self.cache,
            self.policy,
            cfg=self.scheduler_cfg,
            on_result=self._on_result,
            route_priorities={route.id: route.priority for route in self.routes},
        )
        # discovery.rescanSecs and discovery.debounceMs as test-configs/config.json sets them.
        self.sources = [
            SpoolSource(route_config(route), self, debounce_secs=0.25, rescan_interval_secs=15.0)
            for route in self.routes
        ]

    def _cell_factory(self, cell_id: str, device: Optional[str], providers) -> MeasuringCell:
        """Build one real executor cell, wrapped so its exchanges are timed.

        Args:
            cell_id: The cell identity.
            device: The device ordinal as a string.
            providers: The providers the cell requests.

        Returns:
            The wrapped handle, not yet started.
        """
        inner = ExecutorCell(cell_id, device, providers, call_timeout_s=900.0)
        return MeasuringCell(inner, self._record_exchange)

    def _record_exchange(self, exchange: CellExchange) -> None:
        """Collect one executor exchange, marking a first load of a digest as cold.

        Cold and reload are told apart here rather than from the residency policy, because the
        policy forgets a digest when it evicts one and a reload would then look cold again.

        Args:
            exchange: What crossed the boundary.
        """
        with self._lock:
            exchange.issued_by = "preload" if self._preloading else "scheduler"
            if exchange.kind == "LoadModel" and exchange.ok:
                exchange.cold = exchange.digest not in self._ever_loaded
                self._ever_loaded.add(exchange.digest)
            self.exchanges.append(exchange)

    def _build_routes(self, route_count: Optional[int]) -> List[Route]:
        """Bind one route to each corpus bundle.

        Priorities are spread across four bands so that route priority is a live term in the
        scheduler's lane ranking rather than a constant.

        Args:
            route_count: How many bundles to use, or None for all of them.

        Returns:
            The routes, in corpus order.
        """
        bundles = self.index["bundles"]
        if route_count is not None:
            bundles = bundles[: int(route_count)]
        routes = []
        for position, entry in enumerate(bundles):
            family = entry["family"]
            pool = self.image_pools.get(family) or ()
            if not pool:
                raise RuntimeError(f"no replay images for the {family} family")
            routes.append(
                Route(
                    id=f"cam-{position:03d}",
                    model_id=entry["modelId"],
                    digest=entry["digest"],
                    root=self.workdir / "spool" / f"cam-{position:03d}",
                    priority=100 + (position % 4) * 25,
                    images=tuple(pool),
                    tier_mib=int(entry["tierMiB"]),
                    base=entry["base"],
                    family=family,
                )
            )
        return routes

    def start(self) -> "Tier3Harness":
        """Bring the executor, the scheduler, and every spool source up.

        Returns:
            This harness.
        """
        for route in self.routes:
            route.root.mkdir(parents=True, exist_ok=True)
        self.supervisor.start()
        self.scheduler.start()
        for source in self.sources:
            source.start()
        return self

    def stop(self) -> None:
        """Take everything down in the order the application takes it down."""
        for source in self.sources:
            try:
                source.stop()
            except Exception:  # noqa: BLE001 - a source that will not stop must not hide the rest
                logger.exception("a spool source did not stop")
        self.scheduler.stop(60.0)
        self.supervisor.stop()
        self.ledger.close()

    # -- the source events the application implements ---------------------------------------

    def discovered(self, route_id: str, source, staged_path) -> None:
        """Admit one verified capture and hand it to the scheduler.

        This is ``ImageProcessor.discovered`` with the parts tier 3 does not measure left out: the
        identity is derived rather than generated, the ledger refuses a duplicate, and the job
        joins its lane at the route's priority.

        Args:
            route_id: The route that found it.
            source: The verified source identity.
            staged_path: The processor-owned copy, which a spool input never has.
        """
        route = next((entry for entry in self.routes if entry.id == route_id), None)
        if route is None:
            return
        model = route.model_ref
        inference_id = derive_inference_id(
            route_id, source.capture_id, source.sha256, source.relative_path, model.digest
        )
        job = Job(
            inference_id=inference_id,
            route_id=route_id,
            source=source,
            model=model,
            transform_version="1",
            state=JobState.READY,
            staged_path=str(route.root / source.relative_path),
        )
        try:
            admitted = self.ledger.admit(job, 64 * 1024)
        except Exception as exc:  # noqa: BLE001 - a refused admission is a finding, never a drop
            with self._lock:
                self.admission_failures.append(f"{route_id}: {exc}")
            return
        if not admitted:
            return
        with self._lock:
            arrival = self._arrivals.get(source.capture_id or "")
            if arrival is not None:
                self._arrivals[inference_id] = arrival
        self.scheduler.submit(job, route.priority)

    def invalid(self, route_id: str, relative_path: str, reason: str) -> None:
        """Record an input the source refused.

        Args:
            route_id: The route that owns it.
            relative_path: The path as configured.
            reason: The stable token the source produced.
        """
        with self._lock:
            self.invalids.append((route_id, relative_path, reason))

    # -- the result pipeline ----------------------------------------------------------------

    def _on_result(self, job, result) -> None:
        """Record what one job cost and take it to a terminal state.

        Args:
            job: The job as the ledger holds it.
            result: The inference result, successful or terminally failed.
        """
        now = time.monotonic()
        with self._lock:
            arrival = self._arrivals.pop(job.inference_id, None)
        timings = result.timings
        self.jobs.append(
            JobRecord(
                inference_id=job.inference_id,
                route_id=job.route_id,
                digest=job.model.digest,
                status=result.status,
                queue_ms=float(timings.queue_ms),
                inference_ms=float(timings.inference_ms),
                total_ms=float(timings.total_ms),
                e2e_ms=(now - arrival) * 1000.0 if arrival is not None else float("nan"),
                providers=",".join(result.providers or ()),
                gpu_device=str(result.gpu_device or ""),
                error=str(result.error or "")[:200],
            )
        )
        try:
            if result.status == "SUCCEEDED":
                self._complete(job)
            else:
                self._retain(job)
        except Exception as exc:  # noqa: BLE001 - a stuck job is a finding, not a crash
            with self._lock:
                self.completion_failures.append(f"{job.inference_id}: {type(exc).__name__}: {exc}")

    def _complete(self, job) -> None:
        """Take a successful job the whole way: commit, confirm, clean up.

        The path is the application's, edge for edge (DESIGN.md section 7): the result and one
        gating outbox row land in one transaction, the row is confirmed, the cleanup intent is
        persisted before the file is touched, and only then does the input go away.

        Args:
            job: The job in INFERENCING.
        """
        body = json.dumps(
            {"inferenceId": job.inference_id, "status": "SUCCEEDED", "route": job.route_id}
        ).encode("utf-8")
        self.ledger.commit_result(
            job.inference_id,
            body,
            None,
            [
                OutboxRow(
                    id=None,
                    inference_id=job.inference_id,
                    topic=RESULT_TOPIC.format(route=job.route_id),
                    encoded_bytes=body,
                    gating=True,
                )
            ],
        )
        for row in self.ledger.outbox_for(job.inference_id):
            self.ledger.mark_published(row.id)
        source_path = Path(job.staged_path)
        self.ledger.record_cleanup_intent(
            CleanupIntent(
                inference_id=job.inference_id,
                action=CompletionAction.DELETE,
                source_path=str(source_path),
                source_sha256=job.source.sha256,
                target_path=None,
                members=(f"{source_path.name}.json",),
            )
        )
        sidecar = source_path.with_name(source_path.name + ".json")
        source_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        self.ledger.complete_cleanup(job.inference_id, "DELETED")

    def _retain(self, job) -> None:
        """Take a terminally failed job to RETAINED_FAILED, leaving the input in place.

        A job blocked on configuration has no completing edge in the DESIGN.md section 7 diagram,
        so it is left where it is and reported.

        Args:
            job: The job as the ledger holds it.
        """
        current = self.ledger.get(job.inference_id)
        if current is None or current.state is not JobState.PROCESSING_EXHAUSTED:
            return
        self.ledger.record_cleanup_intent(
            CleanupIntent(
                inference_id=job.inference_id,
                action=CompletionAction.RETAIN,
                source_path=str(job.staged_path),
                source_sha256=job.source.sha256,
                target_path=None,
                members=(),
            )
        )
        self.ledger.complete_cleanup(job.inference_id, "RETAINED")

    # -- arrival patterns -------------------------------------------------------------------

    def _emit(self, route: Route) -> None:
        """Lay one capture into one route's spool and nudge its source.

        Args:
            route: The route to write for.
        """
        with self._lock:
            ordinal = self._ordinal
            self._ordinal += 1
        image = route.images[ordinal % len(route.images)].read_bytes()
        record = write_capture(route.root, route.id, ordinal, image)
        with self._lock:
            self._arrivals[record["captureId"]] = record["at"]
            self._emitted += 1
        self.sources[self.routes.index(route)].nudge()

    def _pace(self, deadline: float) -> None:
        """Sleep until one arrival's slot is up.

        Args:
            deadline: The monotonic clock reading the next arrival is due at.
        """
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _drive_uniform(self, arrivals: int, rate: float) -> int:
        """Write one capture per slot, round-robin over every route.

        Args:
            arrivals: How many captures to write.
            rate: Captures per second.

        Returns:
            How many were written.
        """
        interval = 1.0 / float(rate)
        start = time.monotonic()
        for index in range(arrivals):
            self._pace(start + index * interval)
            self._emit(self.routes[index % len(self.routes)])
        return arrivals

    def _drive_zipf(self, arrivals: int, rate: float) -> int:
        """Write at a fixed rate, choosing routes with a Zipf skew.

        Args:
            arrivals: How many captures to write.
            rate: Captures per second.

        Returns:
            How many were written.
        """
        count = len(self.routes)
        ranks = np.arange(1, count + 1, dtype=float)
        weights = ranks ** (-ZIPF_EXPONENT)
        weights = weights / weights.sum()
        rng = np.random.default_rng(int(self.index["seed"]))
        interval = 1.0 / float(rate)
        start = time.monotonic()
        for index in range(arrivals):
            self._pace(start + index * interval)
            self._emit(self.routes[int(rng.choice(count, p=weights))])
        return arrivals

    def _drive_burst(self, arrivals: int, rate: float) -> int:
        """Fire every route at once, in waves, with a quiet gap between them.

        Args:
            arrivals: The arrival budget, which rounds down to whole waves.
            rate: Ignored: a synchronized burst is by definition unpaced.

        Returns:
            How many captures were written.
        """
        count = len(self.routes)
        waves = max(1, arrivals // count)
        written = 0
        for wave in range(waves):
            if wave:
                time.sleep(BURST_INTERVAL_SECS)
            for route in self.routes:
                self._emit(route)
                written += 1
        return written

    def _drive_prefetch(self, arrivals: int, rate: float) -> int:
        """Warm the next segment of routes before their arrivals reach the spool.

        This is what ``preload-model`` is for (DESIGN.md section 13): the operator knows which
        generations the next shift needs and stages and warms them while the queue is quiet. The
        harness waits for the queue to go quiet, warms the segment the way ``ArtifactManager.warm``
        does -- load with golden warmup, then release the session -- and only then writes.

        Args:
            arrivals: How many captures to write.
            rate: Captures per second inside a segment.

        Returns:
            How many were written.
        """
        interval = 1.0 / float(rate)
        written = 0
        for offset in range(0, len(self.routes), PREFETCH_SEGMENT):
            segment = self.routes[offset : offset + PREFETCH_SEGMENT]
            if written >= arrivals:
                break
            self._quiet(PREFETCH_QUIET_SECS)
            self.preload([route.digest for route in segment])
            budget = min(arrivals - written, len(segment) * 3)
            start = time.monotonic()
            for index in range(budget):
                self._pace(start + index * interval)
                self._emit(segment[index % len(segment)])
                written += 1
        return written

    def _quiet(self, timeout: float) -> None:
        """Wait for the scheduler to have nothing queued, or give up.

        Args:
            timeout: How long to wait, in seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.scheduler.queued() == 0:
                return
            time.sleep(0.25)

    def preload(self, digests) -> int:
        """Stage and warm generations ahead of their work, as ``preload-model`` does.

        The component's ``preload-model`` verb runs ``ArtifactManager.warm``: one ``LoadModel``
        with golden warmup on an alive cell, then an ``Unload`` that gives the session back, so
        residency stays the scheduler's alone (DESIGN.md section 9 step 5). A cell that already has
        a request in flight refuses the second one, which is counted rather than retried: that
        refusal is the measurement.

        Args:
            digests: The bundle digests to warm.

        Returns:
            How many were warmed.
        """
        self._preloading = True
        try:
            return self._preload(digests)
        finally:
            self._preloading = False

    def _preload(self, digests) -> int:
        """Warm each generation on an alive cell and give the session straight back.

        Args:
            digests: The bundle digests to warm.

        Returns:
            How many were warmed.
        """
        warmed = 0
        for digest in digests:
            bundle = self.cache.get(digest)
            if bundle is None:
                continue
            message = LoadModel(
                digest=digest,
                bundle_root=str(bundle.root),
                providers=tuple(self.runtime.providers),
                provider_policy=bundle.manifest.provider_policy,
                providers_permitted=tuple(bundle.manifest.providers_permitted),
                warmup=True,
                required_provider=self.runtime.requiredProvider,
                allow_cpu_only=bool(self.runtime.allowCpuOnly),
            )
            for cell in [entry for entry in self.supervisor.cells() if entry.is_alive()]:
                try:
                    reply = self.supervisor.call(cell, message, 900.0)
                except CellError as exc:
                    with self._lock:
                        self.preload_refusals += 1
                    logger.info("preload of %s was refused: %s", digest[:19], exc)
                    break
                except Exception:  # noqa: BLE001 - a dead cell is one cell, not the answer
                    logger.exception("preload of %s failed on %s", digest[:19], cell.cell_id)
                    continue
                if isinstance(reply, Loaded):
                    try:
                        self.supervisor.call(cell, Unload(digest), 900.0)
                    except Exception:  # noqa: BLE001 - the warmup answered; the unload is hygiene
                        logger.exception("preload could not release %s", digest[:19])
                    warmed += 1
                    with self._lock:
                        self.preloads += 1
                break
        return warmed

    # -- running one pattern, and what it measured ------------------------------------------

    def _reset(self) -> None:
        """Start a fresh measurement window."""
        with self._lock:
            self.exchanges.clear()
            self.jobs.clear()
            self.invalids.clear()
            self.admission_failures.clear()
            self.completion_failures.clear()
            self.preload_refusals = 0
            self.preloads = 0
            self._emitted = 0
        self._counters_before = dict(self.scheduler.counters)
        self._recycles_before = int(self.supervisor.recycle_count)
        self._admitted_before = sum(self.ledger.counts_by_state().values())
        self._peak_resident_mib = 0
        self._min_free_mib = None

    def budget_mib(self) -> int:
        """Return the device memory the residency budget allows the scheduler to hold.

        This is the bound ``ResidencyPolicy.admit`` enforces on the residency map:
        ``residentMemoryBudgetPercent`` of the installed memory, less the CUDA context each cell
        holds, which is the cell's overhead rather than any model's and comes off the budget once
        (DESIGN.md section 10.2). ``reserveMiB`` is a separate bound, applied to the device's free
        memory rather than to the map, so it is not subtracted here.

        Returns:
            The budget in MiB. Before the first load no cell has a context yet, so the figure is
            the whole share.
        """
        reading = self._probe.snapshot(self.device)
        total = int(reading.total_mib or 0)
        share = total * int(self.gpu.residentMemoryBudgetPercent) // 100
        return max(share - self.context_mib(), 0)

    def context_mib(self) -> int:
        """Return the CUDA context the cells on this device hold, as they measured it.

        Returns:
            The total in MiB, or ``0`` before any cell has loaded a model.
        """
        return sum(
            int(cell.get("contextMib") or 0) for cell in self.scheduler.status()["cells"]
        )

    def outstanding(self) -> int:
        """Return how many jobs are not in a terminal state."""
        counts = self.ledger.counts_by_state()
        return sum(count for state, count in counts.items() if state not in TERMINAL_STATES)

    def _drain(self, timeout: float, stall_secs: float = DRAIN_STALL_SECS) -> bool:
        """Wait for every emitted capture to be admitted and every job to finish.

        The wait ends early when nothing is moving: a job in ``BLOCKED_CONFIGURATION`` has no
        outgoing edge in the DESIGN.md section 7 diagram, so a run that produces one would
        otherwise sit here until the whole budget is gone. A stall is reported as a failure to
        drain, which is what the gate asserts on.

        Args:
            timeout: How long to wait, in seconds.
            stall_secs: How long nothing may move before the wait gives up.

        Returns:
            ``True`` when the queue drained inside the budget.
        """
        deadline = time.monotonic() + timeout
        last_change = time.monotonic()
        previous = None
        while time.monotonic() < deadline:
            self.sample_device()
            admitted = sum(self.ledger.counts_by_state().values()) - self._admitted_before
            outstanding = self.outstanding()
            if admitted >= self._emitted and outstanding == 0:
                return True
            progress = (admitted, outstanding, len(self.jobs), self.scheduler.queued())
            if progress != previous:
                previous, last_change = progress, time.monotonic()
            elif time.monotonic() - last_change > stall_secs:
                logger.error(
                    "the queue stopped moving with %d job(s) not terminal: %s",
                    outstanding,
                    {state.value: count for state, count in self.ledger.counts_by_state().items()},
                )
                return False
            time.sleep(0.5)
        return False

    def sample_device(self) -> None:
        """Take one reading of what the scheduler holds resident and what the device has left.

        Both are peaks the gates are judged against: the residency map must stay inside the budget
        DESIGN.md section 10.2 sets, and the device must never run out.
        """
        resident = 0
        for cell in self.scheduler.status()["cells"]:
            resident += sum(int(value) for value in cell["residentMib"].values())
        self._peak_resident_mib = max(self._peak_resident_mib, resident)
        reading = self._probe.snapshot(self.device)
        if reading.known:
            free = int(reading.free_mib)
            self._min_free_mib = free if self._min_free_mib is None else min(self._min_free_mib, free)

    def run_pattern(
        self, name: str, arrivals: int, rate: float, drain_timeout: float = 900.0
    ) -> Dict[str, Any]:
        """Drive one arrival pattern end to end and return what it measured.

        Args:
            name: One of ``uniform``, ``zipf``, ``burst``, ``prefetch``.
            arrivals: The arrival budget.
            rate: Arrivals per second for the paced patterns.
            drain_timeout: How long to wait for the queue to finish after the last arrival.

        Returns:
            The measurement record for this pattern.

        Raises:
            ValueError: The pattern is not one of the four.
        """
        drivers = {
            "uniform": self._drive_uniform,
            "zipf": self._drive_zipf,
            "burst": self._drive_burst,
            "prefetch": self._drive_prefetch,
        }
        if name not in drivers:
            raise ValueError(f"unknown arrival pattern: {name!r}")
        self._reset()
        started = time.monotonic()
        written = drivers[name](int(arrivals), float(rate))
        arrival_secs = time.monotonic() - started
        drained = self._drain(drain_timeout)
        return self._summary(name, written, arrival_secs, time.monotonic() - started, drained)

    def _summary(
        self, name: str, written: int, arrival_secs: float, wall_secs: float, drained: bool
    ) -> Dict[str, Any]:
        """Reduce one pattern's window to the record the results file keeps.

        Args:
            name: The pattern.
            written: How many captures were laid down.
            arrival_secs: How long the arrivals took to write.
            wall_secs: How long the whole window took, arrivals plus drain.
            drained: Whether the queue emptied inside the drain budget.

        Returns:
            The record.
        """
        with self._lock:
            exchanges = list(self.exchanges)
            jobs = list(self.jobs)
            invalids = list(self.invalids)
            admission_failures = list(self.admission_failures)
            completion_failures = list(self.completion_failures)
            preloads, refusals = self.preloads, self.preload_refusals
        scheduled = [entry for entry in exchanges if entry.issued_by == "scheduler"]
        warmed = [entry for entry in exchanges if entry.issued_by == "preload"]
        loads = [entry for entry in scheduled if entry.kind == "LoadModel" and entry.ok]
        cold = [entry for entry in loads if entry.cold]
        reload = [entry for entry in loads if not entry.cold]
        unloads = [entry for entry in scheduled if entry.kind == "Unload"]
        failed_loads = [entry for entry in exchanges if entry.kind == "LoadModel" and not entry.ok]
        preload_loads = [entry for entry in warmed if entry.kind == "LoadModel" and entry.ok]
        counters = {
            key: int(value) - int(self._counters_before.get(key, 0))
            for key, value in self.scheduler.counters.items()
        }
        succeeded = [entry for entry in jobs if entry.status == "SUCCEEDED"]
        states = {state.value: count for state, count in self.ledger.counts_by_state().items()}
        return {
            "pattern": name,
            "arrivals": written,
            "arrivalSecs": round(arrival_secs, 1),
            "wallSecs": round(wall_secs, 1),
            "drained": drained,
            "jobs": {
                "results": len(jobs),
                "succeeded": len(succeeded),
                "failed": len(jobs) - len(succeeded),
                "invalidInputs": len(invalids),
                "admissionFailures": admission_failures,
                "completionFailures": completion_failures,
            },
            "latencyMs": {
                "queue": percentiles([entry.queue_ms for entry in succeeded]),
                "inference": percentiles([entry.inference_ms for entry in succeeded]),
                "cellTotal": percentiles([entry.total_ms for entry in succeeded]),
                "endToEnd": percentiles(
                    [entry.e2e_ms for entry in succeeded if entry.e2e_ms == entry.e2e_ms]
                ),
            },
            "loads": {
                "cold": len(cold),
                "coldMs": percentiles([entry.ms for entry in cold]),
                "coldDeviceMiB": percentiles([float(entry.device_mib) for entry in cold]),
                "reload": len(reload),
                "reloadMs": percentiles([entry.ms for entry in reload]),
                "failed": len(failed_loads),
                "failureCodes": sorted({entry.detail for entry in failed_loads}),
            },
            "evictions": {
                "unloads": len(unloads),
                "unloadMs": percentiles([entry.ms for entry in unloads]),
                "freedMiB": percentiles([float(entry.device_mib) for entry in unloads]),
            },
            "oom": {
                "loadPressure": sum(1 for entry in exchanges if entry.pressure),
                "jobErrors": sorted(
                    {entry.error for entry in jobs if entry.status != "SUCCEEDED" and entry.error}
                ),
            },
            "providers": sorted({entry.providers for entry in succeeded if entry.providers}),
            "gpuDevices": sorted({entry.gpu_device for entry in succeeded if entry.gpu_device}),
            "prefetch": {
                "warmed": preloads,
                "refusedByBusyCell": refusals,
                "warmMs": percentiles([entry.ms for entry in preload_loads]),
            },
            "device": {
                "peakResidentMiB": self._peak_resident_mib,
                "minFreeMiB": self._min_free_mib,
                "budgetMiB": self.budget_mib(),
                "contextMiB": self.context_mib(),
                "reserveMiB": int(self.gpu.reserveMiB),
            },
            "recycles": int(self.supervisor.recycle_count) - self._recycles_before,
            "schedulerCounters": counters,
            "schedulerStatus": scheduler_summary(self.scheduler.status()),
            "ledgerStates": states,
        }


def scheduler_summary(status: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a scheduler status to what a committed results file keeps.

    The lane list is one entry per model generation, so a forty-bundle corpus carries forty of
    them and almost all are empty by the time a pattern ends. Only the lanes that still hold work
    or are blocked say anything, so only those are kept.

    Args:
        status: What ``Scheduler.status`` returned.

    Returns:
        The reduced status.
    """
    lanes = [
        lane
        for lane in status.get("lanes", [])
        if lane.get("queued") or lane.get("blockedReason") or lane.get("blockedUntilMs")
    ]
    return {
        "paused": status.get("paused"),
        "queued": status.get("queued"),
        "retryWaiting": status.get("retryWaiting"),
        "recycleCount": status.get("recycleCount"),
        "lanes": status.get("lanes", []) and len(status["lanes"]),
        "lanesWithWorkOrBlocked": lanes,
        "cells": status.get("cells", []),
        "counters": status.get("counters", {}),
    }


def percentiles(values) -> Dict[str, float]:
    """Reduce a sample to the shape the results file records.

    Args:
        values: The observations, in any order.

    Returns:
        ``count``, ``p50``, ``p95``, ``p99``, ``min``, ``max``, and ``mean``, rounded to two
        decimals. An empty sample reports a count of zero and nothing else.
    """
    sample = [float(value) for value in values if value == value]
    if not sample:
        return {"count": 0}
    array = np.asarray(sample, dtype=float)
    return {
        "count": len(sample),
        "p50": round(float(np.percentile(array, 50)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "p99": round(float(np.percentile(array, 99)), 2),
        "min": round(float(array.min()), 2),
        "max": round(float(array.max()), 2),
        "mean": round(float(array.mean()), 2),
    }
