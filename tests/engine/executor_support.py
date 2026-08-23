"""Stand-ins for the executor boundary, so the WP4b suites test decisions rather than hardware.

A :class:`FakeCell` answers the protocol without a subprocess, an ONNX Runtime session, or a GPU,
and every reply is scripted, which is how failures no CPU host can produce -- an out-of-memory, a
poisoned CUDA context, a cell that dies mid-job -- get exercised deterministically.
"""

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_processor.engine.cell import CellDead, CellError  # noqa: E402
from image_processor.engine.protocol import (  # noqa: E402
    CUDA_PROVIDER,
    CellStats,
    Infer,
    LoadModel,
    Loaded,
    Stats,
    Unload,
    Unloaded,
)
from image_processor.types import (  # noqa: E402
    Decision,
    Family,
    InferenceResult,
    Job,
    JobState,
    ModelRef,
    NormalizedOutput,
    Outcome,
    SourceIdentity,
    SourceKind,
    Timings,
)

#: A digest that looks like the real thing without being any particular bundle.
DIGEST_A = "sha256:" + "a" * 64

#: A second digest, for lane and eviction tests.
DIGEST_B = "sha256:" + "b" * 64

#: A third digest.
DIGEST_C = "sha256:" + "c" * 64


class ManualClock:
    """A millisecond clock a test moves by hand."""

    def __init__(self, start: int = 1_700_000_000_000) -> None:
        """Initialize the clock.

        Args:
            start: The starting wall clock in milliseconds.
        """
        self.now = start

    def __call__(self) -> int:
        """Return the current time in milliseconds."""
        return self.now

    def advance(self, ms: int) -> int:
        """Move the clock forward.

        Args:
            ms: Milliseconds to advance.

        Returns:
            The new time.
        """
        self.now += int(ms)
        return self.now


def ok_result(inference_id: str, outcome: Outcome = Outcome.CLEAR) -> InferenceResult:
    """Build a successful inference result.

    Args:
        inference_id: The job identity.
        outcome: The decision outcome to report.

    Returns:
        The result.
    """
    return InferenceResult(
        inference_id=inference_id,
        status="SUCCEEDED",
        normalized=NormalizedOutput(family=Family.CLASSIFICATION),
        decision=Decision(
            outcome=outcome, passed=outcome is Outcome.CLEAR, confidence=0.99, threshold=0.5,
            rule="pass",
        ),
        providers=[CUDA_PROVIDER],
        gpu_device="0",
        gpu_class="NVIDIA Test GPU",
        timings=Timings(1.0, 0.0, 1.0, 1.0, 1.0, 4.0),
        memory_high_water_mib=512,
    )


def failed_result(inference_id: str, error_class: str, code: str = "TEST") -> InferenceResult:
    """Build a failed inference result.

    Args:
        inference_id: The job identity.
        error_class: One of the protocol error classes.
        code: The stable code to embed in the error.

    Returns:
        The result.
    """
    return InferenceResult(
        inference_id=inference_id,
        status="FAILED",
        normalized=None,
        decision=None,
        providers=[CUDA_PROVIDER],
        gpu_device="0",
        gpu_class="NVIDIA Test GPU",
        timings=Timings(1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        memory_high_water_mib=None,
        error=f"{code}: scripted {error_class} failure",
        error_class=error_class,
    )


class FakeCell:
    """An executor cell that answers the protocol from a script.

    Attributes:
        calls: Every request the cell received, in order.
        resident: The digests it is holding.
        on_load: Called with the :class:`LoadModel` to produce a reply, or ``None`` for the
            default success. Raising from it simulates a dead cell.
        on_infer: Called with the :class:`Infer` to produce a reply, or ``None`` for the default
            success.
        on_unload: Called with the :class:`Unload` to produce a reply, or ``None`` for the default.
    """

    def __init__(
        self,
        cell_id: str = "gpu0-0",
        device="0",
        providers=(CUDA_PROVIDER,),
        total_mib: int = 8192,
        free_mib: int = 8192,
        load_mib: int = 1024,
        load_ms: float = 120.0,
    ) -> None:
        """Build a fake cell with an imagined device."""
        self.cell_id = cell_id
        self.device = device
        self.providers = tuple(providers)
        self.total_mib = total_mib
        self.free_mib = free_mib
        self.load_mib = load_mib
        self.load_ms = load_ms
        self.calls = []
        self.resident = {}
        self.started = 0
        self.stopped = 0
        self.alive = False
        self.broken = None
        self.pid = 4242
        self.on_load = None
        self.on_infer = None
        self.on_unload = None
        self._replies = deque()
        self._in_flight = None

    # -- lifecycle -------------------------------------------------------------------------

    def start(self):
        """Bring the fake cell up."""
        self.started += 1
        self.alive = True
        self.broken = None
        self.resident = {}
        self._replies.clear()
        self._in_flight = None
        return self

    def stop(self, timeout_s=None) -> None:
        """Take the fake cell down.

        Args:
            timeout_s: Ignored.
        """
        self.stopped += 1
        self.alive = False
        self._in_flight = None

    def is_alive(self) -> bool:
        """Report whether the fake cell is up and not broken."""
        return self.alive and self.broken is None

    def uptime_s(self) -> float:
        """Report a fixed uptime."""
        return 1.0

    def mark_broken(self, reason: str) -> None:
        """Refuse further requests.

        Args:
            reason: Why.
        """
        self.broken = reason

    def die(self) -> None:
        """Simulate the child exiting without notice."""
        self.alive = False

    # -- protocol --------------------------------------------------------------------------

    def _answer(self, message):
        """Produce the reply for one request.

        Args:
            message: The request.

        Returns:
            The reply.
        """
        self.calls.append(message)
        if isinstance(message, LoadModel):
            reply = self.on_load(message) if self.on_load else None
            if reply is None:
                self.resident[message.digest] = self.load_mib
                self.free_mib = max(self.free_mib - self.load_mib, 0)
                reply = Loaded(
                    digest=message.digest,
                    providers_assigned=tuple(message.providers),
                    load_ms=self.load_ms,
                    device_mib=self.load_mib,
                    gpu_device=self.device,
                    gpu_class="NVIDIA Test GPU",
                )
            return reply
        if isinstance(message, Infer):
            reply = self.on_infer(message) if self.on_infer else None
            return reply if reply is not None else ok_result(message.inference_id)
        if isinstance(message, Unload):
            reply = self.on_unload(message) if self.on_unload else None
            if reply is None:
                held = self.resident.pop(message.digest, None)
                self.free_mib = min(self.free_mib + (held or 0), self.total_mib)
                reply = Unloaded(
                    digest=message.digest,
                    freed_mib=held or 0,
                    was_resident=held is not None,
                    expected_mib=held or 0,
                )
            else:
                self.resident.pop(message.digest, None)
            return reply
        if isinstance(message, Stats):
            return CellStats(
                resident=tuple(self.resident),
                device_free_mib=self.free_mib,
                device_total_mib=self.total_mib,
                uptime_s=1.0,
                cell_id=self.cell_id,
                gpu_device=self.device,
                gpu_class="NVIDIA Test GPU",
                resident_mib=dict(self.resident),
            )
        raise CellError(f"the fake cell has no answer for {type(message).__name__}")

    def send(self, message) -> None:
        """Accept one request.

        Args:
            message: The request.

        Raises:
            CellDead: The cell is down or broken.
        """
        if not self.is_alive():
            raise CellDead(f"cell {self.cell_id} is not running")
        self._in_flight = message
        self._replies.append(self._answer(message))

    def receive(self, timeout_s=None):
        """Return the reply to the request in flight.

        Args:
            timeout_s: Ignored.

        Returns:
            The reply.

        Raises:
            CellDead: The cell died before answering.
        """
        if not self.is_alive():
            self._in_flight = None
            raise CellDead(f"cell {self.cell_id} exited before it answered")
        self._in_flight = None
        return self._replies.popleft()

    def call(self, message, timeout_s=None):
        """Send one request and return its reply.

        Args:
            message: The request.
            timeout_s: Ignored.

        Returns:
            The reply.
        """
        self.send(message)
        return self.receive(timeout_s)


class FakeSupervisor:
    """A supervisor that owns fake cells and records every recycle.

    Attributes:
        recycles: ``(cell_id, reason)`` for every recycle asked for.
        recycle_count: How many recycles happened.
        refuse_recycle: When set, the reason a recycle raises instead of restarting.
    """

    def __init__(
        self,
        cells=None,
        providers=(CUDA_PROVIDER,),
        required_provider=None,
        allow_cpu_only=False,
        load_concurrency_per_gpu: int = 1,
    ) -> None:
        """Build the supervisor over its cells."""
        self._cells = list(cells or [FakeCell()])
        for cell in self._cells:
            cell.start()
        self.providers = tuple(providers)
        self.required_provider = required_provider
        self.allow_cpu_only = allow_cpu_only
        self.load_concurrency_per_gpu = load_concurrency_per_gpu
        self.recycles = []
        self.recycle_count = 0
        self.refuse_recycle = None

    def cells(self) -> list:
        """Return the cells."""
        return list(self._cells)

    def healthy(self) -> bool:
        """Report whether every cell is alive."""
        return all(cell.is_alive() for cell in self._cells)

    def recycle(self, cell, reason: str):
        """Restart one cell.

        Args:
            cell: The cell handle.
            reason: Why.

        Returns:
            The request that was in flight, or ``None``.
        """
        self.recycles.append((cell.cell_id, reason))
        self.recycle_count += 1
        drained = cell._in_flight
        if self.refuse_recycle:
            from image_processor.engine.supervisor import SupervisorError

            raise SupervisorError("RESTART_BUDGET_EXHAUSTED", self.refuse_recycle)
        cell.stop()
        cell.start()
        return drained


class FakeManifest:
    """The manifest fields the scheduler reads off a cached bundle."""

    def __init__(
        self,
        estimated_device_mib: int = 1024,
        provider_policy: str = "preferListed",
        providers_permitted=(CUDA_PROVIDER,),
        transform_version: str = "t1",
    ) -> None:
        """Build the manifest stand-in."""
        self.estimated_device_mib = estimated_device_mib
        self.provider_policy = provider_policy
        self.providers_permitted = list(providers_permitted)
        self.transform_version = transform_version


class FakeBundle:
    """A cached bundle stand-in."""

    def __init__(self, digest: str, root="/bundles/x", manifest=None) -> None:
        """Build the bundle stand-in."""
        self.digest = digest
        self.root = root
        self.manifest = manifest or FakeManifest()


class FakeCache:
    """A bundle cache stand-in.

    Attributes:
        bundles: Digest to :class:`FakeBundle`.
        raises: Digest to the exception ``get`` raises for it.
    """

    def __init__(self, bundles=None) -> None:
        """Build the cache stand-in."""
        self.bundles = dict(bundles or {})
        self.raises = {}

    def add(self, digest: str, **kwargs) -> FakeBundle:
        """Add one bundle.

        Args:
            digest: The bundle digest.
            **kwargs: Passed to :class:`FakeManifest`.

        Returns:
            The bundle.
        """
        bundle = FakeBundle(digest, manifest=FakeManifest(**kwargs))
        self.bundles[digest] = bundle
        return bundle

    def get(self, digest: str, verify: bool = False):
        """Return one bundle.

        Args:
            digest: The bundle digest.
            verify: Ignored.

        Returns:
            The bundle, or ``None`` when it is not cached.

        Raises:
            Exception: Whatever ``raises`` holds for this digest.
        """
        if digest in self.raises:
            raise self.raises[digest]
        return self.bundles.get(digest)


def build_job(
    inference_id: str,
    digest: str = DIGEST_A,
    state: JobState = JobState.READY,
    route_id: str = "cam-01",
    staged_path: str = "/spool/cam-01/capture.jpg",
    sha256: str = "d" * 64,
    attempts: int = 0,
    transform_version: str = "t1",
) -> Job:
    """Build a job value for a scheduler test.

    Args:
        inference_id: The job identity.
        digest: The bundle digest it is pinned to.
        state: The state to build it in.
        route_id: The owning route.
        staged_path: The file the cell would read.
        sha256: The input digest.
        attempts: The attempt count.
        transform_version: The transform generation.

    Returns:
        The job.
    """
    return Job(
        inference_id=inference_id,
        route_id=route_id,
        source=SourceIdentity(
            kind=SourceKind.SPOOL,
            route_id=route_id,
            relative_path="2026/08/22/capture.jpg",
            bytes=4096,
            sha256=sha256,
            capture_id=inference_id,
            camera_id="cam-01",
        ),
        model=ModelRef(id="line-clearance", version="2026.08.20", digest=digest),
        transform_version=transform_version,
        state=state,
        attempts=attempts,
        staged_path=staged_path,
    )


def admit_ready(ledger, inference_id: str, digest: str = DIGEST_A, **kwargs) -> Job:
    """Admit one job into a real ledger and leave it ``READY``.

    Args:
        ledger: The ledger.
        inference_id: The job identity.
        digest: The bundle digest.
        **kwargs: Passed to :func:`build_job`.

    Returns:
        The job as the ledger holds it.
    """
    job = build_job(inference_id, digest=digest, state=JobState.READY, **kwargs)
    assert ledger.admit(job, 1024) is True
    return ledger.get(inference_id)


def sleeping_entrypoint(config, connection):
    """A cell subprocess that takes a request and never answers it.

    Args:
        config: The cell configuration. Ignored.
        connection: The pipe to the parent.
    """
    import time

    while True:
        try:
            connection.recv()
        except (EOFError, OSError):
            return
        time.sleep(30)


def dying_entrypoint(config, connection):
    """A cell subprocess that exits the moment it is asked anything.

    Args:
        config: The cell configuration. Ignored.
        connection: The pipe to the parent.
    """
    import os

    try:
        connection.recv()
    except (EOFError, OSError):
        return
    os._exit(9)
