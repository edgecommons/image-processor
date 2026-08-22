"""Shared builders for the WP3 ledger suite: job values, legal-edge walks, and outbox rows."""

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processor.ledger import Ledger, OutboxRow  # noqa: E402
from image_processor.ledger.schema import TRANSITIONS  # noqa: E402
from image_processor.types import (  # noqa: E402
    Job,
    JobState,
    ModelRef,
    SourceIdentity,
    SourceKind,
)

MODEL = ModelRef(id="line-clearance", version="2026.08.20", digest="sha256:" + "1" * 64)


class StepClock:
    """A monotonic millisecond clock that advances one tick per read."""

    def __init__(self, start: int = 1_700_000_000_000, step: int = 1) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> int:
        self.now += self.step
        return self.now


def build_job(
    inference_id: str = "job-1",
    state: JobState = JobState.DISCOVERED,
    route_id: str = "cam-01",
    relative_path: str = "2026/08/22/capture-1.jpg",
    sha256: str = "a" * 64,
    attempts: int = 0,
    **kwargs,
) -> Job:
    """Build a job value for a test.

    Args:
        inference_id: The job identity.
        state: The state to admit at.
        route_id: The owning route.
        relative_path: The input's path relative to its route root.
        sha256: The input digest.
        attempts: The starting attempt count.
        **kwargs: Passed through to :class:`~image_processor.types.Job`.

    Returns:
        The job.
    """
    source = SourceIdentity(
        kind=SourceKind.SPOOL,
        route_id=route_id,
        relative_path=relative_path,
        bytes=4096,
        sha256=sha256,
        capture_id=kwargs.pop("capture_id", "capture-1"),
        camera_id="cam-01",
    )
    return Job(
        inference_id=inference_id,
        route_id=route_id,
        source=source,
        model=kwargs.pop("model", MODEL),
        transform_version=kwargs.pop("transform_version", "t1"),
        state=state,
        attempts=attempts,
        **kwargs,
    )


def path_to(target: JobState) -> list:
    """Return the shortest legal edge sequence from ``DISCOVERED`` to ``target``.

    Args:
        target: The state to reach.

    Returns:
        The states to pass through, excluding ``DISCOVERED``.

    Raises:
        AssertionError: The state is unreachable from ``DISCOVERED``.
    """
    if target is JobState.DISCOVERED:
        return []
    edges: dict = {}
    for src, dst in TRANSITIONS:
        edges.setdefault(src, []).append(dst)
    queue = deque([(JobState.DISCOVERED, [])])
    seen = {JobState.DISCOVERED}
    while queue:
        state, trail = queue.popleft()
        for nxt in sorted(edges.get(state, []), key=lambda s: s.value):
            if nxt in seen:
                continue
            if nxt is target:
                return trail + [nxt]
            seen.add(nxt)
            queue.append((nxt, trail + [nxt]))
    raise AssertionError(f"{target.value} is unreachable from DISCOVERED")


def drive(ledger: Ledger, inference_id: str, target: JobState) -> Job:
    """Walk a job along legal edges until it reaches ``target``.

    Args:
        ledger: The ledger holding the job.
        inference_id: The job identity.
        target: The state to reach.

    Returns:
        The job as it now stands.
    """
    job = ledger.get(inference_id)
    assert job is not None
    for state in path_to(target):
        job = ledger.transition(job.inference_id, job.state, state)
    return job


def admitted(ledger: Ledger, target: JobState = JobState.DISCOVERED, **kwargs) -> Job:
    """Admit a job and drive it to ``target``.

    Args:
        ledger: The ledger to admit into.
        target: The state to leave the job in.
        **kwargs: Passed to :func:`build_job`.

    Returns:
        The job as it now stands.
    """
    reserve_bytes = kwargs.pop("reserve_bytes", 1024)
    job = build_job(**kwargs)
    assert ledger.admit(job, reserve_bytes) is True
    return drive(ledger, job.inference_id, target)


def row(inference_id: str = "job-1", topic: str = "ecv1/d/image-processor/cam-01/app/inference/result",
        payload: bytes = b"body", gating: bool = True) -> OutboxRow:
    """Build an outbox row for a test."""
    return OutboxRow(
        id=None, inference_id=inference_id, topic=topic, encoded_bytes=payload, gating=gating
    )
