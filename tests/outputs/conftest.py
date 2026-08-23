"""Builders for the outputs suites (WP6).

Every helper builds the real value type, so a test that asserts about a body is asserting about
what the component would actually publish.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from image_processor.types import (
    ClassScore,
    Decision,
    Detection,
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

DIGEST = "sha256:" + "ab" * 32
HEX = "cd" * 32


def make_source(**overrides) -> SourceIdentity:
    """Build one verified input identity."""
    base = {
        "kind": SourceKind.SPOOL,
        "route_id": "clearance-cam-01",
        "relative_path": "2026/08/22/cap-0001.jpg",
        "bytes": 4182,
        "sha256": HEX,
        "capture_id": "cap-0001",
        "camera_id": "cam-01",
    }
    base.update(overrides)
    return SourceIdentity(**base)


def make_job(**overrides) -> Job:
    """Build one durable job."""
    source = overrides.pop("source", None) or make_source()
    base = {
        "inference_id": "01k5inference0001",
        "route_id": source.route_id,
        "source": source,
        "model": ModelRef("line-clearance", "2026.08.20", DIGEST),
        "transform_version": "1.0.0",
        "state": JobState.INFERENCING,
    }
    base.update(overrides)
    return Job(**base)


def make_timings(**overrides) -> Timings:
    """Build one set of stage timings."""
    base = {
        "queue_ms": 12.4,
        "model_load_ms": 0.0,
        "preprocess_ms": 3.1,
        "inference_ms": 5.8,
        "postprocess_ms": 0.9,
        "total_ms": 22.2,
    }
    base.update(overrides)
    return Timings(**base)


def make_classes(count: int = 2) -> NormalizedOutput:
    """Build a classification output of ``count`` classes."""
    return NormalizedOutput(
        family=Family.CLASSIFICATION,
        classes=[
            ClassScore(label=f"class-{index}", index=index, score=1.0 / (index + 1))
            for index in range(count)
        ],
    )


def make_detections(count: int = 2) -> NormalizedOutput:
    """Build a detection output of ``count`` boxes."""
    return NormalizedOutput(
        family=Family.DETECTION,
        detections=[
            Detection(
                label=f"part-{index}",
                index=index,
                score=0.9,
                box=(0.1 * index, 0.2, 0.3, 0.4),
            )
            for index in range(count)
        ],
    )


def make_result(**overrides) -> InferenceResult:
    """Build one executor answer."""
    base = {
        "inference_id": "01k5inference0001",
        "status": "SUCCEEDED",
        "normalized": make_classes(),
        "decision": Decision(Outcome.CLEAR, True, 0.99, 0.9, "pass"),
        "providers": ["CUDAExecutionProvider"],
        "gpu_device": "0",
        "gpu_class": "sm_120",
        "timings": make_timings(),
        "memory_high_water_mib": 512,
    }
    base.update(overrides)
    return InferenceResult(**base)


def make_failure(error: str = "DECODE_FAILED: the image is truncated", **overrides):
    """Build one failed executor answer."""
    return make_result(
        status="FAILED",
        normalized=None,
        decision=None,
        providers=[],
        gpu_device=None,
        gpu_class=None,
        error=error,
        error_class="permanent",
        **overrides,
    )


@pytest.fixture
def job() -> Job:
    """One durable job in flight."""
    return make_job()


@pytest.fixture
def result() -> InferenceResult:
    """One successful executor answer."""
    return make_result()
