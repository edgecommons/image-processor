"""The result body: what it says, what it refuses to say, and what it never says silently."""

from __future__ import annotations

import json

import pytest

from image_processor.outputs.result import (
    DEFAULT_ERROR_CODE,
    ResultError,
    ResultLimits,
    body_bytes,
    build_result_body,
    fits_budget,
    split_error,
    validate_result_body,
)
from image_processor.types import Decision, Outcome
from tests.outputs.conftest import (
    make_classes,
    make_detections,
    make_failure,
    make_job,
    make_result,
    make_source,
)
from image_processor.types import SourceKind


def test_a_successful_result_validates_against_the_shipped_schema(job, result):
    body = build_result_body(job, result)

    validate_result_body(body)
    assert body["schemaVersion"] == "1.0"
    assert body["inferenceId"] == job.inference_id
    assert body["routeId"] == "clearance-cam-01"
    assert body["status"] == "SUCCEEDED"
    assert body["source"]["captureId"] == "cap-0001"
    assert body["model"]["providers"] == ["CUDAExecutionProvider"]
    assert body["model"]["gpu"] == {"deviceId": "0", "class": "sm_120"}
    assert body["model"]["transformVersion"] == "1.0.0"
    assert body["decision"] == {
        "outcome": "CLEAR",
        "pass": True,
        "confidence": 0.99,
        "threshold": 0.9,
        "rule": "pass",
    }
    assert body["timingsMs"]["total"] == 22.2


def test_the_reported_providers_are_the_assignment_not_the_preference(job):
    result = make_result(providers=["CPUExecutionProvider"])

    body = build_result_body(job, result)

    assert body["model"]["providers"] == ["CPUExecutionProvider"]


def test_a_failure_that_never_reached_a_session_reports_no_provider(job):
    body = build_result_body(job, make_failure())

    assert body["model"]["providers"] == ["none"]
    assert body["model"]["gpu"] is None


def test_a_failed_inference_holds_and_carries_its_code(job):
    body = build_result_body(job, make_failure())

    validate_result_body(body)
    assert body["status"] == "FAILED"
    assert body["decision"]["outcome"] == "HOLD"
    assert body["decision"]["pass"] is False
    assert body["error"] == {
        "code": "DECODE_FAILED",
        "message": "the image is truncated",
        "class": "permanent",
    }


def test_a_success_with_no_decision_still_holds(job):
    body = build_result_body(job, make_result(decision=None))

    assert body["decision"]["outcome"] == "HOLD"
    assert body["decision"]["rule"] == "no decision rules evaluated"


def test_a_success_without_an_output_is_refused(job):
    with pytest.raises(ResultError) as failure:
        build_result_body(job, make_result(normalized=None))

    assert failure.value.code == "NO_NORMALIZED_OUTPUT"


@pytest.mark.parametrize(
    "error, code, message",
    [
        ("DECODE_FAILED: truncated", "DECODE_FAILED", "truncated"),
        ("something went wrong", DEFAULT_ERROR_CODE, "something went wrong"),
        ("", DEFAULT_ERROR_CODE, "the executor reported no detail"),
        ("lower: case", DEFAULT_ERROR_CODE, "lower: case"),
        ("ONLY_CODE:", "ONLY_CODE", "ONLY_CODE"),
    ],
)
def test_an_executor_error_splits_into_a_stable_code(error, code, message):
    assert split_error(error) == (code, message)


def test_a_body_over_the_budget_is_bounded_and_says_so(job):
    result = make_result(normalized=make_detections(200))
    limits = ResultLimits(max_items=200, max_body_bytes=2048)
    artifacts = {
        "evidenceId": job.inference_id,
        "localRelativePath": "cam-01/cap.inference.json",
        "sha256": "ef" * 32,
        "bytes": 9182,
    }

    body = build_result_body(job, result, artifacts=artifacts, limits=limits)

    validate_result_body(body)
    assert body["outputs"]["truncated"] is True
    assert len(body["outputs"]["detections"]) < 200
    assert fits_budget(body, limits)
    assert body["artifacts"]["evidenceId"] == job.inference_id


def test_a_result_that_must_be_bounded_without_evidence_is_refused(job):
    result = make_result(normalized=make_detections(200))

    with pytest.raises(ResultError) as failure:
        build_result_body(job, result, limits=ResultLimits(max_items=200, max_body_bytes=1024))

    assert failure.value.code == "EVIDENCE_REQUIRED"


def test_the_manifest_cardinality_bounds_the_collection(job):
    class _Manifest:
        max_result_items = 3
        transform_version = "1.0.0"

    body = build_result_body(job, make_result(normalized=make_classes(10)), _Manifest())

    assert len(body["outputs"]["classes"]) == 3


def test_a_body_that_fits_is_not_marked_truncated(job, result):
    body = build_result_body(job, result, limits=ResultLimits())

    assert body["outputs"]["truncated"] is False
    assert "artifacts" not in body


def test_numpy_scalars_survive_serialization(job):
    numpy = pytest.importorskip("numpy")
    normalized = make_classes(1)
    segments = {"defect": {"pixels": numpy.int64(12), "fraction": numpy.float32(0.5), "bbox": None}}
    from image_processor.types import Family, NormalizedOutput

    body = build_result_body(
        job,
        make_result(normalized=NormalizedOutput(family=Family.SEGMENTATION, segments=segments)),
    )

    assert json.loads(body_bytes(body))["outputs"]["segments"]["defect"]["pixels"] == 12


def test_an_inline_source_reports_its_correlation(job):
    source = make_source(
        kind=SourceKind.INLINE,
        relative_path="ab/abcd.png",
        capture_id=None,
        camera_id=None,
        correlation_id="corr-1",
        captured_at_ms=1789012506010,
    )

    body = build_result_body(make_job(source=source), make_result())

    assert body["source"]["kind"] == "inline"
    assert body["source"]["correlationId"] == "corr-1"
    assert body["source"]["capturedAtMs"] == 1789012506010
    assert "captureId" not in body["source"]


def test_an_invalid_body_names_the_offending_path(job, result):
    body = build_result_body(job, result)
    body["source"]["sha256"] = "not-a-digest"

    with pytest.raises(ResultError) as failure:
        validate_result_body(body)

    assert failure.value.code == "RESULT_SCHEMA_INVALID"
    assert "source/sha256" in failure.value.message


def test_a_body_carrying_a_value_json_cannot_express_is_refused():
    with pytest.raises(ResultError) as failure:
        body_bytes({"value": float("nan")})

    assert failure.value.code == "BODY_NOT_SERIALIZABLE"
