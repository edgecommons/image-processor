"""The authoritative inference result body (DESIGN.md 12.1, LLD 8).

This module owns one document: the body of the ``ImageInferenceResult`` message published on
``ecv1/{device}/image-processor/{routeId}/app/inference/result``. It is the only cleanup-gating
output the component has (D-IP-6), so everything about it is deliberate.

* It is built from the job and the executor cell answer, never from anything a producer said.
* It is validated against ``schemas/inference-result.schema.json`` before it is prepared, because
  a body that does not satisfy the contract must fail here rather than on a consumer gate.
* A failed inference still produces a body, and its decision is ``HOLD``. A missing, failed,
  stale, or unverified inference is never ``CLEAR``.
* Every collection is bounded. When the body no longer fits the message budget the caller writes
  the full result to the evidence sidecar and publishes the summary with ``truncated`` set and
  ``artifacts`` naming the sidecar, so a decision-bearing result is never shortened silently.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from jsonschema import Draft202012Validator

from image_processor.config.validate import INFERENCE_RESULT_SCHEMA, load_schema
from image_processor.types import Family, Outcome, SourceKind

logger = logging.getLogger(__name__)

#: The body contract version this component emits.
RESULT_SCHEMA_VERSION = "1.0"

#: The envelope header name of the result message.
RESULT_MESSAGE_NAME = "ImageInferenceResult"

#: The envelope header version of the result message.
RESULT_MESSAGE_VERSION = "1.0"

#: The ``app`` channel the result is published on.
RESULT_CHANNEL = "inference/result"

#: The inference runtime this component reports. Phase 1 has exactly one.
DEFAULT_RUNTIME = "onnxruntime"

#: The schema ceiling on any published collection.
SCHEMA_MAX_ITEMS = 1000

#: The schema bound on the failure detail.
MAX_ERROR_MESSAGE_CHARS = 1024

#: The stable code a failure carries when the executor reported none.
DEFAULT_ERROR_CODE = "INFERENCE_FAILED"

_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ResultError(Exception):
    """A result body cannot be built, or does not satisfy its contract.

    Attributes:
        code: Stable SCREAMING_SNAKE code.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class ResultLimits:
    """What one published result message may cost.

    Attributes:
        max_items: The largest collection the published message carries, before the byte budget
            bounds it further.
        max_body_bytes: The message budget. A body over it is published as a bounded summary and
            the full result goes to the evidence sidecar.
    """

    max_items: int = 100
    max_body_bytes: int = 32768


#: The budget a route uses when it configures none.
DEFAULT_LIMITS = ResultLimits()


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    """Return the cached schema validator for the result body."""
    return Draft202012Validator(load_schema(INFERENCE_RESULT_SCHEMA))


def _plain(value: Any) -> Any:
    """Return a value using only JSON-shaped types.

    A task family computes with numpy, so a score can arrive as ``numpy.float32`` and a pixel
    count as ``numpy.int64``. Both serialize only after they are converted, and converting here
    keeps every family free of that concern.

    Args:
        value: Any value from a normalized output.

    Returns:
        The same structure using dicts, lists, and Python scalars.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(entry) for entry in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(value)
    return str(value)


def body_bytes(body: dict) -> bytes:
    """Serialize a result body to the bytes the ledger retains.

    Args:
        body: The result body.

    Returns:
        Compact UTF-8 JSON.

    Raises:
        ResultError: ``BODY_NOT_SERIALIZABLE`` when the body carries a value JSON cannot express.
    """
    try:
        return json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResultError("BODY_NOT_SERIALIZABLE", str(exc)) from exc


def fits_budget(body: dict, limits: ResultLimits = DEFAULT_LIMITS) -> bool:
    """Report whether a body fits the message budget.

    Args:
        body: The result body.
        limits: The route limits.

    Returns:
        ``True`` when the serialized body is within ``limits.max_body_bytes``.
    """
    return len(body_bytes(body)) <= limits.max_body_bytes


def validate_result_body(body: dict) -> None:
    """Check a body against ``schemas/inference-result.schema.json``.

    Args:
        body: The result body.

    Raises:
        ResultError: ``RESULT_SCHEMA_INVALID`` naming the first offending path.
    """
    errors = sorted(_validator().iter_errors(body), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    first = errors[0]
    where = "/".join(str(part) for part in first.absolute_path) or "(root)"
    raise ResultError("RESULT_SCHEMA_INVALID", f"{where}: {first.message}")


def _source_block(source: Any) -> dict:
    """Build the ``source`` block from a verified input identity."""
    kind = source.kind.value if isinstance(source.kind, SourceKind) else str(source.kind)
    block: dict = {
        "kind": kind,
        "relativePath": source.relative_path,
        "bytes": int(source.bytes),
        "sha256": source.sha256,
    }
    if source.capture_id:
        block["captureId"] = source.capture_id
    if source.camera_id:
        block["cameraId"] = source.camera_id
    if source.correlation_id:
        block["correlationId"] = source.correlation_id
    if source.captured_at_ms is not None:
        block["capturedAtMs"] = int(source.captured_at_ms)
    return block


def _model_block(job: Any, result: Any, manifest: Any) -> dict:
    """Build the ``model`` block: the pinned generation and the observed assignment.

    The providers are the session actual assignment, never the configured preference, so a result
    never implies a device it did not run on. A failure that never reached a session has no
    assignment at all, and reports that rather than inventing one.
    """
    providers = [str(name) for name in (result.providers or [])]
    block: dict = {
        "id": job.model.id,
        "version": job.model.version,
        "digest": job.model.digest,
        "runtime": DEFAULT_RUNTIME,
        "providers": providers or ["none"],
    }
    transform_version = job.transform_version or getattr(manifest, "transform_version", None)
    if transform_version:
        block["transformVersion"] = str(transform_version)
    if result.gpu_device is not None or result.gpu_class is not None:
        gpu: dict = {}
        if result.gpu_device is not None:
            gpu["deviceId"] = str(result.gpu_device)
        if result.gpu_class is not None:
            gpu["class"] = str(result.gpu_class)
        block["gpu"] = gpu
    else:
        block["gpu"] = None
    return block


def _decision_block(decision: Any) -> dict:
    """Build the ``decision`` block from the rules verdict."""
    outcome = (
        decision.outcome.value if isinstance(decision.outcome, Outcome) else str(decision.outcome)
    )
    block: dict = {"outcome": outcome, "pass": bool(decision.passed)}
    block["confidence"] = None if decision.confidence is None else float(decision.confidence)
    block["threshold"] = None if decision.threshold is None else float(decision.threshold)
    if decision.rule:
        block["rule"] = str(decision.rule)[:512]
    return block


def _held_decision(rule: str) -> dict:
    """Build the decision a failed inference reports.

    A failed, missing, stale, or unverified inference is ``HOLD``, never ``CLEAR`` (DESIGN.md 15).
    Saying so explicitly means a consumer reading ``decision.outcome`` always gets an answer.
    """
    return {
        "outcome": Outcome.HOLD.value,
        "pass": False,
        "confidence": None,
        "threshold": None,
        "rule": rule[:512],
    }


def _item_cap(manifest: Any, limits: Optional[ResultLimits]) -> int:
    """Return how many items of one collection the body may carry."""
    declared = getattr(manifest, "max_result_items", None)
    cap = SCHEMA_MAX_ITEMS if not declared else int(declared)
    if limits is not None:
        cap = min(cap, int(limits.max_items))
    return max(0, min(cap, SCHEMA_MAX_ITEMS))


def _outputs_block(normalized: Any, manifest: Any, limits: Optional[ResultLimits]) -> dict:
    """Build the ``outputs`` block: only the family own collection is populated."""
    cap = _item_cap(manifest, limits)
    family = normalized.family
    block: dict = {"family": family.value if isinstance(family, Family) else str(family)}
    if normalized.classes:
        block["classes"] = [
            {"label": str(entry.label), "index": int(entry.index), "score": float(entry.score)}
            for entry in list(normalized.classes)[:cap]
        ]
    if normalized.detections:
        block["detections"] = [
            {
                "label": str(entry.label),
                "index": int(entry.index),
                "score": float(entry.score),
                "box": [float(value) for value in entry.box],
            }
            for entry in list(normalized.detections)[:cap]
        ]
    if normalized.segments:
        ordered = sorted(
            normalized.segments.items(),
            key=lambda item: (-float(item[1].get("pixels", 0) or 0), str(item[0])),
        )
        block["segments"] = {str(label): _plain(value) for label, value in ordered[:cap]}
    if normalized.anomaly:
        block["anomaly"] = _plain(dict(normalized.anomaly))
    block["truncated"] = False
    return block


def _timings_block(timings: Any) -> dict:
    """Build the ``timingsMs`` block."""
    return {
        "queue": max(float(timings.queue_ms), 0.0),
        "modelLoad": max(float(timings.model_load_ms), 0.0),
        "preprocess": max(float(timings.preprocess_ms), 0.0),
        "inference": max(float(timings.inference_ms), 0.0),
        "postprocess": max(float(timings.postprocess_ms), 0.0),
        "total": max(float(timings.total_ms), 0.0),
    }


def split_error(error: Optional[str]) -> tuple:
    """Split an executor error string into its stable code and its detail.

    The executor boundary reports ``CODE: detail`` (``engine/protocol.py``). A string without a
    recognizable code is still reported, under :data:`DEFAULT_ERROR_CODE`, because a failure that
    cannot be classified is still a failure.

    Args:
        error: The error string, or ``None``.

    Returns:
        The ``(code, message)`` pair.
    """
    text = (error or "").strip()
    if not text:
        return DEFAULT_ERROR_CODE, "the executor reported no detail"
    head, separator, tail = text.partition(":")
    if separator and _CODE.match(head.strip()):
        return head.strip(), (tail.strip()[:MAX_ERROR_MESSAGE_CHARS] or head.strip())
    return DEFAULT_ERROR_CODE, text[:MAX_ERROR_MESSAGE_CHARS]


def _error_block(result: Any) -> dict:
    """Build the ``error`` block of a failed result."""
    code, message = split_error(result.error)
    block = {"code": code, "message": message}
    if result.error_class:
        block["class"] = str(result.error_class)
    return block


def _artifacts_block(artifacts: Optional[dict]) -> Optional[dict]:
    """Build the ``artifacts`` block from the installed evidence sidecar."""
    if not artifacts:
        return None
    block: dict = {}
    for key in ("evidenceId", "localRelativePath", "sha256"):
        value = artifacts.get(key)
        if value:
            block[key] = str(value)
    if artifacts.get("bytes") is not None:
        block["bytes"] = int(artifacts["bytes"])
    return block or None


def _drop_one(outputs: dict) -> bool:
    """Remove the least informative item of the widest collection.

    Returns:
        ``True`` when something was removed, ``False`` when nothing is left to remove.
    """
    sizes = {
        key: len(outputs.get(key) or ())
        for key in ("classes", "detections", "segments")
        if outputs.get(key)
    }
    if not sizes:
        return False
    widest = max(sizes, key=lambda key: sizes[key])
    collection = outputs[widest]
    if isinstance(collection, dict):
        smallest = min(
            collection, key=lambda label: (float(collection[label].get("pixels", 0) or 0), label)
        )
        collection.pop(smallest)
    else:
        collection.pop()
    if not collection:
        outputs.pop(widest)
    return True


def _bound(body: dict, limits: ResultLimits) -> bool:
    """Shrink a body collections until it fits the message budget.

    Args:
        body: The result body, modified in place.
        limits: The route limits.

    Returns:
        Whether anything was dropped, which is what ``outputs.truncated`` reports.
    """
    if fits_budget(body, limits):
        return False
    outputs = body.get("outputs")
    if not isinstance(outputs, dict):
        return False
    dropped = False
    while not fits_budget(body, limits) and _drop_one(outputs):
        dropped = True
    if dropped:
        outputs["truncated"] = True
    return dropped


def build_result_body(
    job: Any,
    result: Any,
    manifest: Any = None,
    artifacts: Optional[dict] = None,
    limits: Optional[ResultLimits] = None,
    *,
    validate: bool = True,
) -> dict:
    """Build one ``ImageInferenceResult`` body.

    Args:
        job: The durable job the result belongs to.
        result: The executor cell :class:`~image_processor.types.InferenceResult`.
        manifest: The bundle manifest of the pinned generation, or ``None`` when the model never
            loaded and there is nothing to read a cardinality or a transform version from.
        artifacts: The installed evidence sidecar as ``evidenceId``, ``localRelativePath``,
            ``sha256``, and ``bytes``, or ``None`` when the route writes none.
        limits: The message budget. ``None`` builds the full result, which is what the evidence
            sidecar carries.
        validate: Whether to check the body against the shipped schema. Only a test deliberately
            building an invalid body turns it off.

    Returns:
        The body, ready to hand to ``app().prepare()``.

    Raises:
        ResultError: The body cannot be serialized, needs an evidence sidecar it was not given,
            or does not satisfy the contract.
    """
    body: dict = {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "inferenceId": job.inference_id,
        "routeId": job.route_id,
        "status": "SUCCEEDED" if result.status == "SUCCEEDED" else "FAILED",
        "source": _source_block(job.source),
        "model": _model_block(job, result, manifest),
        "timingsMs": _timings_block(result.timings),
    }
    if body["status"] == "SUCCEEDED":
        if result.normalized is None:
            raise ResultError("NO_NORMALIZED_OUTPUT", "a succeeded result carries an output")
        body["outputs"] = _outputs_block(result.normalized, manifest, limits)
        body["decision"] = (
            _decision_block(result.decision)
            if result.decision is not None
            else _held_decision("no decision rules evaluated")
        )
    else:
        error = _error_block(result)
        body["error"] = error
        body["decision"] = _held_decision(error["code"])
        if result.normalized is not None:
            body["outputs"] = _outputs_block(result.normalized, manifest, limits)
    block = _artifacts_block(artifacts)
    if block:
        body["artifacts"] = block
    if limits is not None and _bound(body, limits) and "artifacts" not in body:
        raise ResultError(
            "EVIDENCE_REQUIRED", "a truncated result names the sidecar holding the full result"
        )
    if validate:
        validate_result_body(body)
    return body
