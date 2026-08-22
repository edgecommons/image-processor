"""Shared value types for ImageProcessor (LLD §2).

Every package imports from here and nowhere else for these shapes. The dataclasses are frozen:
a job, a manifest, or a result is a value, and the ledger is the only place state changes.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobState(str, Enum):
    """Durable job states (DESIGN.md §7)."""

    DISCOVERED = "DISCOVERED"
    READY = "READY"
    INPUT_INVALID = "INPUT_INVALID"
    QUARANTINED = "QUARANTINED"
    CLAIMED = "CLAIMED"
    WAITING_MODEL = "WAITING_MODEL"
    INFERENCING = "INFERENCING"
    BLOCKED_CONFIGURATION = "BLOCKED_CONFIGURATION"
    RETRY_WAIT = "RETRY_WAIT"
    PROCESSING_EXHAUSTED = "PROCESSING_EXHAUSTED"
    RETAINED_FAILED = "RETAINED_FAILED"
    RESULT_COMMITTED = "RESULT_COMMITTED"
    PUBLISH_PENDING = "PUBLISH_PENDING"
    PUBLISHED = "PUBLISHED"
    PUBLISH_EXHAUSTED = "PUBLISH_EXHAUSTED"
    CLEANUP_PENDING = "CLEANUP_PENDING"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    COMPLETED = "COMPLETED"


#: States from which a job never leaves without an operator action or a retry timer.
TERMINAL_STATES = frozenset(
    {
        JobState.QUARANTINED,
        JobState.RETAINED_FAILED,
        JobState.COMPLETED,
    }
)


class SourceKind(str, Enum):
    """Where an input image came from (DESIGN.md §4)."""

    SPOOL = "spool"
    INLINE = "inline"
    REFERENCE = "reference"


class Family(str, Enum):
    """Built-in task families (DESIGN.md §8.1, D-IP-12)."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    ANOMALY = "anomaly"


class Outcome(str, Enum):
    """Decision outcome. Anything that is not a verified pass is HOLD or FAIL, never CLEAR."""

    CLEAR = "CLEAR"
    HOLD = "HOLD"
    FAIL = "FAIL"


class CompletionAction(str, Enum):
    """What happens to the owned input copy after the result is confirmed (DESIGN.md §5, §7)."""

    ARCHIVE = "archive"
    DELETE = "delete"
    RETAIN = "retain"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class ModelRef:
    """An immutable model identity. ``digest`` is ``sha256:<hex>`` of the bundle tarball."""

    id: str
    version: str
    digest: str


@dataclass(frozen=True)
class SourceIdentity:
    """Identity and provenance of one input image."""

    kind: SourceKind
    route_id: str
    relative_path: str
    bytes: int
    sha256: str
    capture_id: Optional[str] = None
    camera_id: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    captured_at_ms: Optional[int] = None


@dataclass(frozen=True)
class Job:
    """One durable unit of work: an input under a pinned model generation."""

    inference_id: str
    route_id: str
    source: SourceIdentity
    model: ModelRef
    transform_version: str
    state: JobState
    attempts: int = 0
    next_attempt_at_ms: Optional[int] = None
    staged_path: Optional[str] = None
    config_generation: int = 0


@dataclass(frozen=True)
class TensorSpec:
    """One model input or output. A ``"N"`` dimension marks a dynamic batch axis."""

    name: str
    dtype: str
    shape: tuple  # tuple[int | str, ...]


@dataclass(frozen=True)
class BundleManifest:
    """A parsed, schema-valid ``manifest.json`` (DESIGN.md §8)."""

    schema_version: int
    model_id: str
    version: str
    files: dict  # relative path -> sha256 hex
    min_onnxruntime: str
    providers_permitted: list
    provider_policy: str
    inputs: list  # list[TensorSpec]
    outputs: list  # list[TensorSpec]
    dynamic_batch: bool
    family: Family
    family_params: dict
    preprocess: dict
    decision_rules: dict
    max_result_items: int
    estimated_device_mib: int
    warmup: list
    tolerances: dict
    compatibility_keys: dict
    provenance: dict
    key_id: Optional[str]
    transform_version: str


@dataclass(frozen=True)
class CachedBundle:
    """A verified bundle promoted into the content-addressed cache."""

    digest: str
    root: Path
    manifest: BundleManifest
    model_path: Path


@dataclass(frozen=True)
class ClassScore:
    label: str
    index: int
    score: float


@dataclass(frozen=True)
class Detection:
    """One detection; ``box`` is ``(x, y, w, h)`` normalized to ``[0, 1]`` of the source image."""

    label: str
    index: int
    score: float
    box: tuple  # tuple[float, float, float, float]


@dataclass(frozen=True)
class NormalizedOutput:
    """Task-family output in the normalized shape the decision rules evaluate (DESIGN.md §8.1)."""

    family: Family
    classes: list = field(default_factory=list)  # list[ClassScore]
    detections: list = field(default_factory=list)  # list[Detection]
    segments: dict = field(default_factory=dict)  # {label: {"pixels": int, "bbox": [x, y, w, h]}}
    anomaly: dict = field(default_factory=dict)  # {"score": float, "threshold": float, "summary": {...}}
    raw_shapes: dict = field(default_factory=dict)  # output name -> shape


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    passed: bool
    confidence: Optional[float]
    threshold: Optional[float]
    rule: str


@dataclass(frozen=True)
class Timings:
    queue_ms: float
    model_load_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


@dataclass(frozen=True)
class InferenceResult:
    """An executor cell's answer for one job."""

    inference_id: str
    status: str  # "SUCCEEDED" | "FAILED"
    normalized: Optional[NormalizedOutput]
    decision: Optional[Decision]
    providers: list  # the actual session provider assignment
    gpu_device: Optional[str]
    gpu_class: Optional[str]
    timings: Timings
    memory_high_water_mib: Optional[int]
    error: Optional[str] = None
    error_class: Optional[str] = None  # "transient" | "permanent" | "contaminating"


@dataclass(frozen=True)
class CleanupIntent:
    """Write-ahead record of a file mutation (DESIGN.md §7)."""

    inference_id: str
    action: CompletionAction
    source_path: str
    source_sha256: str
    target_path: Optional[str]
    members: tuple  # tuple[str, ...] companion files moved with the image


def derive_inference_id(
    route_id: str,
    capture_id: Optional[str],
    source_sha256: str,
    normalized_source_id: str,
    model_digest: str,
) -> str:
    """Derive the stable ``inferenceId`` for a logical key (DESIGN.md §4.3).

    The preferred key is ``routeId + captureId + modelDigest``; without a durable capture identity
    the key is ``routeId + normalizedSourceId + sourceSha256 + modelDigest``. The id is the
    BLAKE2b-128 of the key, base32 without padding, so re-discovering the same input under the
    same model yields the same id and a new model digest yields a new one.
    """
    if capture_id:
        key = f"1|{route_id}|{capture_id}|{model_digest}"
    else:
        key = f"2|{route_id}|{normalized_source_id}|{source_sha256}|{model_digest}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()
