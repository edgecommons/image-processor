"""One image through the real chain, and the record a golden is made of.

There is no executor cell yet, so the suite runs the same sequence a cell will: decode the file
with the component's decoder, build the feed with the family's ``preprocess``, run the graph on
``CPUExecutionProvider``, read the tensors with the family's ``postprocess``, and evaluate the
manifest's decision rules. Nothing about the model is inferred here; every choice comes from the
staged manifest.

The record a run produces is deliberately small. It holds what a golden compares, and nothing
that would make the committed file large: scores and boxes rounded to six places, per-class pixel
fractions rather than masks, and the decision the rules reached.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import onnxruntime as ort

from image_processor.engine.cell_main import onnxruntime_module, provider_options
from image_processor.engine.decision import decide
from image_processor.engine.decode import DecodeLimits, decode_image
from image_processor.engine.families import family_for
from image_processor.engine.protocol import CPU_PROVIDER, CUDA_PROVIDER
from image_processor.types import BundleManifest, CachedBundle
from tools.update_goldens import round_number

#: The provider the committed goldens were produced on. Every other provider is asserted against
#: them, which is what makes CPU-to-CUDA parity a comparison rather than two separate baselines.
PROVIDERS = [CPU_PROVIDER]

#: The arena bound a CUDA session in this suite is given, in MiB. The tier-2 graphs are small; the
#: bound is here so a lab run cannot take the whole card away from anything else on it.
CUDA_ARENA_MIB = 4096


def open_session(
    bundle: CachedBundle, provider: str = CPU_PROVIDER, device_id: int = 0
) -> ort.InferenceSession:
    """Open a session on a staged bundle's graph, on one execution provider.

    The runtime is imported through the executor cell's own accessor, which preloads the NVIDIA
    libraries the CUDA provider links against; without that the provider library fails to load and
    the session lands on CPU. The provider options are the cell's own as well
    (``engine.cell_main.provider_options``), so a CUDA session here is configured the way a CUDA
    session in a cell is: the device ordinal, the arena bound, and ``kSameAsRequested`` so the arena
    does not double itself (DESIGN.md 10.2).

    The assignment is then checked, because ONNX Runtime does not raise when it cannot build a
    provider: it drops it and runs on CPU. A parity leg that accepted that would compare CPU
    against CPU and report agreement.

    Args:
        bundle: The staged bundle.
        provider: The execution provider to request.
        device_id: The CUDA device ordinal, when the provider is CUDA.

    Returns:
        The session, assigned the provider that was asked for.

    Raises:
        RuntimeError: The session was not assigned the requested provider.
    """
    runtime = onnxruntime_module()
    requested = [provider]
    options = provider_options(
        requested, device_id, CUDA_ARENA_MIB if provider == CUDA_PROVIDER else None
    )
    session_options = runtime.SessionOptions()
    session_options.log_severity_level = 3
    session = runtime.InferenceSession(
        str(bundle.model_path),
        sess_options=session_options,
        providers=requested,
        provider_options=[dict(entry) for entry in options],
    )
    assigned = list(session.get_providers())
    if provider not in assigned:
        raise RuntimeError(
            f"{bundle.manifest.model_id} was asked for {provider} and the session was assigned "
            f"{assigned}. ONNX Runtime falls back silently, so a run that accepted this would "
            "compare CPU against CPU and call it parity."
        )
    return session


def infer(session: ort.InferenceSession, manifest: BundleManifest, data: bytes) -> tuple:
    """Run one encoded image through decode, preprocess, the graph, postprocess, and the rules.

    Args:
        session: The open session.
        manifest: The staged manifest.
        data: The encoded image bytes.

    Returns:
        A ``(normalized, decision, session_ms)`` triple. The milliseconds cover the graph alone,
        not decode or the transforms, so a CPU reading and a CUDA reading compare like for like.
    """
    family = family_for(manifest)
    image = decode_image(data, DecodeLimits())
    feed = family.preprocess(image, manifest)
    names = [tensor.name for tensor in session.get_outputs()]
    started = time.perf_counter()
    outputs = dict(zip(names, session.run(names, feed)))
    session_ms = (time.perf_counter() - started) * 1000.0
    normalized = family.postprocess(outputs, manifest, (image.shape[0], image.shape[1]))
    return normalized, decide(normalized, manifest.decision_rules), session_ms


def _decision_record(decision) -> Dict[str, Any]:
    """Reduce a decision to the fields a golden records.

    Args:
        decision: The :class:`~image_processor.types.Decision` the rules produced.

    Returns:
        The recorded decision.
    """
    return {
        "outcome": decision.outcome.value,
        "pass": bool(decision.passed),
        "confidence": round_number(decision.confidence),
        "threshold": round_number(decision.threshold),
        "rule": decision.rule,
    }


def record(name: str, normalized, decision) -> Dict[str, Any]:
    """Build the golden record for one image.

    Args:
        name: How the image is named in the golden, relative to its dataset.
        normalized: The family's normalized output.
        decision: The decision the rules reached.

    Returns:
        The record, with every number rounded so the committed file is small and stable.

    Raises:
        ValueError: When the family has no recording, which means a family was added without one.
    """
    family = normalized.family.value
    entry: Dict[str, Any] = {"image": name, "decision": _decision_record(decision)}
    if family == "classification":
        entry["classes"] = [
            {"label": item.label, "index": item.index, "score": round_number(item.score)}
            for item in normalized.classes
        ]
    elif family == "detection":
        entry["detections"] = [
            {
                "label": item.label,
                "index": item.index,
                "score": round_number(item.score),
                "box": [round_number(value) for value in item.box],
            }
            for item in normalized.detections
        ]
    elif family == "segmentation":
        entry["segments"] = {
            label: {"pixels": int(value["pixels"]), "fraction": round_number(value["fraction"])}
            for label, value in sorted(normalized.segments.items())
        }
    elif family == "anomaly":
        entry["anomaly"] = {
            "score": round_number(normalized.anomaly["score"]),
            "threshold": round_number(normalized.anomaly["threshold"]),
            "anomalous": bool(normalized.anomaly["anomalous"]),
        }
    else:  # pragma: no cover - Family has exactly four members
        raise ValueError(f"no golden record for family {family!r}")
    return entry


def golden_document(model, records, tolerances: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the golden file for one model.

    Args:
        model: The :class:`~tests.live_models.bundles.LiveModel` the records came from.
        records: The per-image records, in a fixed order.
        tolerances: The tolerances the comparison applies, recorded so the file says how it is
            read.

    Returns:
        The golden document.
    """
    return {
        "model": model.key,
        "family": model.family,
        "bundle": {
            "modelId": model.document["modelId"],
            "version": model.document["version"],
            "assetId": model.asset_id,
        },
        "provider": PROVIDERS[0],
        "onnxruntime": ort.__version__,
        "tolerances": tolerances or {},
        "images": list(records),
    }


def read_image(path: Path) -> bytes:
    """Read one corpus image.

    Args:
        path: The image file.

    Returns:
        The encoded bytes.
    """
    return Path(path).read_bytes()
