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

from pathlib import Path
from typing import Any, Dict, Optional

import onnxruntime as ort

from image_processor.engine.decision import decide
from image_processor.engine.decode import DecodeLimits, decode_image
from image_processor.engine.families import family_for
from image_processor.types import BundleManifest, CachedBundle
from tools.update_goldens import round_number

#: Execution provider the tier-2 suite runs on. CUDA parity is the lab leg (D-IP-14).
PROVIDERS = ["CPUExecutionProvider"]


def open_session(bundle: CachedBundle) -> ort.InferenceSession:
    """Open a session on a staged bundle's graph.

    Args:
        bundle: The staged bundle.

    Returns:
        The session, pinned to :data:`PROVIDERS`.
    """
    return ort.InferenceSession(str(bundle.model_path), providers=PROVIDERS)


def infer(session: ort.InferenceSession, manifest: BundleManifest, data: bytes) -> tuple:
    """Run one encoded image through decode, preprocess, the graph, postprocess, and the rules.

    Args:
        session: The open session.
        manifest: The staged manifest.
        data: The encoded image bytes.

    Returns:
        A ``(normalized, decision)`` pair.
    """
    family = family_for(manifest)
    image = decode_image(data, DecodeLimits())
    feed = family.preprocess(image, manifest)
    names = [tensor.name for tensor in session.get_outputs()]
    outputs = dict(zip(names, session.run(names, feed)))
    normalized = family.postprocess(outputs, manifest, (image.shape[0], image.shape[1]))
    return normalized, decide(normalized, manifest.decision_rules)


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
