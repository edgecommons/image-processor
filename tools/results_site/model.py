"""The document the site is: runs, models, entries, and the merge of one run into another.

``data.js`` is a single assignment to ``window.RESULTS``. Everything the page draws is in it, so
the site never fetches anything and works identically from ``file://`` and from a server.

The document is keyed by run. A run is one execution provider on one device -- ``cpu``, or
``cuda-nvidia-geforce-rtx-5080`` -- and building the same run again replaces it rather than
appending to it, which is what lets a CPU build and a CUDA build of the same corpus sit in one
site and be compared model by model.

Shape::

    {
      schemaVersion, generated,
      runs:      [{runId, provider, gpu, host, onnxruntime, generated}],
      models:    [{key, runId, suite, family, corpus, card{...}}],
      entries:   [{id, runId, modelKey, image{...}, timings{...}, decision, resultBody, summary}],
      refusals:  [{runId, name, bytes, expected, observed, refused}]
    }

``models`` carries one row per run and model rather than one per model, because the provider
assignment and the device are properties of the run: the same MobileNetV2 is a CPU session in one
row and a CUDA session in another, and a site that stated only one of them would be claiming a
device the numbers next to it did not come from.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from tools.results_site import SITE_SCHEMA_VERSION

#: The prefix ``data.js`` assigns through, and what the loader looks for when merging.
ASSIGNMENT = "window.RESULTS = "

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Reduce a string to lowercase words joined by single hyphens.

    Args:
        text: Any string.

    Returns:
        The slug, which is empty only when the input carried no letters or digits.
    """
    return _NON_SLUG.sub("-", str(text).lower()).strip("-")


def run_id(provider: str, gpu: Optional[str]) -> str:
    """Name the run one provider on one device produces.

    The name is derived rather than generated, so building the same leg twice replaces it. That is
    the whole of the merge rule: a CPU rebuild refreshes the CPU numbers and leaves the CUDA ones
    alone.

    Args:
        provider: The execution provider the session was assigned.
        gpu: The device name, or ``None`` on a CPU session.

    Returns:
        The run id.
    """
    short = slug(provider.replace("ExecutionProvider", "")) or "unknown"
    return f"{short}-{slug(gpu)}" if gpu else short


def entry_id(run: str, model_key: str, name: str) -> str:
    """Name one entry, stably.

    The id goes in the URL fragment, so it has to survive a rebuild: the same picture under the
    same model in the same run keeps its link.

    Args:
        run: The run id.
        model_key: The model key.
        name: The image name, relative to its corpus.

    Returns:
        A twelve-character id.
    """
    key = f"{run}|{model_key}|{name}".encode("utf-8")
    return "e" + hashlib.blake2b(key, digest_size=6).hexdigest()


def file_slug(name: str) -> str:
    """Turn a corpus-relative image name into one safe path segment.

    Two VisA images are called ``000.JPG``, one in ``Normal`` and one in ``Anomaly``, so the last
    path element alone is not a name. Flattening the whole relative path keeps them apart and
    keeps the site one directory deep per model.

    Args:
        name: The corpus-relative image name, with ``/`` separators.

    Returns:
        The flattened name.
    """
    return str(name).replace("\\", "/").strip("/").replace("/", "__")


def summarize(body: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a result body to the one line the gallery and the summary card read.

    Args:
        body: The wire-shaped result body.

    Returns:
        The summary, whose shape follows the task family: the top class for a classification, the
        labels found for a detection, the widest classes for a segmentation, the reading and the
        threshold for an anomaly.
    """
    outputs = body.get("outputs") or {}
    family = outputs.get("family") or "unknown"
    if family == "classification":
        classes = outputs.get("classes") or []
        top = classes[0] if classes else None
        return {
            "family": family,
            "label": top["label"] if top else None,
            "score": round(float(top["score"]), 6) if top else None,
            "classes": len(classes),
        }
    if family == "detection":
        detections = outputs.get("detections") or []
        labels: List[str] = []
        for item in detections:
            if item["label"] not in labels:
                labels.append(item["label"])
        return {
            "family": family,
            "count": len(detections),
            "labels": labels,
            "topScore": round(float(detections[0]["score"]), 6) if detections else None,
        }
    if family == "segmentation":
        segments = outputs.get("segments") or {}
        widest = sorted(
            segments.items(), key=lambda item: (-float(item[1].get("fraction", 0.0)), item[0])
        )[:3]
        return {
            "family": family,
            "classes": len(segments),
            "top": [
                {"label": label, "fraction": round(float(value.get("fraction", 0.0)), 6)}
                for label, value in widest
            ],
        }
    anomaly = outputs.get("anomaly") or {}
    return {
        "family": family,
        "score": round(float(anomaly.get("score", 0.0)), 6),
        "threshold": round(float(anomaly.get("threshold", 0.0)), 6),
        "anomalous": bool(anomaly.get("anomalous", False)),
        "direction": anomaly.get("direction"),
    }


def model_card(manifest: Any, providers: Iterable[str], gpu: Optional[str]) -> Dict[str, Any]:
    """Describe the model generation one run of one model actually loaded.

    ``providers`` is the session assignment ONNX Runtime reports, not the preference that was
    asked for. The runtime drops a provider it cannot build and carries on, so the assignment is
    the only statement about where the work ran that is worth printing.

    Args:
        manifest: The staged :class:`~image_processor.types.BundleManifest`.
        providers: The session actual provider assignment.
        gpu: The device name NVML reported, or ``None``.

    Returns:
        The card.
    """
    inputs = list(manifest.inputs)
    first = inputs[0] if inputs else None
    preprocess = dict(manifest.preprocess or {})
    resize = dict(preprocess.get("resize") or {})
    return {
        "modelId": manifest.model_id,
        "version": manifest.version,
        "transformVersion": manifest.transform_version,
        "providers": list(providers),
        "gpu": gpu,
        "inputName": first.name if first else None,
        "inputShape": [str(value) for value in first.shape] if first else [],
        "inputDtype": first.dtype if first else None,
        "providersPermitted": list(manifest.providers_permitted),
        "providerPolicy": manifest.provider_policy,
        "maxResultItems": manifest.max_result_items,
        "estimatedDeviceMiB": manifest.estimated_device_mib,
        "preprocess": {
            "colorOrder": preprocess.get("colorOrder"),
            "resize": resize.get("mode"),
            "size": (
                [resize["width"], resize["height"]]
                if resize.get("width") and resize.get("height")
                else None
            ),
            "layout": preprocess.get("layout"),
            "dtype": preprocess.get("dtype"),
            "scale": preprocess.get("scale"),
        },
        "signed": bool(manifest.key_id),
        "keyId": manifest.key_id,
    }


def preprocess_summary(card: Dict[str, Any]) -> str:
    """State one card preprocessing in a single line.

    Args:
        card: A card from :func:`model_card`.

    Returns:
        The one-line summary, for a card that has no room for a block.
    """
    block = card.get("preprocess") or {}
    parts = [str(block.get("colorOrder") or "?"), str(block.get("resize") or "?")]
    size = block.get("size")
    if size:
        parts.append(f"{size[0]}x{size[1]}")
    parts.append(str(block.get("layout") or "?"))
    parts.append(str(block.get("dtype") or "?"))
    return " ".join(parts)


def document(
    generated: str,
    runs: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
    refusals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble one build into the site document.

    Args:
        generated: When the build ran, ISO 8601 in UTC.
        runs: The run rows this build produced. A build produces exactly one.
        models: The model rows, one per run and model.
        entries: The per-image entries.
        refusals: The bad-input rows, when the build ran the synthetic suite.

    Returns:
        The document.
    """
    return {
        "schemaVersion": SITE_SCHEMA_VERSION,
        "generated": generated,
        "runs": list(runs),
        "models": list(models),
        "entries": list(entries),
        "refusals": list(refusals or []),
    }


class MergeError(Exception):
    """An existing ``data.js`` cannot be merged into."""


def parse_data_js(text: str) -> Dict[str, Any]:
    """Read a ``data.js`` back into its document.

    Args:
        text: The file content.

    Returns:
        The parsed document.

    Raises:
        MergeError: The file is not a ``data.js`` this tool wrote, or its schema version is one
            this tool does not know how to merge into.
    """
    start = text.find(ASSIGNMENT)
    if start < 0:
        raise MergeError(f"the file does not assign {ASSIGNMENT.strip()}")
    payload = text[start + len(ASSIGNMENT):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MergeError(f"the assigned value is not JSON: {exc}") from exc
    version = parsed.get("schemaVersion")
    if version != SITE_SCHEMA_VERSION:
        raise MergeError(
            f"the site was built with data model {version}, this tool writes {SITE_SCHEMA_VERSION}"
        )
    return parsed


def merge(existing: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a fresh build into a site that already holds other runs.

    Every row the fresh build carries names its run, so merging is a replacement of exactly those
    run ids and nothing else. A CUDA build merged onto a CPU site leaves every CPU number where it
    was; a second CPU build replaces the first.

    Args:
        existing: The document read out of the site.
        fresh: The document this build produced.

    Returns:
        The merged document.
    """
    replaced = {run["runId"] for run in fresh["runs"]}

    def kept(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [row for row in rows if row.get("runId") not in replaced]

    runs = kept(existing.get("runs", [])) + list(fresh["runs"])
    models = kept(existing.get("models", [])) + list(fresh["models"])
    entries = kept(existing.get("entries", [])) + list(fresh["entries"])
    refusals = kept(existing.get("refusals", [])) + list(fresh.get("refusals", []))
    return {
        "schemaVersion": SITE_SCHEMA_VERSION,
        "generated": fresh["generated"],
        "runs": sorted(runs, key=lambda row: row["runId"]),
        "models": sorted(models, key=lambda row: (row["runId"], row["suite"], row["key"])),
        "entries": entries,
        "refusals": refusals,
    }