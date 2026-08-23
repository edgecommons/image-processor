"""Signed bundles built around the real models (DESIGN.md section 8, section 9).

Each real export is wrapped in the bundle the component actually installs: ``model.onnx``,
``labels.json``, ``transforms.json``, and a ``manifest.json`` that declares the tensors, the task
family and its parameters, the preprocessing, and the decision rules. The directory is packed and
signed by ``tools/make_bundle.py`` and staged through ``image_processor.bundles.stage_bundle``,
so the tier-2 suite exercises digest verification, Ed25519 signature verification, bounded
extraction, per-file digests, and family validation on real payloads rather than on fixtures.

The manifests here are the reference statement of how each export is described in the grammar the
task families read: what YOLOX's letterbox and BGR convention look like as a ``preprocess`` block,
how an SSD head's already-decoded tensors are named, and which output an auxiliary-head
segmentation export declares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from image_processor.bundles import BundleCache, generate_keypair, stage_bundle
from image_processor.engine.families import family_for
from image_processor.types import CachedBundle
from tests.live_models import labels as label_sets
from tools import build_anomaly_model
from tools.make_bundle import make_bundle

#: The bundle-manifest schema. WP1 owns ``schemas/model-bundle-manifest.schema.json``; until that
#: lands, WP2's stand-in carries the same DESIGN.md section 8 key names and is what the bundle
#: suite validates against.
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "bundles" / "manifest.schema.json"

#: One over 255, the scale that turns 8-bit samples into the unit range.
UNIT_SCALE = 1.0 / 255.0

#: ImageNet channel statistics, in the unit range, as torchvision exports expect them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#: The signing key id every tier-2 bundle is signed with.
KEY_ID = "tier2-test-publisher"

#: The rule both detectors carry: the frame is clear when nothing was found, or when nothing that
#: was found is a person. It exercises an ``any`` group, ``absent``, and a claim about every match.
DETECTION_RULES = {
    "pass": {
        "any": [
            {"path": "$.detections[*].label", "op": "absent"},
            {"path": "$.detections[*].label", "op": "!=", "value": "person"},
        ]
    },
    "threshold": 0.3,
    "outcomeOnPass": "CLEAR",
    "outcomeOnFail": "HOLD",
}


@dataclass(frozen=True)
class LiveModel:
    """One real model and everything needed to serve it.

    Attributes:
        key: Short name, which is also the golden file name.
        asset_id: The ``tests/assets.json`` id the graph comes from.
        filename: The graph's file name inside the asset directory.
        family: The task family the manifest declares.
        version: The bundle version.
        document: The authored ``manifest.json``, without the ``files`` digests that
            ``make_bundle`` computes.
        dataset: Which image set drives this model, ``"imagenette"``, ``"coco"``, or ``"visa"``.
    """

    key: str
    asset_id: str
    filename: str
    family: str
    version: str
    document: Dict[str, Any]
    dataset: str


def _base(model_id: str, version: str, family: str) -> Dict[str, Any]:
    """Start a manifest document with the fields every bundle shares.

    Args:
        model_id: The bundle's model id.
        version: The bundle version.
        family: The task family.

    Returns:
        The shared part of the manifest document.
    """
    return {
        "schemaVersion": 1,
        "modelId": model_id,
        "version": version,
        "minOnnxRuntime": "1.17.0",
        "providersPermitted": ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "providerPolicy": "preferred",
        "family": family,
        "warmup": [],
        "tolerances": {"absolute": 1e-4},
        "compatibilityKeys": {},
        "provenance": {"corpus": "tier-2", "pinnedBy": "tests/assets.json"},
        "keyId": KEY_ID,
        "transformVersion": "1",
    }


def _classifier(model_id: str, input_name: str, output_name: str, batch_axis: str,
                imagenet_labels: List[str]) -> Dict[str, Any]:
    """Build the manifest of an ImageNet classifier.

    Both classifiers take a 224 by 224 centre crop in the unit range, normalized by the ImageNet
    channel statistics, and return one logit per class.

    Args:
        model_id: The bundle's model id.
        input_name: The graph's input tensor name.
        output_name: The graph's output tensor name.
        batch_axis: The name the graph gives its dynamic batch dimension.
        imagenet_labels: The thousand ImageNet-1k class names.

    Returns:
        The manifest document.
    """
    document = _base(model_id, "1.0.0", "classification")
    document.update(
        {
            "inputs": [
                {"name": input_name, "dtype": "float32", "shape": [batch_axis, 3, 224, 224]}
            ],
            "outputs": [{"name": output_name, "dtype": "float32", "shape": [batch_axis, 1000]}],
            "dynamicBatch": True,
            "familyParams": {
                "labels": imagenet_labels,
                "activation": "softmax",
                "topK": 5,
                "scoreThreshold": 0.0,
                "outputName": output_name,
            },
            "preprocess": {
                "colorOrder": "RGB",
                "resize": {
                    "mode": "centerCrop",
                    "width": 224,
                    "height": 224,
                    "interpolation": "bilinear",
                },
                "scale": UNIT_SCALE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "layout": "NCHW",
                "dtype": "float32",
                "inputName": input_name,
            },
            "decisionRules": {
                "pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.5},
                "confidence": "$.classes[0].score",
                "threshold": 0.5,
                "outcomeOnPass": "CLEAR",
                "outcomeOnFail": "HOLD",
            },
            "maxResultItems": 5,
            "estimatedDeviceMiB": 256,
        }
    )
    return document


def _yolox(model_id: str, size: int, anchors: int) -> Dict[str, Any]:
    """Build the manifest of a YOLOX export.

    YOLOX exports its head undecoded: one row per grid cell of a box offset, a log size, an
    objectness, and eighty class scores, with objectness and the class scores already passed
    through a sigmoid inside the graph. The demo preprocessing letterboxes onto a grey 114 canvas
    at the top left, keeps the OpenCV BGR channel order, and does not normalize, so the manifest
    says exactly that.

    Args:
        model_id: The bundle's model id.
        size: The square model input size in pixels.
        anchors: How many grid cells the three strides describe at that size.

    Returns:
        The manifest document.
    """
    document = _base(model_id, "0.1.1rc0", "detection")
    document.update(
        {
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, size, size]}],
            "outputs": [{"name": "output", "dtype": "float32", "shape": [1, anchors, 85]}],
            "dynamicBatch": False,
            "familyParams": {
                "decode": "yoloxGrid",
                "strides": [8, 16, 32],
                "objectness": True,
                "objectnessActivation": "none",
                "scoreActivation": "none",
                "labels": label_sets.COCO_80,
                "scoreThreshold": 0.3,
                "iouThreshold": 0.45,
                "maxDetections": 20,
                "outputName": "output",
            },
            "preprocess": {
                "colorOrder": "BGR",
                "resize": {
                    "mode": "letterbox",
                    "width": size,
                    "height": size,
                    "padMode": "topLeft",
                    "padColor": [114, 114, 114],
                    "interpolation": "bilinear",
                },
                "scale": 1.0,
                "mean": 0.0,
                "std": 1.0,
                "layout": "NCHW",
                "dtype": "float32",
                "inputName": "images",
            },
            "decisionRules": DETECTION_RULES,
            "maxResultItems": 20,
            "estimatedDeviceMiB": 256,
        }
    )
    return document


def _ssd() -> Dict[str, Any]:
    """Build the manifest of the SSD-MobileNetV1 export.

    The export carries its own resizing and suppression: it takes the picture at its original size
    as unsigned bytes in NHWC order and returns boxes already decoded, already suppressed, and
    normalized to that picture, with one-based COCO category ids. The manifest therefore asks for
    no resize at all, which makes the model canvas the source image and the box mapping the
    identity.

    Returns:
        The manifest document.
    """
    document = _base("ssd-mobilenetv1", "12", "detection")
    document.update(
        {
            "inputs": [
                {"name": "inputs", "dtype": "uint8", "shape": [1, "height", "width", 3]}
            ],
            "outputs": [
                {"name": "detection_boxes", "dtype": "float32", "shape": [1, "detections", 4]},
                {"name": "detection_scores", "dtype": "float32", "shape": [1, "detections"]},
                {"name": "detection_classes", "dtype": "float32", "shape": [1, "detections"]},
                {"name": "num_detections", "dtype": "float32", "shape": [1]},
            ],
            "dynamicBatch": False,
            "familyParams": {
                "decode": "decodedBoxes",
                "outputNames": {
                    "boxes": "detection_boxes",
                    "scores": "detection_scores",
                    "classes": "detection_classes",
                    "count": "num_detections",
                },
                "boxFormat": "yxyx",
                "boxCoordinates": "normalized",
                "scoresLayout": "perBox",
                "scoreActivation": "none",
                "classIndexOffset": 1,
                "applyNms": False,
                "labels": label_sets.coco_90(),
                "scoreThreshold": 0.3,
                "maxDetections": 20,
            },
            "preprocess": {
                "colorOrder": "RGB",
                "resize": {"mode": "none"},
                "layout": "NHWC",
                "dtype": "uint8",
                "inputName": "inputs",
            },
            "decisionRules": DETECTION_RULES,
            "maxResultItems": 20,
            "estimatedDeviceMiB": 256,
        }
    )
    return document


def _fcn() -> Dict[str, Any]:
    """Build the manifest of the FCN-ResNet50 segmentation export.

    The graph returns two class maps, the head's ``out`` and the auxiliary classifier's ``aux``.
    The segmentation family reads one class map, so the manifest declares ``out`` and names it in
    ``familyParams.outputName``; ``aux`` is left undeclared and the family never looks at it.

    Returns:
        The manifest document.
    """
    document = _base("fcn-resnet50", "12", "segmentation")
    document.update(
        {
            "inputs": [
                {"name": "input", "dtype": "float32", "shape": ["batch", 3, "height", "width"]}
            ],
            "outputs": [
                {"name": "out", "dtype": "float32", "shape": ["batch", 21, "height", "width"]}
            ],
            "dynamicBatch": True,
            "familyParams": {
                "mode": "argmax",
                "outputLayout": "NCHW",
                "activation": "none",
                "outputName": "out",
                "labels": label_sets.VOC_21,
                "minPixels": 0,
            },
            "preprocess": {
                "colorOrder": "RGB",
                "resize": {
                    "mode": "stretch",
                    "width": 520,
                    "height": 520,
                    "interpolation": "bilinear",
                },
                "scale": UNIT_SCALE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "layout": "NCHW",
                "dtype": "float32",
                "inputName": "input",
            },
            "decisionRules": {
                "pass": {"path": "$.segments.person.fraction", "op": "<=", "value": 0.02},
                "confidence": "$.segments.person.fraction",
                "threshold": 0.02,
                "outcomeOnPass": "CLEAR",
                "outcomeOnFail": "HOLD",
            },
            "maxResultItems": 21,
            "estimatedDeviceMiB": 512,
        }
    )
    return document


def patchcore_document(size: int, threshold: float) -> Dict[str, Any]:
    """Build the manifest of the PatchCore anomaly export.

    PatchCore returns an anomaly map rather than a score, so the manifest reduces the map by its
    maximum, which is what the method scores an image by, and records the threshold the memory
    bank was built with.

    Args:
        size: The square model input size in pixels.
        threshold: The decision threshold in the map's own units.

    Returns:
        The manifest document.
    """
    document = _base("patchcore-visa-capsules", "1.0.0", "anomaly")
    document.update(
        {
            "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 3, size, size]}],
            "outputs": [{"name": "anomaly_map", "dtype": "float32", "shape": [1, 1, size, size]}],
            "dynamicBatch": False,
            "familyParams": {
                "source": "mapMax",
                "activation": "none",
                "direction": "higherIsAnomalous",
                "threshold": threshold,
                "outputName": "anomaly_map",
            },
            "preprocess": build_anomaly_model.preprocess_block(size),
            "decisionRules": {
                "pass": {"path": "$.anomaly.anomalous", "op": "==", "value": False},
                "confidence": "$.anomaly.score",
                "threshold": "$.anomaly.threshold",
                "outcomeOnPass": "CLEAR",
                "outcomeOnFail": "FAIL",
            },
            "maxResultItems": 1,
            "estimatedDeviceMiB": 512,
        }
    )
    return document


def live_models(imagenet_labels: List[str], patchcore: Optional[Dict[str, Any]] = None) -> Dict[str, LiveModel]:
    """Describe every model of the tier-2 corpus.

    Args:
        imagenet_labels: The thousand ImageNet-1k class names, read from the pinned synset.
        patchcore: The PatchCore build record, with ``imageSize`` and ``threshold``, or ``None``
            when the anomaly model has not been built on this machine.

    Returns:
        The models by key.
    """
    models = {
        "mobilenetv2-12": LiveModel(
            key="mobilenetv2-12",
            asset_id="model-mobilenetv2-12",
            filename="mobilenetv2-12.onnx",
            family="classification",
            version="1.0.0",
            document=_classifier("mobilenetv2", "input", "output", "batch_size", imagenet_labels),
            dataset="imagenette",
        ),
        "resnet50-v1-12": LiveModel(
            key="resnet50-v1-12",
            asset_id="model-resnet50-v1-12",
            filename="resnet50-v1-12.onnx",
            family="classification",
            version="1.0.0",
            document=_classifier(
                "resnet50-v1", "data", "resnetv17_dense0_fwd", "N", imagenet_labels
            ),
            dataset="imagenette",
        ),
        "yolox-nano": LiveModel(
            key="yolox-nano",
            asset_id="model-yolox-nano",
            filename="yolox_nano.onnx",
            family="detection",
            version="0.1.1rc0",
            document=_yolox("yolox-nano", 416, 3549),
            dataset="coco",
        ),
        "yolox-s": LiveModel(
            key="yolox-s",
            asset_id="model-yolox-s",
            filename="yolox_s.onnx",
            family="detection",
            version="0.1.1rc0",
            document=_yolox("yolox-s", 640, 8400),
            dataset="coco",
        ),
        "ssd-mobilenetv1-12": LiveModel(
            key="ssd-mobilenetv1-12",
            asset_id="model-ssd-mobilenetv1-12",
            filename="ssd_mobilenet_v1_12.onnx",
            family="detection",
            version="12",
            document=_ssd(),
            dataset="coco",
        ),
        "fcn-resnet50-12": LiveModel(
            key="fcn-resnet50-12",
            asset_id="model-fcn-resnet50-12",
            filename="fcn-resnet50-12.onnx",
            family="segmentation",
            version="12",
            document=_fcn(),
            dataset="coco",
        ),
    }
    if patchcore is not None:
        models["patchcore-visa-capsules"] = LiveModel(
            key="patchcore-visa-capsules",
            asset_id="model-patchcore-visa-capsules",
            filename="model.onnx",
            family="anomaly",
            version="1.0.0",
            document=patchcore_document(int(patchcore["imageSize"]), float(patchcore["threshold"])),
            dataset="visa",
        )
    return models


def write_source(model: LiveModel, graph: Path, src_dir: Path) -> Path:
    """Lay out one bundle source directory.

    Args:
        model: The model being packed.
        graph: The ONNX graph to copy in.
        src_dir: The directory to build, which is created.

    Returns:
        The directory written.
    """
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "model.onnx").write_bytes(graph.read_bytes())
    params = model.document.get("familyParams", {})
    payload = params.get("labels") or {"classes": params.get("numClasses", 1)}
    (src_dir / "labels.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    transforms = {
        "transformVersion": model.document["transformVersion"],
        "preprocess": model.document["preprocess"],
    }
    (src_dir / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
    )
    (src_dir / "manifest.json").write_text(
        json.dumps(model.document, indent=2) + "\n", encoding="utf-8"
    )
    return src_dir


def stage(model: LiveModel, graph: Path, workdir: Path, cache_root: Path,
          private_key: bytes, public_key: bytes) -> CachedBundle:
    """Pack, sign, and stage one model, exactly as the component would install it.

    Args:
        model: The model to bundle.
        graph: The fetched ONNX graph.
        workdir: A scratch directory for the source tree, the tarball, and staging.
        cache_root: The content-addressed cache to promote into.
        private_key: The Ed25519 private key PEM to sign ``manifest.json`` with.
        public_key: The raw public key the staging call trusts under :data:`KEY_ID`.

    Returns:
        The staged, verified bundle.
    """
    src = write_source(model, graph, workdir / "src" / model.key)
    archive = workdir / f"{model.key}.tar"
    digest = make_bundle(
        src_dir=src,
        out_path=archive,
        key=private_key,
        key_id=KEY_ID,
        compress=False,
        schema_path=SCHEMA_PATH,
    )
    cache = BundleCache(cache_root, schema_path=SCHEMA_PATH)
    return stage_bundle(
        uri=str(archive),
        digest=digest,
        staging_root=workdir / "staging",
        cache=cache,
        signing_required=True,
        trusted_keys={KEY_ID: public_key},
        schema_path=SCHEMA_PATH,
        model_id=model.document["modelId"],
        version=model.document["version"],
        available_providers=["CPUExecutionProvider"],
        validators=[lambda manifest: family_for(manifest).validate_manifest(manifest)],
    )


def keypair() -> tuple:
    """Generate the Ed25519 keypair the tier-2 bundles are signed with.

    A fresh key per run keeps a signing key out of the repository and still proves the whole
    signed path: sign, trust by key id, verify before extraction.

    Returns:
        A ``(private_pem, raw_public_key)`` pair.
    """
    private_pem, _, public_raw = generate_keypair()
    return private_pem, public_raw
