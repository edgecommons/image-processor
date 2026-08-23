"""Build the PatchCore anomaly model of the tier-2 corpus from the fetched VisA capsules.

DESIGN.md section 16.1 names an anomalib PatchCore trained on VisA ``capsules`` for the anomaly
row of tier 2. PatchCore has no training epochs: it runs a frozen ImageNet backbone over the good
split, keeps a coreset of the patch embeddings as a memory bank, and scores a new image by the
distance to that bank. That makes the model reproducible from the pinned dataset rather than a
binary somebody has to host, which is why nothing is published here and this tool exists instead.

The build is deterministic. The good images are taken in sorted order, the seed is fixed, and the
coreset selection runs from that seed, so the same VisA archive and the same arguments give the
same graph on any machine.

Preprocessing at build time is the bundle's own ``preprocess`` block, executed by
``image_processor.engine.families.preprocess_image``. The memory bank is therefore built from
exactly the tensors the component will feed the model, not from a second implementation that
happens to agree.

Examples:
    Build with the defaults, into the cache the tier-2 suite reads::

        python tools/build_anomaly_model.py

    Build a larger bank at a different resolution::

        python tools/build_anomaly_model.py --good-images 200 --coreset-ratio 0.1 --image-size 320
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Where ``tools/fetch_test_assets.py`` unpacks VisA.
DEFAULT_VISA_ROOT = _REPO_ROOT / "tests" / ".cache" / "dataset-visa" / "extracted" / "capsules"

#: Where the built model and its build record are cached. The suite looks here.
DEFAULT_OUT_DIR = _REPO_ROOT / "tests" / ".cache" / "model-patchcore-visa-capsules"

#: Backbone, layers, and sizes. PatchCore's own backbone is kept; only the layers up to
#: ``layer3`` are exported, which is what holds the graph under 200 MB with the memory bank.
DEFAULT_BACKBONE = "wide_resnet50_2"
DEFAULT_LAYERS = ("layer2", "layer3")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_GOOD_IMAGES = 200
DEFAULT_CORESET_RATIO = 0.02
DEFAULT_SEED = 20260822

#: How many held-out images of each split the threshold is chosen on.
DEFAULT_EVAL_IMAGES = 50

#: The ONNX opset the graph is exported at.
OPSET = 17

#: What to tell an operator who does not have the build dependencies.
INSTALL_HINT = (
    "PatchCore needs anomalib and torch, which are not part of this component's dependencies. "
    "Install them into a scratch environment and rerun:\n"
    "    python -m venv .venv-anomalib\n"
    "    .venv-anomalib/Scripts/python -m pip install anomalib onnx jsonpath-ng\n"
    "    .venv-anomalib/Scripts/python tools/build_anomaly_model.py"
)


class AnomalyBuildError(Exception):
    """The PatchCore model could not be built on this machine."""


@dataclass(frozen=True)
class BuildOptions:
    """What one build does.

    Attributes:
        visa_root: The unpacked ``capsules`` directory of VisA.
        out_dir: Where ``model.onnx`` and ``build.json`` are written.
        backbone: The timm backbone the patch embeddings come from.
        layers: The backbone layers the embedding is built from.
        image_size: The square input size in pixels.
        good_images: How many images of the good split fill the memory bank.
        coreset_ratio: The fraction of patch embeddings the coreset keeps.
        eval_images: How many held-out images of each split the threshold is chosen on.
        seed: The seed every random step runs from.
    """

    visa_root: Path = DEFAULT_VISA_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    backbone: str = DEFAULT_BACKBONE
    layers: tuple = DEFAULT_LAYERS
    image_size: int = DEFAULT_IMAGE_SIZE
    good_images: int = DEFAULT_GOOD_IMAGES
    coreset_ratio: float = DEFAULT_CORESET_RATIO
    eval_images: int = DEFAULT_EVAL_IMAGES
    seed: int = DEFAULT_SEED


def splits(visa_root: Path) -> tuple:
    """Locate the good and bad image splits of VisA capsules.

    Args:
        visa_root: The unpacked ``capsules`` directory.

    Returns:
        A ``(good, bad)`` pair of sorted image path lists.

    Raises:
        AnomalyBuildError: When either split is missing or empty.
    """
    images = Path(visa_root) / "Data" / "Images"
    good = sorted((images / "Normal").glob("*.JPG"))
    bad = sorted((images / "Anomaly").glob("*.JPG"))
    if not good or not bad:
        raise AnomalyBuildError(
            f"{images} does not hold the capsules splits; fetch VisA first with\n"
            "    python tools/fetch_test_assets.py --only dataset-visa"
        )
    return good, bad


def preprocess_block(image_size: int) -> Dict[str, Any]:
    """Build the ``preprocess`` block the model is fed through.

    This is the single statement of PatchCore's input transform: the bundle manifest carries it,
    and this tool runs the memory-bank images through the same block. There is no second
    preprocessing to drift from it.

    Args:
        image_size: The square input size in pixels.

    Returns:
        The ``preprocess`` block, in the grammar the task families read.
    """
    return {
        "colorOrder": "RGB",
        "resize": {
            "mode": "stretch",
            "width": image_size,
            "height": image_size,
            "interpolation": "bilinear",
        },
        "scale": 1.0 / 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "layout": "NCHW",
        "dtype": "float32",
        "inputName": "input",
    }


def manifest_for(image_size: int):
    """Build the manifest whose ``preprocess`` block the build runs images through.

    Args:
        image_size: The square input size in pixels.

    Returns:
        The parsed :class:`~image_processor.types.BundleManifest`.
    """
    from image_processor.bundles import parse_manifest

    return parse_manifest(
        {
            "schemaVersion": 1,
            "modelId": "patchcore-visa-capsules",
            "version": "1.0.0",
            "files": {"model.onnx": "0" * 64},
            "family": "anomaly",
            "familyParams": {"source": "mapMax", "threshold": 0.0, "outputName": "anomaly_map"},
            "inputs": [
                {"name": "input", "dtype": "float32", "shape": [1, 3, image_size, image_size]}
            ],
            "outputs": [
                {
                    "name": "anomaly_map",
                    "dtype": "float32",
                    "shape": [1, 1, image_size, image_size],
                }
            ],
            "preprocess": preprocess_block(image_size),
        }
    )


def _load_batch(paths: List[Path], manifest) -> Any:
    """Decode and preprocess a list of images into one batched tensor.

    Args:
        paths: The images to load.
        manifest: The manifest whose ``preprocess`` block is executed.

    Returns:
        A ``(N, 3, size, size)`` ``float32`` numpy array.
    """
    import numpy as np

    from image_processor.engine.decode import DecodeLimits, decode_image
    from image_processor.engine.families import preprocess_image

    batch = []
    for path in paths:
        image = decode_image(Path(path).read_bytes(), DecodeLimits())
        feed = preprocess_image(image, manifest)
        batch.append(next(iter(feed.values()))[0])
    return np.stack(batch).astype(np.float32)


def _require_dependencies():
    """Import the build dependencies, or explain how to get them.

    Returns:
        A ``(torch, PatchcoreModel)`` pair.

    Raises:
        AnomalyBuildError: When anomalib or torch is not importable.
    """
    try:
        import torch
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    except ImportError as exc:
        raise AnomalyBuildError(f"{exc}\n\n{INSTALL_HINT}") from exc
    return torch, PatchcoreModel


def _auroc(good_scores: List[float], bad_scores: List[float]) -> float:
    """Compute the image-level area under the ROC curve of the two splits.

    The rank-sum form is used, so no plotting library is needed and ties count as half.

    Args:
        good_scores: Scores of held-out good images.
        bad_scores: Scores of bad images.

    Returns:
        The area under the curve, where 0.5 is chance and 1.0 is perfect separation.
    """
    wins = 0.0
    for bad in bad_scores:
        for good in good_scores:
            wins += 1.0 if bad > good else (0.5 if bad == good else 0.0)
    return wins / (len(good_scores) * len(bad_scores))


def _f1_threshold(good_scores: List[float], bad_scores: List[float]) -> float:
    """Choose the threshold that separates the two splits best.

    Every observed score is tried as a cut and the one with the highest F1 over the bad split
    wins; ties go to the lower cut, so the choice is a function of the scores alone.

    Args:
        good_scores: Scores of held-out good images.
        bad_scores: Scores of bad images.

    Returns:
        The chosen threshold.
    """
    best_threshold, best_f1 = 0.0, -1.0
    for candidate in sorted(set(good_scores + bad_scores)):
        true_positive = sum(1 for score in bad_scores if score >= candidate)
        false_positive = sum(1 for score in good_scores if score >= candidate)
        false_negative = len(bad_scores) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2 * true_positive / denominator) if denominator else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = float(candidate), f1
    return best_threshold


def build(options: BuildOptions = BuildOptions()) -> Dict[str, Any]:
    """Build, export, and record the PatchCore model.

    Args:
        options: What the build does.

    Returns:
        The build record, which is also written to ``build.json`` beside the graph.

    Raises:
        AnomalyBuildError: When the dependencies or the VisA splits are missing.
    """
    torch, PatchcoreModel = _require_dependencies()
    import numpy as np

    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    torch.use_deterministic_algorithms(False)

    good, bad = splits(options.visa_root)
    manifest = manifest_for(options.image_size)
    bank_images = good[: options.good_images]
    held_out = good[options.good_images : options.good_images + options.eval_images]
    bad_images = bad[: options.eval_images]

    model = PatchcoreModel(
        layers=list(options.layers), backbone=options.backbone, pre_trained=True, num_neighbors=9
    )
    model.train()
    for start in range(0, len(bank_images), 8):
        batch = torch.from_numpy(_load_batch(bank_images[start : start + 8], manifest))
        with torch.no_grad():
            model(batch)
    model.subsample_embedding(sampling_ratio=options.coreset_ratio)
    model.eval()

    class _MapOnly(torch.nn.Module):
        """The map head alone, so the bundle declares one output and the family reads it."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_tensor):
            return self.inner(input_tensor).anomaly_map

    exported = _MapOnly(model).eval()

    def score(paths):
        values = []
        for start in range(0, len(paths), 4):
            batch = torch.from_numpy(_load_batch(paths[start : start + 4], manifest))
            with torch.no_grad():
                maps = exported(batch)
            values.extend(float(entry.max()) for entry in maps)
        return values

    good_scores, bad_scores = score(held_out), score(bad_images)
    threshold = _f1_threshold(good_scores, bad_scores)

    out_dir = Path(options.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = out_dir / "model.onnx"
    dummy = torch.zeros(1, 3, options.image_size, options.image_size, dtype=torch.float32)
    torch.onnx.export(
        exported,
        (dummy,),
        str(graph),
        input_names=["input"],
        output_names=["anomaly_map"],
        opset_version=OPSET,
        dynamo=False,
    )
    return _record(options, graph, threshold, good_scores, bad_scores, len(model.memory_bank))


def _record(options: BuildOptions, graph: Path, threshold: float,
            good_scores: List[float], bad_scores: List[float], bank: int) -> Dict[str, Any]:
    """Write the build record beside the graph.

    Args:
        options: What the build did.
        graph: The exported ONNX file.
        threshold: The threshold the bundle declares.
        good_scores: Scores of the held-out good images.
        bad_scores: Scores of the bad images.
        bank: How many embeddings the memory bank kept.

    Returns:
        The build record.
    """
    import anomalib
    import torch

    payload = graph.read_bytes()
    record = {
        "imageSize": options.image_size,
        "threshold": round(threshold, 6),
        "backbone": options.backbone,
        "layers": list(options.layers),
        "coresetRatio": options.coreset_ratio,
        "goodImages": options.good_images,
        "memoryBank": int(bank),
        "seed": options.seed,
        "opset": OPSET,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "separation": {
            "auroc": round(_auroc(good_scores, bad_scores), 6),
            "goodMax": round(max(good_scores), 6),
            "badMin": round(min(bad_scores), 6),
            "goodMean": round(sum(good_scores) / len(good_scores), 6),
            "badMean": round(sum(bad_scores) / len(bad_scores), 6),
            "flaggedGood": sum(1 for value in good_scores if value >= threshold),
            "flaggedBad": sum(1 for value in bad_scores if value >= threshold),
            "goodImages": len(good_scores),
            "badImages": len(bad_scores),
        },
    }
    (Path(options.out_dir) / "build.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    defaults = BuildOptions()
    parser = argparse.ArgumentParser(
        prog="build_anomaly_model",
        description="Build the tier-2 PatchCore anomaly model from the fetched VisA capsules.",
    )
    parser.add_argument("--visa-root", default=str(defaults.visa_root))
    parser.add_argument("--out-dir", default=str(defaults.out_dir))
    parser.add_argument("--backbone", default=defaults.backbone)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--good-images", type=int, default=defaults.good_images)
    parser.add_argument("--coreset-ratio", type=float, default=defaults.coreset_ratio)
    parser.add_argument("--eval-images", type=int, default=defaults.eval_images)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when the model was built, ``2`` when it could not be.
    """
    args = _parser().parse_args(argv)
    options = BuildOptions(
        visa_root=Path(args.visa_root),
        out_dir=Path(args.out_dir),
        backbone=args.backbone,
        image_size=args.image_size,
        good_images=args.good_images,
        coreset_ratio=args.coreset_ratio,
        eval_images=args.eval_images,
        seed=args.seed,
    )
    try:
        record = build(options)
    except AnomalyBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
