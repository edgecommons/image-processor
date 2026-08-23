"""The label sets the tier-2 bundles carry.

A bundle names its classes in ``labels.json`` and repeats them in ``familyParams.labels``, because
that is what the task families read (``family_labels``). Three label sets cover the corpus:
ImageNet-1k for the two classifiers, COCO for the two detector conventions, and Pascal VOC for the
segmentation model.

The ImageNet names come from the pinned ``synset.txt`` rather than from a copy in this repository,
so the label order is the one the models were exported against and is checked by SHA-256 like
every other asset.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

#: COCO's 80 detection classes, in the index order YOLOX produces.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

#: The COCO category ids between 1 and 90 that the 80-class set does not use.
COCO_UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)

#: Pascal VOC's 21 segmentation classes, index 0 first.
VOC_21 = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def coco_90() -> List[str]:
    """Build the 90-entry COCO label set an SSD export indexes into.

    A TensorFlow Object Detection export reports one-based COCO category ids, and those ids run to
    90 with ten unused slots. The bundle declares ``classIndexOffset: 1``, so index zero is
    category one; the unused slots keep the rest of the list aligned.

    Returns:
        Ninety labels, where entry ``i`` is COCO category ``i + 1``.
    """
    names = iter(COCO_80)
    return [
        f"unused-{identifier}" if identifier in COCO_UNUSED_IDS else next(names)
        for identifier in range(1, 91)
    ]


def imagenet_1000(synset_path: Path) -> List[str]:
    """Read the ImageNet-1k class names from a pinned ``synset.txt``.

    Each line is a WordNet id and a comma-separated list of names for that class. The first name
    is taken, so ``n01440764 tench, Tinca tinca`` becomes ``tench``.

    Args:
        synset_path: The fetched ``synset.txt``.

    Returns:
        One thousand class names, in the index order the models produce.

    Raises:
        ValueError: When the file does not hold exactly one thousand usable lines.
    """
    labels: List[str] = []
    for line in Path(synset_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        _, _, names = stripped.partition(" ")
        labels.append(names.split(",")[0].strip() or stripped)
    if len(labels) != 1000:
        raise ValueError(f"{synset_path} holds {len(labels)} classes, ImageNet-1k has 1000")
    return labels


def wordnet_ids(synset_path: Path) -> List[str]:
    """Read the WordNet ids from a pinned ``synset.txt``.

    Imagenette names its directories by WordNet id, so this is what turns a directory into an
    expected class index.

    Args:
        synset_path: The fetched ``synset.txt``.

    Returns:
        One thousand WordNet ids, in class-index order.
    """
    return [
        line.split(" ", 1)[0].strip()
        for line in Path(synset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
