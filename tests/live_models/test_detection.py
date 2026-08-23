"""YOLOX and SSD-MobileNetV1 against the COCO slice (DESIGN.md section 16.1, tier 2).

The two models exercise the detection family's two head conventions. YOLOX exports an undecoded
anchor-free grid, so the family rebuilds the grid from the strides and decodes it; SSD exports
boxes that are already decoded and suppressed, so the family reads them and maps them back. Both
report boxes in the same normalized source coordinates, which is what makes one golden format
serve both.
"""

from __future__ import annotations

import pytest

from tests.live_models import relative_name, require_live_models
from tests.live_models import verify

require_live_models()

#: The three detectors of the corpus.
DETECTORS = ["yolox-nano", "yolox-s", "ssd-mobilenetv1-12"]


@pytest.mark.parametrize("key", DETECTORS)
def test_detector_matches_its_golden(key, staged, coco_images, update_goldens_mode, provider):
    """The staged detector reproduces its committed golden on the COCO slice."""
    model, records = verify.run_corpus(staged, key, coco_images, relative_name)
    assert len(records) == len(coco_images)
    verify.check(model, records, update_goldens_mode, provider=provider)


@pytest.mark.parametrize("key", DETECTORS)
def test_detector_boxes_are_inside_the_source_image(key, staged, coco_images):
    """Every reported box lies in the unit square of the picture the camera took.

    A letterboxed canvas, a transposed corner order, or a pixel-versus-normalized mix-up all show
    up here as a box outside the image.
    """
    _, records = verify.run_corpus(staged, key, coco_images, relative_name)
    for entry in records:
        for detection in entry["detections"]:
            x, y, width, height = detection["box"]
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"{entry['image']}: origin {x}, {y}"
            assert width > 0.0 and height > 0.0, f"{entry['image']}: empty box {detection['box']}"
            assert x + width <= 1.000001 and y + height <= 1.000001, (
                f"{entry['image']}: box {detection['box']} runs outside the image"
            )


def test_the_two_yolox_sizes_agree_on_what_is_in_the_frame(staged, coco_images):
    """YOLOX-Nano and YOLOX-S see the same kinds of object on most of the slice.

    The two models differ in capacity, not in convention, so a disagreement about which labels are
    present points at the decode rather than at the model.
    """
    _, small = verify.run_corpus(staged, "yolox-nano", coco_images, relative_name)
    _, large = verify.run_corpus(staged, "yolox-s", coco_images, relative_name)
    agreed = 0
    for nano, size_s in zip(small, large):
        nano_labels = {item["label"] for item in nano["detections"]}
        s_labels = {item["label"] for item in size_s["detections"]}
        if nano_labels & s_labels:
            agreed += 1
    assert agreed >= len(coco_images) * 0.7, (
        f"the two YOLOX sizes shared a label on {agreed} of {len(coco_images)} images"
    )
