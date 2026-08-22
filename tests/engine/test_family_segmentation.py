"""Segmentation postprocessing (DESIGN.md §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import FamilyError
from image_processor.engine.families.segmentation import SegmentationFamily
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec

FAMILY = SegmentationFamily()
PREPROCESS = {
    "resize": {"mode": "stretch", "width": 4, "height": 4, "interpolation": "nearest"},
    "layout": "NCHW",
    "dtype": "float32",
}


def _manifest(params, **overrides):
    overrides.setdefault("outputs", [spec("mask", (1, 1, 4, 4))])
    overrides.setdefault("inputs", [spec("images", (1, 3, 4, 4))])
    return make_manifest(
        family=Family.SEGMENTATION, family_params=params, preprocess=PREPROCESS, **overrides
    )


def test_a_thresholded_channel_yields_a_count_a_fraction_and_a_region():
    plane = np.zeros((1, 1, 4, 4), np.float32)
    plane[0, 0, 1:3, 2:4] = 1.0
    m = _manifest({"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5})
    result = FAMILY.postprocess({"mask": plane}, m, (4, 4))
    entry = result.segments["defect"]
    assert entry["pixels"] == 4
    assert entry["fraction"] == pytest.approx(0.25)
    assert entry["bbox"] == pytest.approx([0.5, 0.25, 0.5, 0.5])


def test_a_clean_image_still_reports_its_positive_class_with_no_pixels():
    m = _manifest({"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5})
    result = FAMILY.postprocess({"mask": np.zeros((1, 1, 4, 4), np.float32)}, m, (4, 4))
    assert result.segments["defect"] == {"pixels": 0, "fraction": 0.0, "bbox": None}


def test_a_declared_activation_is_applied_before_the_threshold():
    plane = np.full((1, 1, 4, 4), 1.0, np.float32)
    m = _manifest(
        {"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.6, "activation": "sigmoid"}
    )
    result = FAMILY.postprocess({"mask": plane}, m, (4, 4))
    assert result.segments["defect"]["pixels"] == 16


def test_argmax_reports_every_named_class():
    logits = np.zeros((1, 3, 4, 4), np.float32)
    logits[0, 0] = 0.4
    logits[0, 1, 0:2, 0:2] = 1.0
    m = _manifest(
        {"mode": "argmax", "labels": ["background", "part", "defect"]},
        outputs=[spec("mask", (1, 3, 4, 4))],
    )
    result = FAMILY.postprocess({"mask": logits}, m, (4, 4))
    assert result.segments["part"]["pixels"] == 4
    assert result.segments["background"]["pixels"] == 12
    assert result.segments["defect"] == {"pixels": 0, "fraction": 0.0, "bbox": None}


def test_an_ignored_class_is_left_out_entirely():
    logits = np.zeros((1, 3, 4, 4), np.float32)
    logits[0, 0] = 0.4
    m = _manifest(
        {"mode": "argmax", "labels": ["background", "part", "defect"], "ignoreIndex": 0},
        outputs=[spec("mask", (1, 3, 4, 4))],
    )
    result = FAMILY.postprocess({"mask": logits}, m, (4, 4))
    assert "background" not in result.segments
    assert set(result.segments) == {"part", "defect"}


def test_a_minimum_pixel_count_drops_specks():
    logits = np.zeros((1, 3, 4, 4), np.float32)
    logits[0, 0] = 0.4
    logits[0, 2, 0, 0] = 1.0
    m = _manifest(
        {"mode": "argmax", "labels": ["background", "part", "defect"], "minPixels": 2},
        outputs=[spec("mask", (1, 3, 4, 4))],
    )
    result = FAMILY.postprocess({"mask": logits}, m, (4, 4))
    assert set(result.segments) == {"background"}


def test_a_channels_last_class_map_is_transposed():
    logits = np.zeros((1, 4, 4, 3), np.float32)
    logits[0, :, :, 0] = 0.4
    logits[0, 0:2, 0:2, 1] = 1.0
    m = _manifest(
        {"mode": "argmax", "labels": ["background", "part", "defect"], "outputLayout": "NHWC"},
        outputs=[spec("mask", (1, 4, 4, 3))],
    )
    result = FAMILY.postprocess({"mask": logits}, m, (4, 4))
    assert result.segments["part"]["pixels"] == 4


def test_a_bare_two_dimensional_map_is_read_as_one_channel():
    plane = np.zeros((4, 4), np.float32)
    plane[0, 0] = 1.0
    m = _manifest(
        {"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5},
        outputs=[spec("mask", (4, 4))],
    )
    result = FAMILY.postprocess({"mask": plane}, m, (4, 4))
    assert result.segments["defect"]["pixels"] == 1


def test_a_coarse_class_map_scales_its_region_onto_the_source():
    plane = np.zeros((1, 1, 2, 2), np.float32)
    plane[0, 0, 0, 0] = 1.0
    m = _manifest(
        {"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5},
        outputs=[spec("mask", (1, 1, 2, 2))],
    )
    result = FAMILY.postprocess({"mask": plane}, m, (4, 4))
    assert result.segments["defect"]["bbox"] == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_a_class_map_whose_channels_disagree_with_the_labels_is_refused():
    m = _manifest(
        {"mode": "argmax", "labels": ["background", "part", "defect"]},
        outputs=[spec("mask", (1, "C", 4, 4))],
    )
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"mask": np.zeros((1, 2, 4, 4), np.float32)}, m, (4, 4))
    assert caught.value.code == "CLASS_DIM_MISMATCH"


def test_threshold_mode_refuses_a_multi_channel_map():
    m = _manifest(
        {"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5},
        outputs=[spec("mask", (1, "C", 4, 4))],
    )
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"mask": np.zeros((1, 3, 4, 4), np.float32)}, m, (4, 4))
    assert caught.value.code == "UNSUPPORTED_OUTPUT_SHAPE"


def test_preprocess_delegates_to_the_shared_transform():
    m = _manifest({"mode": "threshold", "labels": ["clean", "defect"], "threshold": 0.5})
    assert FAMILY.preprocess(np.zeros((4, 4, 3), np.uint8), m)["images"].shape == (1, 3, 4, 4)
