"""Preprocessing geometry, normalization, layout, and refusals (LLD §6)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import (
    FamilyError,
    ResizePlan,
    SourceMapper,
    input_binding,
    preprocess_image,
    resize_plan,
    validate_preprocess,
)
from tests.engine.conftest import gradient, make_manifest, spec


def _manifest(preprocess, **overrides):
    shape = overrides.pop("input_shape", (1, 3, 8, 8))
    return make_manifest(preprocess=preprocess, inputs=[spec("images", shape)], **overrides)


def test_stretch_produces_the_declared_canvas_and_scale():
    m = _manifest(
        {
            "resize": {"mode": "stretch", "width": 8, "height": 8, "interpolation": "nearest"},
            "scale": 1 / 255,
            "layout": "NCHW",
            "dtype": "float32",
        }
    )
    fed = preprocess_image(gradient(4, 16), m)["images"]
    assert fed.shape == (1, 3, 8, 8)
    assert fed.dtype == np.float32
    assert 0.0 <= float(fed.min()) and float(fed.max()) <= 1.0


def test_letterbox_centers_the_image_and_fills_the_rest_with_the_pad_colour():
    m = _manifest(
        {
            "resize": {
                "mode": "letterbox",
                "width": 8,
                "height": 8,
                "interpolation": "nearest",
                "padColor": [114, 114, 114],
                "padMode": "center",
            },
            "scale": 1.0,
            "layout": "NHWC",
            "dtype": "float32",
        },
        input_shape=(1, 8, 8, 3),
    )
    fed = preprocess_image(gradient(4, 8), m)["images"][0]
    assert fed.shape == (8, 8, 3)
    assert np.allclose(fed[0], 114.0)
    assert np.allclose(fed[7], 114.0)
    assert not np.allclose(fed[2], 114.0)


def test_letterbox_top_left_puts_the_image_in_the_corner():
    plan = {
        "resize": {
            "mode": "letterbox",
            "width": 8,
            "height": 8,
            "interpolation": "nearest",
            "padColor": 200,
            "padMode": "topLeft",
        },
        "scale": 1.0,
        "layout": "NHWC",
        "dtype": "float32",
    }
    fed = preprocess_image(gradient(4, 8), _manifest(plan, input_shape=(1, 8, 8, 3)))["images"][0]
    assert not np.allclose(fed[0], 200.0)
    assert np.allclose(fed[7], 200.0)


def test_center_crop_covers_the_canvas_without_padding():
    m = _manifest(
        {
            "resize": {"mode": "centerCrop", "width": 8, "height": 8, "interpolation": "nearest"},
            "scale": 1.0,
            "layout": "NHWC",
            "dtype": "float32",
        },
        input_shape=(1, 8, 8, 3),
    )
    fed = preprocess_image(gradient(4, 16), m)["images"][0]
    assert fed.shape == (8, 8, 3)


def test_resize_mode_none_feeds_the_source_geometry_unchanged():
    m = _manifest(
        {"resize": {"mode": "none"}, "scale": 1.0, "layout": "NHWC", "dtype": "float32"},
        input_shape=(1, "H", "W", 3),
    )
    fed = preprocess_image(gradient(5, 7), m)["images"]
    assert fed.shape == (1, 5, 7, 3)


def test_bgr_reverses_the_channels_before_anything_else():
    plan = {
        "colorOrder": "BGR",
        "resize": {"mode": "none"},
        "scale": 1.0,
        "layout": "NHWC",
        "dtype": "float32",
    }
    image = np.zeros((2, 2, 3), np.uint8)
    image[..., 0] = 10
    image[..., 2] = 30
    fed = preprocess_image(image, _manifest(plan, input_shape=(1, 2, 2, 3)))["images"][0]
    assert np.allclose(fed[..., 0], 30.0)
    assert np.allclose(fed[..., 2], 10.0)


def test_scale_then_mean_and_standard_deviation_are_applied_per_channel():
    plan = {
        "resize": {"mode": "none"},
        "scale": 1 / 255,
        "mean": [0.5, 0.0, 0.25],
        "std": [0.5, 2.0, 1.0],
        "layout": "NHWC",
        "dtype": "float32",
    }
    image = np.full((2, 2, 3), 255, np.uint8)
    fed = preprocess_image(image, _manifest(plan, input_shape=(1, 2, 2, 3)))["images"][0]
    assert np.allclose(fed[..., 0], (1.0 - 0.5) / 0.5, atol=1e-6)
    assert np.allclose(fed[..., 1], 1.0 / 2.0, atol=1e-6)
    assert np.allclose(fed[..., 2], (1.0 - 0.25) / 1.0, atol=1e-6)


def test_a_uint8_feed_skips_normalization_and_rounds():
    plan = {"resize": {"mode": "none"}, "layout": "NHWC", "dtype": "uint8"}
    fed = preprocess_image(gradient(3, 3), _manifest(plan, input_shape=(1, 3, 3, 3)))["images"]
    assert fed.dtype == np.uint8


def test_float16_is_available_for_a_half_precision_graph():
    plan = {"resize": {"mode": "none"}, "scale": 1 / 255, "layout": "NCHW", "dtype": "float16"}
    fed = preprocess_image(gradient(2, 2), _manifest(plan, input_shape=(1, 3, 2, 2)))["images"]
    assert fed.dtype == np.float16


def test_sixteen_bit_maps_into_the_eight_bit_range_by_default():
    plan = {"resize": {"mode": "none"}, "scale": 1.0, "layout": "NHWC", "dtype": "float32"}
    image = np.full((2, 2, 3), 65535, np.uint16)
    fed = preprocess_image(image, _manifest(plan, input_shape=(1, 2, 2, 3)))["images"][0]
    assert np.allclose(fed, 255.0, atol=1e-3)


def test_raw_bit_depth_keeps_the_full_range_and_scales_the_pad_colour():
    plan = {
        "resize": {
            "mode": "letterbox",
            "width": 4,
            "height": 4,
            "interpolation": "nearest",
            "padColor": 128,
        },
        "highBitDepthMode": "raw",
        "scale": 1.0,
        "layout": "NHWC",
        "dtype": "float32",
    }
    image = np.full((1, 4, 3), 65535, np.uint16)
    fed = preprocess_image(image, _manifest(plan, input_shape=(1, 4, 4, 3)))["images"][0]
    assert np.allclose(fed[1], 65535.0)
    assert np.allclose(fed[0], 128.0 * 257.0)
    assert np.allclose(fed[3], 128.0 * 257.0)


@pytest.mark.parametrize(
    "block, code",
    [
        ({"resize": {"mode": "warp", "width": 8, "height": 8}}, "UNSUPPORTED_RESIZE_MODE"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8, "interpolation": "magic"}}, "UNSUPPORTED_INTERPOLATION"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8, "padMode": "corner"}}, "UNSUPPORTED_PAD_MODE"),
        ({"resize": {"mode": "stretch", "height": 8}}, "MISSING_FAMILY_PARAM"),
        ({"resize": {"mode": "stretch", "width": 0, "height": 8}}, "INVALID_FAMILY_PARAM"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8, "padColor": [1, 2]}}, "INVALID_FAMILY_PARAM"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "colorOrder": "RBG"}, "UNSUPPORTED_COLOR_ORDER"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "layout": "CHWN"}, "UNSUPPORTED_LAYOUT"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "dtype": "int8"}, "UNSUPPORTED_DTYPE"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "highBitDepthMode": "keep"}, "UNSUPPORTED_BIT_DEPTH_MODE"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "std": 0.0}, "INVALID_FAMILY_PARAM"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "mean": [1, 2]}, "INVALID_FAMILY_PARAM"),
        ({"resize": {"mode": "stretch", "width": 8, "height": 8}, "scale": "half"}, "INVALID_FAMILY_PARAM"),
        (
            {"resize": {"mode": "stretch", "width": 8, "height": 8}, "dtype": "uint8", "scale": 0.5},
            "PREPROCESS_UINT8_NORMALIZATION",
        ),
    ],
)
def test_an_unusable_preprocess_block_is_refused(block, code):
    with pytest.raises(FamilyError) as caught:
        validate_preprocess(_manifest(block))
    assert caught.value.code == code


def test_a_declared_input_shape_that_contradicts_the_resize_target_is_refused():
    block = {"resize": {"mode": "stretch", "width": 8, "height": 4}, "layout": "NCHW"}
    with pytest.raises(FamilyError) as caught:
        validate_preprocess(_manifest(block, input_shape=(1, 3, 8, 8)))
    assert caught.value.code == "INPUT_SHAPE_MISMATCH"

    block = {"resize": {"mode": "stretch", "width": 4, "height": 8}, "layout": "NCHW"}
    with pytest.raises(FamilyError) as caught:
        validate_preprocess(_manifest(block, input_shape=(1, 3, 8, 8)))
    assert caught.value.code == "INPUT_SHAPE_MISMATCH"


def test_a_dynamic_input_side_satisfies_the_resize_check():
    block = {"resize": {"mode": "stretch", "width": 8, "height": 8}, "layout": "NHWC"}
    validate_preprocess(_manifest(block, input_shape=("N", "H", "W", 3)))


def test_an_input_of_the_wrong_rank_is_refused():
    block = {"resize": {"mode": "none"}}
    with pytest.raises(FamilyError) as caught:
        validate_preprocess(_manifest(block, input_shape=(1, 2, 3, 4, 5)))
    assert caught.value.code == "UNSUPPORTED_INPUT_RANK"


def test_the_batch_axis_follows_the_declared_rank_and_the_explicit_override():
    rank_four = _manifest({"resize": {"mode": "none"}}, input_shape=(1, 3, 8, 8))
    assert input_binding(rank_four) == ("images", True)

    rank_three = _manifest({"resize": {"mode": "none"}}, input_shape=(3, 8, 8))
    assert input_binding(rank_three) == ("images", False)

    forced = _manifest({"resize": {"mode": "none"}, "batchAxis": False}, input_shape=(1, 3, 8, 8))
    assert input_binding(forced) == ("images", False)


def test_an_unknown_rank_falls_back_to_the_dynamic_batch_flag():
    m = _manifest({"resize": {"mode": "none"}}, input_shape=(1, 2, 3, 4, 5), dynamic_batch=True)
    assert input_binding(m) == ("images", True)


def test_a_named_input_must_be_declared():
    m = _manifest({"resize": {"mode": "none"}, "inputName": "pixels"})
    with pytest.raises(FamilyError) as caught:
        input_binding(m)
    assert caught.value.code == "MISSING_INPUT"


def test_a_manifest_without_inputs_is_refused():
    m = make_manifest(inputs=[])
    with pytest.raises(FamilyError) as caught:
        input_binding(m)
    assert caught.value.code == "MISSING_INPUT"


def test_an_image_that_is_not_three_channel_hwc_is_refused():
    m = _manifest({"resize": {"mode": "none"}})
    with pytest.raises(FamilyError) as caught:
        preprocess_image(np.zeros((4, 4), np.uint8), m)
    assert caught.value.code == "INVALID_IMAGE"


def test_a_zero_sized_source_is_refused():
    with pytest.raises(FamilyError) as caught:
        SourceMapper.build(ResizePlan("stretch", 8, 8, 0, (0, 0, 0), "center"), (0, 4))
    assert caught.value.code == "INVALID_IMAGE"


def test_preprocess_rejects_a_zero_standard_deviation_at_run_time():
    block = {"resize": {"mode": "none"}, "std": [1.0, 0.0, 1.0], "layout": "NHWC"}
    with pytest.raises(FamilyError) as caught:
        preprocess_image(gradient(2, 2), _manifest(block, input_shape=(1, 2, 2, 3)))
    assert caught.value.code == "INVALID_FAMILY_PARAM"
