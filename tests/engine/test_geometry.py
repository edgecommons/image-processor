"""Coordinate mapping between the model canvas and the source image (LLD §6)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import ResizePlan, SourceMapper, resize_plan
from tests.engine.conftest import make_manifest


def _plan(mode, width=64, height=64, pad_mode="center"):
    return ResizePlan(mode, width, height, 0, (114.0, 114.0, 114.0), pad_mode)


def _forward(mapper: SourceMapper, x: float, y: float) -> tuple:
    """Map a source pixel onto the model canvas, the inverse of ``SourceMapper.to_source``."""
    return x * mapper.scale_x + mapper.pad_x, y * mapper.scale_y + mapper.pad_y


@pytest.mark.parametrize("mode", ["letterbox", "stretch", "centerCrop", "none"])
@pytest.mark.parametrize("source", [(64, 128), (128, 64), (64, 64), (37, 91)])
def test_a_source_point_survives_the_round_trip_through_the_model_canvas(mode, source):
    mapper = SourceMapper.build(_plan(mode), source)
    for point in [(0.0, 0.0), (5.5, 7.25), (float(source[1]) / 2, float(source[0]) / 3)]:
        canvas = _forward(mapper, *point)
        back = mapper.to_source(np.array(canvas, dtype=np.float64))
        assert back[0] == pytest.approx(point[0], abs=1e-9)
        assert back[1] == pytest.approx(point[1], abs=1e-9)


def test_letterbox_padding_is_whole_pixels_and_centred():
    mapper = SourceMapper.build(_plan("letterbox"), (64, 128))
    assert mapper.scale_x == mapper.scale_y == 0.5
    assert (mapper.pad_x, mapper.pad_y) == (0.0, 16.0)


def test_top_left_letterboxing_leaves_no_offset():
    mapper = SourceMapper.build(_plan("letterbox", pad_mode="topLeft"), (64, 128))
    assert (mapper.pad_x, mapper.pad_y) == (0.0, 0.0)


def test_a_box_drawn_in_the_padding_is_clipped_to_the_picture():
    mapper = SourceMapper.build(_plan("letterbox"), (64, 128))
    boxes = mapper.normalized_boxes(np.array([[0.0, 0.0, 64.0, 64.0]]))
    assert boxes[0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_a_known_letterboxed_box_maps_to_the_hand_computed_source_box():
    mapper = SourceMapper.build(_plan("letterbox"), (64, 128))
    region = mapper.normalized_region(8.0, 24.0, 24.0, 40.0)
    assert region == pytest.approx([0.125, 0.25, 0.25, 0.5])


def test_stretch_scales_each_axis_independently():
    mapper = SourceMapper.build(_plan("stretch", 32, 16), (64, 128))
    assert (mapper.scale_x, mapper.scale_y) == (0.25, 0.25)
    region = mapper.normalized_region(0.0, 0.0, 16.0, 8.0)
    assert region == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_center_crop_offsets_are_negative_padding():
    mapper = SourceMapper.build(_plan("centerCrop", 32, 32), (32, 64))
    assert mapper.scale_x == 1.0
    assert mapper.pad_x == -16.0
    assert mapper.pad_y == 0.0


def test_resize_mode_none_maps_the_source_onto_itself():
    mapper = SourceMapper.build(_plan("none"), (20, 40))
    assert (mapper.in_w, mapper.in_h) == (40, 20)
    assert mapper.normalized_region(0.0, 0.0, 40.0, 20.0) == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_a_pad_colour_given_as_one_number_fills_all_three_channels():
    m = make_manifest(preprocess={"resize": {"mode": "letterbox", "width": 4, "height": 4, "padColor": 7}})
    assert resize_plan(m).pad_color == (7.0, 7.0, 7.0)
