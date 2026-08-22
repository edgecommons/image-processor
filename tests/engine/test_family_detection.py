"""Detection postprocessing for both head conventions (DESIGN.md §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import FamilyError
from image_processor.engine.families.detection import DetectionFamily
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec

FAMILY = DetectionFamily()
LABELS = ["bolt", "nut", "washer"]

#: A 32 by 32 canvas with strides 8 and 16 gives 16 plus 4 cells.
GRID_PREPROCESS = {
    "resize": {"mode": "stretch", "width": 32, "height": 32, "interpolation": "nearest"},
    "layout": "NCHW",
    "dtype": "float32",
}


def _grid_manifest(extra=None, **overrides):
    params = {
        "decode": "yoloxGrid",
        "labels": LABELS,
        "strides": [8, 16],
        "scoreThreshold": 0.25,
        "iouThreshold": 0.45,
    }
    params.update(extra or {})
    overrides.setdefault("outputs", [spec("output", (1, 20, 8))])
    overrides.setdefault("inputs", [spec("images", (1, 3, 32, 32))])
    return make_manifest(
        family=Family.DETECTION,
        family_params=params,
        preprocess=GRID_PREPROCESS,
        **overrides,
    )


def _grid_rows(count=20, width=8):
    return np.zeros((count, width), dtype=np.float32)


def test_a_grid_cell_decodes_to_the_hand_computed_box():
    rows = _grid_rows()
    rows[9] = [0.0, 0.0, np.log(2.0), np.log(2.0), 0.9, 0.9, 0.05, 0.05]
    result = FAMILY.postprocess({"output": rows[None]}, _grid_manifest(), (32, 32))
    assert len(result.detections) == 1
    found = result.detections[0]
    assert found.label == "bolt"
    assert found.score == pytest.approx(0.81)
    assert list(found.box) == pytest.approx([0.0, 0.25, 0.5, 0.5])


def test_objectness_can_be_absent_from_the_block():
    rows = _grid_rows(width=7)
    rows[9] = [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1]
    m = _grid_manifest({"objectness": False}, outputs=[spec("output", (1, 20, 7))])
    result = FAMILY.postprocess({"output": rows[None]}, m, (32, 32))
    assert result.detections[0].label == "nut"
    assert result.detections[0].score == pytest.approx(0.8)


def test_declared_activations_are_applied_to_both_blocks():
    rows = _grid_rows()
    rows[9] = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    m = _grid_manifest({"scoreActivation": "softmax", "objectnessActivation": "sigmoid"})
    result = FAMILY.postprocess({"output": rows[None]}, m, (32, 32))
    expected = 0.5 * float(np.exp(2.0) / (np.exp(2.0) + 2.0))
    assert result.detections[0].score == pytest.approx(expected)


def test_a_grid_that_does_not_match_the_strides_is_refused():
    m = _grid_manifest()
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"output": _grid_rows(count=19)[None]}, m, (32, 32))
    assert caught.value.code == "OUTPUT_DIM_MISMATCH"


def test_a_stride_that_does_not_divide_the_canvas_is_refused():
    m = _grid_manifest({"strides": [7]})
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"output": _grid_rows()[None]}, m, (32, 32))
    assert caught.value.code == "INVALID_FAMILY_PARAM"


def test_an_enormous_log_size_stays_finite_and_clipped():
    rows = _grid_rows()
    rows[9] = [0.0, 0.0, 500.0, 500.0, 0.9, 0.9, 0.05, 0.05]
    result = FAMILY.postprocess({"output": rows[None]}, _grid_manifest(), (32, 32))
    assert list(result.detections[0].box) == pytest.approx([0.0, 0.0, 1.0, 1.0])


DECODED_PREPROCESS = dict(GRID_PREPROCESS)


def _decoded_manifest(extra=None, **overrides):
    params = {
        "decode": "decodedBoxes",
        "labels": LABELS,
        "scoreThreshold": 0.25,
        "iouThreshold": 0.45,
    }
    params.update(extra or {})
    overrides.setdefault(
        "outputs",
        [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
    )
    overrides.setdefault("inputs", [spec("images", (1, 3, 32, 32))])
    return make_manifest(
        family=Family.DETECTION,
        family_params=params,
        preprocess=DECODED_PREPROCESS,
        **overrides,
    )


def _decoded_outputs(boxes, scores, classes):
    return {
        "boxes": np.asarray(boxes, np.float32)[None],
        "scores": np.asarray(scores, np.float32)[None],
        "classes": np.asarray(classes, np.float32)[None],
    }


def test_normalized_corner_boxes_map_onto_the_source():
    outputs = _decoded_outputs(
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.8, 0.0],
        [0, 1, 0],
    )
    result = FAMILY.postprocess(outputs, _decoded_manifest(), (32, 32))
    assert [entry.label for entry in result.detections] == ["bolt", "nut"]
    assert list(result.detections[0].box) == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_the_yxyx_order_real_ssd_exports_use_is_understood():
    outputs = _decoded_outputs(
        [[0.25, 0.0, 0.75, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.0, 0.0],
        [0, 0, 0],
    )
    result = FAMILY.postprocess(outputs, _decoded_manifest({"boxFormat": "yxyx"}), (32, 32))
    assert list(result.detections[0].box) == pytest.approx([0.0, 0.25, 0.5, 0.5])


def test_centre_and_size_boxes_are_understood():
    outputs = _decoded_outputs(
        [[16.0, 16.0, 16.0, 16.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.0, 0.0],
        [0, 0, 0],
    )
    m = _decoded_manifest({"boxFormat": "cxcywh", "boxCoordinates": "pixels"})
    result = FAMILY.postprocess(outputs, m, (32, 32))
    assert list(result.detections[0].box) == pytest.approx([0.25, 0.25, 0.5, 0.5])


def test_reversed_corners_are_ordered_before_use():
    outputs = _decoded_outputs(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.0, 0.0],
        [0, 0, 0],
    )
    result = FAMILY.postprocess(outputs, _decoded_manifest(), (32, 32))
    assert list(result.detections[0].box) == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_per_class_scores_pick_their_own_class():
    outputs = {
        "boxes": np.array([[[0.0, 0.0, 0.5, 0.5]]], np.float32),
        "scores": np.array([[[0.1, 0.2, 0.7]]], np.float32),
    }
    m = _decoded_manifest(
        {"scoresLayout": "perClass"},
        outputs=[spec("boxes", (1, 1, 4)), spec("scores", (1, 1, 3))],
    )
    result = FAMILY.postprocess(outputs, m, (32, 32))
    assert result.detections[0].label == "washer"
    assert result.detections[0].score == pytest.approx(0.7)


def test_a_one_based_class_id_is_shifted_back_onto_the_label_list():
    outputs = _decoded_outputs(
        [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.0, 0.0],
        [3, 0, 0],
    )
    result = FAMILY.postprocess(outputs, _decoded_manifest({"classIndexOffset": 1}), (32, 32))
    assert result.detections[0].label == "washer"


def test_a_detection_count_output_truncates_the_block():
    outputs = _decoded_outputs(
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.25, 0.25]],
        [0.9, 0.8, 0.7],
        [0, 1, 2],
    )
    outputs["count"] = np.array([1.0], np.float32)
    m = _decoded_manifest(
        {"outputNames": {"count": "count"}},
        outputs=[
            spec("boxes", (1, 3, 4)),
            spec("scores", (1, 3)),
            spec("classes", (1, 3)),
            spec("count", (1,)),
        ],
    )
    result = FAMILY.postprocess(outputs, m, (32, 32))
    assert len(result.detections) == 1


def test_suppression_can_be_turned_off_for_a_head_that_already_ran_it():
    outputs = _decoded_outputs(
        [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
        [0.9, 0.8, 0.0],
        [0, 0, 0],
    )
    kept = FAMILY.postprocess(outputs, _decoded_manifest({"applyNms": False}), (32, 32))
    assert len(kept.detections) == 2
    suppressed = FAMILY.postprocess(outputs, _decoded_manifest(), (32, 32))
    assert len(suppressed.detections) == 1


def test_blocks_that_disagree_on_how_many_boxes_there_are_are_refused():
    outputs = {
        "boxes": np.zeros((1, 3, 4), np.float32),
        "scores": np.zeros((1, 2), np.float32),
        "classes": np.zeros((1, 3), np.float32),
    }
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess(outputs, _decoded_manifest(), (32, 32))
    assert caught.value.code == "OUTPUT_DIM_MISMATCH"


def test_a_box_block_that_does_not_end_in_four_values_is_refused():
    outputs = {
        "boxes": np.zeros((1, 3, 5), np.float32),
        "scores": np.zeros((1, 3), np.float32),
        "classes": np.zeros((1, 3), np.float32),
    }
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess(outputs, _decoded_manifest(), (32, 32))
    assert caught.value.code == "OUTPUT_DIM_MISMATCH"


def test_the_result_budget_caps_the_detections():
    outputs = _decoded_outputs(
        [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7], [0.8, 0.8, 1.0, 1.0]],
        [0.9, 0.8, 0.7],
        [0, 1, 2],
    )
    m = _decoded_manifest({"maxDetections": 3}, max_result_items=2)
    assert len(FAMILY.postprocess(outputs, m, (32, 32)).detections) == 2
