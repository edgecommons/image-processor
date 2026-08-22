"""Class-aware non-maximum suppression (LLD §6, DESIGN.md §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import apply_activation, nms, sigmoid, softmax
from image_processor.engine.families import FamilyError

BOX_A = [8.0, 24.0, 24.0, 40.0]
BOX_B = [10.0, 26.0, 26.0, 42.0]
BOX_FAR = [36.0, 20.0, 60.0, 44.0]


def test_an_overlapping_box_of_the_same_class_is_suppressed():
    keep = nms(np.array([BOX_A, BOX_B]), np.array([0.9, 0.6]), np.array([0, 0]), 0.45)
    assert keep == [0]


def test_the_same_pixels_under_two_classes_both_survive():
    keep = nms(np.array([BOX_A, BOX_A]), np.array([0.9, 0.6]), np.array([0, 2]), 0.45)
    assert keep == [0, 1]


def test_a_looser_threshold_keeps_a_partly_overlapping_box():
    keep = nms(np.array([BOX_A, BOX_B]), np.array([0.9, 0.6]), np.array([0, 0]), 0.7)
    assert keep == [0, 1]


def test_results_come_back_highest_score_first():
    keep = nms(
        np.array([BOX_A, BOX_FAR]), np.array([0.2, 0.8]), np.array([0, 1]), 0.45
    )
    assert keep == [1, 0]


def test_the_cap_stops_the_scan_early():
    boxes = np.array([BOX_A, BOX_FAR, [0.0, 0.0, 4.0, 4.0]])
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 1, 2]), 0.45, max_items=2)
    assert keep == [0, 1]


def test_an_empty_input_keeps_nothing():
    assert nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.5) == []


def test_zero_area_boxes_never_suppress_each_other():
    boxes = np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
    keep = nms(boxes, np.array([0.9, 0.8]), np.array([0, 0]), 0.5)
    assert keep == [0, 1]


def test_equal_scores_keep_their_input_order():
    boxes = np.array([BOX_A, BOX_FAR])
    keep = nms(boxes, np.array([0.5, 0.5]), np.array([0, 1]), 0.45)
    assert keep == [0, 1]


def test_softmax_is_stable_at_extreme_logits():
    scores = softmax(np.array([1000.0, 1000.0, 999.0]))
    assert scores.sum() == pytest.approx(1.0)
    assert scores[0] == pytest.approx(scores[1])


def test_sigmoid_is_stable_in_both_tails():
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)


def test_an_unknown_activation_is_refused():
    with pytest.raises(FamilyError) as caught:
        apply_activation(np.zeros(3), "tanh")
    assert caught.value.code == "UNSUPPORTED_ACTIVATION"


def test_the_none_activation_passes_values_through():
    assert apply_activation(np.array([2.0, 3.0]), "none").tolist() == [2.0, 3.0]
