"""Classification postprocessing (DESIGN.md §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import FamilyError
from image_processor.engine.families.classification import ClassificationFamily
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec

FAMILY = ClassificationFamily()


def _manifest(params=None, **overrides):
    overrides.setdefault("outputs", [spec("logits", (1, 3))])
    return make_manifest(
        family=Family.CLASSIFICATION,
        family_params=params if params is not None else {"labels": ["a", "b", "c"]},
        **overrides,
    )


def test_scores_are_ranked_and_labelled():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "none", "topK": 3})
    result = FAMILY.postprocess({"logits": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8))
    assert [entry.label for entry in result.classes] == ["b", "c", "a"]
    assert [entry.index for entry in result.classes] == [1, 2, 0]
    assert result.classes[0].score == pytest.approx(0.7)
    assert result.family is Family.CLASSIFICATION


def test_softmax_normalizes_the_reported_scores():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "softmax"})
    result = FAMILY.postprocess({"logits": np.array([[1.0, 1.0, 1.0]])}, m, (8, 8))
    assert sum(entry.score for entry in result.classes) == pytest.approx(1.0)


def test_sigmoid_scores_each_class_independently():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "sigmoid"})
    result = FAMILY.postprocess({"logits": np.array([[0.0, 0.0, 0.0]])}, m, (8, 8))
    assert all(entry.score == pytest.approx(0.5) for entry in result.classes)


def test_top_k_bounds_the_report():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "none", "topK": 2})
    result = FAMILY.postprocess({"logits": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8))
    assert len(result.classes) == 2


def test_the_result_budget_bounds_the_report_further():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "none", "topK": 3}, max_result_items=1)
    result = FAMILY.postprocess({"logits": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8))
    assert len(result.classes) == 1


def test_a_score_floor_drops_weak_classes():
    m = _manifest(
        {"labels": ["a", "b", "c"], "activation": "none", "topK": 3, "scoreThreshold": 0.5}
    )
    result = FAMILY.postprocess({"logits": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8))
    assert [entry.label for entry in result.classes] == ["b"]


def test_a_head_without_a_batch_axis_is_read_the_same_way():
    m = _manifest({"labels": ["a", "b", "c"], "activation": "none"}, outputs=[spec("logits", (3,))])
    result = FAMILY.postprocess({"logits": np.array([0.1, 0.7, 0.2])}, m, (8, 8))
    assert result.classes[0].label == "b"


def test_a_class_count_without_labels_yields_positional_names():
    m = _manifest({"numClasses": 3, "activation": "none"})
    result = FAMILY.postprocess({"logits": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8))
    assert result.classes[0].label == "class-1"


def test_a_named_output_is_selected_from_several():
    m = _manifest(
        {"labels": ["a", "b", "c"], "activation": "none", "outputName": "head"},
        outputs=[spec("other", (1, 3)), spec("head", (1, 3))],
    )
    result = FAMILY.postprocess(
        {"other": np.zeros((1, 3)), "head": np.array([[0.1, 0.7, 0.2]])}, m, (8, 8)
    )
    assert result.classes[0].label == "b"
    assert set(result.raw_shapes) == {"other", "head"}


def test_a_missing_output_is_reported():
    m = _manifest({"labels": ["a", "b", "c"], "outputName": "head"})
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"logits": np.zeros((1, 3))}, m, (8, 8))
    assert caught.value.code == "MISSING_OUTPUT"


def test_a_manifest_with_no_declared_outputs_cannot_choose_one():
    m = _manifest({"labels": ["a", "b", "c"]}, outputs=[])
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"a": np.zeros((1, 3)), "b": np.zeros((1, 3))}, m, (8, 8))
    assert caught.value.code == "MISSING_OUTPUT"


def test_more_than_one_sample_in_a_batch_is_refused():
    m = _manifest({"labels": ["a", "b", "c"]})
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"logits": np.zeros((2, 3))}, m, (8, 8))
    assert caught.value.code == "UNSUPPORTED_BATCH"


def test_a_head_of_the_wrong_rank_is_refused_at_run_time():
    m = _manifest({"labels": ["a", "b", "c"]})
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"logits": np.zeros((1, 1, 3, 3))}, m, (8, 8))
    assert caught.value.code == "UNSUPPORTED_OUTPUT_RANK"


def test_preprocess_delegates_to_the_shared_transform():
    m = _manifest({"labels": ["a", "b", "c"]})
    fed = FAMILY.preprocess(np.zeros((8, 8, 3), np.uint8), m)
    assert fed["images"].shape == (1, 3, 8, 8)
