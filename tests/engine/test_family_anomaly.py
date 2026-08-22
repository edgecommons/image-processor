"""Anomaly postprocessing (DESIGN.md §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import FamilyError
from image_processor.engine.families.anomaly import AnomalyFamily
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec

FAMILY = AnomalyFamily()
PREPROCESS = {
    "resize": {"mode": "stretch", "width": 4, "height": 4, "interpolation": "nearest"},
    "layout": "NCHW",
    "dtype": "float32",
}


def _manifest(params, **overrides):
    overrides.setdefault("outputs", [spec("score", (1,))])
    overrides.setdefault("inputs", [spec("images", (1, 3, 4, 4))])
    return make_manifest(
        family=Family.ANOMALY, family_params=params, preprocess=PREPROCESS, **overrides
    )


def test_a_scalar_head_is_compared_against_the_declared_threshold():
    m = _manifest({"source": "scalar", "threshold": 0.5})
    below = FAMILY.postprocess({"score": np.array([0.2], np.float32)}, m, (4, 4))
    assert below.anomaly == {
        "score": pytest.approx(0.2),
        "threshold": 0.5,
        "anomalous": False,
        "direction": "higherIsAnomalous",
    }
    above = FAMILY.postprocess({"score": np.array([0.9], np.float32)}, m, (4, 4))
    assert above.anomaly["anomalous"] is True


def test_a_model_that_scores_similarity_can_declare_the_other_direction():
    m = _manifest({"source": "scalar", "threshold": 0.5, "direction": "lowerIsAnomalous"})
    assert FAMILY.postprocess({"score": np.array([0.2])}, m, (4, 4)).anomaly["anomalous"] is True
    assert FAMILY.postprocess({"score": np.array([0.9])}, m, (4, 4)).anomaly["anomalous"] is False


def test_a_declared_activation_is_applied_to_the_raw_score():
    m = _manifest({"source": "scalar", "threshold": 0.4, "activation": "sigmoid"})
    result = FAMILY.postprocess({"score": np.array([0.0])}, m, (4, 4))
    assert result.anomaly["score"] == pytest.approx(0.5)


def test_a_linear_normalization_rescales_and_clamps():
    m = _manifest(
        {"source": "scalar", "threshold": 0.5, "normalization": {"min": 10.0, "max": 20.0}}
    )
    assert FAMILY.postprocess({"score": np.array([15.0])}, m, (4, 4)).anomaly["score"] == pytest.approx(0.5)
    assert FAMILY.postprocess({"score": np.array([99.0])}, m, (4, 4)).anomaly["score"] == pytest.approx(1.0)
    assert FAMILY.postprocess({"score": np.array([0.0])}, m, (4, 4)).anomaly["score"] == pytest.approx(0.0)


def test_a_map_head_reduces_to_a_score_and_a_bounded_summary():
    heatmap = np.zeros((1, 1, 4, 4), np.float32)
    heatmap[0, 0, 1:3, 2:4] = 0.8
    m = _manifest(
        {"source": "mapMax", "threshold": 0.5}, outputs=[spec("heatmap", (1, 1, 4, 4))]
    )
    result = FAMILY.postprocess({"heatmap": heatmap}, m, (4, 4))
    assert result.anomaly["score"] == pytest.approx(0.8)
    summary = result.anomaly["summary"]
    assert summary["max"] == pytest.approx(0.8)
    assert summary["min"] == pytest.approx(0.0)
    assert summary["mean"] == pytest.approx(0.8 * 4 / 16)
    assert summary["aboveThresholdPixels"] == 4
    assert summary["fraction"] == pytest.approx(0.25)
    assert summary["bbox"] == pytest.approx([0.5, 0.25, 0.5, 0.5])


def test_a_map_can_be_reduced_by_its_mean_instead():
    heatmap = np.zeros((1, 1, 4, 4), np.float32)
    heatmap[0, 0, 0, 0] = 1.6
    m = _manifest(
        {"source": "mapMean", "threshold": 0.5}, outputs=[spec("heatmap", (1, 1, 4, 4))]
    )
    result = FAMILY.postprocess({"heatmap": heatmap}, m, (4, 4))
    assert result.anomaly["score"] == pytest.approx(0.1)
    assert result.anomaly["anomalous"] is False


def test_a_map_with_nothing_above_the_threshold_reports_no_region():
    m = _manifest({"source": "mapMax", "threshold": 0.5}, outputs=[spec("heatmap", (1, 1, 4, 4))])
    result = FAMILY.postprocess({"heatmap": np.zeros((1, 1, 4, 4), np.float32)}, m, (4, 4))
    assert result.anomaly["summary"]["bbox"] is None
    assert result.anomaly["summary"]["aboveThresholdPixels"] == 0


def test_a_channels_last_map_with_one_channel_is_accepted():
    heatmap = np.zeros((1, 4, 4, 1), np.float32)
    heatmap[0, 0, 0, 0] = 0.9
    m = _manifest({"source": "mapMax", "threshold": 0.5}, outputs=[spec("heatmap", (1, 4, 4, 1))])
    assert FAMILY.postprocess({"heatmap": heatmap}, m, (4, 4)).anomaly["score"] == pytest.approx(0.9)


def test_a_multi_channel_map_is_refused():
    m = _manifest({"source": "mapMax", "threshold": 0.5}, outputs=[spec("heatmap", (1, 4, 4, 3))])
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"heatmap": np.zeros((1, 4, 4, 3), np.float32)}, m, (4, 4))
    assert caught.value.code == "UNSUPPORTED_OUTPUT_SHAPE"


def test_a_map_of_the_wrong_rank_is_refused():
    m = _manifest({"source": "mapMax", "threshold": 0.5}, outputs=[spec("heatmap", (1, 4))])
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"heatmap": np.zeros(4, np.float32)}, m, (4, 4))
    assert caught.value.code == "UNSUPPORTED_OUTPUT_SHAPE"


def test_a_scalar_head_that_returns_several_values_is_refused():
    m = _manifest({"source": "scalar", "threshold": 0.5})
    with pytest.raises(FamilyError) as caught:
        FAMILY.postprocess({"score": np.zeros(3, np.float32)}, m, (4, 4))
    assert caught.value.code == "UNSUPPORTED_OUTPUT_SHAPE"


def test_preprocess_delegates_to_the_shared_transform():
    m = _manifest({"source": "scalar", "threshold": 0.5})
    assert FAMILY.preprocess(np.zeros((4, 4, 3), np.uint8), m)["images"].shape == (1, 3, 4, 4)
