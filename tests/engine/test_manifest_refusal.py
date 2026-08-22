"""Manifest refusals: an unreadable head never reaches a session (D-IP-12, DESIGN.md §9 step 4)."""

from __future__ import annotations

import pytest

from image_processor.engine.families import FAMILIES, FamilyError, family_for, family_labels
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec

CLASSIFICATION_PREPROCESS = {
    "resize": {"mode": "stretch", "width": 8, "height": 8},
    "layout": "NCHW",
    "dtype": "float32",
}
SMALL_INPUT = [spec("images", (1, 3, 8, 8))]


def _manifest(family, params, outputs, **overrides):
    overrides.setdefault("preprocess", CLASSIFICATION_PREPROCESS)
    overrides.setdefault("inputs", SMALL_INPUT)
    return make_manifest(family=family, family_params=params, outputs=outputs, **overrides)


def _refused(manifest) -> str:
    with pytest.raises(FamilyError) as caught:
        FAMILIES[Family(manifest.family)].validate_manifest(manifest)
    return caught.value.code


def test_every_family_accepts_a_well_formed_manifest(corpus):
    for name in corpus.expected["bundles"]:
        manifest = corpus.manifest(name)
        family_for(manifest).validate_manifest(manifest)


def test_a_family_that_is_not_built_in_is_refused():
    manifest = make_manifest(family="pose")
    with pytest.raises(FamilyError) as caught:
        family_for(manifest)
    assert caught.value.code == "UNSUPPORTED_FAMILY"


def test_a_family_serving_the_wrong_manifest_refuses_it():
    manifest = _manifest(Family.DETECTION, {"labels": ["a"]}, [spec("logits", (1, 1))])
    with pytest.raises(FamilyError) as caught:
        FAMILIES[Family.CLASSIFICATION].validate_manifest(manifest)
    assert caught.value.code == "FAMILY_MISMATCH"


def test_a_manifest_that_names_no_classes_is_refused():
    with pytest.raises(FamilyError) as caught:
        family_labels(make_manifest(family_params={}))
    assert caught.value.code == "MISSING_FAMILY_PARAM"


@pytest.mark.parametrize(
    "params, code",
    [
        ({"labels": []}, "INVALID_FAMILY_PARAM"),
        ({"labels": [1, 2]}, "INVALID_FAMILY_PARAM"),
        ({"labels": "a,b"}, "INVALID_FAMILY_PARAM"),
        ({"labels": ["a", "b"], "numClasses": 3}, "LABEL_COUNT_MISMATCH"),
        ({"numClasses": 0}, "INVALID_FAMILY_PARAM"),
    ],
)
def test_an_unusable_label_set_is_refused(params, code):
    with pytest.raises(FamilyError) as caught:
        family_labels(make_manifest(family_params=params))
    assert caught.value.code == code


@pytest.mark.parametrize(
    "params, outputs, code",
    [
        ({"labels": ["a", "b", "c"]}, [spec("a", (1, 3)), spec("b", (1, 3))], "UNSUPPORTED_OUTPUT_COUNT"),
        ({"labels": ["a", "b", "c"]}, [spec("logits", (1, 1, 3, 3))], "UNSUPPORTED_OUTPUT_RANK"),
        ({"labels": ["a", "b", "c"]}, [spec("logits", (1, 4))], "CLASS_DIM_MISMATCH"),
        ({"labels": ["a"], "activation": "relu"}, [spec("logits", (1, 1))], "UNSUPPORTED_ACTIVATION"),
        ({"labels": ["a"], "topK": 0}, [spec("logits", (1, 1))], "INVALID_FAMILY_PARAM"),
        ({"labels": ["a"], "scoreThreshold": 2.0}, [spec("logits", (1, 1))], "INVALID_FAMILY_PARAM"),
    ],
)
def test_a_classification_head_that_cannot_be_read_is_refused(params, outputs, code):
    assert _refused(_manifest(Family.CLASSIFICATION, params, outputs)) == code


DETECTION_PREPROCESS = {
    "resize": {"mode": "stretch", "width": 32, "height": 32},
    "layout": "NCHW",
    "dtype": "float32",
}
DETECTION_INPUT = [spec("images", (1, 3, 32, 32))]
LABELS = ["bolt", "nut", "washer"]


def _detection(params, outputs, **overrides):
    merged = {"labels": LABELS}
    merged.update(params)
    return _manifest(
        Family.DETECTION,
        merged,
        outputs,
        preprocess=DETECTION_PREPROCESS,
        inputs=DETECTION_INPUT,
        **overrides,
    )


@pytest.mark.parametrize(
    "params, outputs, code",
    [
        ({"decode": "anchors"}, [spec("output", (1, 20, 8))], "UNSUPPORTED_DECODE"),
        ({"decode": "yoloxGrid", "strides": [8, 16]}, [spec("a", (1, 20, 8)), spec("b", (1, 2))], "UNSUPPORTED_OUTPUT_COUNT"),
        ({"decode": "yoloxGrid", "strides": [8, 16]}, [spec("output", (8,))], "UNSUPPORTED_OUTPUT_RANK"),
        ({"decode": "yoloxGrid"}, [spec("output", (1, 20, 8))], "MISSING_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": []}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": [0]}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": [8, 16]}, [spec("output", (1, 20, 9))], "OUTPUT_DIM_MISMATCH"),
        ({"decode": "yoloxGrid", "strides": [8, 16]}, [spec("output", (1, 21, 8))], "OUTPUT_DIM_MISMATCH"),
        ({"decode": "yoloxGrid", "strides": [7]}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": [8, 16], "scoreActivation": "relu"}, [spec("output", (1, 20, 8))], "UNSUPPORTED_ACTIVATION"),
        ({"decode": "yoloxGrid", "strides": [8, 16], "objectnessActivation": "softmax"}, [spec("output", (1, 20, 8))], "UNSUPPORTED_ACTIVATION"),
        ({"decode": "yoloxGrid", "strides": [8, 16], "iouThreshold": 1.5}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": [8, 16], "scoreThreshold": "high"}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
        ({"decode": "yoloxGrid", "strides": [8, 16], "maxDetections": -1}, [spec("output", (1, 20, 8))], "INVALID_FAMILY_PARAM"),
    ],
)
def test_a_grid_head_that_cannot_be_read_is_refused(params, outputs, code):
    assert _refused(_detection(params, outputs)) == code


@pytest.mark.parametrize(
    "params, outputs, code",
    [
        ({"decode": "decodedBoxes"}, [spec("boxes", (1, 3, 4))], "UNSUPPORTED_OUTPUT_COUNT"),
        (
            {"decode": "decodedBoxes"},
            [spec("boxes", (1, 3, 5)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "OUTPUT_DIM_MISMATCH",
        ),
        (
            {"decode": "decodedBoxes"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3))],
            "MISSING_OUTPUT",
        ),
        (
            {"decode": "decodedBoxes", "scoresLayout": "perClass"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3, 4))],
            "CLASS_DIM_MISMATCH",
        ),
        (
            {"decode": "decodedBoxes", "scoresLayout": "everything"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "UNSUPPORTED_SCORES_LAYOUT",
        ),
        (
            {"decode": "decodedBoxes", "boxFormat": "ltrb"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "UNSUPPORTED_BOX_FORMAT",
        ),
        (
            {"decode": "decodedBoxes", "boxCoordinates": "metres"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "UNSUPPORTED_BOX_COORDINATES",
        ),
        (
            {"decode": "decodedBoxes", "classIndexOffset": 1.5},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "INVALID_FAMILY_PARAM",
        ),
        (
            {"decode": "decodedBoxes", "outputNames": {"boxes": 7}},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "INVALID_FAMILY_PARAM",
        ),
        (
            {"decode": "decodedBoxes", "outputNames": "boxes"},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "INVALID_FAMILY_PARAM",
        ),
        (
            {"decode": "decodedBoxes", "outputNames": {"count": "num"}},
            [spec("boxes", (1, 3, 4)), spec("scores", (1, 3)), spec("classes", (1, 3))],
            "MISSING_OUTPUT",
        ),
    ],
)
def test_a_decoded_box_head_that_cannot_be_read_is_refused(params, outputs, code):
    assert _refused(_detection(params, outputs)) == code


SMALL_PREPROCESS = {
    "resize": {"mode": "stretch", "width": 4, "height": 4},
    "layout": "NCHW",
    "dtype": "float32",
}
TINY_INPUT = [spec("images", (1, 3, 4, 4))]


def _small(family, params, outputs):
    return _manifest(family, params, outputs, preprocess=SMALL_PREPROCESS, inputs=TINY_INPUT)


@pytest.mark.parametrize(
    "params, outputs, code",
    [
        ({"labels": ["a", "b"]}, [spec("m", (1, 1, 4, 4)), spec("n", (1, 1))], "UNSUPPORTED_OUTPUT_COUNT"),
        ({"labels": ["a", "b"]}, [spec("m", (4,))], "UNSUPPORTED_OUTPUT_RANK"),
        ({"labels": ["a", "b"], "mode": "watershed"}, [spec("m", (1, 1, 4, 4))], "UNSUPPORTED_SEGMENTATION_MODE"),
        ({"labels": ["a", "b"], "outputLayout": "CHWN"}, [spec("m", (1, 1, 4, 4))], "UNSUPPORTED_LAYOUT"),
        ({"labels": ["a", "b"], "activation": "relu"}, [spec("m", (1, 1, 4, 4))], "UNSUPPORTED_ACTIVATION"),
        ({"labels": ["a", "b"], "minPixels": -1}, [spec("m", (1, 1, 4, 4))], "INVALID_FAMILY_PARAM"),
        ({"labels": ["a", "b", "c"], "mode": "argmax"}, [spec("m", (1, 2, 4, 4))], "CLASS_DIM_MISMATCH"),
        ({"labels": ["a", "b"], "mode": "argmax", "ignoreIndex": 5}, [spec("m", (1, 2, 4, 4))], "INVALID_FAMILY_PARAM"),
        ({"labels": ["a", "b"], "mode": "argmax", "ignoreIndex": "background"}, [spec("m", (1, 2, 4, 4))], "INVALID_FAMILY_PARAM"),
        ({"labels": ["a", "b"], "mode": "threshold"}, [spec("m", (1, 3, 4, 4))], "UNSUPPORTED_OUTPUT_SHAPE"),
        ({"labels": ["a", "b"], "mode": "threshold", "positiveLabel": ""}, [spec("m", (1, 1, 4, 4))], "INVALID_FAMILY_PARAM"),
        ({"labels": ["a", "b"], "mode": "threshold", "threshold": "half"}, [spec("m", (1, 1, 4, 4))], "INVALID_FAMILY_PARAM"),
    ],
)
def test_a_segmentation_head_that_cannot_be_read_is_refused(params, outputs, code):
    assert _refused(_small(Family.SEGMENTATION, params, outputs)) == code


@pytest.mark.parametrize(
    "params, outputs, code",
    [
        ({"threshold": 0.5}, [spec("s", (1,)), spec("t", (1,))], "UNSUPPORTED_OUTPUT_COUNT"),
        ({"threshold": 0.5, "source": "quantile"}, [spec("s", (1,))], "UNSUPPORTED_ANOMALY_SOURCE"),
        ({"threshold": 0.5, "activation": "softmax"}, [spec("s", (1,))], "UNSUPPORTED_ACTIVATION"),
        ({"threshold": 0.5, "direction": "sideways"}, [spec("s", (1,))], "UNSUPPORTED_ANOMALY_DIRECTION"),
        ({}, [spec("s", (1,))], "MISSING_FAMILY_PARAM"),
        ({"threshold": "high"}, [spec("s", (1,))], "INVALID_FAMILY_PARAM"),
        ({"threshold": 0.5, "normalization": 3}, [spec("s", (1,))], "INVALID_FAMILY_PARAM"),
        ({"threshold": 0.5, "normalization": {"min": 1.0}}, [spec("s", (1,))], "INVALID_FAMILY_PARAM"),
        ({"threshold": 0.5, "normalization": {"min": 1.0, "max": 1.0}}, [spec("s", (1,))], "INVALID_FAMILY_PARAM"),
        ({"threshold": 0.5}, [spec("s", (1, 1, 4, 4))], "UNSUPPORTED_OUTPUT_RANK"),
        ({"threshold": 0.5}, [spec("s", (1, 3))], "UNSUPPORTED_OUTPUT_SHAPE"),
        ({"threshold": 0.5, "source": "mapMax"}, [spec("s", (1,))], "UNSUPPORTED_OUTPUT_RANK"),
    ],
)
def test_an_anomaly_head_that_cannot_be_read_is_refused(params, outputs, code):
    assert _refused(_small(Family.ANOMALY, params, outputs)) == code


@pytest.mark.parametrize(
    "family, params, outputs",
    [
        (Family.DETECTION, {"labels": LABELS, "decode": "yoloxGrid", "strides": [8]}, [spec("output", (1, 16, 8))]),
        (Family.SEGMENTATION, {"labels": ["a", "b"]}, [spec("m", (1, 1, 8, 8))]),
        (Family.ANOMALY, {"threshold": 0.5}, [spec("s", (1,))]),
    ],
)
def test_every_family_refuses_a_manifest_that_belongs_to_another(family, params, outputs):
    manifest = _manifest(Family.CLASSIFICATION, params, outputs)
    with pytest.raises(FamilyError) as caught:
        FAMILIES[family].validate_manifest(manifest)
    assert caught.value.code == "FAMILY_MISMATCH"
