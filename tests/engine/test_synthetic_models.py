"""Every synthetic model, end to end, against its hand-computed answer (DESIGN.md §16.1, tier 1).

This is the suite that makes tier 1 worth running: the ONNX graph is executed on
``CPUExecutionProvider`` (D-IP-14), fed by the real preprocessing, read by the real
postprocessing, and judged by the real decision rules -- and the answer is compared against
``expected.json``, which was computed arithmetically rather than recorded from a previous run.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from image_processor.engine.decision import decide
from image_processor.engine.decode import DecodeLimits, decode_image
from image_processor.engine.families import family_for

#: Tolerance for a score that a float32 graph and a float64 oracle both compute.
SCORE_TOLERANCE = 1e-5

#: Tolerance for a coordinate, which is exact arithmetic on both sides.
GEOMETRY_TOLERANCE = 1e-6


def _run(corpus, bundle_key, image_relative):
    """Decode, preprocess, infer, postprocess, and decide one image against one bundle."""
    bundle = corpus.expected["bundles"][bundle_key]
    directory = corpus.path(bundle["path"])
    manifest = corpus.manifest(bundle_key)
    family = family_for(manifest)
    family.validate_manifest(manifest)

    image = decode_image(corpus.read(image_relative), DecodeLimits())
    feed = family.preprocess(image, manifest)
    session = ort.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
    names = [tensor.name for tensor in session.get_outputs()]
    outputs = dict(zip(names, session.run(names, feed)))
    normalized = family.postprocess(outputs, manifest, (image.shape[0], image.shape[1]))
    return normalized, decide(normalized, manifest.decision_rules)


def _cases(corpus):
    for key, bundle in corpus.expected["bundles"].items():
        for case in bundle["cases"]:
            yield pytest.param(key, case, id=f"{key}:{case['image'].split('/')[-1]}")


def test_the_corpus_covers_all_four_families(corpus):
    families = {bundle["family"] for bundle in corpus.expected["bundles"].values()}
    assert families == {"classification", "detection", "segmentation", "anomaly"}


def test_the_corpus_covers_both_detection_conventions(corpus):
    keys = set(corpus.expected["bundles"])
    assert "synthetic-detection-grid-1.0.0" in keys
    assert "synthetic-detection-decoded-1.0.0" in keys


def _assert_classes(actual, expected, where):
    assert [entry.label for entry in actual] == [entry["label"] for entry in expected], where
    assert [entry.index for entry in actual] == [entry["index"] for entry in expected], where
    for got, want in zip(actual, expected):
        assert got.score == pytest.approx(want["score"], abs=SCORE_TOLERANCE), where


def _assert_detections(actual, expected, where):
    assert [entry.label for entry in actual] == [entry["label"] for entry in expected], where
    assert [entry.index for entry in actual] == [entry["index"] for entry in expected], where
    for got, want in zip(actual, expected):
        assert got.score == pytest.approx(want["score"], abs=SCORE_TOLERANCE), where
        assert list(got.box) == pytest.approx(want["box"], abs=GEOMETRY_TOLERANCE), where


def _assert_segments(actual, expected, where):
    assert set(actual) == set(expected), where
    for label, want in expected.items():
        got = actual[label]
        assert got["pixels"] == want["pixels"], f"{where}:{label}"
        assert got["fraction"] == pytest.approx(want["fraction"]), f"{where}:{label}"
        if want["bbox"] is None:
            assert got["bbox"] is None, f"{where}:{label}"
        else:
            assert got["bbox"] == pytest.approx(want["bbox"], abs=GEOMETRY_TOLERANCE), f"{where}:{label}"


def _assert_anomaly(actual, expected, where):
    assert actual["threshold"] == pytest.approx(expected["threshold"]), where
    assert actual["anomalous"] is expected["anomalous"], where
    assert actual["direction"] == expected["direction"], where
    assert actual["score"] == pytest.approx(expected["score"], abs=SCORE_TOLERANCE), where
    if "summary" not in expected:
        assert "summary" not in actual, where
        return
    summary = actual["summary"]
    for key in ("min", "max", "mean", "fraction"):
        assert summary[key] == pytest.approx(expected["summary"][key], abs=SCORE_TOLERANCE), f"{where}:{key}"
    assert summary["aboveThresholdPixels"] == expected["summary"]["aboveThresholdPixels"], where
    if expected["summary"]["bbox"] is None:
        assert summary["bbox"] is None, where
    else:
        assert summary["bbox"] == pytest.approx(expected["summary"]["bbox"], abs=GEOMETRY_TOLERANCE), where


def _assert_decision(actual, expected, where):
    assert actual.outcome.value == expected["outcome"], where
    assert actual.passed is expected["passed"], where
    assert actual.rule == expected["rule"], where
    for attribute in ("confidence", "threshold"):
        want = expected[attribute]
        got = getattr(actual, attribute)
        if want is None:
            assert got is None, f"{where}:{attribute}"
        else:
            assert got == pytest.approx(want, abs=SCORE_TOLERANCE), f"{where}:{attribute}"


ASSERTIONS = {
    "classes": _assert_classes,
    "detections": _assert_detections,
    "segments": _assert_segments,
    "anomaly": _assert_anomaly,
}


@pytest.mark.parametrize(
    "family",
    ["classification", "detection", "segmentation", "anomaly"],
)
def test_every_synthetic_model_of_a_family_answers_as_the_oracle_says(corpus, family):
    checked = 0
    for key, bundle in corpus.expected["bundles"].items():
        if bundle["family"] != family:
            continue
        for case in bundle["cases"]:
            where = f"{key}:{case['image']}"
            normalized, decision = _run(corpus, key, case["image"])
            for field, want in case["expected"].items():
                ASSERTIONS[field](getattr(normalized, field), want, where)
            _assert_decision(decision, case["decision"], where)
            assert normalized.raw_shapes, where
            checked += 1
    assert checked > 0


def test_both_detection_conventions_agree_on_the_same_image(corpus):
    grid, _ = _run(corpus, "synthetic-detection-grid-1.0.0", "images/detect-scene.png")
    decoded, _ = _run(corpus, "synthetic-detection-decoded-1.0.0", "images/detect-scene.png")
    assert [entry.label for entry in grid.detections] == [entry.label for entry in decoded.detections]
    for left, right in zip(grid.detections, decoded.detections):
        assert left.score == pytest.approx(right.score, abs=SCORE_TOLERANCE)
        assert list(left.box) == pytest.approx(list(right.box), abs=GEOMETRY_TOLERANCE)


def test_a_bundle_whose_rules_are_broken_holds_rather_than_clears(corpus):
    manifest = corpus.manifest("synthetic-classification-1.0.0")
    family = family_for(manifest)
    image = decode_image(corpus.read("images/solid-black.png"), DecodeLimits())
    import onnxruntime

    directory = corpus.path(corpus.expected["bundles"]["synthetic-classification-1.0.0"]["path"])
    session = onnxruntime.InferenceSession(
        str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    names = [tensor.name for tensor in session.get_outputs()]
    outputs = dict(zip(names, session.run(names, family.preprocess(image, manifest))))
    normalized = family.postprocess(outputs, manifest, (image.shape[0], image.shape[1]))

    assert decide(normalized, manifest.decision_rules).outcome.value == "CLEAR"
    broken = dict(manifest.decision_rules)
    broken["pass"] = {"path": "$.nowhere.at.all", "op": ">=", "value": 1}
    assert decide(normalized, broken).outcome.value == "HOLD"
