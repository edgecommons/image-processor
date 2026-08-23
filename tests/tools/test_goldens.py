"""The tier-2 golden comparison, exercised on synthetic records.

The tolerances are what decides whether a nightly real-model run is a pass or a regression, so
they are tested here rather than only through a run that needs the corpus: each one is pushed just
inside and just outside its bound.
"""

from __future__ import annotations

import json

import pytest

from tools import update_goldens as goldens


def classification(labels, scores=None):
    """Build a classification record from a label list."""
    scores = scores or [0.9 - 0.1 * index for index in range(len(labels))]
    return {
        "image": "a.jpg",
        "classes": [
            {"label": label, "index": index, "score": score}
            for index, (label, score) in enumerate(zip(labels, scores))
        ],
        "decision": {"outcome": "CLEAR"},
    }


def detection(entries, outcome="CLEAR"):
    """Build a detection record from ``(label, score, box)`` triples."""
    return {
        "image": "a.jpg",
        "detections": [
            {"label": label, "index": 0, "score": score, "box": list(box)}
            for label, score, box in entries
        ],
        "decision": {"outcome": outcome},
    }


def segmentation(fractions):
    """Build a segmentation record from a label-to-fraction mapping."""
    return {
        "image": "a.jpg",
        "segments": {
            label: {"pixels": int(value * 1000), "fraction": value}
            for label, value in fractions.items()
        },
        "decision": {"outcome": "CLEAR"},
    }


def anomaly(score, anomalous, outcome="CLEAR"):
    """Build an anomaly record."""
    return {
        "image": "a.jpg",
        "anomaly": {"score": score, "threshold": 0.5, "anomalous": anomalous},
        "decision": {"outcome": outcome},
    }


def test_round_number_leaves_non_numbers_alone():
    assert goldens.round_number(1.23456789) == 1.234568
    assert goldens.round_number(True) is True
    assert goldens.round_number(None) is None
    assert goldens.round_number("x") == "x"


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], 1.0),
        ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], 0.0),
        ([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], 0.25),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 1.0),
    ],
)
def test_iou(first, second, expected):
    assert goldens.iou(first, second) == pytest.approx(expected)


def test_classification_accepts_a_reordered_tail():
    golden = classification(["tench", "goldfish", "shark", "ray", "eel"])
    run = classification(["tench", "goldfish", "shark", "ray", "barracuda"])
    assert goldens.compare_classification(golden, run) == []


def test_classification_refuses_a_changed_top_one():
    golden = classification(["tench", "goldfish", "shark", "ray", "eel"])
    run = classification(["goldfish", "tench", "shark", "ray", "eel"])
    problems = goldens.compare_classification(golden, run)
    assert len(problems) == 1 and "top-1 label is 'goldfish'" in problems[0]


def test_classification_refuses_too_little_overlap():
    golden = classification(["tench", "goldfish", "shark", "ray", "eel"])
    run = classification(["tench", "cod", "carp", "pike", "perch"])
    problems = goldens.compare_classification(golden, run)
    assert any("share 1 entries" in message for message in problems)


def test_classification_refuses_an_empty_list():
    golden = classification(["tench"])
    run = {"image": "a.jpg", "classes": [], "decision": {"outcome": "CLEAR"}}
    assert "classes is empty" in goldens.compare_classification(golden, run)[0]


def test_detection_accepts_a_box_within_tolerance():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection([("person", 0.93, [0.105, 0.105, 0.40, 0.40])])
    assert goldens.compare_detection(golden, run) == []


def test_detection_refuses_a_box_below_the_iou_floor():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection([("person", 0.90, [0.30, 0.30, 0.40, 0.40])])
    problems = goldens.compare_detection(golden, run)
    assert "no match with IoU >= 0.9" in problems[0]


def test_detection_refuses_a_score_outside_tolerance():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection([("person", 0.80, [0.10, 0.10, 0.40, 0.40])])
    problems = goldens.compare_detection(golden, run)
    assert "difference 0.1000 > 0.05" in problems[0]


def test_detection_refuses_a_relabelled_box():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection([("dog", 0.90, [0.10, 0.10, 0.40, 0.40])])
    problems = goldens.compare_detection(golden, run)
    assert "best overlap of that label is 0.000" in problems[0]
    assert "extra 'dog'" in problems[1]


def test_detection_reports_an_extra_box():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection(
        [("person", 0.90, [0.10, 0.10, 0.40, 0.40]), ("dog", 0.50, [0.60, 0.60, 0.20, 0.20])]
    )
    problems = goldens.compare_detection(golden, run)
    assert len(problems) == 1 and "extra 'dog'" in problems[0]


def test_detection_matches_the_best_overlapping_candidate():
    golden = detection([("person", 0.90, [0.10, 0.10, 0.40, 0.40])])
    run = detection(
        [
            ("person", 0.60, [0.101, 0.101, 0.40, 0.40]),
            ("person", 0.90, [0.100, 0.100, 0.40, 0.40]),
        ]
    )
    problems = goldens.compare_detection(golden, run)
    assert len(problems) == 1 and "extra 'person'" in problems[0]


def test_segmentation_accepts_a_fraction_within_tolerance():
    golden = segmentation({"background": 0.80, "person": 0.20})
    run = segmentation({"background": 0.79, "person": 0.21})
    assert goldens.compare_segmentation(golden, run) == []


def test_segmentation_refuses_a_moved_fraction():
    golden = segmentation({"background": 0.80, "person": 0.20})
    run = segmentation({"background": 0.70, "person": 0.30})
    problems = goldens.compare_segmentation(golden, run)
    assert len(problems) == 2 and "difference 0.1000 > 0.02" in problems[0]


def test_segmentation_reports_a_new_or_missing_class():
    golden = segmentation({"background": 1.0})
    run = segmentation({"person": 1.0})
    problems = goldens.compare_segmentation(golden, run)
    assert "'background'] is missing" in problems[0]
    assert "'person'] is new" in problems[1]


def test_anomaly_accepts_a_score_within_tolerance():
    assert goldens.compare_anomaly(anomaly(0.60, True), anomaly(0.609, True)) == []


def test_anomaly_refuses_a_moved_score_or_a_flipped_verdict():
    problems = goldens.compare_anomaly(anomaly(0.60, True), anomaly(0.40, False))
    assert "difference 0.20000 > 0.01" in problems[0]
    assert "anomalous is False" in problems[1]


def test_anomaly_refuses_a_missing_record():
    assert goldens.compare_anomaly({}, anomaly(0.6, True)) == [
        "anomaly is missing from the golden or from the run"
    ]


def test_compare_record_also_compares_the_decision():
    golden = anomaly(0.60, True, outcome="FAIL")
    run = anomaly(0.60, True, outcome="CLEAR")
    problems = goldens.compare_record("anomaly", golden, run)
    assert problems == ["decision outcome is 'CLEAR', the golden records 'FAIL'"]


def test_compare_record_dispatches_on_the_family():
    for family in goldens.COMPARISONS:
        assert family in goldens.DEFAULT_TOLERANCES
    five = classification(["a", "b", "c", "d", "e"])
    assert goldens.compare_record("classification", five, five) == []
    with pytest.raises(KeyError):
        goldens.compare_record("pose", {}, {})


def test_write_and_load_a_golden(tmp_path):
    document = {"model": "demo", "family": "anomaly", "images": [anomaly(0.6, True)]}
    path = goldens.write_golden(document, tmp_path)
    assert path == goldens.golden_path("demo", tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == document
    assert goldens.load_golden("demo", tmp_path) == document
    assert goldens.load_golden("absent", tmp_path) is None


def test_main_runs_pytest_with_the_switches():
    seen = {}

    def runner(command, env):
        seen["command"] = command
        seen["env"] = env
        return 0

    assert goldens.main([], runner=runner) == 0
    assert "--update-goldens" in seen["command"]
    assert seen["command"][-4:] == ["-o", "addopts=", "-q", "--update-goldens"]
    assert seen["env"][goldens.LIVE_ENV] == "1"
    assert seen["env"][goldens.UPDATE_ENV] == "1"


def test_main_passes_a_selection_and_extra_arguments():
    seen = {}

    def runner(command, env):
        seen["command"] = command
        return 3

    assert goldens.main(["--only", "yolox-nano", "--pytest-arg=-x"], runner=runner) == 3
    assert "-k" in seen["command"]
    assert "yolox_nano" in seen["command"]
    assert seen["command"][-1] == "-x"


def test_the_committed_goldens_are_small_and_readable():
    """Every committed golden parses, names its tolerances, and stays under fifty kilobytes."""
    directory = goldens.GOLDEN_DIR
    files = sorted(directory.glob("*.json"))
    assert files, f"{directory} holds no goldens"
    for path in files:
        assert path.stat().st_size <= 50 * 1024, f"{path.name} is {path.stat().st_size} bytes"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["model"] == path.stem
        assert document["family"] in goldens.COMPARISONS
        assert document["tolerances"] == goldens.DEFAULT_TOLERANCES[document["family"]]
        assert 0 < len(document["images"]) <= 20
        assert len({entry["image"] for entry in document["images"]}) == len(document["images"])
