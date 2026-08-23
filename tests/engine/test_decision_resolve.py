"""Reading one value out of a document with the rule grammar (WP6, the decision mirror)."""

from __future__ import annotations

import pytest

from image_processor.engine.decision import resolve_path

BODY = {
    "status": "SUCCEEDED",
    "decision": {"outcome": "CLEAR", "pass": True, "confidence": 0.997, "threshold": None},
    "outputs": {"classes": [{"label": "clear", "score": 0.9}, {"label": "other", "score": 0.1}]},
}


@pytest.mark.parametrize(
    "path, expected",
    [
        ("$.status", "SUCCEEDED"),
        ("$.decision.pass", True),
        ("$.decision.confidence", 0.997),
        ("$.outputs.classes[0].label", "clear"),
        ("$.outputs.classes[*].label", "clear"),
    ],
)
def test_a_path_resolves_to_its_first_match(path, expected):
    assert resolve_path(BODY, path) == expected


@pytest.mark.parametrize("path", ["$.nope", "$.decision.missing", "$.outputs.classes[9].label"])
def test_a_path_that_names_nothing_returns_the_default(path):
    assert resolve_path(BODY, path, "absent") == "absent"


@pytest.mark.parametrize("path", ["", None, 7, "$.[[["])
def test_an_unusable_path_returns_the_default(path):
    assert resolve_path(BODY, path, "absent") == "absent"


def test_a_normalized_output_is_readable_the_same_way():
    from image_processor.types import ClassScore, Family, NormalizedOutput

    normalized = NormalizedOutput(
        family=Family.CLASSIFICATION, classes=[ClassScore("clear", 0, 0.9)]
    )

    assert resolve_path(normalized, "$.family") == "classification"


def test_a_null_in_the_document_is_returned_as_none():
    assert resolve_path(BODY, "$.decision.threshold", "absent") is None
