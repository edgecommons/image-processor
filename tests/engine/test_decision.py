"""Decision-rule evaluation (DESIGN.md §8.1, §15)."""

from __future__ import annotations

import pytest

from image_processor.engine.decision import decide
from image_processor.types import ClassScore, Detection, Family, NormalizedOutput, Outcome


def _classes():
    return NormalizedOutput(
        family=Family.CLASSIFICATION,
        classes=[
            ClassScore(label="clean", index=0, score=0.97),
            ClassScore(label="dirty", index=1, score=0.03),
        ],
    )


def _detections():
    return NormalizedOutput(
        family=Family.DETECTION,
        detections=[
            Detection(label="bolt", index=0, score=0.81, box=(0.1, 0.2, 0.3, 0.4)),
            Detection(label="washer", index=2, score=0.63, box=(0.1, 0.2, 0.3, 0.4)),
        ],
    )


def _rules(overrides=None):
    rules = {
        "pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9},
        "confidence": "$.classes[0].score",
        "threshold": 0.9,
        "outcomeOnPass": "CLEAR",
        "outcomeOnFail": "HOLD",
    }
    rules.update(overrides or {})
    return rules


BARE = {"confidence": None, "threshold": None}


def test_a_satisfied_rule_clears_and_names_itself():
    decision = decide(_classes(), _rules())
    assert decision.outcome is Outcome.CLEAR
    assert decision.passed is True
    assert decision.confidence == pytest.approx(0.97)
    assert decision.threshold == pytest.approx(0.9)
    assert decision.rule == "pass"


def test_a_failed_rule_takes_the_configured_outcome_and_names_what_failed():
    decision = decide(_classes(), _rules({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.99}}))
    assert decision.outcome is Outcome.HOLD
    assert decision.passed is False
    assert decision.rule == "pass: $.classes[0].score >= 0.99"
    assert decision.confidence == pytest.approx(0.97)


def test_a_failure_can_be_configured_to_fail_outright():
    rules = _rules(
        {"pass": {"path": "$.classes[0].label", "op": "==", "value": "dirty"}, "outcomeOnFail": "FAIL"}
    )
    assert decide(_classes(), rules).outcome is Outcome.FAIL


@pytest.mark.parametrize(
    "operator, value, expected",
    [
        (">=", 0.97, True),
        (">", 0.97, False),
        ("<=", 0.97, True),
        ("<", 0.97, False),
        ("==", 0.97, True),
        ("!=", 0.97, False),
    ],
)
def test_every_comparison_operator_is_applied(operator, value, expected):
    rules = _rules({"pass": {"path": "$.classes[0].score", "op": operator, "value": value}})
    assert decide(_classes(), rules).passed is expected


def test_exists_and_absent_test_the_match_set_itself():
    output = _detections()
    assert decide(output, _rules({"pass": {"path": "$.detections[0]", "op": "exists"}, **BARE})).passed
    assert decide(output, _rules({"pass": {"path": "$.segments.defect", "op": "absent"}, **BARE})).passed
    assert not decide(output, _rules({"pass": {"path": "$.segments.defect", "op": "exists"}, **BARE})).passed


def test_count_compares_how_many_values_matched():
    output = _detections()
    two = _rules({"pass": {"path": "$.detections[*]", "op": "count>=", "value": 2}, **BARE})
    assert decide(output, two).passed is True
    three = _rules({"pass": {"path": "$.detections[*]", "op": "count>=", "value": 3}, **BARE})
    assert decide(output, three).passed is False


def test_a_multi_valued_path_is_a_claim_about_every_match():
    output = _detections()
    every = _rules({"pass": {"path": "$.detections[*].label", "op": "!=", "value": "screw"}, **BARE})
    assert decide(output, every).passed is True
    one = _rules({"pass": {"path": "$.detections[*].label", "op": "!=", "value": "washer"}, **BARE})
    assert decide(output, one).passed is False


def test_all_needs_every_child_and_any_needs_one():
    strong = {"path": "$.classes[0].score", "op": ">=", "value": 0.9}
    weak = {"path": "$.classes[0].score", "op": ">=", "value": 0.99}
    assert decide(_classes(), _rules({"pass": {"all": [strong, strong]}})).passed is True
    assert decide(_classes(), _rules({"pass": {"all": [strong, weak]}})).passed is False
    assert decide(_classes(), _rules({"pass": {"any": [weak, strong]}})).passed is True
    assert decide(_classes(), _rules({"pass": {"any": [weak, weak]}})).passed is False


def test_a_group_failure_names_the_child_that_decided_it():
    strong = {"path": "$.classes[0].score", "op": ">=", "value": 0.9}
    weak = {"path": "$.classes[0].score", "op": ">=", "value": 0.99}
    decision = decide(_classes(), _rules({"pass": {"all": [strong, weak]}}))
    assert decision.rule == "pass.all[1]: $.classes[0].score >= 0.99"
    decision = decide(_classes(), _rules({"pass": {"any": [weak, weak]}}))
    assert decision.rule == "pass.any: none of 2 matched"


def test_nested_groups_are_evaluated():
    strong = {"path": "$.classes[0].score", "op": ">=", "value": 0.9}
    weak = {"path": "$.classes[0].score", "op": ">=", "value": 0.99}
    rules = _rules({"pass": {"all": [{"any": [weak, strong]}, strong]}})
    assert decide(_classes(), rules).passed is True


def test_the_enum_family_is_addressable_as_its_wire_value():
    rules = _rules({"pass": {"path": "$.family", "op": "==", "value": "classification"}, **BARE})
    assert decide(_classes(), rules).passed is True


def test_a_box_element_is_addressable_by_index():
    rules = _rules({"pass": {"path": "$.detections[0].box[2]", "op": ">=", "value": 0.3}, **BARE})
    assert decide(_detections(), rules).passed is True


@pytest.mark.parametrize(
    "rules, where",
    [
        ({}, "rules"),
        ({"pass": "$.classes[0].score >= 0.9"}, "pass"),
        ({"pass": {"path": "$.classes[0].score", "op": "~=", "value": 0.9}}, "pass"),
        ({"pass": {"path": "$$$[[", "op": ">=", "value": 0.9}}, "pass"),
        ({"pass": {"path": 7, "op": ">=", "value": 0.9}}, "pass"),
        ({"pass": {"path": "$.segments.defect.pixels", "op": ">=", "value": 1}}, "pass"),
        ({"pass": {"path": "$.classes[0].label", "op": ">=", "value": 0.9}}, "pass"),
        ({"pass": {"all": []}}, "pass"),
        ({"pass": {"all": "everything"}}, "pass"),
        ({"pass": {"path": "$.detections[*]", "op": "count>=", "value": "two"}}, "pass"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "outcomeOnPass": "MAYBE"}, "outcomeOnPass"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "outcomeOnFail": "CLEAR"}, "outcomeOnFail"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "confidence": "$.nowhere"}, "confidence"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "confidence": "$.classes[0].label"}, "confidence"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "threshold": "$.nowhere"}, "threshold"),
        ({"pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.9}, "threshold": True}, "threshold"),
    ],
)
def test_a_rule_that_cannot_be_evaluated_holds_and_says_so(rules, where):
    decision = decide(_classes(), rules)
    assert decision.outcome is Outcome.HOLD
    assert decision.passed is False
    assert decision.confidence is None
    assert decision.threshold is None
    assert decision.rule.startswith(f"UNEVALUABLE:{where}")


def test_a_non_mapping_rule_set_holds():
    decision = decide(_classes(), ["pass"])
    assert decision.outcome is Outcome.HOLD
    assert decision.rule.startswith("UNEVALUABLE:rules")


def test_fail_on_empty_turns_a_missing_path_into_a_plain_failure():
    rules = {
        "pass": {"path": "$.segments.defect.pixels", "op": "<", "value": 10},
        "outcomeOnPass": "CLEAR",
        "outcomeOnFail": "HOLD",
        "failOnEmpty": True,
    }
    decision = decide(_classes(), rules)
    assert decision.outcome is Outcome.HOLD
    assert decision.passed is False
    assert decision.rule == "pass: $.segments.defect.pixels < 10"


def test_a_threshold_may_be_read_from_the_output_itself():
    output = NormalizedOutput(
        family=Family.ANOMALY, anomaly={"score": 0.2, "threshold": 0.5, "anomalous": False}
    )
    rules = {
        "pass": {"path": "$.anomaly.anomalous", "op": "==", "value": False},
        "confidence": "$.anomaly.score",
        "threshold": "$.anomaly.threshold",
        "outcomeOnPass": "CLEAR",
        "outcomeOnFail": "HOLD",
    }
    decision = decide(output, rules)
    assert decision.outcome is Outcome.CLEAR
    assert decision.threshold == pytest.approx(0.5)
    assert decision.confidence == pytest.approx(0.2)


def test_a_plain_mapping_may_stand_in_for_a_normalized_output():
    document = {"classes": [{"label": "clean", "index": 0, "score": 0.95}]}
    assert decide(document, _rules()).outcome is Outcome.CLEAR


def test_an_outcome_on_pass_other_than_clear_is_honoured():
    rules = _rules({"outcomeOnPass": "HOLD"})
    assert decide(_classes(), rules).outcome is Outcome.HOLD


def test_a_path_that_cannot_be_applied_to_the_document_holds(monkeypatch):
    from image_processor.engine import decision as module

    class Exploding:
        def find(self, _document):
            raise RuntimeError("no")

    monkeypatch.setattr(module, "_compiled", lambda path: Exploding())
    decision = decide(_classes(), _rules())
    assert decision.outcome is Outcome.HOLD
    assert "could not be evaluated" in decision.rule


def test_count_with_a_non_numeric_bound_holds():
    rules = _rules({"pass": {"path": "$.detections[*]", "op": "count>=", "value": "two"}, **BARE})
    decision = decide(_detections(), rules)
    assert decision.outcome is Outcome.HOLD
    assert decision.rule.startswith("UNEVALUABLE:pass")


def test_an_unexpected_failure_inside_evaluation_still_holds(monkeypatch):
    from image_processor.engine import decision as module

    def explode(_normalized):
        raise TypeError("that is not an output")

    monkeypatch.setattr(module, "_document", explode)
    decision = decide(_classes(), _rules())
    assert decision.outcome is Outcome.HOLD
    assert decision.passed is False
    assert decision.rule.startswith("UNEVALUABLE:rules: TypeError")
