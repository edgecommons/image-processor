"""Decision rules over a normalized task output (DESIGN.md §8.1, §15; LLD §6).

A bundle carries its own rule set, so the component's judgement about an image is the model
author's judgement rather than something hard-coded here. The rules are JSONPath expressions over
:func:`dataclasses.asdict` of the family's :class:`~image_processor.types.NormalizedOutput`, which
means a rule can name any field any family produces, and a new field is addressable the day it
exists.

The one invariant that overrides everything else: **a rule that cannot be evaluated yields
``HOLD``, never ``CLEAR``**. A missing path, a malformed expression, a non-numeric confidence, a
threshold that resolves to nothing, an outcome mapping that names ``CLEAR`` as its failure branch:
all of them are the same answer, and it is the safe one. ``decision.rule`` says which rule
produced it, so the operator sees whether the image failed or the rule set did.

Rule grammar::

    {
      "pass": <expression>,
      "confidence": "$.classes[0].score",
      "threshold": 0.9,
      "outcomeOnPass": "CLEAR",
      "outcomeOnFail": "HOLD",
      "failOnEmpty": false
    }

An ``<expression>`` is a leaf ``{"path": "$...", "op": <operator>, "value": <literal>}`` or a
grouping ``{"all": [<expression>, ...]}`` or ``{"any": [<expression>, ...]}``. A leaf whose path
matches several values is a universal claim: every match must satisfy the comparison, so
``$.detections[*].label != "washer"`` means "no detection is a washer". ``exists`` and ``absent``
test the match set itself and are never unevaluable; ``count>=`` compares its size. For every other
operator an empty match set is unevaluable, and therefore ``HOLD``, unless ``failOnEmpty`` says an
empty match set is a plain failure.
"""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from jsonpath_ng.ext import parse as parse_jsonpath

from image_processor.types import Decision, NormalizedOutput, Outcome

logger = logging.getLogger(__name__)

#: Operators a leaf expression may use.
OPERATORS = (">=", ">", "<=", "<", "==", "!=", "exists", "absent", "count>=")

#: Operators that compare the matched values rather than the match set.
_COMPARISONS = (">=", ">", "<=", "<", "==", "!=")



class _Unevaluable(Exception):
    """Raised inside rule evaluation when the rule set, not the image, is at fault.

    Attributes:
        where: The rule location, such as ``pass.all[1]`` or ``confidence``.
        reason: What could not be evaluated.
    """

    def __init__(self, where: str, reason: str) -> None:
        """Initialize the marker.

        Args:
            where: The rule location.
            reason: What could not be evaluated.
        """
        super().__init__(f"{where}: {reason}")
        self.where = where
        self.reason = reason


@lru_cache(maxsize=512)
def _compiled(path: str):
    """Parse and cache one JSONPath expression.

    Args:
        path: The expression source.

    Returns:
        The parsed jsonpath-ng expression.
    """
    return parse_jsonpath(path)


def _jsonable(value: Any) -> Any:
    """Convert a dataclass dump into plain JSON-shaped values.

    Enum members become their values so ``$.family == "detection"`` reads naturally, and tuples
    become lists so an index into a box behaves like an index into JSON.

    Args:
        value: Any value from :func:`dataclasses.asdict`.

    Returns:
        The same structure using only dicts, lists, and scalars.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(entry) for entry in value]
    return value


def _document(normalized: Any) -> dict:
    """Build the document the rules are evaluated against.

    Args:
        normalized: A :class:`~image_processor.types.NormalizedOutput`, or a mapping already in
            that shape.

    Returns:
        A plain dictionary.
    """
    if isinstance(normalized, NormalizedOutput):
        return _jsonable(dataclasses.asdict(normalized))
    return _jsonable(normalized)


def _matches(document: dict, path: Any, where: str) -> list:
    """Evaluate one JSONPath against the document.

    Args:
        document: The normalized output as a plain dictionary.
        path: The expression source.
        where: The rule location, for the error.

    Returns:
        The matched values, in document order.

    Raises:
        _Unevaluable: When the path is not a string, does not parse, or cannot be applied.
    """
    if not isinstance(path, str) or not path:
        raise _Unevaluable(where, f"path {path!r} is not a JSONPath string")
    try:
        expression = _compiled(path)
    except Exception as error:
        raise _Unevaluable(where, f"path {path!r} does not parse: {error}") from error
    try:
        return [match.value for match in expression.find(document)]
    except Exception as error:
        raise _Unevaluable(where, f"path {path!r} could not be evaluated: {error}") from error


def _numeric(value: Any):
    """Return a value as a float when it is a real number.

    Args:
        value: The candidate.

    Returns:
        The value as a ``float``, or ``None`` when it is not a number. Booleans are not numbers
        here: a rule comparing a flag with ``>=`` is a broken rule, not an implicit cast.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _compare(observed: Any, operator: str, expected: Any, where: str) -> bool:
    """Apply one comparison operator to one matched value.

    Args:
        observed: The value the path matched.
        operator: One of :data:`_COMPARISONS`.
        expected: The literal from the rule.
        where: The rule location, for the error.

    Returns:
        Whether the comparison holds.

    Raises:
        _Unevaluable: When an ordering operator is applied to something that is not a number.
    """
    if operator == "==":
        return bool(observed == expected)
    if operator == "!=":
        return bool(observed != expected)
    left, right = _numeric(observed), _numeric(expected)
    if left is None or right is None:
        raise _Unevaluable(
            where, f"operator {operator!r} needs two numbers, got {observed!r} and {expected!r}"
        )
    if operator == ">=":
        return left >= right
    if operator == ">":
        return left > right
    if operator == "<=":
        return left <= right
    return left < right


def _leaf(document: dict, expression: dict, where: str, fail_on_empty: bool) -> tuple:
    """Evaluate one comparison expression.

    Args:
        document: The normalized output as a plain dictionary.
        expression: The leaf rule.
        where: The rule location.
        fail_on_empty: Whether a path matching nothing is a plain failure rather than a broken
            rule.

    Returns:
        A ``(passed, label)`` pair, where the label names this rule for ``decision.rule``.

    Raises:
        _Unevaluable: When the operator is unknown, the path matches nothing and ``failOnEmpty``
            is not set, or a comparison cannot be made.
    """
    operator = expression.get("op")
    if operator not in OPERATORS:
        raise _Unevaluable(where, f"operator {operator!r} is not one of {list(OPERATORS)}")
    path = expression.get("path")
    expected = expression.get("value")
    label = (
        f"{where}: {path} {operator}"
        if operator in ("exists", "absent")
        else f"{where}: {path} {operator} {expected!r}"
    )
    found = _matches(document, path, where)
    if operator == "exists":
        return bool(found), label
    if operator == "absent":
        return not found, label
    if not found:
        if fail_on_empty:
            return False, label
        raise _Unevaluable(where, f"path {path!r} matched nothing")
    if operator == "count>=":
        wanted = _numeric(expected)
        if wanted is None:
            raise _Unevaluable(where, f"operator 'count>=' needs a number, got {expected!r}")
        return len(found) >= wanted, label
    for observed in found:
        if not _compare(observed, operator, expected, where):
            return False, label
    return True, label


def _evaluate(document: dict, expression: Any, where: str, fail_on_empty: bool) -> tuple:
    """Evaluate one expression, which may group others.

    Args:
        document: The normalized output as a plain dictionary.
        expression: A leaf, an ``all`` group, or an ``any`` group.
        where: The rule location.
        fail_on_empty: Whether a path matching nothing is a plain failure.

    Returns:
        A ``(passed, label)`` pair naming the rule that decided the answer.

    Raises:
        _Unevaluable: When the expression is not a rule, or a child cannot be evaluated.
    """
    if not isinstance(expression, dict):
        raise _Unevaluable(where, f"{expression!r} is not a rule expression")
    for connective in ("all", "any"):
        if connective in expression:
            children = expression[connective]
            if not isinstance(children, (list, tuple)) or not children:
                raise _Unevaluable(where, f"{connective!r} needs a non-empty list of expressions")
            labels = []
            for index, child in enumerate(children):
                passed, label = _evaluate(
                    document, child, f"{where}.{connective}[{index}]", fail_on_empty
                )
                labels.append(label)
                if connective == "all" and not passed:
                    return False, label
                if connective == "any" and passed:
                    return True, label
            return connective == "all", f"{where}.{connective}: none of {len(labels)} matched"
    return _leaf(document, expression, where, fail_on_empty)


def _outcome(rules: dict, key: str, fallback: Outcome) -> Outcome:
    """Read one outcome name from the rule set.

    Args:
        rules: The rule set.
        key: ``"outcomeOnPass"`` or ``"outcomeOnFail"``.
        fallback: The outcome used when the key is absent.

    Returns:
        The named :class:`~image_processor.types.Outcome`.

    Raises:
        _Unevaluable: When the name is not an outcome.
    """
    if key not in rules:
        return fallback
    try:
        return Outcome(rules[key])
    except ValueError as error:
        raise _Unevaluable(key, f"{rules[key]!r} is not an outcome") from error


def _scalar_at(document: dict, spec: Any, where: str) -> float:
    """Resolve a number that a rule set gives either literally or as a JSONPath.

    Args:
        document: The normalized output as a plain dictionary.
        spec: A number, or a JSONPath string.
        where: The rule location, for the error.

    Returns:
        The value as a ``float``.

    Raises:
        _Unevaluable: When the path matches nothing, or the value is not a number. A configured
            confidence or threshold that resolves to nothing is a broken rule set, and a broken
            rule set holds.
    """
    literal = _numeric(spec)
    if literal is not None:
        return literal
    found = _matches(document, spec, where)
    if not found:
        raise _Unevaluable(where, f"path {spec!r} matched nothing")
    value = _numeric(found[0])
    if value is None:
        raise _Unevaluable(where, f"path {spec!r} resolved to {found[0]!r}, which is not a number")
    return value


def _held(where: str, reason: str) -> Decision:
    """Build the fail-safe decision for a rule set that could not be evaluated.

    Args:
        where: The rule location.
        reason: What could not be evaluated.

    Returns:
        A ``HOLD`` decision with no confidence and no threshold: a number that could not be read is
        not a number to report.
    """
    return Decision(
        outcome=Outcome.HOLD,
        passed=False,
        confidence=None,
        threshold=None,
        rule=f"UNEVALUABLE:{where}: {reason}",
    )


def decide(normalized: Any, rules: dict) -> Decision:
    """Apply a bundle's decision rules to one normalized task output.

    Args:
        normalized: The :class:`~image_processor.types.NormalizedOutput` a task family produced,
            or a mapping already in that shape.
        rules: The manifest's ``decisionRules`` block.

    Returns:
        The :class:`~image_processor.types.Decision`. On any evaluation failure the outcome is
        ``HOLD``, ``passed`` is ``False``, and ``rule`` starts with ``UNEVALUABLE:`` and names what
        could not be evaluated. ``CLEAR`` is only ever returned by a rule that evaluated and
        passed.
    """
    try:
        document = _document(normalized)
        if not isinstance(rules, dict) or "pass" not in rules:
            raise _Unevaluable("rules", "no pass rule is configured")
        on_pass = _outcome(rules, "outcomeOnPass", Outcome.CLEAR)
        on_fail = _outcome(rules, "outcomeOnFail", Outcome.HOLD)
        if on_fail is Outcome.CLEAR:
            raise _Unevaluable("outcomeOnFail", "a failing rule cannot clear an image")
        fail_on_empty = bool(rules.get("failOnEmpty", False))
        confidence: Optional[float] = None
        if rules.get("confidence") is not None:
            confidence = _scalar_at(document, rules["confidence"], "confidence")
        threshold: Optional[float] = None
        if rules.get("threshold") is not None:
            threshold = _scalar_at(document, rules["threshold"], "threshold")
        passed, label = _evaluate(document, rules["pass"], "pass", fail_on_empty)
    except _Unevaluable as error:
        logger.warning("decision rule could not be evaluated: %s", error)
        return _held(error.where, error.reason)
    except Exception as error:  # noqa: BLE001 - a broken rule set holds the image, it does not crash
        logger.error("decision rules raised: %s", error, exc_info=True)
        return _held("rules", f"{type(error).__name__}: {error}")
    return Decision(
        outcome=on_pass if passed else on_fail,
        passed=bool(passed),
        confidence=confidence,
        threshold=threshold,
        rule="pass" if passed else label,
    )


# WP6 -- the decision mirror reads a few normalized values out of the committed result body with
# the same JSONPath grammar the rules use, and this package owns the jsonpath-ng dependency
# (LLD 3). Exposing the lookup here keeps it that way.
def resolve_path(document: Any, path: str, default: Any = None) -> Any:
    """Return the single value a JSONPath names in a document.

    Args:
        document: Any JSON-shaped value.
        path: The JSONPath expression.
        default: What to return when the path names nothing, does not parse, or cannot be
            applied. A mirror publishes nothing rather than guessing.

    Returns:
        The first match in document order, or ``default``.
    """
    if not isinstance(path, str) or not path:
        return default
    try:
        matches = _compiled(path).find(_document(document))
    except Exception as error:  # noqa: BLE001 - an unusable path resolves to nothing
        logger.debug("path %r could not be evaluated: %s", path, error)
        return default
    if not matches:
        return default
    return matches[0].value
