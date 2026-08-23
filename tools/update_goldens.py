"""Golden comparison for the tier-2 real-model suite, and the tool that regenerates it.

The tier-2 suite (DESIGN.md section 16.1, D-IP-19) runs real models against real images and
compares the answer to a small JSON golden committed under ``tests/goldens/``. Models and images
are never committed, so the golden is the only record of what the corpus produced; this module
owns both halves of that record: the tolerances a comparison applies, and the tool that writes a
new golden from a real run.

A comparison is deliberately not an equality check. The same graph on the same image gives
slightly different last digits on a different ONNX Runtime build, so each family is compared on
what a change in behavior would actually move:

* classification: the top-1 label is the same, and at least four of the top five labels are shared.
* detection: every golden detection is matched by a detection of the same label whose box overlaps
  it by an intersection over union of at least 0.9, with the score within 0.05.
* segmentation: every class holds the same fraction of the class map within 0.02.
* anomaly: the score is within 0.01 and the verdict is the same.

Examples:
    Regenerate every golden from a real run::

        python tools/update_goldens.py

    Regenerate one model's golden::

        python tools/update_goldens.py --only mobilenetv2-12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Repository root, so the defaults work from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the committed goldens live.
GOLDEN_DIR = REPO_ROOT / "tests" / "goldens"

#: Environment variable that turns the tier-2 suite on.
LIVE_ENV = "EC_LIVE_MODELS"

#: Environment variable that makes the tier-2 suite write goldens instead of asserting them.
UPDATE_ENV = "EC_UPDATE_GOLDENS"

#: How many of the top labels must be shared for a classification comparison to pass.
TOP_K_OVERLAP = 4

#: How many labels the shared count is taken from.
TOP_K = 5

#: Smallest accepted intersection over union between a golden box and its match.
IOU_MIN = 0.9

#: Largest accepted difference between a golden detection score and its match.
SCORE_TOLERANCE = 0.05

#: Largest accepted difference between a golden class pixel fraction and the run's.
FRACTION_TOLERANCE = 0.02

#: Largest accepted difference between a golden anomaly score and the run's.
ANOMALY_TOLERANCE = 0.01

#: Decimal places every recorded number is rounded to, so a golden is small and stable.
PLACES = 6

#: The tolerances a golden file records, so the file says how it is read.
DEFAULT_TOLERANCES = {
    "classification": {"topK": TOP_K, "topKOverlap": TOP_K_OVERLAP},
    "detection": {"iouMin": IOU_MIN, "scoreTolerance": SCORE_TOLERANCE},
    "segmentation": {"fractionTolerance": FRACTION_TOLERANCE},
    "anomaly": {"scoreTolerance": ANOMALY_TOLERANCE},
}


def round_number(value: Any, places: int = PLACES) -> Any:
    """Round a number for recording, leaving anything else alone.

    Args:
        value: The value to record.
        places: Decimal places to keep.

    Returns:
        The rounded float, or the value unchanged when it is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), places)


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute the intersection over union of two ``[x, y, w, h]`` boxes.

    Args:
        first: The first box, normalized to the source image.
        second: The second box, in the same space.

    Returns:
        The overlap ratio in ``[0, 1]``. Two empty boxes overlap perfectly, because a model that
        found nothing agrees with a golden that recorded nothing.
    """
    ax0, ay0, aw, ah = (float(value) for value in first)
    bx0, by0, bw, bh = (float(value) for value in second)
    ax1, ay1 = ax0 + max(aw, 0.0), ay0 + max(ah, 0.0)
    bx1, by1 = bx0 + max(bw, 0.0), by0 + max(bh, 0.0)
    left, top = max(ax0, bx0), max(ay0, by0)
    right, bottom = min(ax1, bx1), min(ay1, by1)
    overlap = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - overlap
    if union <= 0.0:
        return 1.0
    return overlap / union


def compare_classification(
    golden: Dict[str, Any], actual: Dict[str, Any], top_k_overlap: int = TOP_K_OVERLAP
) -> List[str]:
    """Compare one classification record against its golden.

    Args:
        golden: The recorded answer, with a ``classes`` list.
        actual: The answer this run produced, in the same shape.
        top_k_overlap: How many recorded labels must also appear in the run.

    Returns:
        One message per difference. An empty list means the record matches.
    """
    problems: List[str] = []
    expected = list(golden.get("classes") or [])
    observed = list(actual.get("classes") or [])
    if not expected or not observed:
        return [f"classes is empty: golden has {len(expected)}, the run has {len(observed)}"]
    if expected[0]["label"] != observed[0]["label"]:
        problems.append(
            f"top-1 label is {observed[0]['label']!r}, "
            f"the golden records {expected[0]['label']!r}"
        )
    shared = {entry["label"] for entry in expected} & {entry["label"] for entry in observed}
    if len(shared) < top_k_overlap:
        problems.append(
            f"top-{len(expected)} labels share {len(shared)} entries, "
            f"at least {top_k_overlap} required; golden "
            f"{[entry['label'] for entry in expected]}, "
            f"run {[entry['label'] for entry in observed]}"
        )
    return problems


def compare_detection(
    golden: Dict[str, Any],
    actual: Dict[str, Any],
    iou_min: float = IOU_MIN,
    score_tolerance: float = SCORE_TOLERANCE,
    boundary_allowance: int = 0,
) -> List[str]:
    """Compare one detection record against its golden.

    Every recorded detection must be matched by one of the same label whose box overlaps it by at
    least ``iou_min`` and whose score is within ``score_tolerance``. A detection the run produced
    and the golden does not record is reported too, so an extra box is a failure rather than a
    silent addition.

    Args:
        golden: The recorded answer, with a ``detections`` list.
        actual: The answer this run produced, in the same shape.
        iou_min: Smallest accepted overlap between a recorded box and its match.
        score_tolerance: Largest accepted score difference.
        boundary_allowance: How many unmatched boxes -- a golden entry with no match, or an
            extra the golden does not record -- the record may carry before they count as
            differences. Zero on the golden provider; a non-golden provider shifts NMS and
            score-threshold boundaries, and a single flipped box is expected there.

    Returns:
        One message per difference. An empty list means the record matches.
    """
    problems: List[str] = []
    boundary: List[str] = []
    expected = list(golden.get("detections") or [])
    observed = list(actual.get("detections") or [])
    unmatched = list(range(len(observed)))
    for position, entry in enumerate(expected):
        same_label = [index for index in unmatched if observed[index]["label"] == entry["label"]]
        overlaps = {index: iou(entry["box"], observed[index]["box"]) for index in same_label}
        candidates = [index for index, value in overlaps.items() if value >= iou_min]
        if not candidates:
            best = max(overlaps.values(), default=0.0)
            boundary.append(
                f"detections[{position}] {entry['label']!r} at {entry['box']} has no match with "
                f"IoU >= {iou_min}; the best overlap of that label is {best:.3f}"
            )
            continue
        chosen = max(candidates, key=lambda index: overlaps[index])
        unmatched.remove(chosen)
        difference = abs(float(entry["score"]) - float(observed[chosen]["score"]))
        if difference > score_tolerance:
            problems.append(
                f"detections[{position}] {entry['label']!r} scores "
                f"{observed[chosen]['score']}, the golden records {entry['score']} "
                f"(difference {difference:.4f} > {score_tolerance})"
            )
    for index in unmatched:
        boundary.append(
            f"the run found an extra {observed[index]['label']!r} at "
            f"{observed[index]['box']} scoring {observed[index]['score']}, "
            f"which the golden does not record"
        )
    if len(boundary) > boundary_allowance:
        problems.extend(boundary)
    return problems


def compare_segmentation(
    golden: Dict[str, Any],
    actual: Dict[str, Any],
    fraction_tolerance: float = FRACTION_TOLERANCE,
) -> List[str]:
    """Compare one segmentation record against its golden.

    Args:
        golden: The recorded answer, with a ``segments`` mapping of label to ``fraction``.
        actual: The answer this run produced, in the same shape.
        fraction_tolerance: Largest accepted difference in the fraction of the class map a class
            claims.

    Returns:
        One message per difference. An empty list means the record matches.
    """
    problems: List[str] = []
    expected = dict(golden.get("segments") or {})
    observed = dict(actual.get("segments") or {})
    for label in sorted(set(expected) | set(observed)):
        if label not in expected:
            problems.append(f"segments[{label!r}] is new in this run")
            continue
        if label not in observed:
            problems.append(f"segments[{label!r}] is missing from this run")
            continue
        difference = abs(float(expected[label]["fraction"]) - float(observed[label]["fraction"]))
        if difference > fraction_tolerance:
            problems.append(
                f"segments[{label!r}] covers {observed[label]['fraction']:.4f} of the class map, "
                f"the golden records {expected[label]['fraction']:.4f} "
                f"(difference {difference:.4f} > {fraction_tolerance})"
            )
    return problems


def compare_anomaly(
    golden: Dict[str, Any],
    actual: Dict[str, Any],
    score_tolerance: float = ANOMALY_TOLERANCE,
    relative: bool = False,
) -> List[str]:
    """Compare one anomaly record against its golden.

    Args:
        golden: The recorded answer, with an ``anomaly`` object.
        actual: The answer this run produced, in the same shape.
        score_tolerance: Largest accepted score difference.
        relative: Whether the tolerance also scales with the score's magnitude
            (``max(score_tolerance, 0.001 * |score|)``). The anomaly score is unnormalized, so
            an absolute tolerance recorded on one provider is too tight for another.

    Returns:
        One message per difference. An empty list means the record matches.
    """
    problems: List[str] = []
    expected = dict(golden.get("anomaly") or {})
    observed = dict(actual.get("anomaly") or {})
    if not expected or not observed:
        return ["anomaly is missing from the golden or from the run"]
    difference = abs(float(expected["score"]) - float(observed["score"]))
    if relative:
        magnitude = max(abs(float(expected["score"])), abs(float(observed["score"])))
        score_tolerance = max(score_tolerance, 0.001 * magnitude)
    if difference > score_tolerance:
        problems.append(
            f"anomaly score is {observed['score']}, the golden records {expected['score']} "
            f"(difference {difference:.5f} > {score_tolerance})"
        )
    if bool(expected["anomalous"]) != bool(observed["anomalous"]):
        problems.append(
            f"anomalous is {observed['anomalous']}, the golden records {expected['anomalous']}"
        )
    return problems


#: The comparison each family is read with.
COMPARISONS = {
    "classification": compare_classification,
    "detection": compare_detection,
    "segmentation": compare_segmentation,
    "anomaly": compare_anomaly,
}


#: Per-family comparison keywords for a provider the goldens were NOT recorded on. The goldens
#: are CPU-recorded (the golden provider gets an empty profile); another provider is held to the
#: same answers with a bounded allowance for numeric boundary effects: one flipped box per image
#: and a relative anomaly tolerance. Decision outcomes are never given an allowance.
NON_GOLDEN_PROFILE: Dict[str, Dict[str, Any]] = {
    "detection": {"iou_min": 0.85, "score_tolerance": 0.1, "boundary_allowance": 1},
    "anomaly": {"relative": True},
}


def comparison_profile(provider: str, golden_provider: str = "CPUExecutionProvider") -> Dict[str, Dict[str, Any]]:
    """Return the per-family comparison keywords for a run on ``provider``.

    Args:
        provider: The execution provider the run used.
        golden_provider: The provider the goldens were recorded on.

    Returns:
        An empty mapping on the golden provider, :data:`NON_GOLDEN_PROFILE` otherwise.
    """
    return {} if provider == golden_provider else NON_GOLDEN_PROFILE


def compare_record(
    family: str,
    golden: Dict[str, Any],
    actual: Dict[str, Any],
    profile: Dict[str, Dict[str, Any]] | None = None,
) -> List[str]:
    """Compare one image's record against its golden, with the family's tolerances.

    The decision outcome is compared for every family: a change in the answer that leaves the
    numbers inside tolerance but flips CLEAR to HOLD is exactly the change the tier-2 suite is
    there to catch.

    Args:
        family: The task family the record came from.
        golden: The recorded answer.
        actual: The answer this run produced.

    Returns:
        One message per difference. An empty list means the record matches.

    Raises:
        KeyError: When the family has no comparison, which means a family was added without one.
    """
    keywords = dict((profile or {}).get(family) or {})
    problems = list(COMPARISONS[family](golden, actual, **keywords))
    expected = (golden.get("decision") or {}).get("outcome")
    observed = (actual.get("decision") or {}).get("outcome")
    if expected != observed:
        problems.append(f"decision outcome is {observed!r}, the golden records {expected!r}")
    return problems


def golden_path(model: str, golden_dir: Path = GOLDEN_DIR) -> Path:
    """Resolve the golden file of one model.

    Args:
        model: The model key, such as ``mobilenetv2-12``.
        golden_dir: Where the goldens live.

    Returns:
        The path of the golden file, whether or not it exists.
    """
    return Path(golden_dir) / f"{model}.json"


def load_golden(model: str, golden_dir: Path = GOLDEN_DIR) -> Optional[Dict[str, Any]]:
    """Read one model's golden.

    Args:
        model: The model key.
        golden_dir: Where the goldens live.

    Returns:
        The parsed golden, or ``None`` when the model has none yet.
    """
    path = golden_path(model, golden_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_golden(document: Dict[str, Any], golden_dir: Path = GOLDEN_DIR) -> Path:
    """Write one model's golden.

    Args:
        document: The golden to record. Its ``model`` names the file.
        golden_dir: Where the goldens live.

    Returns:
        The path written.
    """
    path = golden_path(str(document["model"]), golden_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="update_goldens",
        description=(
            "Run the tier-2 real-model suite and write tests/goldens/ from what it produced. "
            "Fetch the assets with tools/fetch_test_assets.py first."
        ),
    )
    parser.add_argument(
        "--only", help="run one model's test file, for example mobilenetv2-12 or yolox-nano"
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="pytest_args",
        help="extra argument to pass through to pytest; repeatable",
    )
    return parser


def main(argv: Optional[List[str]] = None, runner=subprocess.call) -> int:
    """Run the tier-2 suite in golden-writing mode.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.
        runner: What executes the pytest command. Injected so a test can watch the command
            without running a real suite.

    Returns:
        The exit status pytest returned.
    """
    args = _parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(REPO_ROOT / "tests" / "live_models"),
        "-o",
        "addopts=",
        "-q",
        "--update-goldens",
    ]
    if args.only:
        command += ["-k", args.only.replace("-", "_")]
    command += list(args.pytest_args)
    environment = dict(os.environ)
    environment[LIVE_ENV] = "1"
    environment[UPDATE_ENV] = "1"
    print("running: " + " ".join(command))
    return int(runner(command, env=environment))


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
