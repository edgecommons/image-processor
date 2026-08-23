"""Running one model over the corpus, then asserting or rewriting its golden.

Every tier-2 test is the same shape: stage the bundle, run a fixed image list through it, and
either compare the records to ``tests/goldens/<model>.json`` or write that file from the run. This
module holds the shape so each test file states only which model and which images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import pytest

from tests.live_models import runner
from tools import update_goldens


def run_corpus(staged: Callable, key: str, images: List[Path], name_of: Callable) -> tuple:
    """Run one model over a list of images.

    Args:
        staged: The session fixture that builds and stages a bundle.
        key: The model key.
        images: The images to run, in the order the golden records them.
        name_of: How to name one image in the golden.

    Returns:
        A ``(model, records)`` pair.
    """
    model, bundle, session, record_ms = staged(key)
    records = []
    for image in images:
        normalized, decision, session_ms = runner.infer(
            session, bundle.manifest, runner.read_image(image)
        )
        record_ms(session_ms)
        records.append(runner.record(name_of(image), normalized, decision))
    return model, records


def check(model, records: List[Dict], update: bool, golden_dir: Path = update_goldens.GOLDEN_DIR):
    """Assert the records against the model's golden, or write the golden from them.

    Args:
        model: The :class:`~tests.live_models.bundles.LiveModel` the records came from.
        records: The per-image records.
        update: Whether this run writes the golden instead of asserting it.
        golden_dir: Where the goldens live.

    Returns:
        The golden that was asserted against or written.
    """
    tolerances = update_goldens.DEFAULT_TOLERANCES[model.family]
    if update:
        document = runner.golden_document(model, records, tolerances)
        path = update_goldens.write_golden(document, golden_dir)
        print(f"wrote {path} ({path.stat().st_size} bytes, {len(records)} images)")
        return document

    golden = update_goldens.load_golden(model.key, golden_dir)
    if golden is None:
        pytest.fail(
            f"tests/goldens/{model.key}.json is missing; "
            f"run python tools/update_goldens.py --only {model.key}"
        )
    expected = {entry["image"]: entry for entry in golden["images"]}
    observed = {entry["image"]: entry for entry in records}
    assert sorted(expected) == sorted(observed), (
        "the corpus selection changed: "
        f"golden has {len(expected)} images, the run produced {len(observed)}"
    )
    problems: List[str] = []
    for name in sorted(expected):
        for message in update_goldens.compare_record(
            model.family, expected[name], observed[name]
        ):
            problems.append(f"{name}: {message}")
    assert not problems, "\n".join([f"{model.key} differs from its golden:"] + problems)
    return golden
