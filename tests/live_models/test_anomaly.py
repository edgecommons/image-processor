"""PatchCore on VisA capsules (DESIGN.md section 16.1, tier 2).

The anomaly model is the one member of the corpus that is built rather than fetched: PatchCore
needs no training epochs, only a memory bank taken from the good split, so
``tools/build_anomaly_model.py`` reconstructs it deterministically from the pinned VisA archive
and nothing binary is hosted anywhere.

The suite therefore skips with an instruction rather than failing when the model has not been
built on this machine.
"""

from __future__ import annotations

import pytest

from tests.live_models import relative_name, require_live_models
from tests.live_models import verify

require_live_models()

#: The model key, which is also the golden file name.
KEY = "patchcore-visa-capsules"


@pytest.fixture(scope="module", autouse=True)
def require_built_model(patchcore_build):
    """Skip the module when the PatchCore model has not been built here.

    Args:
        patchcore_build: The build record, empty when the model is absent.
    """
    if not patchcore_build:
        pytest.skip(
            "the PatchCore anomaly model is not in tests/.cache; build it with\n"
            "    python tools/fetch_test_assets.py --only dataset-visa\n"
            "    python tools/build_anomaly_model.py"
        )


def test_anomaly_matches_its_golden(staged, visa_capsule_images, update_goldens_mode, provider):
    """The staged PatchCore reproduces its committed golden on the capsules slice."""
    model, records = verify.run_corpus(staged, KEY, visa_capsule_images, relative_name)
    assert len(records) == len(visa_capsule_images)
    verify.check(model, records, update_goldens_mode, provider=provider)


def test_the_bad_split_scores_higher_than_the_good_split(staged, visa_capsule_images):
    """Capsules with a defect score above capsules without one, on average.

    This is the property the memory bank is built for, and the one a broken preprocessing chain
    destroys first.
    """
    _, records = verify.run_corpus(staged, KEY, visa_capsule_images, relative_name)
    half = len(records) // 2
    good = [entry["anomaly"]["score"] for entry in records[:half]]
    bad = [entry["anomaly"]["score"] for entry in records[half:]]
    assert sum(bad) / len(bad) > sum(good) / len(good), (
        f"good mean {sum(good) / len(good):.4f}, bad mean {sum(bad) / len(bad):.4f}"
    )


def test_the_build_record_reports_the_separation(patchcore_build):
    """The build records how well the memory bank separates the two splits.

    ``tools/build_anomaly_model.py`` measures the area under the ROC curve on held-out images and
    writes it beside the graph, so the corpus carries the quality of its own anomaly model rather
    than assuming it.
    """
    separation = patchcore_build["separation"]
    assert separation["goodImages"] and separation["badImages"]
    assert 0.0 <= separation["auroc"] <= 1.0
    assert patchcore_build["bytes"] < 200 * 2**20, (
        f"the exported graph is {patchcore_build['bytes']} bytes; the corpus keeps it under 200 MB"
    )
