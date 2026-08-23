"""Fixtures for the tier-2 real-model suite.

Every bundle is built, signed, and staged once for the whole session: packing a 134 MB graph and
re-hashing it per test would dominate the run, and the staged bundle is immutable anyway. The
corpus selection is fixed and sorted, so the same twenty images drive every run and a golden means
the same thing on every machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from tests.live_models import asset_path, require_asset
from tests.live_models import bundles as bundle_support
from tests.live_models import labels as label_sets
from tests.live_models import runner
from tools import update_goldens

#: How many images each model is measured on. Twenty keeps a golden well under fifty kilobytes.
IMAGE_COUNT = 20

#: Where the built PatchCore model and its build record are cached.
PATCHCORE_ASSET = "model-patchcore-visa-capsules"


@pytest.fixture(scope="session")
def update_goldens_mode(pytestconfig) -> bool:
    """Whether this run writes goldens rather than asserting them.

    Args:
        pytestconfig: The pytest configuration.

    Returns:
        ``True`` when ``--update-goldens`` was passed or ``EC_UPDATE_GOLDENS=1`` is set.
    """
    passed = bool(pytestconfig.getoption("--update-goldens", default=False))
    return passed or os.environ.get(update_goldens.UPDATE_ENV) == "1"


@pytest.fixture(scope="session")
def imagenet_labels() -> List[str]:
    """The thousand ImageNet-1k class names, from the pinned synset.

    Returns:
        The class names in index order.
    """
    return label_sets.imagenet_1000(
        require_asset("labels-imagenet-synset", "synset.txt")
    )


@pytest.fixture(scope="session")
def patchcore_build() -> Dict:
    """The PatchCore build record, or an empty mapping when the model was never built.

    Returns:
        The parsed ``build.json`` of the anomaly model, or ``{}``.
    """
    path = asset_path(PATCHCORE_ASSET, "build.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def models(imagenet_labels, patchcore_build) -> Dict[str, bundle_support.LiveModel]:
    """Every model of the corpus, keyed by its golden name.

    Args:
        imagenet_labels: The ImageNet class names.
        patchcore_build: The anomaly model's build record, if it has one.

    Returns:
        The models by key.
    """
    return bundle_support.live_models(imagenet_labels, patchcore_build or None)


@pytest.fixture(scope="session")
def staged(tmp_path_factory, models):
    """Build, sign, and stage one bundle per model, once for the session.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.
        models: The corpus.

    Returns:
        A callable taking a model key and returning a ``(model, bundle, session)`` triple.
    """
    workdir = tmp_path_factory.mktemp("tier2")
    private_key, public_key = bundle_support.keypair()
    built: Dict[str, tuple] = {}

    def get(key: str):
        if key not in built:
            model = models.get(key)
            if model is None:
                pytest.skip(f"{key} is not available on this machine")
            graph = require_asset(model.asset_id, model.filename)
            bundle = bundle_support.stage(
                model, graph, workdir, workdir / "cache", private_key, public_key
            )
            built[key] = (model, bundle, runner.open_session(bundle))
        return built[key]

    return get


@pytest.fixture(scope="session")
def imagenette_images() -> List[Path]:
    """Two validation images from each of Imagenette's ten classes.

    The directory name is the WordNet id of the ImageNet class, so the expected class is known
    without any annotation file.

    Returns:
        Twenty image paths, sorted by class and then by name.
    """
    root = require_asset("dataset-imagenette2-160", "extracted", "imagenette2-160", "val")
    chosen: List[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        chosen.extend(sorted(directory.glob("*.JPEG"))[: IMAGE_COUNT // 10])
    return chosen


@pytest.fixture(scope="session")
def coco_images() -> List[Path]:
    """The first twenty images of the pinned COCO val2017 slice.

    Returns:
        Twenty image paths, sorted by image id.
    """
    root = require_asset("dataset-coco-val2017-slice")
    return sorted(root.glob("*.jpg"))[:IMAGE_COUNT]


@pytest.fixture(scope="session")
def visa_capsule_images() -> List[Path]:
    """Ten good and ten bad VisA capsule images.

    Returns:
        Twenty image paths, the good split first, each half sorted by name.
    """
    root = require_asset(
        "dataset-visa", "extracted", "capsules", "Data", "Images"
    )
    good = sorted((root / "Normal").glob("*.JPG"))[: IMAGE_COUNT // 2]
    bad = sorted((root / "Anomaly").glob("*.JPG"))[: IMAGE_COUNT // 2]
    return good + bad
