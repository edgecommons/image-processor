"""Fixtures for the tier-2 real-model suite.

Every bundle is built, signed, and staged once for the whole session: packing a 134 MB graph and
re-hashing it per test would dominate the run, and the staged bundle is immutable anyway. The
corpus selection is fixed and sorted, so the same twenty images drive every run and a golden means
the same thing on every machine.

The suite runs once per execution provider. ``CPUExecutionProvider`` is always one of them and is
the provider the committed goldens were produced on; with ``EC_NVIDIA`` set on a machine that has
the CUDA provider, ``CUDAExecutionProvider`` is a second, and every model is asserted against the
same goldens with the same tolerances. That comparison is the CPU-to-CUDA parity gate D-IP-14 asks
for: not two baselines, one baseline and a second provider held to it. The session time of every
image is collected per provider and printed as a table when the session ends.
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

#: The environment variable that adds the CUDA leg.
NVIDIA_ENV = "EC_NVIDIA"


def providers_under_test() -> List[str]:
    """Return the execution providers this run measures.

    Returns:
        ``CPUExecutionProvider`` always, and ``CUDAExecutionProvider`` as well when ``EC_NVIDIA``
        is set.
    """
    chosen = [runner.CPU_PROVIDER]
    if os.environ.get(NVIDIA_ENV):
        chosen.append(runner.CUDA_PROVIDER)
    return chosen


@pytest.fixture(
    scope="session",
    params=providers_under_test(),
    ids=lambda name: name.replace("ExecutionProvider", "").lower(),
)
def provider(request) -> str:
    """The execution provider this parametrization runs every model on.

    Returns:
        The provider name.

    Raises:
        Skipped: The installed ONNX Runtime does not offer it.
    """
    import onnxruntime as ort

    if request.param not in ort.get_available_providers():
        pytest.skip(f"this onnxruntime build has no {request.param}")
    return request.param


@pytest.fixture(scope="session")
def latencies() -> Dict:
    """Collect the graph time of every image, keyed by provider and model.

    Yields:
        The accumulator. On teardown it prints one row per model and provider.
    """
    collected: Dict = {}
    yield collected
    if not collected:
        return
    print()
    print("tier-2 graph time (ms), per image")
    print(f"{'model':<28}{'provider':<10}{'n':>5}{'p50':>9}{'p95':>9}{'mean':>9}")
    for (name, provider_name), samples in sorted(collected.items()):
        ordered = sorted(samples)
        count = len(ordered)
        p50 = ordered[int(count * 0.50)] if count else 0.0
        p95 = ordered[min(int(count * 0.95), count - 1)] if count else 0.0
        mean = sum(ordered) / count if count else 0.0
        short = provider_name.replace("ExecutionProvider", "").lower()
        print(f"{name:<28}{short:<10}{count:>5}{p50:>9.2f}{p95:>9.2f}{mean:>9.2f}")


@pytest.fixture(scope="session")
def update_goldens_mode(pytestconfig, provider) -> bool:
    """Whether this run writes goldens rather than asserting them.

    Goldens are produced on ``CPUExecutionProvider`` alone, which is what makes every other
    provider's run a parity comparison. An update asked for while measuring another provider would
    overwrite the baseline with the thing being compared to it, so it is refused rather than
    quietly ignored.

    Args:
        pytestconfig: The pytest configuration.
        provider: The execution provider this parametrization runs on.

    Returns:
        ``True`` when ``--update-goldens`` was passed or ``EC_UPDATE_GOLDENS=1`` is set.

    Raises:
        Failed: An update was asked for on a provider that does not produce goldens.
    """
    passed = bool(pytestconfig.getoption("--update-goldens", default=False))
    wanted = passed or os.environ.get(update_goldens.UPDATE_ENV) == "1"
    if wanted and provider != runner.CPU_PROVIDER:
        pytest.fail(
            f"goldens are written on {runner.CPU_PROVIDER}; {provider} is asserted against them"
        )
    return wanted



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
def bundle_store(tmp_path_factory, models):
    """Build, sign, and stage one bundle per model, once for the whole session.

    The store is provider-independent on purpose: packing and hashing a 134 MB graph costs more
    than every inference in the suite put together, so the CUDA leg reuses the bundle the CPU leg
    staged and only opens a second session on it.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.
        models: The corpus.

    Returns:
        A callable taking a model key and returning a ``(model, bundle)`` pair.
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
            built[key] = (
                model,
                bundle_support.stage(
                    model, graph, workdir, workdir / "cache", private_key, public_key
                ),
            )
        return built[key]

    return get


@pytest.fixture(scope="session")
def staged(bundle_store, provider, latencies):
    """Open one session per model on the provider this parametrization measures.

    Args:
        bundle_store: The staged bundles.
        provider: The execution provider to run on.
        latencies: The graph-time accumulator.

    Returns:
        A callable taking a model key and returning ``(model, bundle, session, record_ms)``, where
        ``record_ms`` takes one graph time and files it under this model and provider.
    """
    opened: Dict[str, tuple] = {}

    def get(key: str):
        if key not in opened:
            model, bundle = bundle_store(key)
            session = runner.open_session(bundle, provider)
            samples = latencies.setdefault((model.key, provider), [])
            opened[key] = (model, bundle, session, samples.append)
        return opened[key]

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
