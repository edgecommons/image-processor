"""Session fixtures for the tier-3 suite.

The rig is expensive to build -- one CUDA context, forty spool sources, a 23 GiB corpus -- so it is
built once for the session and every arrival pattern runs against it. That is also what makes the
measurement honest: the second pattern starts with whatever the first left resident, which is what
a component that has been up for a week looks like.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from tests.nvidia import (
    corpus_root,
    device_ordinal,
    gpu_slug,
    results_dir,
    setting,
)
from tests.nvidia.harness import Tier3Harness

#: Where the tier-2 corpora the spool writer replays live.
CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"

#: How many images each family's replay pool holds. The pool cycles, so this only bounds how long
#: it takes before a route repeats an image.
POOL_SIZE = 24


@pytest.fixture(scope="session")
def corpus() -> Path:
    """The synthesized tier-3 corpus.

    Returns:
        The corpus root.
    """
    root = corpus_root()
    if not (root / "corpus.json").is_file():
        pytest.skip(
            f"{root}/corpus.json is missing; build it with "
            f"python tools/synth_corpus.py --out {root}"
        )
    return root


@pytest.fixture(scope="session")
def image_pools() -> Dict[str, Tuple[Path, ...]]:
    """The tier-2 images each task family replays.

    Returns:
        Imagenette for the classifiers, the COCO slice for the detectors.
    """
    pools = {}
    imagenette = CACHE_ROOT / "dataset-imagenette2-160" / "extracted" / "imagenette2-160" / "val"
    coco = CACHE_ROOT / "dataset-coco-val2017-slice"
    for family, root, pattern in (
        ("classification", imagenette, "**/*.JPEG"),
        ("detection", coco, "*.jpg"),
    ):
        if not root.exists():
            pytest.skip(f"{root} is missing; run python tools/fetch_test_assets.py")
        chosen = sorted(root.glob(pattern))[:POOL_SIZE]
        if not chosen:
            pytest.skip(f"{root} holds no images to replay")
        pools[family] = tuple(chosen)
    return pools


@pytest.fixture(scope="session")
def workdir() -> Path:
    """Scratch space for the ledger and the route spools.

    The default is a local Linux path rather than pytest's temporary directory, because under WSL2
    a temporary directory on the Windows mount is both slow and outside inotify's reach, and the
    spool source watches its root.

    Returns:
        The directory, emptied before the run.
    """
    import shutil

    configured = os.environ.get("EC_NVIDIA_WORK")
    root = Path(configured) if configured else Path.home() / "ip-tier3-work"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def harness(corpus, image_pools, workdir):
    """The running rig: real subsystems on the configured device.

    Yields:
        The started harness.
    """
    rig = Tier3Harness(
        corpus_root=corpus,
        image_pools=image_pools,
        workdir=workdir,
        device=device_ordinal(),
        route_count=setting("ROUTES", 0) or None,
    )
    rig.start()
    try:
        yield rig
    finally:
        rig.stop()


class Results:
    """What a run measured, and the file it is written to.

    Attributes:
        patterns: One record per arrival pattern, in the order they ran.
        meta: What the run was: device, corpus, knobs, versions.
    """

    def __init__(self, meta: Dict) -> None:
        """Start an empty run record.

        Args:
            meta: The run's identifying metadata.
        """
        self.meta = dict(meta)
        self.patterns: List[Dict] = []

    def add(self, record: Dict) -> None:
        """Keep one pattern's measurement.

        Args:
            record: What ``Tier3Harness.run_pattern`` returned.
        """
        self.patterns.append(record)

    def by_pattern(self, name: str) -> Dict:
        """Return one pattern's record.

        Args:
            name: The pattern name.

        Returns:
            The record, or an empty mapping when that pattern did not run.
        """
        return next((entry for entry in self.patterns if entry["pattern"] == name), {})

    def document(self) -> Dict:
        """Assemble the results document.

        Returns:
            The metadata and every pattern record.
        """
        return {"schemaVersion": 1, **self.meta, "patterns": self.patterns}

    def write(self, directory: Path) -> Path:
        """Write the results file.

        Args:
            directory: Where results are kept.

        Returns:
            The file written, named ``<gpu-class>-<date>.json``.
        """
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{gpu_slug(self.meta.get('gpuClass', ''))}-{date.today().isoformat()}.json"
        path = directory / name
        path.write_text(json.dumps(self.document(), indent=2) + "\n", encoding="utf-8")
        return path


@pytest.fixture(scope="session")
def results(harness, corpus):
    """Collect the run's measurements and write them out at the end.

    Yields:
        The results accumulator.
    """
    import onnxruntime as ort

    from image_processor.engine.residency import NvmlProbe

    reading = NvmlProbe().snapshot(harness.device)
    index = harness.index
    record = Results(
        {
            "gpuClass": reading.device_class or "unknown",
            "deviceTotalMiB": int(reading.total_mib or 0),
            "device": str(harness.device),
            "onnxruntime": ort.__version__,
            "providers": list(harness.runtime.providers),
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "corpus": {
                "root": str(corpus),
                "bundles": len(index["bundles"]),
                "routes": len(harness.routes),
                "tiersMiB": index["tiersMiB"],
                "totalModelMiB": index["totalModelMiB"],
                "seed": index["seed"],
            },
            "budget": {
                "residentMemoryBudgetPercent": harness.gpu.residentMemoryBudgetPercent,
                "reserveMiB": harness.gpu.reserveMiB,
                "budgetMiB": harness.budget_mib(),
            },
            "arrivals": {"perPattern": setting("ARRIVALS", 96), "ratePerSec": setting("RATE", 8.0)},
        }
    )
    yield record
    path = record.write(results_dir())
    print(f"\ntier-3 results: {path}")
