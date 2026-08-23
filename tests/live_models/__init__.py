"""The tier-2 real-model suite (DESIGN.md section 16.1, D-IP-17, D-IP-19).

Tier 1 proves the task families against synthetic graphs whose answers are computed
arithmetically. This tier proves them against real exports: a MobileNetV2 and a ResNet-50 from
the ONNX Model Zoo, YOLOX-Nano and YOLOX-S from the Megvii releases, an SSD-MobileNetV1, an
FCN-ResNet50, and a PatchCore built on VisA capsules. Each one is packed into a signed bundle,
staged through the real cache, run on ``CPUExecutionProvider``, and compared to a committed JSON
golden.

The suite is off by default: models and images are large, they come over the network, and per-PR
CI is meant to stay fast and offline (D-IP-19). Set ``EC_LIVE_MODELS=1`` to run it, after fetching
the corpus with ``tools/fetch_test_assets.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where ``tools/fetch_test_assets.py`` puts the verified corpus.
CACHE_ROOT = REPO_ROOT / "tests" / ".cache"

#: The environment variable that turns the suite on.
LIVE_ENV = "EC_LIVE_MODELS"


def require_live_models() -> None:
    """Skip the calling module unless the tier-2 suite is switched on.

    Call this at the top of every test module in this package. The skip is module level, so a
    machine without the corpus collects the suite without importing a single model.
    """
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(
            f"set {LIVE_ENV}=1 and run tools/fetch_test_assets.py to run the tier-2 real-model suite",
            allow_module_level=True,
        )


def asset_path(asset_id: str, *parts: str) -> Path:
    """Resolve a path inside one fetched asset.

    Args:
        asset_id: The id from ``tests/assets.json``.
        *parts: Path segments below the asset directory.

    Returns:
        The path, whether or not it exists.
    """
    return CACHE_ROOT.joinpath(asset_id, *parts)


def require_asset(asset_id: str, *parts: str) -> Path:
    """Resolve a fetched asset, skipping the test when it is not in the cache.

    Args:
        asset_id: The id from ``tests/assets.json``.
        *parts: Path segments below the asset directory.

    Returns:
        The existing path.
    """
    path = asset_path(asset_id, *parts)
    if not path.exists():
        pytest.skip(f"{path} is missing; run python tools/fetch_test_assets.py --only {asset_id}")
    return path


def relative_name(path: Path) -> str:
    """Name one corpus image the way a golden records it.

    Args:
        path: The image in the fetched cache.

    Returns:
        The path relative to the cache root, in POSIX form, so a golden reads the same on every
        machine.
    """
    return Path(path).resolve().relative_to(CACHE_ROOT.resolve()).as_posix()
