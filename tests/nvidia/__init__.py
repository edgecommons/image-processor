"""The tier-3 NVIDIA residency and burst suite (DESIGN.md section 16.1 tier 3, D-IP-13, D-IP-14).

Everything here needs a real CUDA device, ``onnxruntime-gpu``, and the synthesized corpus that
``tools/synth_corpus.py`` builds, so the suite runs only when ``EC_NVIDIA`` is set: the desktop
RTX 5080 under WSL2 and ``lab-5950x``'s RTX 2080 Super. What it proves is what neither a CPU host
nor tier 2 can: that the scheduler, the residency policy, and the executor supervisor hold up when
the configured model set is several times larger than device memory, under four arrival patterns.

The suite is a measurement as much as a gate. Every run writes
``tests/nvidia/results/<gpu-class>-<date>.json``, and those numbers are the Phase 0 SLO baseline
DESIGN.md section 17 records.

Environment:
    ``EC_NVIDIA``: set to any non-empty value to run the suite.
    ``EC_NVIDIA_DEVICE``: the CUDA device ordinal. Defaults to ``0``.
    ``EC_NVIDIA_CORPUS``: the corpus root built by ``tools/synth_corpus.py``. Defaults to
        ``~/ip-corpus``.
    ``EC_NVIDIA_ARRIVALS``: images per arrival pattern. Defaults to ``96``.
    ``EC_NVIDIA_RATE``: arrivals per second for the paced patterns. Defaults to ``8``.
    ``EC_NVIDIA_ROUTES``: how many of the corpus bundles to bind routes to. Defaults to all.
    ``EC_NVIDIA_PATTERNS``: comma-separated subset of ``uniform,zipf,burst,prefetch``.
    ``EC_NVIDIA_RESULTS``: where to write the results file. Defaults to ``tests/nvidia/results``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

#: Repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the results of a run are written.
RESULTS_DIR = REPO_ROOT / "tests" / "nvidia" / "results"

#: The environment variable that turns the suite on.
NVIDIA_ENV = "EC_NVIDIA"

#: The four arrival patterns DESIGN.md section 16.1 names.
PATTERNS = ("uniform", "zipf", "burst", "prefetch")


def require_nvidia() -> None:
    """Skip the calling module unless the tier-3 suite is switched on.

    Call this at the top of every test module in this package, so a machine without a device
    collects the suite without importing ``onnxruntime`` or touching the corpus.
    """
    if not os.environ.get(NVIDIA_ENV):
        pytest.skip(
            f"set {NVIDIA_ENV}=1 on a CUDA host, and build the corpus with "
            "tools/synth_corpus.py, to run the tier-3 residency suite",
            allow_module_level=True,
        )


def device_ordinal() -> int:
    """Return the CUDA device ordinal this run uses."""
    return int(os.environ.get("EC_NVIDIA_DEVICE", "0"))


def corpus_root() -> Path:
    """Return the corpus root this run reads.

    Returns:
        The directory ``tools/synth_corpus.py`` wrote, from ``EC_NVIDIA_CORPUS`` or ``~/ip-corpus``.
    """
    configured = os.environ.get("EC_NVIDIA_CORPUS")
    return Path(configured) if configured else Path.home() / "ip-corpus"


def results_dir() -> Path:
    """Return the directory a run writes its results file into."""
    configured = os.environ.get("EC_NVIDIA_RESULTS")
    return Path(configured) if configured else RESULTS_DIR


def setting(name: str, default):
    """Read one numeric knob from the environment.

    Args:
        name: The environment variable, without the ``EC_NVIDIA_`` prefix.
        default: The value to use when it is unset.

    Returns:
        The configured value, converted to the type of ``default``.
    """
    raw = os.environ.get(f"EC_NVIDIA_{name}")
    if raw is None or raw == "":
        return default
    return type(default)(raw)


def patterns() -> tuple:
    """Return the arrival patterns this run drives, in order.

    Returns:
        The subset of :data:`PATTERNS` named by ``EC_NVIDIA_PATTERNS``, or all of them.
    """
    raw = os.environ.get("EC_NVIDIA_PATTERNS")
    if not raw:
        return PATTERNS
    chosen = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in PATTERNS]
    if unknown:
        raise ValueError(f"unknown arrival pattern(s): {', '.join(unknown)}")
    return tuple(chosen)


def gpu_slug(device_class: str) -> str:
    """Turn a GPU product name into the stem of a results file name.

    Args:
        device_class: The name NVML reports, such as ``NVIDIA GeForce RTX 5080``.

    Returns:
        A lowercase, hyphenated slug, such as ``nvidia-geforce-rtx-5080``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", str(device_class).lower()).strip("-")
    return slug or "unknown-gpu"
