"""Shared fixtures for the engine suite.

The corpus is built once per session into a temporary directory: it is deterministic, so one build
serves every test, and it is temporary, so nothing binary ever lands in the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_processor.types import BundleManifest, Family, TensorSpec
from tests.fixtures.build import build, load_bundle_manifest


class Corpus:
    """The generated corpus and its oracle.

    Attributes:
        root: The directory the corpus was built into.
        expected: The parsed ``expected.json`` document.
    """

    def __init__(self, root: Path, expected: dict) -> None:
        """Initialize the handle.

        Args:
            root: The corpus root.
            expected: The oracle document.
        """
        self.root = root
        self.expected = expected

    def path(self, relative: str) -> Path:
        """Resolve one corpus-relative path.

        Args:
            relative: The path recorded in the oracle.

        Returns:
            The absolute path.
        """
        return self.root / relative

    def read(self, relative: str) -> bytes:
        """Read one corpus file.

        Args:
            relative: The path recorded in the oracle.

        Returns:
            The file bytes.
        """
        return self.path(relative).read_bytes()

    def manifest(self, bundle: str) -> BundleManifest:
        """Load one bundle's manifest.

        Args:
            bundle: The bundle key in the oracle, such as ``synthetic-classification-1.0.0``.

        Returns:
            The manifest dataclass.
        """
        return load_bundle_manifest(self.path(self.expected["bundles"][bundle]["path"]))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Corpus:
    """Build the tier-1 corpus once for the whole session.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        The corpus handle.
    """
    root = tmp_path_factory.mktemp("corpus")
    return Corpus(root, build(root))


def spec(name: str, shape, dtype: str = "float32") -> TensorSpec:
    """Build one tensor declaration.

    Args:
        name: The tensor name.
        shape: The declared shape.
        dtype: The element type name.

    Returns:
        The tensor spec.
    """
    return TensorSpec(name=name, dtype=dtype, shape=tuple(shape))


def make_manifest(**overrides) -> BundleManifest:
    """Build a manifest for a unit test, overriding only what the test is about.

    Args:
        **overrides: Any :class:`~image_processor.types.BundleManifest` field.

    Returns:
        The manifest dataclass.
    """
    base = {
        "schema_version": 1,
        "model_id": "unit",
        "version": "1.0.0",
        "files": {},
        "min_onnxruntime": "1.17.0",
        "providers_permitted": ["CPUExecutionProvider"],
        "provider_policy": "preferred",
        "inputs": [spec("images", (1, 3, 8, 8))],
        "outputs": [spec("logits", (1, 3))],
        "dynamic_batch": False,
        "family": Family.CLASSIFICATION,
        "family_params": {"labels": ["a", "b", "c"]},
        "preprocess": {
            "resize": {"mode": "stretch", "width": 8, "height": 8},
            "layout": "NCHW",
            "dtype": "float32",
        },
        "decision_rules": {},
        "max_result_items": 8,
        "estimated_device_mib": 8,
        "warmup": [],
        "tolerances": {},
        "compatibility_keys": {},
        "provenance": {},
        "key_id": None,
        "transform_version": "1",
    }
    base.update(overrides)
    return BundleManifest(**base)


def gradient(height: int, width: int, dtype=np.uint8) -> np.ndarray:
    """Build a deterministic ``HWC`` image for preprocessing tests.

    Args:
        height: Image height.
        width: Image width.
        dtype: ``numpy.uint8`` or ``numpy.uint16``.

    Returns:
        An ``(height, width, 3)`` array whose channels differ from each other.
    """
    top = 255 if dtype == np.uint8 else 65535
    columns = np.linspace(0, top, width)
    rows = np.linspace(0, top, height)
    red = np.broadcast_to(columns[None, :], (height, width))
    green = np.broadcast_to(rows[:, None], (height, width))
    blue = np.full((height, width), top // 2, dtype=float)
    return np.rint(np.stack([red, green, blue], axis=2)).astype(dtype)
