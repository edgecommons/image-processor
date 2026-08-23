"""Put the component root on `sys.path` so `import image_processor.pipeline` resolves from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- WP1 (contracts) ---------------------------------------------------------------------
# Fixtures shared by tests/config and tests/schemas. The DESIGN.md examples are read out of
# the document itself rather than copied, so an edit to the design that the schemas no longer
# accept fails the suite instead of drifting quietly.

import json
import re

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

#: A syntactically valid stand-in for the digest placeholders DESIGN.md writes.
DIGEST = "sha256:" + "ab" * 32
HEX256 = "ab" * 32
ULID = "01KZ8Q4M7N3P5R7T9V1X3Z5B7D"

_PLACEHOLDERS = {"01K...": ULID, "sha256:...": DIGEST, "sha256:<tarball-digest>": DIGEST}
_PLACEHOLDERS_BY_KEY = {
    "sha256": HEX256,
    "class": "sm_120",
    "relativePath": "2026/08/22/cap-0001.jpg",
    "localRelativePath": "cam-01/cap-0001.inference.json",
}


def _design_block(heading: str, fence: str) -> str:
    """Return the first fenced block under a DESIGN.md heading."""
    text = (COMPONENT_ROOT / "DESIGN.md").read_text(encoding="utf-8")
    match = re.search(re.escape(heading) + r".*?```" + fence + r"\n(.*?)\n```", text, re.S)
    assert match is not None, f"DESIGN.md has no {fence} block under {heading!r}"
    return match.group(1)


def _resolve_placeholders(node, key=None):
    """Replace DESIGN.md's illustrative ellipses with values of the right shape."""
    if isinstance(node, dict):
        return {name: _resolve_placeholders(value, name) for name, value in node.items()}
    if isinstance(node, list):
        return [_resolve_placeholders(value, key) for value in node]
    if isinstance(node, str):
        if node in _PLACEHOLDERS:
            return _PLACEHOLDERS[node]
        if "..." in node and key in _PLACEHOLDERS_BY_KEY:
            return _PLACEHOLDERS_BY_KEY[key]
    return node


def _load_schema(relative: str) -> dict:
    return json.loads((COMPONENT_ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def component_root() -> Path:
    """The repository root, which is where the shipped schemas live."""
    return COMPONENT_ROOT


@pytest.fixture(scope="session")
def config_schema() -> dict:
    """The component's own configuration contract."""
    return _load_schema("config.schema.json")


@pytest.fixture(scope="session")
def result_schema() -> dict:
    """The `app/inference/result` body contract."""
    return _load_schema("schemas/inference-result.schema.json")


@pytest.fixture(scope="session")
def manifest_schema() -> dict:
    """The model bundle manifest contract."""
    return _load_schema("schemas/model-bundle-manifest.schema.json")


@pytest.fixture(scope="session")
def design_config_example() -> dict:
    """DESIGN.md §11's configuration example, with its digest placeholders resolved."""
    return _resolve_placeholders(json.loads(_design_block("## 11. Configuration shape", "jsonc")))


@pytest.fixture(scope="session")
def design_result_example() -> dict:
    """DESIGN.md §12.1's result example, with its illustrative ellipses resolved."""
    return _resolve_placeholders(json.loads(_design_block("### 12.1 Inference result", "json")))


# WP7 -- the tier-2 real-model suite writes its goldens from a real run rather than by hand. The
# switch is registered here because pytest only reads `pytest_addoption` from the root conftest,
# so `--update-goldens` is valid whether the suite is run on its own or with everything else.
def pytest_addoption(parser):
    """Register the golden-writing switch (WP7).

    Args:
        parser: pytest's option parser.
    """
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="write tests/goldens/ from this run instead of asserting against it",
    )
