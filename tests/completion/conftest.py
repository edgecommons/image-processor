"""Fixtures for the WP3 completion suite."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from completion_support import FakeFs, Policy, StepClock  # noqa: E402

from image_processor.ledger import Ledger  # noqa: E402


@pytest.fixture()
def ledger(tmp_path):
    """A file-backed ledger, closed at the end of the test."""
    store = Ledger(tmp_path / "state.db", synchronous="NORMAL", clock=StepClock())
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def fs() -> FakeFs:
    """A fake filesystem where the archive tree is a second device."""
    return FakeFs()


@pytest.fixture()
def policy() -> Policy:
    """The default route completion policy."""
    return Policy()
