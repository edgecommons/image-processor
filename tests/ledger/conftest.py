"""Fixtures for the WP3 ledger suite."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_support import StepClock  # noqa: E402

from image_processor.ledger import Ledger  # noqa: E402


@pytest.fixture()
def clock() -> StepClock:
    """A deterministic millisecond clock."""
    return StepClock()


@pytest.fixture()
def db_path(tmp_path) -> Path:
    """The state database path for one test."""
    return tmp_path / "state" / "image-processor.db"


@pytest.fixture()
def ledger(db_path, clock):
    """A file-backed ledger, closed at the end of the test."""
    store = Ledger(db_path, synchronous="NORMAL", clock=clock)
    try:
        yield store
    finally:
        store.close()
