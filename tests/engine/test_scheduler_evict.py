"""Operator eviction of a resident model generation (DESIGN.md 13, the `evict-model` verb, WP6)."""

from __future__ import annotations

import pytest

from image_processor.engine.protocol import LoadModel, Unload
from image_processor.ledger import Ledger
from tests.engine.executor_support import DIGEST_A, DIGEST_B, FakeCell, admit_ready
from tests.engine.test_scheduler import Harness


@pytest.fixture
def ledger(tmp_path):
    """A real ledger in a temporary directory."""
    store = Ledger(tmp_path / "state.db", synchronous="OFF")
    try:
        yield store
    finally:
        store.close()


def _resident(harness) -> None:
    """Get one generation resident by running a job through it."""
    admit_ready(harness.ledger, "01k5job0001", DIGEST_A)
    harness.scheduler.submit(harness.ledger.get("01k5job0001"))
    harness.scheduler.run_once()


def test_evicting_an_idle_generation_unloads_it(ledger):
    harness = Harness(ledger)
    _resident(harness)
    harness.cells[0].lease_burst = 0
    harness.scheduler._lanes[DIGEST_A].burst_remaining = 0

    outcome = harness.scheduler.evict(DIGEST_A)

    assert outcome["evicted"] is True
    assert outcome["digest"] == DIGEST_A
    assert outcome["cells"] == ["gpu0-0"]
    assert [call.digest for call in harness.sent(0, Unload)] == [DIGEST_A]
    assert DIGEST_A not in harness.scheduler._resident["gpu0-0"]


def test_a_leased_generation_is_refused_rather_than_taken_away(ledger):
    harness = Harness(ledger)
    _resident(harness)
    harness.scheduler._lanes[DIGEST_A].burst_remaining = 4

    outcome = harness.scheduler.evict(DIGEST_A)

    assert outcome["evicted"] is False
    assert "leased" in outcome["reason"]
    assert harness.sent(0, Unload) == []


def test_a_generation_that_is_not_resident_says_so(ledger):
    harness = Harness(ledger)

    outcome = harness.scheduler.evict(DIGEST_B)

    assert outcome == {
        "evicted": False,
        "digest": DIGEST_B,
        "cells": [],
        "reason": "the generation is not resident",
    }


def test_a_cell_that_will_not_release_is_recycled_and_reported(ledger):
    from image_processor.engine.protocol import Unloaded

    harness = Harness(ledger)
    _resident(harness)
    harness.scheduler._lanes[DIGEST_A].burst_remaining = 0
    # the cell says it let go of nothing, which is the sticky-memory case DESIGN.md 10.4 recycles
    harness.cells[0].on_unload = lambda message: Unloaded(
        digest=message.digest, freed_mib=0, was_resident=True, expected_mib=4096
    )

    outcome = harness.scheduler.evict(DIGEST_A)

    assert outcome["evicted"] is False
    assert outcome["reason"] == "the cell did not release it"
    assert harness.supervisor.recycles, "the cell that would not release was recycled"
