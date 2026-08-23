"""Operator eviction of a resident model generation (DESIGN.md 13, the `evict-model` verb, WP6)."""

from __future__ import annotations

import pytest

from image_processor.engine.protocol import LoadModel, Unload, Unloaded
from image_processor.engine.scheduler import _Pass
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


# -- the context is not the model's to give back (DESIGN.md 10.4) -------------------------------


def test_evicting_the_first_model_off_a_cell_that_built_a_context_does_not_recycle_it(ledger):
    # the shape WP10 measured on both cards: a 271 MiB context and a 198 MiB model, and an unload
    # that returns the model
    cell = FakeCell(context_mib=271, load_mib=198)
    harness = Harness(ledger, cells=[cell])
    _resident(harness)
    harness.scheduler._lanes[DIGEST_A].burst_remaining = 0

    outcome = harness.scheduler.evict(DIGEST_A)

    assert outcome["evicted"] is True
    assert harness.supervisor.recycle_count == 0
    assert harness.scheduler.counters["recycles"] == 0


def test_an_unload_that_really_does_not_reclaim_still_recycles_the_cell(ledger):
    # the same numbers with the context charged to the model, which is what the cell used to
    # report: 198 MiB back out of 469, under the reclaim ratio, and the cell is recycled. The
    # guard is intact; what changed is that a healthy cell no longer trips it.
    cell = FakeCell(context_mib=271, load_mib=198)
    harness = Harness(ledger, cells=[cell])
    _resident(harness)
    harness.scheduler._lanes[DIGEST_A].burst_remaining = 0
    cell.on_unload = lambda message: Unloaded(
        digest=message.digest, freed_mib=198, was_resident=True, expected_mib=469
    )

    harness.scheduler.evict(DIGEST_A)

    assert harness.supervisor.recycle_count == 1


def test_the_scheduler_learns_the_context_from_the_load_and_from_the_stats(ledger):
    cell = FakeCell(context_mib=256, load_mib=512)
    harness = Harness(ledger, cells=[cell])

    assert harness.scheduler._context.get("gpu0-0", 0) == 0
    _resident(harness)

    assert harness.scheduler._context["gpu0-0"] == 256
    assert harness.scheduler.status()["cells"][0]["contextMib"] == 256
    assert harness.scheduler._resident["gpu0-0"][DIGEST_A].size_mib == 512


def test_a_restarted_cell_holds_no_context_until_it_loads_again(ledger):
    cell = FakeCell(context_mib=256, load_mib=512)
    harness = Harness(ledger, cells=[cell])
    _resident(harness)

    harness.supervisor.recycle(cell, "a test restart")
    harness.scheduler._memory(cell, _Pass(now_ms=harness.clock()))

    assert harness.scheduler._context["gpu0-0"] == 0


def test_the_context_is_taken_off_the_budget_when_a_model_is_admitted(ledger):
    cell = FakeCell(context_mib=256, load_mib=512)
    harness = Harness(ledger, cells=[cell])
    _resident(harness)
    seen: list = []
    admit = harness.policy.admit
    harness.policy.admit = lambda *args, **kwargs: (
        seen.append(kwargs.get("context_mib")) or admit(*args, **kwargs)
    )

    admit_ready(harness.ledger, "01k5job0002", DIGEST_B)
    harness.scheduler.submit(harness.ledger.get("01k5job0002"))
    harness.scheduler.run_once()

    assert seen and seen[0] == 256
