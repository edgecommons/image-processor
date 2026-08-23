"""The queue depth and age queries the operator surfaces read (WP6)."""

from __future__ import annotations

import pytest

from image_processor.types import JobState
from tests.ledger.ledger_support import build_job


def test_counting_by_state_covers_every_route(ledger):
    ledger.admit(build_job("01k5job0001", JobState.READY, route_id="cam-01"), 0)
    ledger.admit(build_job("01k5job0002", JobState.READY, route_id="cam-01"), 0)
    ledger.admit(build_job("01k5job0003", JobState.READY, route_id="cam-02"), 0)
    ledger.transition("01k5job0002", JobState.READY, JobState.CLAIMED)

    assert ledger.counts_by_state() == {JobState.READY: 2, JobState.CLAIMED: 1}
    assert ledger.counts_by_state("cam-01") == {JobState.READY: 1, JobState.CLAIMED: 1}
    assert ledger.counts_by_state("cam-03") == {}


def test_the_oldest_job_is_the_one_admitted_first(ledger, clock):
    clock.now = 1_000
    ledger.admit(build_job("01k5job0001", JobState.READY, route_id="cam-01"), 0)
    clock.now = 5_000
    ledger.admit(build_job("01k5job0002", JobState.READY, route_id="cam-02"), 0)

    # the clock ticks once per read, so the recorded stamps are the tick after each set point
    assert ledger.oldest_created_at_ms([JobState.READY]) == 1_001
    assert ledger.oldest_created_at_ms([JobState.READY], "cam-02") == 5_001
    assert ledger.oldest_created_at_ms([JobState.PUBLISHED]) is None
    assert ledger.oldest_created_at_ms([]) is None
