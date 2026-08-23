"""Forgetting announced inputs so a new model generation sees them again (WP6, DESIGN.md 4.3)."""

from __future__ import annotations

from tests.sources.conftest import spool_route, write_capture

DATA = b"a finalized capture"


def test_forgetting_everything_makes_the_next_walk_announce_it_again(tmp_path, events):
    write_capture(tmp_path, "a.jpg", DATA)
    source = _source(tmp_path, events)
    assert source.rescan() == 1
    assert source.rescan() == 0

    assert source.forget() == 1

    assert source.rescan() == 1
    assert len(events.discovered_calls) == 2


def test_forgetting_one_pair_leaves_the_others_announced(tmp_path, events):
    write_capture(tmp_path, "a.jpg", DATA)
    write_capture(tmp_path, "b.jpg", DATA + b"!")
    source = _source(tmp_path, events)
    source.rescan()
    announced = sorted(source.seen())

    assert source.forget([announced[0]]) == 1

    assert source.rescan() == 1
    assert len(source.seen()) == 2


def test_forgetting_a_pair_nobody_announced_changes_nothing(tmp_path, events):
    source = _source(tmp_path, events)

    assert source.forget([("nope.jpg", "ab" * 32)]) == 0


def _source(root, events):
    """Build a spool source over a camera-shaped route."""
    from image_processor.sources.spool import SpoolSource

    return SpoolSource(spool_route(root, camera={}), events)
