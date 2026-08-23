"""The tier-4 line-clearance renderer: determinism, layout, and the playlist directory.

The scenes feed camera-adapter's ``sim`` playlist pattern (D-IP-18), so the same seed has to give
the same bytes: a harness that renders a different frame on every run cannot be replayed, and a
golden taken from it means nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from tests.live_models import line_clearance


def test_a_frame_is_a_function_of_its_seed_and_index():
    first, names, boxes = line_clearance.render_frame(3, foreign=1)
    second, again, same_boxes = line_clearance.render_frame(3, foreign=1)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert names == again and boxes == same_boxes


def test_different_indices_render_different_frames():
    first, _, _ = line_clearance.render_frame(0, foreign=1)
    second, _, _ = line_clearance.render_frame(1, foreign=1)
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_a_different_seed_renders_a_different_frame():
    first, _, _ = line_clearance.render_frame(0, foreign=1, seed=1)
    second, _, _ = line_clearance.render_frame(0, foreign=1, seed=2)
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_a_clean_frame_carries_no_foreign_object():
    image, names, boxes = line_clearance.render_frame(0)
    assert names == () and boxes == ()
    assert image.size == line_clearance.DEFAULT_SIZE


def test_a_contaminated_frame_reports_a_box_inside_the_tray():
    _, names, boxes = line_clearance.render_frame(5, foreign=2)
    assert len(names) == 2 and len(boxes) == 2
    assert len(set(names)) == 2
    for x, y, width, height in boxes:
        assert 0.0 < x < 1.0 and 0.0 < y < 1.0
        assert width > 0.0 and height > 0.0
        assert x + width <= 1.0 and y + height <= 1.0


def test_a_clean_and_a_contaminated_frame_differ():
    clean, _, _ = line_clearance.render_frame(7)
    dirty, _, _ = line_clearance.render_frame(7, foreign=1)
    difference = np.abs(
        np.asarray(clean, dtype=np.int16) - np.asarray(dirty, dtype=np.int16)
    )
    assert difference.max() > 20
    changed = float((difference.sum(axis=2) > 12).mean())
    assert 0.001 < changed < 0.2, f"a foreign object changed {changed:.4f} of the frame"


@pytest.mark.parametrize("size", [(640, 480), (800, 600)])
def test_the_playlist_directory_is_replayable(tmp_path, size):
    scenes = line_clearance.render_playlist(tmp_path / "playlist", clean=3, dirty=3, size=size)
    directory = tmp_path / "playlist"
    frames = sorted(path.name for path in directory.glob("*.jpg"))

    assert len(scenes) == 6
    assert frames == sorted(scene.name for scene in scenes)
    assert sum(1 for scene in scenes if scene.foreign) == 3
    for name in frames:
        with Image.open(directory / name) as image:
            assert image.format == "JPEG"
            assert image.size == size
            assert image.size[0] >= 640 and image.size[1] >= 480

    index = json.loads((directory / "scenes.json").read_text(encoding="utf-8"))
    assert index["size"] == list(size)
    assert [frame["name"] for frame in index["frames"]] == [scene.name for scene in scenes]
    assert set(directory.iterdir()) == {directory / "scenes.json"} | {
        directory / name for name in frames
    }


def test_the_playlist_bytes_are_stable(tmp_path):
    first = line_clearance.render_playlist(tmp_path / "one", clean=2, dirty=2)
    second = line_clearance.render_playlist(tmp_path / "two", clean=2, dirty=2)
    assert [scene.name for scene in first] == [scene.name for scene in second]
    for scene in first:
        assert (tmp_path / "one" / scene.name).read_bytes() == (
            tmp_path / "two" / scene.name
        ).read_bytes()


def test_the_foreign_objects_cycle_through_the_catalogue(tmp_path):
    scenes = line_clearance.render_playlist(tmp_path / "playlist", clean=0, dirty=4)
    named = {name for scene in scenes for name in scene.foreign}
    assert named == {name for name, _ in line_clearance.FOREIGN_OBJECTS}
