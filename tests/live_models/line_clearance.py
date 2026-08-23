"""Procedurally rendered line-clearance scenes for the tier-4 harness (DESIGN.md section 16.1).

Tier 4 replays real imagery through the real camera path: camera-adapter's ``sim`` backend gains a
``playlist`` pattern (D-IP-18) that reads a directory of images and emits them as captures with
genuine sidecars and announcements. This module renders that directory: a tray of capsules on a
conveyor, half of the frames clean and half with a foreign object on the tray.

Rendering only. No model is trained here and nothing is inferred; the scenes are the input a
line-clearance model and the Dallas harness consume.

Every frame is a pure function of the seed and the frame index, so the same call renders the same
bytes, and a scene index records what was drawn so a harness has ground truth without annotating
anything by hand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter

#: Default frame size. Camera-adapter's playlist wants a real capture, not a thumbnail.
DEFAULT_SIZE = (640, 480)

#: JPEG quality the frames are written at.
JPEG_QUALITY = 92

#: The seed the corpus is rendered from.
DEFAULT_SEED = 20260822

#: Conveyor belt colours: body, slat, and the shadow under the tray.
BELT_BODY = (58, 62, 68)
BELT_SLAT = (44, 47, 52)
TRAY_SHADOW = (28, 30, 34)

#: Tray colours: the stainless body and its rim.
TRAY_BODY = (196, 199, 204)
TRAY_RIM = (152, 156, 162)

#: The capsules the tray is loaded with, as body and cap colours.
CAPSULE_COLOURS = (
    ((236, 232, 220), (196, 64, 58)),
    ((236, 232, 220), (46, 96, 168)),
    ((228, 218, 176), (212, 158, 46)),
)

#: The foreign objects a dirty frame carries, each a name and a drawing kind.
FOREIGN_OBJECTS = (
    ("bolt", "bolt"),
    ("washer", "washer"),
    ("glove-fragment", "glove"),
    ("label-scrap", "label"),
)


@dataclass(frozen=True)
class Scene:
    """One rendered frame and what was drawn in it.

    Attributes:
        name: The file name inside the playlist directory.
        foreign: The foreign objects present, empty for a clean frame.
        boxes: One ``[x, y, w, h]`` per foreign object, normalized to the frame.
    """

    name: str
    foreign: Tuple[str, ...]
    boxes: Tuple[Tuple[float, float, float, float], ...]


def _belt(draw: ImageDraw.ImageDraw, size: Tuple[int, int], rng: random.Random) -> None:
    """Paint the conveyor the tray rides on.

    Args:
        draw: The drawing context.
        size: The frame size.
        rng: The frame's random source.
    """
    width, height = size
    draw.rectangle([0, 0, width, height], fill=BELT_BODY)
    offset = rng.randrange(0, 36)
    for x in range(-offset, width + 40, 36):
        draw.polygon(
            [(x, 0), (x + 8, 0), (x + 8 - 26, height), (x - 26, height)], fill=BELT_SLAT
        )
    draw.rectangle([0, 0, width, 10], fill=TRAY_SHADOW)
    draw.rectangle([0, height - 10, width, height], fill=TRAY_SHADOW)


def _tray(draw: ImageDraw.ImageDraw, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Paint the tray and report the rectangle its bed occupies.

    Args:
        draw: The drawing context.
        size: The frame size.

    Returns:
        The ``(left, top, right, bottom)`` of the tray bed in pixels.
    """
    width, height = size
    left, top = int(width * 0.12), int(height * 0.14)
    right, bottom = int(width * 0.88), int(height * 0.86)
    draw.rounded_rectangle([left + 6, top + 8, right + 6, bottom + 8], radius=18, fill=TRAY_SHADOW)
    draw.rounded_rectangle([left, top, right, bottom], radius=18, fill=TRAY_RIM)
    inset = 12
    draw.rounded_rectangle(
        [left + inset, top + inset, right - inset, bottom - inset], radius=10, fill=TRAY_BODY
    )
    return left + inset, top + inset, right - inset, bottom - inset


def _capsules(draw: ImageDraw.ImageDraw, bed: Tuple[int, int, int, int], rng: random.Random,
              columns: int = 6, rows: int = 4) -> None:
    """Load the tray bed with a grid of capsules.

    Args:
        draw: The drawing context.
        bed: The tray bed rectangle.
        rng: The frame's random source.
        columns: How many capsules across.
        rows: How many capsules down.
    """
    left, top, right, bottom = bed
    cell_w = (right - left) / columns
    cell_h = (bottom - top) / rows
    length = cell_w * 0.72
    thickness = cell_h * 0.42
    for row in range(rows):
        for column in range(columns):
            centre_x = left + cell_w * (column + 0.5) + rng.uniform(-2.0, 2.0)
            centre_y = top + cell_h * (row + 0.5) + rng.uniform(-2.0, 2.0)
            body, cap = CAPSULE_COLOURS[(row + column) % len(CAPSULE_COLOURS)]
            x0, y0 = centre_x - length / 2, centre_y - thickness / 2
            x1, y1 = centre_x + length / 2, centre_y + thickness / 2
            draw.rounded_rectangle([x0, y0, x1, y1], radius=thickness / 2, fill=body)
            draw.pieslice(
                [x0, y0, x0 + thickness, y1], start=90, end=270, fill=cap
            )
            draw.rectangle([x0 + thickness / 2, y0, centre_x - length * 0.06, y1], fill=cap)


def _draw_foreign(draw: ImageDraw.ImageDraw, kind: str, centre: Tuple[float, float],
                  scale: float, rng: random.Random) -> Tuple[float, float, float, float]:
    """Draw one foreign object and report the pixel box it occupies.

    Args:
        draw: The drawing context.
        kind: One of ``bolt``, ``washer``, ``glove``, or ``label``.
        centre: Where to put it, in pixels.
        scale: Its size in pixels.
        rng: The frame's random source.

    Returns:
        The ``(x0, y0, x1, y1)`` the object covers.
    """
    x, y = centre
    half = scale / 2
    box = (x - half, y - half, x + half, y + half)
    if kind == "bolt":
        draw.regular_polygon((x, y - half * 0.4, half * 0.55), 6, fill=(96, 100, 108))
        draw.rectangle(
            [x - half * 0.22, y - half * 0.2, x + half * 0.22, y + half], fill=(126, 130, 138)
        )
        for step in range(3):
            offset = y + half * (0.2 + 0.25 * step)
            draw.line([x - half * 0.22, offset, x + half * 0.22, offset], fill=(88, 92, 100))
    elif kind == "washer":
        draw.ellipse(box, fill=(150, 154, 160))
        draw.ellipse(
            [x - half * 0.42, y - half * 0.42, x + half * 0.42, y + half * 0.42],
            fill=TRAY_BODY,
        )
    elif kind == "glove":
        points = [
            (x - half, y + half * 0.4),
            (x - half * 0.5, y - half),
            (x + half * 0.2, y - half * 0.7),
            (x + half, y + half * 0.1),
            (x + half * 0.3, y + half),
        ]
        draw.polygon(points, fill=(84, 156, 214))
    else:
        skew = rng.uniform(-0.25, 0.25) * half
        draw.polygon(
            [
                (x - half, y - half * 0.6 + skew),
                (x + half, y - half * 0.75),
                (x + half * 0.8, y + half * 0.7),
                (x - half * 0.9, y + half * 0.55 + skew),
            ],
            fill=(242, 238, 214),
        )
        for step in range(3):
            offset = y - half * 0.3 + step * half * 0.35
            draw.line([x - half * 0.7, offset, x + half * 0.6, offset], fill=(160, 158, 150))
    return box


def render_frame(index: int, foreign: int = 0, size: Tuple[int, int] = DEFAULT_SIZE,
                 seed: int = DEFAULT_SEED) -> Tuple[Image.Image, Tuple[str, ...], Tuple]:
    """Render one line-clearance frame.

    Args:
        index: The frame number. Every frame of a run gets its own random stream, derived from
            ``seed`` and this number, so a frame is reproducible on its own.
        foreign: How many foreign objects to place on the tray. Zero renders a clean frame.
        size: The frame size in pixels.
        seed: The seed the run is rendered from.

    Returns:
        A ``(image, names, boxes)`` triple, where ``boxes`` are normalized ``[x, y, w, h]``.
    """
    rng = random.Random((seed * 1_000_003) ^ (index * 7919))
    image = Image.new("RGB", size, BELT_BODY)
    draw = ImageDraw.Draw(image)
    _belt(draw, size, rng)
    bed = _tray(draw, size)
    _capsules(draw, bed, rng)

    names: List[str] = []
    boxes: List[Tuple[float, float, float, float]] = []
    left, top, right, bottom = bed
    for slot in range(foreign):
        name, kind = FOREIGN_OBJECTS[(index + slot) % len(FOREIGN_OBJECTS)]
        scale = min(right - left, bottom - top) * rng.uniform(0.14, 0.2)
        centre = (
            rng.uniform(left + scale, right - scale),
            rng.uniform(top + scale, bottom - scale),
        )
        x0, y0, x1, y1 = _draw_foreign(draw, kind, centre, scale, rng)
        names.append(name)
        boxes.append(
            (x0 / size[0], y0 / size[1], (x1 - x0) / size[0], (y1 - y0) / size[1])
        )
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    return image, tuple(names), tuple(boxes)


def render_playlist(dest: Path, clean: int = 12, dirty: int = 12,
                    size: Tuple[int, int] = DEFAULT_SIZE, seed: int = DEFAULT_SEED,
                    foreign_objects: int = 1) -> List[Scene]:
    """Render a playlist directory of clean and contaminated frames.

    The directory holds nothing but the JPEG frames and a ``scenes.json`` index, so
    camera-adapter's ``sim`` playlist pattern can replay it as captures with real sidecars and
    announcements (D-IP-18).

    Args:
        dest: The directory to write. It is created if it does not exist.
        clean: How many frames carry nothing but capsules.
        dirty: How many frames carry a foreign object.
        size: The frame size in pixels. Camera-adapter expects a real capture, so the default is
            640 by 480.
        seed: The seed the run is rendered from.
        foreign_objects: How many foreign objects a contaminated frame carries.

    Returns:
        The scenes, clean frames first, in playlist order.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    scenes: List[Scene] = []
    for position in range(clean + dirty):
        contaminated = position >= clean
        count = foreign_objects if contaminated else 0
        image, names, boxes = render_frame(position, count, size, seed)
        prefix = "foreign" if contaminated else "clean"
        name = f"{prefix}-{position:03d}.jpg"
        image.save(dest / name, "JPEG", quality=JPEG_QUALITY, optimize=False)
        scenes.append(Scene(name=name, foreign=names, boxes=boxes))
    index = {
        "seed": seed,
        "size": list(size),
        "frames": [
            {"name": scene.name, "foreign": list(scene.foreign), "boxes": [list(b) for b in scene.boxes]}
            for scene in scenes
        ],
    }
    (dest / "scenes.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return scenes
