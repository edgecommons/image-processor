"""Drawing what a model reported onto a copy of the picture it reported it about.

Every region in an EdgeCommons inference result -- a detection ``box``, a segment ``bbox``, an
anomaly ``summary.bbox`` -- is ``[x, y, w, h]`` normalized to the source image (DESIGN.md 12.1).
That is the whole reason a viewer can draw one without knowing the model input size, and it is
what this module does: multiply by the canvas, draw, caption.

Two decisions are worth stating.

The canvas is not always the source image. A tier-1 fixture is 32 by 32 pixels, and a two-pixel
outline with a caption on a 32-pixel canvas is not a picture of anything. Small sources are
therefore scaled up with nearest-neighbour sampling until their longest side reaches
:data:`MIN_CANVAS_PX`, which changes no pixel value and no box position, because the boxes are
normalized. A source already that large is drawn at its own size.

Colours are a fixed palette indexed by a digest of the label, so a label keeps its colour between
images, between models, and between runs, and no run order can change it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

#: The longest side a thumbnail is reduced to. A source already smaller is copied unchanged, so a
#: 32-pixel fixture stays 32 pixels rather than being blown up into a blurred square.
THUMB_PX = 320

#: The longest side an overlay canvas is scaled up to before anything is drawn on it.
MIN_CANVAS_PX = 640

#: Outline thickness, in canvas pixels.
OUTLINE_PX = 2

#: The label colours, in the order the digest indexes them. They stay legible against both a dark
#: and a light picture and differ from the fixed red an anomaly region is drawn in.
PALETTE = (
    "#2f81f7",
    "#3fb950",
    "#d29922",
    "#a371f7",
    "#39c5cf",
    "#db6d28",
    "#ff7b72",
    "#7ee787",
    "#f778ba",
    "#bc8cff",
    "#56d4dd",
    "#e3b341",
)

#: The colour every anomaly region is drawn in, whatever the label.
ANOMALY_COLOR = "#f85149"

#: The colour a caption is written in, on top of its filled background.
CAPTION_TEXT = "#ffffff"

#: How much room the caption background leaves around the text, in canvas pixels.
CAPTION_PAD = 3


def label_color(label: str) -> str:
    """Return the palette colour one label always draws in.

    Args:
        label: The class label.

    Returns:
        A hex colour. The same label always returns the same one, on every machine and in every
        run, because the index comes from a digest rather than from Python hashing.
    """
    digest = hashlib.blake2b(str(label).encode("utf-8"), digest_size=2).digest()
    return PALETTE[int.from_bytes(digest, "big") % len(PALETTE)]


def denormalize(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Turn a normalized ``[x, y, w, h]`` region into inclusive canvas pixel corners.

    The result is what Pillow draws a rectangle from, so both corners are inside the canvas: a box
    that reaches the right or bottom edge lands on the last pixel column or row rather than one
    past it, and a box narrower than a pixel still covers one.

    Args:
        box: The normalized region, ``[x, y, w, h]``, with the origin at the top left.
        width: The canvas width in pixels.
        height: The canvas height in pixels.

    Returns:
        The ``(x0, y0, x1, y1)`` corners, inclusive of ``x1`` and ``y1``.

    Raises:
        ValueError: The region is not four numbers, or the canvas has no pixels.
    """
    if len(box) != 4:
        raise ValueError(f"a region is four numbers, got {len(box)}")
    if width < 1 or height < 1:
        raise ValueError(f"a canvas is at least one pixel, got {width} by {height}")
    x, y, w, h = (float(value) for value in box)
    x0 = min(max(round(x * width), 0), width - 1)
    y0 = min(max(round(y * height), 0), height - 1)
    x1 = min(max(round((x + w) * width) - 1, x0), width - 1)
    y1 = min(max(round((y + h) * height) - 1, y0), height - 1)
    return x0, y0, x1, y1


def canvas_size(width: int, height: int, minimum: int = MIN_CANVAS_PX) -> Tuple[int, int]:
    """Return the canvas an overlay of this source is drawn on.

    Args:
        width: The source width.
        height: The source height.
        minimum: The longest side the canvas is brought up to.

    Returns:
        The ``(width, height)`` of the canvas. It is the source size whenever the source already
        reaches ``minimum`` on its longest side; otherwise it is the source scaled by a whole
        number, which keeps a pixel a square block rather than a smear.
    """
    longest = max(width, height)
    if longest >= minimum or longest <= 0:
        return width, height
    factor = -(-minimum // longest)
    return width * factor, height * factor


def _font(size: int):
    """Return a bitmap font of about the requested size.

    Pillow bundles its own font, so the site needs no font file and downloads nothing. A Pillow
    release that cannot scale it falls back to the fixed-size default rather than failing.

    Args:
        size: The requested pixel size.

    Returns:
        A Pillow font.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow older than 10.1 cannot scale the default font
        return ImageFont.load_default()


def _caption(draw, text: str, x: int, y: int, color: str, width: int, height: int, font) -> None:
    """Write one caption with a filled background, kept inside the canvas.

    The caption sits above its region when there is room and inside the region top edge when there
    is not, so a box against the top of the picture still carries its label.

    Args:
        draw: The drawing context.
        text: The caption.
        x: The region left edge.
        y: The region top edge.
        color: The background colour, which is the region own colour.
        width: The canvas width.
        height: The canvas height.
        font: The font to write in.
    """
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    box_w = (right - left) + 2 * CAPTION_PAD
    box_h = (bottom - top) + 2 * CAPTION_PAD
    box_x = min(max(x, 0), max(width - box_w, 0))
    box_y = y - box_h
    if box_y < 0:
        box_y = min(y, max(height - box_h, 0))
    draw.rectangle([box_x, box_y, box_x + box_w - 1, box_y + box_h - 1], fill=color)
    draw.text(
        (box_x + CAPTION_PAD - left, box_y + CAPTION_PAD - top), text, fill=CAPTION_TEXT, font=font
    )


def _region(draw, box, color: str, caption: str, width: int, height: int, font) -> None:
    """Draw one outlined region and its caption.

    Args:
        draw: The drawing context.
        box: The normalized region.
        color: The outline and caption-background colour.
        caption: The caption text, or an empty string for no caption.
        width: The canvas width.
        height: The canvas height.
        font: The font to write the caption in.
    """
    x0, y0, x1, y1 = denormalize(box, width, height)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=OUTLINE_PX)
    if caption:
        _caption(draw, caption, x0, y0, color, width, height, font)


def _banner(draw, text: str, color: str, width: int, font) -> None:
    """Draw a full-width strip across the top of the canvas.

    An anomaly model that reduces its map to a single number has no region to point at, so the
    reading is stated across the picture instead of nowhere.

    Args:
        draw: The drawing context.
        text: The strip text.
        color: The strip colour.
        width: The canvas width.
        font: The font to write in.
    """
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    height = (bottom - top) + 2 * CAPTION_PAD
    draw.rectangle([0, 0, width - 1, height - 1], fill=color)
    draw.text((CAPTION_PAD - left, CAPTION_PAD - top), text, fill=CAPTION_TEXT, font=font)


def detection_regions(outputs: Dict) -> List[Tuple[Sequence[float], str, str]]:
    """Describe every detection as a region, a colour, and a caption.

    Args:
        outputs: The result body ``outputs`` block.

    Returns:
        One ``(box, colour, caption)`` triple per detection.
    """
    regions = []
    for item in outputs.get("detections") or ():
        label = str(item.get("label", "?"))
        regions.append((item["box"], label_color(label), f"{label} {float(item['score']):.2f}"))
    return regions


def segment_regions(outputs: Dict) -> List[Tuple[Sequence[float], str, str]]:
    """Describe every segment that claims a region.

    A class with no bounding box claims no pixels worth drawing, and the background class covers
    the frame, so both are left out unless the background is all the model reported.

    Args:
        outputs: The result body ``outputs`` block.

    Returns:
        One ``(box, colour, caption)`` triple per drawn class, widest first.
    """
    segments = outputs.get("segments") or {}
    drawable = {
        label: value
        for label, value in segments.items()
        if value.get("bbox") and label != "background"
    }
    if not drawable:
        drawable = {label: value for label, value in segments.items() if value.get("bbox")}
    regions = []
    for label, value in sorted(
        drawable.items(), key=lambda item: (-float(item[1].get("pixels", 0) or 0), item[0])
    ):
        pixels = int(value.get("pixels", 0) or 0)
        fraction = float(value.get("fraction", 0.0) or 0.0)
        regions.append(
            (value["bbox"], label_color(label), f"{label} {pixels} px ({fraction * 100:.1f}%)")
        )
    return regions


def draw_overlay(
    source: Image.Image, body: Dict, minimum: int = MIN_CANVAS_PX
) -> Optional[Image.Image]:
    """Draw one result body over a copy of its source image.

    Args:
        source: The decoded source image.
        body: The wire-shaped result body.
        minimum: The longest side the canvas is brought up to.

    Returns:
        The overlay image, or ``None`` for a family that has nothing to draw. Classification is
        the one such family: its answer is a ranking, and the detail view shows the ranking as a
        table where the overlay would be.
    """
    outputs = body.get("outputs") or {}
    family = outputs.get("family")
    if family == "classification":
        return None

    width, height = canvas_size(source.width, source.height, minimum)
    canvas = source.convert("RGB")
    if (width, height) != (source.width, source.height):
        canvas = canvas.resize((width, height), Image.NEAREST)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(11, round(min(width, height) / 34)))

    if family == "detection":
        for box, color, caption in detection_regions(outputs):
            _region(draw, box, color, caption, width, height, font)
    elif family == "segmentation":
        for box, color, caption in segment_regions(outputs):
            _region(draw, box, color, caption, width, height, font)
    elif family == "anomaly":
        anomaly = outputs.get("anomaly") or {}
        score = float(anomaly.get("score", 0.0))
        threshold = float(anomaly.get("threshold", 0.0))
        caption = f"{score:.4f} vs {threshold:.4f}"
        bbox = (anomaly.get("summary") or {}).get("bbox")
        if bbox:
            _region(draw, bbox, ANOMALY_COLOR, caption, width, height, font)
        else:
            flagged = bool(anomaly.get("anomalous"))
            state = "ANOMALOUS" if flagged else "within threshold"
            _banner(draw, f"{state}  {caption}", ANOMALY_COLOR if flagged else PALETTE[1], width, font)
    return canvas


def thumbnail(source: Image.Image, longest: int = THUMB_PX) -> Image.Image:
    """Reduce an image for the gallery grid.

    Args:
        source: The decoded source image.
        longest: The longest side the thumbnail is reduced to.

    Returns:
        The thumbnail. An image already within the bound comes back at its own size, so a
        32-pixel fixture is not blown up into a blurred square; the grid scales it in the browser
        instead, which keeps its pixels square.
    """
    copy = source.convert("RGB")
    if max(copy.width, copy.height) <= longest:
        return copy
    copy.thumbnail((longest, longest), Image.LANCZOS)
    return copy


def save_thumbnail(image: Image.Image, target: Path) -> Path:
    """Write a thumbnail beside the site, choosing the format from its size.

    A reduced photograph is a JPEG, because a few hundred of them as PNG would be most of the
    site. An image small enough to need no reduction is a PNG, because it costs a few hundred
    bytes either way and a fixture with hard edges should not carry JPEG ringing.

    Args:
        image: The thumbnail.
        target: The path to write, without a suffix.

    Returns:
        The path written, with the suffix that was chosen.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if max(image.width, image.height) < THUMB_PX:
        path = target.with_suffix(".png")
        image.save(path, format="PNG", optimize=True)
        return path
    path = target.with_suffix(".jpg")
    image.save(path, format="JPEG", quality=85, optimize=True)
    return path