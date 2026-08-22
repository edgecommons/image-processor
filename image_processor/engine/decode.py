"""Image decode with hard limits (LLD §6, DESIGN.md §15).

Decoding is the component's largest untrusted-input surface: the bytes come from a camera spool or
a trigger message, and an image header can claim an allocation far larger than the file that
carries it. So the order here is fixed and every step is a gate:

1. the byte count is checked before anything parses the buffer;
2. the header is parsed, with Pillow's own decompression-bomb guard armed from
   :attr:`DecodeLimits.max_pixels`;
3. the declared dimensions are checked before a single pixel is allocated;
4. multi-frame and animated containers are refused, because a still image that is really a video
   makes the frame that was inferred on ambiguous; and only then
5. the pixels are read.

Format comes from the bytes, never from a file name. :func:`decode_image` takes a buffer and no
path, so a JPEG named ``.png`` decodes as the JPEG it is and a text file named ``.png`` is refused.
Every failure is permanent: the same bytes fail the same way on every retry, so a decode failure
sends the job to ``INPUT_INVALID`` rather than to ``RETRY_WAIT``.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

#: Pillow modes that carry one 16-bit unsigned sample per pixel.
_UINT16_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N"})

#: Pillow modes with a wider integer sample that may still fit uint16.
_WIDE_INT_MODES = frozenset({"I", "I;32", "I;32B", "I;32L"})

#: Serializes the process-global Image.MAX_IMAGE_PIXELS swap around one decode.
_LIMIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class DecodeLimits:
    """Bounds applied to one image before it is allowed to allocate memory.

    Attributes:
        max_bytes: Largest accepted encoded buffer. Checked before the buffer is parsed.
        max_pixels: Largest accepted pixel count, width times height. Also arms Pillow's own
            decompression-bomb guard, which refuses a header claiming more than twice this.
        max_dim: Largest accepted width or height, independent of the total pixel count.
    """

    max_bytes: int = 64 * 2**20
    max_pixels: int = 50_000_000
    max_dim: int = 16_384


class DecodeError(Exception):
    """A permanent failure to turn bytes into pixels.

    The same bytes fail the same way every time, so a caller retries nothing and marks the input
    invalid.

    Attributes:
        code: Stable SCREAMING_SNAKE code, safe to put on the bus and in metrics.
        message: Operator-readable detail. Never contains image bytes.
        permanent: Always ``True``, so a caller can branch on the attribute rather than the type.
    """

    permanent = True

    def __init__(self, code: str, message: str) -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def is_high_bit_depth(image: np.ndarray) -> bool:
    """Report whether a decoded image kept more than eight bits per sample.

    Args:
        image: An array returned by :func:`decode_image`.

    Returns:
        ``True`` when the samples are ``uint16``, ``False`` when they are ``uint8``.
    """
    return image.dtype == np.uint16


def decode_image(data: bytes, limits: DecodeLimits) -> np.ndarray:
    """Decode one encoded image into an ``HWC`` RGB array.

    The result always has three channels in red, green, blue order, whatever the source had: a
    grayscale source is replicated across the three channels and an alpha channel is dropped. An
    8-bit source decodes to ``uint8``; a 16-bit TIFF or PNG decodes to ``uint16`` with its full
    range preserved, and :func:`is_high_bit_depth` reports which one you were given.

    Args:
        data: The encoded image bytes. The buffer is the only input and no file name is consulted,
            so the format is decided by content.
        limits: The byte, pixel, and dimension bounds to enforce.

    Returns:
        An ``(H, W, 3)`` array of ``uint8`` or ``uint16``.

    Raises:
        DecodeError: With code ``EMPTY_IMAGE``, ``IMAGE_TOO_LARGE``, ``UNREADABLE_IMAGE``,
            ``IMAGE_DIMENSION_EXCEEDED``, ``IMAGE_PIXELS_EXCEEDED``, ``MULTI_FRAME_IMAGE``, or
            ``UNSUPPORTED_PIXEL_FORMAT``.
    """
    if not data:
        raise DecodeError("EMPTY_IMAGE", "the image buffer is empty")
    if len(data) > limits.max_bytes:
        raise DecodeError(
            "IMAGE_TOO_LARGE",
            f"encoded image is {len(data)} bytes, limit is {limits.max_bytes}",
        )

    with _LIMIT_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = limits.max_pixels
        try:
            return _decode_within_limits(data, limits)
        finally:
            Image.MAX_IMAGE_PIXELS = previous


def _decode_within_limits(data: bytes, limits: DecodeLimits) -> np.ndarray:
    """Decode with Pillow's bomb guard already armed.

    Args:
        data: The encoded image bytes.
        limits: The bounds to enforce.

    Returns:
        An ``(H, W, 3)`` array of ``uint8`` or ``uint16``.

    Raises:
        DecodeError: As documented on :func:`decode_image`.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as error:
        raise DecodeError("IMAGE_PIXELS_EXCEEDED", str(error)) from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise DecodeError("UNREADABLE_IMAGE", f"no decoder accepted the bytes: {error}") from error

    with image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise DecodeError("UNREADABLE_IMAGE", f"degenerate image size {width}x{height}")
        if width > limits.max_dim or height > limits.max_dim:
            raise DecodeError(
                "IMAGE_DIMENSION_EXCEEDED",
                f"image is {width}x{height}, per-side limit is {limits.max_dim}",
            )
        if width * height > limits.max_pixels:
            raise DecodeError(
                "IMAGE_PIXELS_EXCEEDED",
                f"image is {width * height} pixels, limit is {limits.max_pixels}",
            )
        frames = getattr(image, "n_frames", 1)
        if frames > 1 or getattr(image, "is_animated", False):
            raise DecodeError(
                "MULTI_FRAME_IMAGE",
                f"container holds {frames} frames; a still image is required",
            )
        return _to_rgb_array(image)


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    """Materialize one Pillow image as an ``HWC`` RGB array.

    Args:
        image: An opened Pillow image whose dimensions already passed the limits.

    Returns:
        An ``(H, W, 3)`` array of ``uint8`` or ``uint16``.

    Raises:
        DecodeError: When the sample format cannot be represented as ``uint8`` or ``uint16``, or
            when the pixels cannot be read.
    """
    mode = image.mode
    try:
        if mode in _UINT16_MODES:
            return _replicate_gray(np.asarray(image, dtype=np.uint16))
        if mode in _WIDE_INT_MODES:
            wide = np.asarray(image)
            if wide.min() < 0 or wide.max() > np.iinfo(np.uint16).max:
                raise DecodeError(
                    "UNSUPPORTED_PIXEL_FORMAT",
                    f"mode {mode!r} carries samples outside the uint16 range",
                )
            return _replicate_gray(wide.astype(np.uint16))
        if mode.startswith("F"):
            raise DecodeError(
                "UNSUPPORTED_PIXEL_FORMAT", "floating-point samples are not supported"
            )
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)
    except DecodeError:
        raise
    except (OSError, ValueError, SyntaxError) as error:
        raise DecodeError("UNREADABLE_IMAGE", f"pixel data could not be read: {error}") from error


def _replicate_gray(plane: np.ndarray) -> np.ndarray:
    """Turn a single-channel plane into a three-channel RGB array of the same dtype.

    A 16-bit source is monochrome in every format this component accepts, so the one plane becomes
    all three channels. That is the same widening ``convert("RGB")`` performs for an 8-bit
    grayscale image, done at 16 bits so the extra range survives.

    Args:
        plane: An ``(H, W)`` or ``(H, W, 1)`` array.

    Returns:
        An ``(H, W, 3)`` array with the plane repeated across the channel axis.

    Raises:
        DecodeError: When the array is not single-channel.
    """
    if plane.ndim == 3 and plane.shape[2] == 1:
        plane = plane[:, :, 0]
    if plane.ndim != 2:
        raise DecodeError(
            "UNSUPPORTED_PIXEL_FORMAT",
            f"expected a single-channel high-bit-depth image, got shape {plane.shape}",
        )
    return np.repeat(plane[:, :, None], 3, axis=2)
