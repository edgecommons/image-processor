"""Decode limits, formats, and refusals (LLD §6, DESIGN.md §15)."""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from PIL import Image

from image_processor.engine.decode import (
    DecodeError,
    DecodeLimits,
    _replicate_gray,
    decode_image,
    is_high_bit_depth,
)

pytestmark = pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")


def _encode(array, image_format="PNG", **options) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format, **options)
    return buffer.getvalue()


def test_every_bad_input_behaves_as_the_oracle_records(corpus):
    for record in corpus.expected["badInputs"]:
        data = corpus.read(record["path"])
        if record["code"] is None:
            image = decode_image(data, DecodeLimits())
            assert list(image.shape) == record["shape"], record["path"]
            assert image.dtype.name == record["dtype"], record["path"]
            if record["dtype"] == "uint16":
                assert int(image.min()) == record["sampleMin"]
                assert int(image.max()) == record["sampleMax"]
                assert is_high_bit_depth(image)
        else:
            with pytest.raises(DecodeError) as caught:
                decode_image(data, DecodeLimits())
            assert caught.value.code == record["code"], record["path"]
            assert caught.value.permanent is True


def test_a_jpeg_named_png_decodes_as_the_jpeg_it_is(corpus):
    record = next(entry for entry in corpus.expected["badInputs"] if "wrong-extension" in entry["path"])
    data = corpus.read(record["path"])
    assert data[:2] == b"\xff\xd8"
    image = decode_image(data, DecodeLimits())
    assert image.shape == (32, 32, 3)
    assert not is_high_bit_depth(image)


def test_sixteen_bit_survives_as_uint16_on_all_three_channels():
    samples = (np.arange(6 * 4, dtype=np.uint16).reshape(4, 6) * np.uint16(2000)).astype(np.uint16)
    image = decode_image(_encode(samples, "TIFF"), DecodeLimits())
    assert image.dtype == np.uint16
    assert image.shape == (4, 6, 3)
    assert np.array_equal(image[:, :, 0], samples)
    assert np.array_equal(image[:, :, 1], samples)
    assert np.array_equal(image[:, :, 2], samples)


def test_byte_budget_is_enforced_before_the_buffer_is_parsed():
    data = _encode(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(DecodeError) as caught:
        decode_image(data, DecodeLimits(max_bytes=4))
    assert caught.value.code == "IMAGE_TOO_LARGE"


def test_an_empty_buffer_is_refused():
    with pytest.raises(DecodeError) as caught:
        decode_image(b"", DecodeLimits())
    assert caught.value.code == "EMPTY_IMAGE"


def test_dimension_and_pixel_budgets_are_separate_gates():
    data = _encode(np.zeros((4, 40, 3), np.uint8))
    with pytest.raises(DecodeError) as caught:
        decode_image(data, DecodeLimits(max_dim=8))
    assert caught.value.code == "IMAGE_DIMENSION_EXCEEDED"

    with pytest.raises(DecodeError) as caught:
        decode_image(data, DecodeLimits(max_pixels=100))
    assert caught.value.code == "IMAGE_PIXELS_EXCEEDED"


def test_grayscale_and_alpha_both_arrive_as_three_channel_rgb():
    grey = decode_image(_encode(np.full((4, 4), 90, np.uint8)), DecodeLimits())
    assert grey.shape == (4, 4, 3)
    assert np.all(grey == 90)

    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    opaque = decode_image(_encode(rgba), DecodeLimits())
    assert opaque.shape == (4, 4, 3)
    assert np.all(opaque[..., 0] == 200)


def test_floating_point_samples_are_refused():
    plane = Image.fromarray(np.zeros((4, 4), np.float32), mode="F")
    buffer = io.BytesIO()
    plane.save(buffer, format="TIFF")
    with pytest.raises(DecodeError) as caught:
        decode_image(buffer.getvalue(), DecodeLimits())
    assert caught.value.code == "UNSUPPORTED_PIXEL_FORMAT"


def test_a_wide_integer_mode_is_narrowed_when_it_fits_and_refused_when_it_does_not(monkeypatch):
    class FakeImage:
        def __init__(self, values, mode):
            self._values = values
            self.mode = mode
            self.size = (values.shape[1], values.shape[0])

        def __array__(self, dtype=None, copy=None):
            return self._values if dtype is None else self._values.astype(dtype)

    from image_processor.engine import decode as module

    fits = FakeImage(np.array([[0, 65535], [1, 2]], dtype=np.int32), "I")
    assert module._to_rgb_array(fits).dtype == np.uint16

    overflows = FakeImage(np.array([[0, 70000], [1, 2]], dtype=np.int32), "I")
    with pytest.raises(DecodeError) as caught:
        module._to_rgb_array(overflows)
    assert caught.value.code == "UNSUPPORTED_PIXEL_FORMAT"


def test_a_multi_channel_high_bit_depth_plane_is_refused():
    with pytest.raises(DecodeError) as caught:
        _replicate_gray(np.zeros((2, 2, 3), np.uint16))
    assert caught.value.code == "UNSUPPORTED_PIXEL_FORMAT"


def test_a_single_channel_hwc_plane_is_accepted():
    widened = _replicate_gray(np.full((2, 2, 1), 7, np.uint16))
    assert widened.shape == (2, 2, 3)


def test_unreadable_pixels_report_the_decode_code(monkeypatch):
    from image_processor.engine import decode as module

    class Broken:
        mode = "RGB"

        def convert(self, mode):
            raise OSError("image file is truncated")

    with pytest.raises(DecodeError) as caught:
        module._to_rgb_array(Broken())
    assert caught.value.code == "UNREADABLE_IMAGE"


def test_the_global_pixel_guard_is_restored_after_a_failed_decode():
    before = Image.MAX_IMAGE_PIXELS
    with pytest.raises(DecodeError):
        decode_image(b"not an image at all", DecodeLimits())
    assert Image.MAX_IMAGE_PIXELS == before


def test_a_degenerate_declared_size_is_refused(monkeypatch):
    from image_processor.engine import decode as module

    class Degenerate:
        size = (0, 4)
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(module.Image, "open", lambda _: Degenerate())
    with pytest.raises(DecodeError) as caught:
        decode_image(b"x", DecodeLimits())
    assert caught.value.code == "UNREADABLE_IMAGE"
