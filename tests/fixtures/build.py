"""Builds the tier-1 test corpus: synthetic models, images, bad inputs, spool fixtures.

DESIGN.md §16.1 makes tier 1 deterministic and network-free. That is only true if the models are
generated rather than downloaded, and only useful if their answers are *known* rather than
recorded from a previous run. So every graph here has fixed weights chosen so the right answer can
be computed by hand, and this module computes each answer arithmetically -- from the image array
and the baked constants, never by running the model. ``expected.json`` is therefore an oracle: a
test that runs the ONNX graph and compares against it is checking the pipeline, not checking that
the pipeline still agrees with itself.

What gets built into the target directory::

    bundles/<modelId>-<version>/   manifest.json, labels.json, transforms.json, model.onnx
    images/                        the images with computable expected outputs
    bad/                           corrupt, truncated, empty, mislabelled, oversized, 16-bit
    spool/<cameraId>/...           camera-shaped JPEG plus its sidecar, written sidecar-first
    expected.json                  the oracle

Nothing here is committed. The builder is fast enough to run per session, and ``onnx`` is a
test-only dependency: the runtime never imports it.

Run it directly to inspect the corpus::

    python tests/fixtures/build.py --out tests/fixtures/out
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

#: Default target directory. Gitignored; nothing binary is committed.
DEFAULT_OUT = Path(__file__).resolve().parent / "out"

#: Seed for every pseudo-random choice, so two runs produce identical bytes.
SEED = 20260822

#: ONNX opset and IR version. IR 9 keeps the graphs loadable by onnxruntime 1.17 and later.
OPSET = 17
IR_VERSION = 9

#: The camera instance the spool fixtures imitate.
CAMERA_ID = "cam-01"

#: Fixed capture timestamps, so the sidecars are byte-identical between runs.
CAPTURE_DAY = "2026/08/22"


def _sha256(data: bytes) -> str:
    """Hash bytes.

    Args:
        data: The bytes to hash.

    Returns:
        The lower-case hex SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


def _value_info(name: str, shape: tuple, dtype: int = TensorProto.FLOAT):
    """Declare one graph input or output.

    Args:
        name: The tensor name.
        shape: The tensor shape.
        dtype: The ONNX element type.

    Returns:
        The value info proto.
    """
    return helper.make_tensor_value_info(name, dtype, list(shape))


def _initializer(name: str, array: np.ndarray):
    """Bake one constant into a graph.

    Args:
        name: The initializer name.
        array: The constant.

    Returns:
        The tensor proto.
    """
    return numpy_helper.from_array(np.ascontiguousarray(array), name)


def _finish(graph) -> bytes:
    """Check a graph and serialize it.

    Args:
        graph: The graph proto.

    Returns:
        The serialized model.
    """
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _input_zero(input_name: str, nodes: list, initializers: list) -> str:
    """Derive an exact scalar zero from a graph input.

    A head whose answer is baked in still has to consume its input, or the graph would declare an
    input it ignores and a session would be free to skip the feed entirely. Multiplying by zero and
    reducing gives exactly zero for any finite input, which is then added to the constant.

    Args:
        input_name: The graph input to consume.
        nodes: The node list to append to.
        initializers: The initializer list to append to.

    Returns:
        The name of the scalar-zero tensor.
    """
    initializers.append(_initializer("zero_scalar", np.float32(0.0)))
    nodes.append(helper.make_node("Mul", [input_name, "zero_scalar"], ["zeroed"]))
    nodes.append(helper.make_node("ReduceSum", ["zeroed"], ["input_zero"], keepdims=0))
    return "input_zero"


def classification_graph() -> bytes:
    """Build the classification model: identity convolution, global pool, linear head.

    The convolution is a 1x1 identity, so global average pooling produces the mean of each colour
    channel over the scaled image, and the linear head passes those three means through unchanged
    while adding a fourth logit fixed at 0.25. The logits are therefore
    ``[mean(R), mean(G), mean(B), 0.25]`` in units of 0 to 1, and the winning class is whichever
    colour dominates -- or ``other`` when none reaches a quarter of full scale.

    Returns:
        The serialized model.
    """
    nodes = [
        helper.make_node("Conv", ["images", "conv_w", "conv_b"], ["features"], kernel_shape=[1, 1]),
        helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], transB=1),
    ]
    initializers = [
        _initializer("conv_w", np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)),
        _initializer("conv_b", np.zeros(3, dtype=np.float32)),
        _initializer(
            "fc_w",
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32),
        ),
        _initializer("fc_b", np.array([0.0, 0.0, 0.0, 0.25], dtype=np.float32)),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic-classification",
        [_value_info("images", (1, 3, 64, 64))],
        [_value_info("logits", (1, 4))],
        initializers,
    )
    return _finish(graph)


def _detection_rows() -> np.ndarray:
    """Bake the one grid tensor both detection variants agree on.

    Five cells carry a detection and the rest are silent. The set is chosen so every branch of the
    decode is exercised by the answer: two boxes of the same class overlap enough to be suppressed,
    two boxes of different classes occupy exactly the same pixels and must both survive, and one
    box scores below the floor.

    Returns:
        An ``(84, 8)`` ``float32`` array of ``[dx, dy, log_w, log_h, objectness, c0, c1, c2]``.
    """
    rows = np.zeros((84, 8), dtype=np.float32)
    two, one_and_a_half = float(np.log(2.0)), float(np.log(1.5))
    rows[34] = [0.0, 0.0, two, two, 0.9, 0.9, 0.05, 0.05]
    rows[35] = [-0.75, 0.25, two, two, 0.8, 0.8, 0.05, 0.05]
    rows[70] = [1.0, 1.0, one_and_a_half, one_and_a_half, 0.95, 0.02, 0.95, 0.03]
    rows[68] = [1.0, 1.0, 0.0, 0.0, 0.7, 0.05, 0.05, 0.9]
    rows[80] = [0.5, 0.5, 0.0, 0.0, 0.2, 0.3, 0.1, 0.1]
    return rows


def detection_grid_graph() -> bytes:
    """Build the YOLOX-style detection model: one anchor-free grid tensor.

    The tensor is baked, so decode and suppression are the only variables under test. Strides 8,
    16, and 32 over a 64 by 64 input give 64, 16, and 4 cells, concatenated in that order.

    Returns:
        The serialized model.
    """
    nodes, initializers = [], []
    zero = _input_zero("images", nodes, initializers)
    initializers.append(_initializer("grid_const", _detection_rows()[None, :, :]))
    nodes.append(helper.make_node("Add", ["grid_const", zero], ["output"]))
    graph = helper.make_graph(
        nodes,
        "synthetic-detection-grid",
        [_value_info("images", (1, 3, 64, 64))],
        [_value_info("output", (1, 84, 8))],
        initializers,
    )
    return _finish(graph)


#: The already-decoded form of the same five detections, in ``yxyx`` normalized to the 64 by 64
#: model canvas, plus a sixth all-zero row that scores nothing.
_SSD_BOXES = np.array(
    [
        [20.0, 36.0, 44.0, 60.0],
        [24.0, 8.0, 40.0, 24.0],
        [26.0, 10.0, 42.0, 26.0],
        [24.0, 8.0, 40.0, 24.0],
        [16.0, 0.0, 24.0, 8.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
) / np.float32(64.0)

#: Per-box confidences for :data:`_SSD_BOXES`, matching the grid variant's decoded scores.
_SSD_SCORES = np.array([0.9025, 0.81, 0.64, 0.63, 0.06, 0.0], dtype=np.float32)

#: Per-box class ids for :data:`_SSD_BOXES`.
_SSD_CLASSES = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32)


def detection_decoded_graph() -> bytes:
    """Build the SSD-style detection model: separate boxes, scores, and classes.

    The three tensors describe exactly the detections the grid variant encodes, so the two
    conventions must produce the same normalized answer from the same source image. Boxes are
    ``yxyx`` normalized to the model canvas, which is what a TensorFlow-exported SSD emits.

    Returns:
        The serialized model.
    """
    nodes, initializers = [], []
    zero = _input_zero("images", nodes, initializers)
    initializers.append(_initializer("boxes_const", _SSD_BOXES[None, :, :]))
    initializers.append(_initializer("scores_const", _SSD_SCORES[None, :]))
    initializers.append(_initializer("classes_const", _SSD_CLASSES[None, :]))
    nodes.append(helper.make_node("Add", ["boxes_const", zero], ["boxes"]))
    nodes.append(helper.make_node("Add", ["scores_const", zero], ["scores"]))
    nodes.append(helper.make_node("Add", ["classes_const", zero], ["classes"]))
    graph = helper.make_graph(
        nodes,
        "synthetic-detection-decoded",
        [_value_info("images", (1, 3, 64, 64))],
        [
            _value_info("boxes", (1, 6, 4)),
            _value_info("scores", (1, 6)),
            _value_info("classes", (1, 6)),
        ],
        initializers,
    )
    return _finish(graph)


#: The sample scale an 8-bit manifest declares, as a JSON-representable double.
SCALE_8_BIT = 1.0 / 255.0

#: The anomaly models' baked reference: a mid-grey image at the same scale the manifest declares.
ANOMALY_REFERENCE_SAMPLE = np.float32(128.0) * np.float32(SCALE_8_BIT)


def segmentation_threshold_graph() -> bytes:
    """Build the binary segmentation model: the mean of the colour channels, per pixel.

    Thresholding that mean at 0.5 selects exactly the pixels that were brighter than mid-grey, so
    a rectangle drawn in white on black yields a pixel count and a bounding region that can be read
    off the image the fixture drew.

    Returns:
        The serialized model.
    """
    nodes = [helper.make_node("ReduceMean", ["images"], ["mask"], axes=[1], keepdims=1)]
    graph = helper.make_graph(
        nodes,
        "synthetic-segmentation-threshold",
        [_value_info("images", (1, 3, 32, 32))],
        [_value_info("mask", (1, 1, 32, 32))],
        [],
    )
    return _finish(graph)


def segmentation_argmax_graph() -> bytes:
    """Build the multi-class segmentation model: three per-pixel class logits.

    Channel 0 is a constant 0.4 background level, channel 1 is the mean brightness, and channel 2
    is how much redder than green a pixel is. A white rectangle therefore wins on brightness and a
    red one wins on redness, while the untouched background stays with the constant, so the argmax
    label map is the picture the fixture drew.

    Returns:
        The serialized model.
    """
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["brightness"], axes=[1], keepdims=1),
        helper.make_node("Split", ["images"], ["red", "green", "blue"], axis=1),
        helper.make_node("Sub", ["red", "green"], ["redness"]),
        helper.make_node("Mul", ["brightness", "zero_c"], ["flat"]),
        helper.make_node("Add", ["flat", "bg_level"], ["background"]),
        helper.make_node("Concat", ["background", "brightness", "redness"], ["classes"], axis=1),
    ]
    initializers = [
        _initializer("zero_c", np.float32(0.0)),
        _initializer("bg_level", np.float32(0.4)),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic-segmentation-argmax",
        [_value_info("images", (1, 3, 32, 32))],
        [_value_info("classes", (1, 3, 32, 32))],
        initializers,
    )
    return _finish(graph)


def _anomaly_difference(nodes: list, initializers: list) -> str:
    """Add the shared mean-absolute-difference body of both anomaly models.

    Args:
        nodes: The node list to append to.
        initializers: The initializer list to append to.

    Returns:
        The name of the per-element absolute-difference tensor.
    """
    initializers.append(
        _initializer("reference", np.full((1, 3, 32, 32), ANOMALY_REFERENCE_SAMPLE, dtype=np.float32))
    )
    nodes.append(helper.make_node("Sub", ["images", "reference"], ["difference"]))
    nodes.append(helper.make_node("Abs", ["difference"], ["absolute"]))
    return "absolute"


def anomaly_scalar_graph() -> bytes:
    """Build the scalar anomaly model: mean absolute difference from a baked reference.

    The reference is a mid-grey image, so a mid-grey input scores exactly zero and a patch of a
    different brightness scores the patch's share of the image times its per-sample difference.

    Returns:
        The serialized model.
    """
    nodes, initializers = [], []
    absolute = _anomaly_difference(nodes, initializers)
    nodes.append(helper.make_node("ReduceMean", [absolute], ["reduced"], keepdims=0))
    initializers.append(_initializer("score_shape", np.array([1], dtype=np.int64)))
    nodes.append(helper.make_node("Reshape", ["reduced", "score_shape"], ["score"]))
    graph = helper.make_graph(
        nodes,
        "synthetic-anomaly-scalar",
        [_value_info("images", (1, 3, 32, 32))],
        [_value_info("score", (1,))],
        initializers,
    )
    return _finish(graph)


def anomaly_map_graph() -> bytes:
    """Build the map anomaly model: the same difference, kept per pixel.

    Reducing only the colour axis leaves one value per pixel, which is what a real anomaly model's
    heatmap output looks like and what the family reduces to a score plus a bounded summary.

    Returns:
        The serialized model.
    """
    nodes, initializers = [], []
    absolute = _anomaly_difference(nodes, initializers)
    nodes.append(helper.make_node("ReduceMean", [absolute], ["heatmap"], axes=[1], keepdims=1))
    graph = helper.make_graph(
        nodes,
        "synthetic-anomaly-map",
        [_value_info("images", (1, 3, 32, 32))],
        [_value_info("heatmap", (1, 1, 32, 32))],
        initializers,
    )
    return _finish(graph)


def quadrant_image(colour: tuple, size: int = 64) -> np.ndarray:
    """Draw three quadrants of one colour and a black fourth.

    Three quarters of one channel at full scale gives that channel a mean of exactly 0.75, which is
    the classification model's logit for that class.

    Args:
        colour: The ``(r, g, b)`` fill.
        size: The square edge in pixels. Must be even.

    Returns:
        An ``(size, size, 3)`` ``uint8`` array.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    image[:half, :half] = colour
    image[:half, half:] = colour
    image[half:, :half] = colour
    return image


def solid_image(colour: tuple, size: int = 64) -> np.ndarray:
    """Fill an image with one colour.

    Args:
        colour: The ``(r, g, b)`` fill.
        size: The square edge in pixels.

    Returns:
        An ``(size, size, 3)`` ``uint8`` array.
    """
    return np.full((size, size, 3), colour, dtype=np.uint8)


def gradient_image(size: int = 64) -> np.ndarray:
    """Draw a horizontal red ramp from black to full scale.

    Args:
        size: The square edge in pixels.

    Returns:
        An ``(size, size, 3)`` ``uint8`` array.
    """
    ramp = np.rint(np.linspace(0.0, 255.0, size)).astype(np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = ramp[None, :]
    return image


def rectangle_image(size: int, box: tuple, colour: tuple) -> np.ndarray:
    """Draw one filled rectangle on black.

    Args:
        size: The square edge in pixels.
        box: The ``(x, y, w, h)`` rectangle in pixels.
        colour: The ``(r, g, b)`` fill.

    Returns:
        An ``(size, size, 3)`` ``uint8`` array.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    x, y, width, height = box
    image[y : y + height, x : x + width] = colour
    return image


def scene_image(width: int, height: int, seed: int) -> np.ndarray:
    """Draw a deterministic non-square scene for the detection fixtures.

    The detection graphs bake their answer, so the pixels only have to be stable, non-square, and
    obviously not a test pattern of one flat colour.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: The generator seed.

    Returns:
        A ``(height, width, 3)`` ``uint8`` array.
    """
    generator = np.random.default_rng(seed)
    columns = np.linspace(0, 200, width, dtype=np.float64)[None, :]
    rows = np.linspace(0, 120, height, dtype=np.float64)[:, None]
    ramp = np.broadcast_to(columns, (height, width))
    tilt = np.broadcast_to(rows, (height, width))
    base = np.stack([ramp + tilt, tilt * 2.0, np.full((height, width), 64.0)], axis=2)
    speckle = generator.integers(0, 24, size=base.shape)
    return np.clip(base + speckle, 0, 255).astype(np.uint8)


def _encode(array: np.ndarray, image_format: str, **options) -> bytes:
    """Encode an array with Pillow.

    Args:
        array: An ``HWC`` ``uint8`` array, or an ``HW`` ``uint16`` array.
        image_format: The Pillow format name.
        **options: Format options passed to Pillow.

    Returns:
        The encoded bytes.
    """
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format, **options)
    return buffer.getvalue()


def _install(root: Path, relative: str, data: bytes) -> dict:
    """Write one fixture file and describe it.

    Args:
        root: The corpus root.
        relative: The path under the root, using forward slashes.
        data: The file content.

    Returns:
        A record with the relative path, byte count, and digest.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": relative, "bytes": len(data), "sha256": _sha256(data)}


def _header_only_png(width: int, height: int) -> bytes:
    """Forge a PNG whose header claims far more pixels than its data holds.

    This is the decompression-bomb case in its cheapest honest form: a real, parseable PNG of a few
    dozen bytes that declares an enormous canvas. A decoder that trusts the header before checking
    it allocates gigabytes; the component checks first.

    Args:
        width: The declared width.
        height: The declared height.

    Returns:
        The forged PNG bytes.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + chunk(b"IEND", b"")
    )


def build_bad_inputs(root: Path, seed: int) -> list:
    """Write the malformed and hostile inputs, and record what each one must do.

    Args:
        root: The corpus root.
        seed: The generator seed.

    Returns:
        One record per fixture. ``code`` is the expected
        :class:`~image_processor.engine.decode.DecodeError` code, or ``None`` for an input that is
        supposed to decode.
    """
    generator = np.random.default_rng(seed)
    records = []

    good_jpeg = _encode(quadrant_image((200, 40, 40), 32), "JPEG", quality=92)
    garbage = bytes(generator.integers(0, 256, size=512, dtype=np.uint8).tolist())

    entries = [
        ("bad/corrupt.jpg", b"\xff\xd8\xff\xe0" + garbage, "UNREADABLE_IMAGE"),
        ("bad/truncated.jpg", good_jpeg[: max(8, len(good_jpeg) * 55 // 100)], "UNREADABLE_IMAGE"),
        ("bad/zero-byte.jpg", b"", "EMPTY_IMAGE"),
        ("bad/not-an-image.png", b"this file is text, whatever its name says\n", "UNREADABLE_IMAGE"),
        ("bad/bomb-dims.png", _header_only_png(20_000, 100), "IMAGE_DIMENSION_EXCEEDED"),
        ("bad/bomb-pixels.png", _header_only_png(8_000, 8_000), "IMAGE_PIXELS_EXCEEDED"),
        ("bad/bomb-declared.png", _header_only_png(30_000, 30_000), "IMAGE_PIXELS_EXCEEDED"),
    ]
    for relative, data, code in entries:
        record = _install(root, relative, data)
        record["code"] = code
        records.append(record)

    mislabelled = _install(root, "bad/wrong-extension.png", good_jpeg)
    mislabelled.update({"code": None, "format": "JPEG", "shape": [32, 32, 3], "dtype": "uint8"})
    records.append(mislabelled)

    samples = (np.arange(16 * 24, dtype=np.uint16).reshape(16, 24) * np.uint16(170)).astype(np.uint16)
    for relative, image_format in (("bad/sixteen-bit.tiff", "TIFF"), ("bad/sixteen-bit.png", "PNG")):
        record = _install(root, relative, _encode(samples, image_format))
        record.update(
            {
                "code": None,
                "dtype": "uint16",
                "shape": [16, 24, 3],
                "sampleMin": int(samples.min()),
                "sampleMax": int(samples.max()),
            }
        )
        records.append(record)

    frames = [Image.fromarray(solid_image((step * 60, 20, 20), 16)) for step in range(3)]
    animated = io.BytesIO()
    frames[0].save(animated, format="GIF", save_all=True, append_images=frames[1:])
    record = _install(root, "bad/animated.gif", animated.getvalue())
    record["code"] = "MULTI_FRAME_IMAGE"
    records.append(record)

    pages = io.BytesIO()
    frames[0].save(pages, format="TIFF", save_all=True, append_images=frames[1:2])
    record = _install(root, "bad/multipage.tiff", pages.getvalue())
    record["code"] = "MULTI_FRAME_IMAGE"
    records.append(record)
    return records


def _terminal_body(index: int, relative: str, data: bytes, absolute: Path) -> dict:
    """Build one camera ``ImageCaptured`` body.

    The shape is camera-adapter's schema-v1 terminal body verbatim, which is also what it writes
    into the on-disk metadata sidecar: the same document, so the two can never disagree about what
    the capture is. Absent stages are omitted rather than fabricated, exactly as the adapter does.

    Args:
        index: The capture ordinal, which seeds every identifier and timestamp.
        relative: The image path relative to the camera's output root.
        data: The installed image bytes.
        absolute: The installed image path.

    Returns:
        The body document, with camelCase keys.
    """
    minute = 3 + index
    stamp = f"2026-08-22T14:{minute:02d}:00Z"
    return {
        "schemaVersion": 1,
        "eventId": f"01K5EVENT{index:017d}",
        "captureId": f"01K5CAPTURE{index:015d}",
        "cameraId": CAMERA_ID,
        "correlationId": f"00000000-0000-4000-8000-{index:012d}",
        "trigger": {
            "type": "schedule",
            "scheduleId": "line-clearance",
            "intendedFireTime": stamp,
        },
        "captureProfile": "line-clearance",
        "captureMode": "simulated",
        "timestamps": {
            "requestedAt": stamp,
            "acquisitionStartedAt": f"2026-08-22T14:{minute:02d}:00.010Z",
            "cameraFrameAt": f"2026-08-22T14:{minute:02d}:00.021Z",
            "frameReceivedAt": f"2026-08-22T14:{minute:02d}:00.022Z",
            "persistedAt": f"2026-08-22T14:{minute:02d}:00.031Z",
            "cameraFrameTimestampQuality": "adapter-receive",
        },
        "durationsMs": {"queue": 1, "acquisition": 11, "encoding": 1, "persistence": 9, "total": 31},
        "image": {
            "absolutePath": absolute.resolve().as_posix(),
            "relativePath": relative,
            "fileUri": absolute.resolve().as_uri(),
            "contentType": "image/jpeg",
            "encoding": "jpeg",
            "bytes": len(data),
            "sha256": _sha256(data),
            "metadataSidecarRelativePath": f"{relative}.json",
        },
        "frame": {"width": 128, "height": 64, "pixelFormat": "JPEG", "sourceEncoding": "jpeg"},
        "camera": {
            "backend": "sim",
            "vendor": "EdgeCommons",
            "model": "sim-playlist",
            "serial": f"sim-{CAMERA_ID}",
        },
        "metadata": {},
    }


def build_spool(root: Path, seed: int, captures: int = 2) -> list:
    """Write camera-shaped spool fixtures, sidecar before image.

    Ordering is the point. camera-adapter installs the metadata sidecar before the image becomes
    visible, which is what makes ``readiness.mode: cameraSidecar`` sound: a visible image with a
    sidecar is complete by construction (DESIGN.md §4.1). A fixture that wrote them the other way
    round would let a broken readiness check pass.

    Args:
        root: The corpus root.
        seed: The generator seed.
        captures: How many captures to write.

    Returns:
        One record per capture, naming both files and the identity they agree on.
    """
    records = []
    camera_root = root / "spool" / CAMERA_ID
    for index in range(captures):
        body_stub = _terminal_body(index, "", b"", camera_root)
        relative = f"{CAPTURE_DAY}/{body_stub['captureId']}.jpg"
        data = _encode(scene_image(128, 64, seed + index), "JPEG", quality=90)
        image_path = camera_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path = image_path.with_name(image_path.name + ".json")

        body = _terminal_body(index, relative, data, image_path)
        sidecar_bytes = (json.dumps(body, indent=2) + "\n").encode("utf-8")
        sidecar_path.write_bytes(sidecar_bytes)
        image_path.write_bytes(data)

        spool_relative = f"spool/{CAMERA_ID}/{relative}"
        records.append(
            {
                "path": spool_relative,
                "sidecar": f"{spool_relative}.json",
                "cameraRoot": f"spool/{CAMERA_ID}",
                "relativePath": relative,
                "captureId": body["captureId"],
                "cameraId": CAMERA_ID,
                "correlationId": body["correlationId"],
                "bytes": len(data),
                "sha256": body["image"]["sha256"],
            }
        )
    return records


def write_bundle(root: Path, spec: dict) -> dict:
    """Write one minimal bundle directory and return its manifest document.

    The layout is the subset of DESIGN.md §8 a task family needs: the graph, the label set, the
    transform block, and the manifest that declares all of it with per-file digests. ``make_bundle``
    (WP2) packs and signs the directory later; nothing here is packed or signed.

    Args:
        root: The corpus root.
        spec: The bundle description: ``modelId``, ``version``, ``onnx`` bytes, ``labels``,
            ``inputs``, ``outputs``, ``family``, ``familyParams``, ``preprocess``,
            ``decisionRules``, and optional ``maxResultItems``, ``dynamicBatch``,
            ``estimatedDeviceMiB``, and ``transformVersion``.

    Returns:
        The manifest document as written.
    """
    name = f"{spec['modelId']}-{spec['version']}"
    directory = root / "bundles" / name
    directory.mkdir(parents=True, exist_ok=True)
    transform_version = spec.get("transformVersion", "1")

    members = {
        "model.onnx": spec["onnx"],
        "labels.json": (json.dumps(spec["labels"], indent=2) + "\n").encode("utf-8"),
        "transforms.json": (
            json.dumps(
                {"transformVersion": transform_version, "preprocess": spec["preprocess"]}, indent=2
            )
            + "\n"
        ).encode("utf-8"),
    }
    for member, data in members.items():
        (directory / member).write_bytes(data)

    manifest = {
        "schemaVersion": 1,
        "modelId": spec["modelId"],
        "version": spec["version"],
        "files": {member: _sha256(data) for member, data in members.items()},
        "minOnnxRuntime": "1.17.0",
        "providersPermitted": ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "providerPolicy": "preferListed",
        "inputs": spec["inputs"],
        "outputs": spec["outputs"],
        "dynamicBatch": spec.get("dynamicBatch", False),
        "family": spec["family"],
        "familyParams": spec["familyParams"],
        "preprocess": spec["preprocess"],
        "decisionRules": spec["decisionRules"],
        "maxResultItems": spec.get("maxResultItems", 16),
        "estimatedDeviceMiB": spec.get("estimatedDeviceMiB", 16),
        "warmup": [],
        "tolerances": {"absolute": 1e-5},
        "compatibilityKeys": {},
        "provenance": {
            "publisher": "edgecommons synthetic corpus",
            "notes": f"generated by tests/fixtures/build.py with seed {SEED}",
        },
        "keyId": None,
        "transformVersion": transform_version,
    }
    (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def manifest_from_document(document: dict):
    """Turn a ``manifest.json`` document into a :class:`~image_processor.types.BundleManifest`.

    This is the camelCase-to-dataclass mapping the engine relies on. WP2's ``load_manifest`` is the
    production path and validates against the JSON Schema first; this one exists so a family test
    can construct a manifest from a fixture bundle without depending on a package it does not own.

    Args:
        document: The parsed ``manifest.json``.

    Returns:
        The manifest dataclass.
    """
    from image_processor.types import BundleManifest, Family, TensorSpec

    def specs(entries):
        return [
            TensorSpec(name=entry["name"], dtype=entry["dtype"], shape=tuple(entry["shape"]))
            for entry in entries
        ]

    return BundleManifest(
        schema_version=document["schemaVersion"],
        model_id=document["modelId"],
        version=document["version"],
        files=document["files"],
        min_onnxruntime=document["minOnnxRuntime"],
        providers_permitted=document["providersPermitted"],
        provider_policy=document["providerPolicy"],
        inputs=specs(document["inputs"]),
        outputs=specs(document["outputs"]),
        dynamic_batch=document["dynamicBatch"],
        family=Family(document["family"]),
        family_params=document["familyParams"],
        preprocess=document["preprocess"],
        decision_rules=document["decisionRules"],
        max_result_items=document["maxResultItems"],
        estimated_device_mib=document["estimatedDeviceMiB"],
        warmup=document["warmup"],
        tolerances=document["tolerances"],
        compatibility_keys=document["compatibilityKeys"],
        provenance=document["provenance"],
        key_id=document.get("keyId"),
        transform_version=document["transformVersion"],
    )


def load_bundle_manifest(bundle_dir: Path):
    """Read a fixture bundle's manifest.

    Args:
        bundle_dir: The bundle directory.

    Returns:
        The manifest dataclass.
    """
    return manifest_from_document(json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8")))


#: Preprocess block for the 64 by 64 classification model.
CLASSIFICATION_PREPROCESS = {
    "colorOrder": "RGB",
    "resize": {"mode": "stretch", "width": 64, "height": 64, "interpolation": "bilinear"},
    "scale": SCALE_8_BIT,
    "mean": 0.0,
    "std": 1.0,
    "layout": "NCHW",
    "dtype": "float32",
    "highBitDepthMode": "scaleTo8Bit",
}


def _detection_preprocess(color_order: str) -> dict:
    """Build the letterbox preprocess block both detection bundles use.

    Args:
        color_order: ``"RGB"`` or ``"BGR"``. Real YOLOX exports take BGR and real SSD exports take
            RGB, so the two fixtures differ here and agree everywhere else.

    Returns:
        The preprocess block.
    """
    return {
        "colorOrder": color_order,
        "resize": {
            "mode": "letterbox",
            "width": 64,
            "height": 64,
            "interpolation": "bilinear",
            "padColor": [114, 114, 114],
            "padMode": "center",
        },
        "scale": 1.0,
        "layout": "NCHW",
        "dtype": "float32",
    }


#: Preprocess block for the 32 by 32 segmentation and anomaly models.
SMALL_PREPROCESS = {
    "colorOrder": "RGB",
    "resize": {"mode": "stretch", "width": 32, "height": 32, "interpolation": "nearest"},
    "scale": SCALE_8_BIT,
    "layout": "NCHW",
    "dtype": "float32",
}

#: The detections both detection bundles must produce from ``images/detect-scene.png``, computed
#: by hand from the baked tensors, the letterbox geometry, and the suppression thresholds.
DETECTION_EXPECTED = [
    {"label": "nut", "index": 1, "score": 0.95 * 0.95, "box": [0.5625, 0.125, 0.375, 0.75]},
    {"label": "bolt", "index": 0, "score": 0.9 * 0.9, "box": [0.125, 0.25, 0.25, 0.5]},
    {"label": "washer", "index": 2, "score": 0.7 * 0.9, "box": [0.125, 0.25, 0.25, 0.5]},
]

#: Labels shared by both detection bundles.
DETECTION_LABELS = ["bolt", "nut", "washer"]

#: The decision rules both detection bundles carry.
DETECTION_RULES = {
    "pass": {
        "all": [
            {"path": "$.detections[*].score", "op": ">=", "value": 0.5},
            {"path": "$.detections[*].label", "op": "!=", "value": "screw"},
        ]
    },
    "confidence": "$.detections[0].score",
    "threshold": 0.5,
    "outcomeOnPass": "CLEAR",
    "outcomeOnFail": "FAIL",
    "failOnEmpty": False,
}


def classification_answer(array: np.ndarray, labels: list, top_k: int) -> list:
    """Compute the classification model's answer arithmetically.

    The graph is an identity convolution, a global average pool, and a pass-through linear head, so
    the logits are the per-channel means of the scaled image plus a fixed fourth logit. This
    reproduces that in double precision without running the graph, which is what makes the
    comparison meaningful.

    Args:
        array: The ``(H, W, 3)`` ``uint8`` source image.
        labels: The class labels.
        top_k: How many classes the manifest reports.

    Returns:
        The expected ``classes`` entries, highest score first.
    """
    means = array.astype(np.float64).mean(axis=(0, 1)) * SCALE_8_BIT
    logits = np.array([means[0], means[1], means[2], 0.25], dtype=np.float64)
    exponentiated = np.exp(logits - logits.max())
    scores = exponentiated / exponentiated.sum()
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {"label": labels[index], "index": int(index), "score": float(scores[index])}
        for index in order
    ]


def _segment(pixels: int, bbox, total: int = 32 * 32) -> dict:
    """Describe one expected segment.

    Args:
        pixels: The pixel count.
        bbox: The normalized ``[x, y, w, h]`` region, or ``None``.
        total: The size of the class map.

    Returns:
        The expected segment entry.
    """
    return {"pixels": pixels, "fraction": pixels / total, "bbox": bbox}


def _decision(outcome: str, passed: bool, confidence, threshold, rule: str) -> dict:
    """Describe one expected decision.

    Args:
        outcome: The expected outcome name.
        passed: Whether the pass rule held.
        confidence: The expected confidence, or ``None``.
        threshold: The expected threshold, or ``None``.
        rule: The expected ``decision.rule``.

    Returns:
        The expected decision record.
    """
    return {
        "outcome": outcome,
        "passed": passed,
        "confidence": confidence,
        "threshold": threshold,
        "rule": rule,
    }


def _write_images(root: Path, seed: int) -> dict:
    """Draw and install every image with a computable expected output.

    Args:
        root: The corpus root.
        seed: The generator seed.

    Returns:
        A mapping of image name to its file record, including its shape.
    """
    drawn = {
        "quadrant-red.png": quadrant_image((255, 0, 0), 64),
        "quadrant-green.png": quadrant_image((0, 255, 0), 64),
        "quadrant-blue.png": quadrant_image((0, 0, 255), 64),
        "solid-black.png": solid_image((0, 0, 0), 64),
        "gradient-red.png": gradient_image(64),
        "detect-scene.png": scene_image(128, 64, seed),
        "seg-rect.png": rectangle_image(32, (4, 8, 16, 16), (255, 255, 255)),
        "seg-red-rect.png": rectangle_image(32, (4, 8, 16, 16), (255, 0, 0)),
        "seg-clean.png": solid_image((0, 0, 0), 32),
        "anomaly-good.png": solid_image((128, 128, 128), 32),
        "anomaly-bad.png": solid_image((128, 128, 128), 32),
    }
    drawn["anomaly-bad.png"][8:24, 8:24] = (255, 255, 255)

    records = {}
    for name, array in drawn.items():
        record = _install(root, f"images/{name}", _encode(array, "PNG"))
        record["shape"] = [int(value) for value in array.shape]
        records[name] = record
    records["_arrays"] = drawn
    return records


def _classification_bundle(root: Path, images: dict) -> dict:
    """Build the classification bundle and its expected answers.

    Args:
        root: The corpus root.
        images: The image records from :func:`_write_images`.

    Returns:
        A single-entry mapping of bundle name to its record.
    """
    labels = ["red", "green", "blue", "other"]
    rules = {
        "pass": {
            "all": [
                {"path": "$.classes[0].label", "op": "==", "value": "other"},
                {"path": "$.classes[0].score", "op": ">=", "value": 0.25},
            ]
        },
        "confidence": "$.classes[0].score",
        "threshold": 0.25,
        "outcomeOnPass": "CLEAR",
        "outcomeOnFail": "HOLD",
        "failOnEmpty": True,
    }
    manifest = write_bundle(
        root,
        {
            "modelId": "synthetic-classification",
            "version": "1.0.0",
            "onnx": classification_graph(),
            "labels": labels,
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 64, 64]}],
            "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 4]}],
            "family": "classification",
            "familyParams": {"labels": labels, "activation": "softmax", "topK": 4},
            "preprocess": CLASSIFICATION_PREPROCESS,
            "decisionRules": rules,
            "maxResultItems": 8,
        },
    )

    cases = []
    for name in (
        "quadrant-red.png",
        "quadrant-green.png",
        "quadrant-blue.png",
        "gradient-red.png",
        "solid-black.png",
    ):
        classes = classification_answer(images["_arrays"][name], labels, 4)
        top = classes[0]
        cleared = top["label"] == "other" and top["score"] >= 0.25
        cases.append(
            {
                "image": f"images/{name}",
                "expected": {"classes": classes},
                "decision": _decision(
                    "CLEAR" if cleared else "HOLD",
                    cleared,
                    top["score"],
                    0.25,
                    "pass" if cleared else "pass.all[0]: $.classes[0].label == 'other'",
                ),
            }
        )
    return {"synthetic-classification-1.0.0": _bundle_record(manifest, cases)}


def _bundle_record(manifest: dict, cases: list) -> dict:
    """Assemble one bundle's entry in the oracle.

    Args:
        manifest: The manifest document as written.
        cases: The expected per-image answers.

    Returns:
        The oracle entry.
    """
    name = f"{manifest['modelId']}-{manifest['version']}"
    return {
        "path": f"bundles/{name}",
        "family": manifest["family"],
        "modelId": manifest["modelId"],
        "version": manifest["version"],
        "cases": cases,
    }


def _detection_bundles(root: Path) -> dict:
    """Build both detection bundles and their shared expected answer.

    Args:
        root: The corpus root.

    Returns:
        A mapping of bundle name to record, one per head convention.
    """
    case = {
        "image": "images/detect-scene.png",
        "expected": {"detections": DETECTION_EXPECTED},
        "decision": _decision("CLEAR", True, DETECTION_EXPECTED[0]["score"], 0.5, "pass"),
    }
    shared = {
        "labels": DETECTION_LABELS,
        "family": "detection",
        "decisionRules": DETECTION_RULES,
        "maxResultItems": 16,
    }
    grid = write_bundle(
        root,
        dict(
            shared,
            modelId="synthetic-detection-grid",
            version="1.0.0",
            onnx=detection_grid_graph(),
            inputs=[{"name": "images", "dtype": "float32", "shape": [1, 3, 64, 64]}],
            outputs=[{"name": "output", "dtype": "float32", "shape": [1, 84, 8]}],
            familyParams={
                "decode": "yoloxGrid",
                "labels": DETECTION_LABELS,
                "strides": [8, 16, 32],
                "objectness": True,
                "scoreActivation": "none",
                "objectnessActivation": "none",
                "scoreThreshold": 0.25,
                "iouThreshold": 0.45,
                "maxDetections": 16,
            },
            preprocess=_detection_preprocess("BGR"),
        ),
    )
    decoded = write_bundle(
        root,
        dict(
            shared,
            modelId="synthetic-detection-decoded",
            version="1.0.0",
            onnx=detection_decoded_graph(),
            inputs=[{"name": "images", "dtype": "float32", "shape": [1, 3, 64, 64]}],
            outputs=[
                {"name": "boxes", "dtype": "float32", "shape": [1, 6, 4]},
                {"name": "scores", "dtype": "float32", "shape": [1, 6]},
                {"name": "classes", "dtype": "float32", "shape": [1, 6]},
            ],
            familyParams={
                "decode": "decodedBoxes",
                "labels": DETECTION_LABELS,
                "scoresLayout": "perBox",
                "boxFormat": "yxyx",
                "boxCoordinates": "normalized",
                "classIndexOffset": 0,
                "outputNames": {"boxes": "boxes", "scores": "scores", "classes": "classes"},
                "applyNms": True,
                "scoreThreshold": 0.25,
                "iouThreshold": 0.45,
                "maxDetections": 16,
            },
            preprocess=_detection_preprocess("RGB"),
        ),
    )
    return {
        "synthetic-detection-grid-1.0.0": _bundle_record(grid, [case]),
        "synthetic-detection-decoded-1.0.0": _bundle_record(decoded, [case]),
    }


#: Where the segmentation fixtures draw their rectangle, normalized to the 32 by 32 source.
RECT_BOX = [4 / 32, 8 / 32, 16 / 32, 16 / 32]

#: The whole image, for a class that covers everything outside the rectangle.
FULL_BOX = [0.0, 0.0, 1.0, 1.0]

#: The decision rules both segmentation bundles carry.
SEGMENTATION_RULES = {
    "pass": {"path": "$.segments.defect.pixels", "op": "<", "value": 100},
    "confidence": "$.segments.defect.fraction",
    "threshold": 100 / 1024,
    "outcomeOnPass": "CLEAR",
    "outcomeOnFail": "HOLD",
    "failOnEmpty": False,
}

#: The rule label reported when the defect pixel count fails.
SEGMENTATION_FAIL_RULE = "pass: $.segments.defect.pixels < 100"


def _segmentation_bundles(root: Path) -> dict:
    """Build both segmentation bundles and their expected answers.

    Args:
        root: The corpus root.

    Returns:
        A mapping of bundle name to record, one per head shape.
    """
    threshold = write_bundle(
        root,
        {
            "modelId": "synthetic-segmentation-threshold",
            "version": "1.0.0",
            "onnx": segmentation_threshold_graph(),
            "labels": ["clean", "defect"],
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 32, 32]}],
            "outputs": [{"name": "mask", "dtype": "float32", "shape": [1, 1, 32, 32]}],
            "family": "segmentation",
            "familyParams": {
                "mode": "threshold",
                "labels": ["clean", "defect"],
                "threshold": 0.5,
                "activation": "none",
                "positiveLabel": "defect",
                "minPixels": 0,
            },
            "preprocess": SMALL_PREPROCESS,
            "decisionRules": SEGMENTATION_RULES,
            "maxResultItems": 8,
        },
    )
    threshold_cases = [
        {
            "image": "images/seg-rect.png",
            "expected": {"segments": {"defect": _segment(256, RECT_BOX)}},
            "decision": _decision("HOLD", False, 0.25, 100 / 1024, SEGMENTATION_FAIL_RULE),
        },
        {
            "image": "images/seg-clean.png",
            "expected": {"segments": {"defect": _segment(0, None)}},
            "decision": _decision("CLEAR", True, 0.0, 100 / 1024, "pass"),
        },
    ]

    labels = ["background", "part", "defect"]
    argmax = write_bundle(
        root,
        {
            "modelId": "synthetic-segmentation-argmax",
            "version": "1.0.0",
            "onnx": segmentation_argmax_graph(),
            "labels": labels,
            "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 32, 32]}],
            "outputs": [{"name": "classes", "dtype": "float32", "shape": [1, 3, 32, 32]}],
            "family": "segmentation",
            "familyParams": {
                "mode": "argmax",
                "labels": labels,
                "outputLayout": "NCHW",
                "minPixels": 0,
            },
            "preprocess": SMALL_PREPROCESS,
            "decisionRules": SEGMENTATION_RULES,
            "maxResultItems": 8,
        },
    )
    argmax_cases = [
        {
            "image": "images/seg-rect.png",
            "expected": {
                "segments": {
                    "background": _segment(768, FULL_BOX),
                    "part": _segment(256, RECT_BOX),
                    "defect": _segment(0, None),
                }
            },
            "decision": _decision("CLEAR", True, 0.0, 100 / 1024, "pass"),
        },
        {
            "image": "images/seg-red-rect.png",
            "expected": {
                "segments": {
                    "background": _segment(768, FULL_BOX),
                    "part": _segment(0, None),
                    "defect": _segment(256, RECT_BOX),
                }
            },
            "decision": _decision("HOLD", False, 0.25, 100 / 1024, SEGMENTATION_FAIL_RULE),
        },
    ]
    return {
        "synthetic-segmentation-threshold-1.0.0": _bundle_record(threshold, threshold_cases),
        "synthetic-segmentation-argmax-1.0.0": _bundle_record(argmax, argmax_cases),
    }


#: The anomaly threshold both anomaly bundles carry.
ANOMALY_THRESHOLD = 0.05

#: The per-sample difference of the bad fixture's patch: white against a mid-grey reference.
PATCH_DIFFERENCE = 127 / 255

#: The patch's share of the 32 by 32 image.
PATCH_FRACTION = 256 / 1024

#: The patch, normalized to the source image.
PATCH_BOX = [8 / 32, 8 / 32, 16 / 32, 16 / 32]

#: The decision rules both anomaly bundles carry.
ANOMALY_RULES = {
    "pass": {"path": "$.anomaly.anomalous", "op": "==", "value": False},
    "confidence": "$.anomaly.score",
    "threshold": "$.anomaly.threshold",
    "outcomeOnPass": "CLEAR",
    "outcomeOnFail": "HOLD",
    "failOnEmpty": True,
}

#: The rule label reported when an image is anomalous.
ANOMALY_FAIL_RULE = "pass: $.anomaly.anomalous == False"


def _anomaly_bundles(root: Path) -> dict:
    """Build both anomaly bundles and their expected answers.

    Args:
        root: The corpus root.

    Returns:
        A mapping of bundle name to record, one per head shape.
    """
    shared = {
        "labels": ["normal", "anomalous"],
        "family": "anomaly",
        "preprocess": SMALL_PREPROCESS,
        "decisionRules": ANOMALY_RULES,
        "maxResultItems": 4,
        "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 32, 32]}],
    }
    scalar_score = PATCH_DIFFERENCE * PATCH_FRACTION
    scalar = write_bundle(
        root,
        dict(
            shared,
            modelId="synthetic-anomaly-scalar",
            version="1.0.0",
            onnx=anomaly_scalar_graph(),
            outputs=[{"name": "score", "dtype": "float32", "shape": [1]}],
            familyParams={
                "source": "scalar",
                "threshold": ANOMALY_THRESHOLD,
                "activation": "none",
                "direction": "higherIsAnomalous",
            },
        ),
    )
    scalar_cases = [
        {
            "image": "images/anomaly-good.png",
            "expected": {
                "anomaly": {
                    "score": 0.0,
                    "threshold": ANOMALY_THRESHOLD,
                    "anomalous": False,
                    "direction": "higherIsAnomalous",
                }
            },
            "decision": _decision("CLEAR", True, 0.0, ANOMALY_THRESHOLD, "pass"),
        },
        {
            "image": "images/anomaly-bad.png",
            "expected": {
                "anomaly": {
                    "score": scalar_score,
                    "threshold": ANOMALY_THRESHOLD,
                    "anomalous": True,
                    "direction": "higherIsAnomalous",
                }
            },
            "decision": _decision("HOLD", False, scalar_score, ANOMALY_THRESHOLD, ANOMALY_FAIL_RULE),
        },
    ]

    heatmap = write_bundle(
        root,
        dict(
            shared,
            modelId="synthetic-anomaly-map",
            version="1.0.0",
            onnx=anomaly_map_graph(),
            outputs=[{"name": "heatmap", "dtype": "float32", "shape": [1, 1, 32, 32]}],
            familyParams={
                "source": "mapMax",
                "threshold": ANOMALY_THRESHOLD,
                "activation": "none",
                "direction": "higherIsAnomalous",
            },
        ),
    )
    map_cases = [
        {
            "image": "images/anomaly-good.png",
            "expected": {
                "anomaly": {
                    "score": 0.0,
                    "threshold": ANOMALY_THRESHOLD,
                    "anomalous": False,
                    "direction": "higherIsAnomalous",
                    "summary": {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "aboveThresholdPixels": 0,
                        "fraction": 0.0,
                        "bbox": None,
                    },
                }
            },
            "decision": _decision("CLEAR", True, 0.0, ANOMALY_THRESHOLD, "pass"),
        },
        {
            "image": "images/anomaly-bad.png",
            "expected": {
                "anomaly": {
                    "score": PATCH_DIFFERENCE,
                    "threshold": ANOMALY_THRESHOLD,
                    "anomalous": True,
                    "direction": "higherIsAnomalous",
                    "summary": {
                        "min": 0.0,
                        "max": PATCH_DIFFERENCE,
                        "mean": scalar_score,
                        "aboveThresholdPixels": 256,
                        "fraction": PATCH_FRACTION,
                        "bbox": PATCH_BOX,
                    },
                }
            },
            "decision": _decision(
                "HOLD", False, PATCH_DIFFERENCE, ANOMALY_THRESHOLD, ANOMALY_FAIL_RULE
            ),
        },
    ]
    return {
        "synthetic-anomaly-scalar-1.0.0": _bundle_record(scalar, scalar_cases),
        "synthetic-anomaly-map-1.0.0": _bundle_record(heatmap, map_cases),
    }


def build(out_dir=DEFAULT_OUT, seed: int = SEED) -> dict:
    """Generate the whole tier-1 corpus into one directory.

    Args:
        out_dir: Where to write. Created if missing; existing files are overwritten.
        seed: The generator seed. Two runs with one seed produce identical bytes.

    Returns:
        The oracle document, which is also written to ``expected.json`` in ``out_dir``.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    images = _write_images(root, seed)
    bundles = {}
    bundles.update(_classification_bundle(root, images))
    bundles.update(_detection_bundles(root))
    bundles.update(_segmentation_bundles(root))
    bundles.update(_anomaly_bundles(root))
    images.pop("_arrays")

    document = {
        "schemaVersion": 1,
        "generator": "tests/fixtures/build.py",
        "seed": seed,
        "images": images,
        "bundles": bundles,
        "badInputs": build_bad_inputs(root, seed),
        "spool": build_spool(root, seed),
    }
    (root / "expected.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main(argv=None) -> int:
    """Run the builder from the command line.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the ImageProcessor tier-1 test corpus.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="target directory")
    parser.add_argument("--seed", type=int, default=SEED, help="generator seed")
    arguments = parser.parse_args(argv)
    document = build(Path(arguments.out), arguments.seed)
    print(
        f"built {len(document['bundles'])} bundles, {len(document['images'])} images, "
        f"{len(document['badInputs'])} bad inputs, {len(document['spool'])} spool captures "
        f"into {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
