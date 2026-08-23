"""Built-in task families and the machinery they share (DESIGN.md §8.1, D-IP-12, LLD §6).

A task family is the only interpreter of a model head this component has. There is no
bundle-supplied code (D-IP-12), so a bundle whose head no family can read is refused at staging
rather than mis-read at inference time: :meth:`TaskFamily.validate_manifest` is the refusal, and it
runs once per bundle, before a session is ever created.

Each family answers three questions about one model:

* what tensors does it want (:meth:`TaskFamily.preprocess`),
* what do its output tensors mean (:meth:`TaskFamily.postprocess`), and
* is this manifest one it can serve at all (:meth:`TaskFamily.validate_manifest`).

Everything here is numpy. Preprocessing, decoding, non-maximum suppression, and coordinate
mapping run identically whether the session behind them is on a CPU or a GPU, which is what makes
the CPU parity suite (D-IP-14) a real comparison rather than a second implementation.

Coordinates are normalized to the source image. A box or a region reported by any family is
``(x, y, w, h)`` in ``[0, 1]`` of the image as it arrived, not of the letterboxed model canvas, so
a consumer needs nothing but the result to draw it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from image_processor.types import BundleManifest, Family, NormalizedOutput, TensorSpec

logger = logging.getLogger(__name__)

#: Pillow resampling filters selectable from ``preprocess.resize.interpolation``.
_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}

#: Resize modes understood by :func:`preprocess_image`.
_RESIZE_MODES = frozenset({"letterbox", "stretch", "centerCrop", "none"})

#: Tensor element types a family may ask for.
_DTYPES = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8}

#: The full-scale value of one 16-bit sample expressed in 8-bit units.
_UINT16_TO_UINT8 = 257.0


class FamilyError(Exception):
    """A manifest a task family refuses, or an output tensor it cannot read.

    Raised from :meth:`TaskFamily.validate_manifest` at staging, which is where a bad head is
    supposed to be caught, and from :meth:`TaskFamily.postprocess` when a session returns something
    the validated manifest said it would not.

    Attributes:
        code: Stable SCREAMING_SNAKE code, safe to put on the bus and in metrics.
        message: Operator-readable detail naming the manifest field at fault.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TaskFamily(Protocol):
    """The interpreter contract for one model head.

    Attributes:
        family: The :class:`~image_processor.types.Family` this implementation serves.
    """

    family: Family

    def validate_manifest(self, m: BundleManifest) -> None:
        """Refuse a manifest this family cannot serve.

        Args:
            m: The parsed bundle manifest.

        Raises:
            FamilyError: When the head, the declared tensors, or ``familyParams`` are unsupported.
        """

    def preprocess(self, image: np.ndarray, m: BundleManifest) -> dict:
        """Turn one decoded image into the session's input feed.

        Args:
            image: An ``(H, W, 3)`` ``uint8`` or ``uint16`` array from
                :func:`~image_processor.engine.decode.decode_image`.
            m: The parsed bundle manifest.

        Returns:
            A mapping of input tensor name to array, ready to feed a session.
        """

    def postprocess(self, outputs: dict, m: BundleManifest, image_hw: tuple) -> NormalizedOutput:
        """Turn the session's output tensors into the normalized task output.

        Args:
            outputs: A mapping of output tensor name to array, as the session returned it.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image, before preprocessing.

        Returns:
            The normalized output the decision rules evaluate.
        """


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax.

    Args:
        values: The logits.
        axis: The axis to normalize over.

    Returns:
        An array of the same shape whose entries along ``axis`` sum to one.
    """
    shifted = values.astype(np.float64) - np.max(values, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute a numerically stable logistic sigmoid.

    Args:
        values: The logits.

    Returns:
        An array of the same shape with entries in ``(0, 1)``.
    """
    wide = values.astype(np.float64)
    positive = wide >= 0
    result = np.empty_like(wide)
    result[positive] = 1.0 / (1.0 + np.exp(-wide[positive]))
    exponentiated = np.exp(wide[~positive])
    result[~positive] = exponentiated / (1.0 + exponentiated)
    return result


def apply_activation(values: np.ndarray, activation: str, axis: int = -1) -> np.ndarray:
    """Apply the activation a manifest declares for a score block.

    Args:
        values: The raw output values.
        activation: ``"none"``, ``"softmax"``, or ``"sigmoid"``.
        axis: The axis a softmax normalizes over.

    Returns:
        The activated values as ``float64``.

    Raises:
        FamilyError: When the activation is not one of the three supported names.
    """
    if activation == "none":
        return values.astype(np.float64)
    if activation == "softmax":
        return softmax(values, axis=axis)
    if activation == "sigmoid":
        return sigmoid(values)
    raise FamilyError(
        "UNSUPPORTED_ACTIVATION",
        f"activation {activation!r} is not one of none, softmax, sigmoid",
    )


def choice(params: dict, key: str, allowed: Iterable, default: str, code: str) -> str:
    """Read a string parameter constrained to a fixed set.

    Args:
        params: The ``familyParams`` or ``preprocess`` mapping.
        key: The parameter name.
        allowed: The accepted values.
        default: The value used when the parameter is absent.
        code: The :class:`FamilyError` code to raise on a value outside the set.

    Returns:
        The selected value.

    Raises:
        FamilyError: When the value is present but outside ``allowed``.
    """
    value = params.get(key, default)
    permitted = list(allowed)
    if value not in permitted:
        raise FamilyError(code, f"{key}={value!r} is not one of {sorted(map(str, permitted))}")
    return str(value)


def number(params: dict, key: str, default: float, low: float, high: float) -> float:
    """Read a numeric parameter constrained to a closed interval.

    Args:
        params: The parameter mapping.
        key: The parameter name.
        default: The value used when the parameter is absent.
        low: Smallest accepted value.
        high: Largest accepted value.

    Returns:
        The value as a ``float``.

    Raises:
        FamilyError: When the value is not a number or falls outside ``[low, high]``.
    """
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FamilyError("INVALID_FAMILY_PARAM", f"{key} must be a number, got {value!r}")
    if not low <= float(value) <= high:
        raise FamilyError("INVALID_FAMILY_PARAM", f"{key}={value} is outside [{low}, {high}]")
    return float(value)


def positive_int(params: dict, key: str, default: Optional[int]) -> int:
    """Read an integer parameter that must be one or more.

    Args:
        params: The parameter mapping.
        key: The parameter name.
        default: The value used when the parameter is absent, or ``None`` to require it.

    Returns:
        The value as an ``int``.

    Raises:
        FamilyError: When the parameter is required and absent, or is not a positive integer.
    """
    if key not in params:
        if default is None:
            raise FamilyError("MISSING_FAMILY_PARAM", f"{key} is required")
        return default
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FamilyError("INVALID_FAMILY_PARAM", f"{key} must be a positive integer, got {value!r}")
    return value


def family_labels(m: BundleManifest) -> list:
    """Resolve the class labels a manifest declares.

    A manifest carries either an explicit ``familyParams.labels`` list, which is the bundle's
    ``labels.json`` content, or a ``familyParams.numClasses`` count that yields positional
    ``class-<i>`` names. A family that classifies anything needs one of the two, so a manifest with
    neither is refused rather than silently labelled by index.

    Args:
        m: The parsed bundle manifest.

    Returns:
        The label list, one entry per class, in class-index order.

    Raises:
        FamilyError: When neither parameter is present, when ``labels`` is not a non-empty list of
            strings, or when both are present and disagree.
    """
    params = m.family_params or {}
    labels = params.get("labels")
    declared = params.get("numClasses")
    if labels is None and declared is None:
        raise FamilyError(
            "MISSING_FAMILY_PARAM",
            "familyParams needs labels or numClasses to name a class",
        )
    if labels is not None:
        if (
            not isinstance(labels, (list, tuple))
            or not labels
            or not all(isinstance(entry, str) for entry in labels)
        ):
            raise FamilyError(
                "INVALID_FAMILY_PARAM",
                "familyParams.labels must be a non-empty list of strings",
            )
        if declared is not None and int(declared) != len(labels):
            raise FamilyError(
                "LABEL_COUNT_MISMATCH",
                f"familyParams.numClasses={declared} but labels has {len(labels)} entries",
            )
        return list(labels)
    return [f"class-{index}" for index in range(positive_int(params, "numClasses", None))]


def label_for(labels: Sequence, index: int) -> str:
    """Name one class index, falling back to a positional name.

    Args:
        labels: The label list from :func:`family_labels`.
        index: The class index a model produced.

    Returns:
        The label at ``index``, or ``class-<index>`` when the index is outside the list.
    """
    if 0 <= index < len(labels):
        return str(labels[index])
    return f"class-{index}"


def single_output(m: BundleManifest, reads: str) -> TensorSpec:
    """Resolve the one output tensor a family reads.

    A manifest declares every graph output (DESIGN.md section 8). When it declares exactly one,
    that is the tensor. When it declares several, ``familyParams.outputName`` names the one this
    family reads and the others are ignored; without that name the manifest is ambiguous and is
    refused at staging.

    Args:
        m: The parsed bundle manifest.
        reads: What the family reads, for the diagnostic (``"classification"``, ``"yoloxGrid"``).

    Returns:
        The declared :class:`~image_processor.types.TensorSpec` the family reads.

    Raises:
        FamilyError: ``UNSUPPORTED_OUTPUT_COUNT`` when no output is declared or several are
            declared without ``familyParams.outputName``; ``MISSING_OUTPUT`` when the named
            output is not declared.
    """
    if not m.outputs:
        raise FamilyError("UNSUPPORTED_OUTPUT_COUNT", f"{reads} reads one output, manifest declares 0")
    name = (m.family_params or {}).get("outputName")
    if isinstance(name, str) and name:
        return output_spec(m, name)
    if len(m.outputs) != 1:
        raise FamilyError(
            "UNSUPPORTED_OUTPUT_COUNT",
            f"{reads} reads one output, manifest declares {len(m.outputs)} and "
            "familyParams.outputName does not say which",
        )
    return m.outputs[0]


def output_spec(m: BundleManifest, name: str) -> TensorSpec:
    """Find one declared output tensor by name.

    Args:
        m: The parsed bundle manifest.
        name: The output tensor name.

    Returns:
        The declared :class:`~image_processor.types.TensorSpec`.

    Raises:
        FamilyError: When the manifest declares no output with that name.
    """
    for spec in m.outputs:
        if spec.name == name:
            return spec
    declared = sorted(spec.name for spec in m.outputs)
    raise FamilyError("MISSING_OUTPUT", f"manifest declares no output named {name!r}; has {declared}")


def static_dim(spec: TensorSpec, axis: int) -> Optional[int]:
    """Read one declared dimension when it is a fixed integer.

    Args:
        spec: The declared tensor.
        axis: The axis to read, which may be negative.

    Returns:
        The dimension as an ``int``, or ``None`` when the axis is dynamic or out of range.
    """
    shape = tuple(spec.shape)
    if not -len(shape) <= axis < len(shape):
        return None
    value = shape[axis]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def pick_output(outputs: dict, m: BundleManifest, name: Optional[str] = None) -> np.ndarray:
    """Select one output tensor from a session result.

    Args:
        outputs: The session outputs, keyed by tensor name.
        m: The parsed bundle manifest.
        name: The tensor to select, or ``None`` to take the manifest's first declared output (or
            the only tensor present).

    Returns:
        The selected tensor as a numpy array.

    Raises:
        FamilyError: When the named tensor is absent, or when no name can be resolved.
    """
    if name is None:
        if len(outputs) == 1:
            return np.asarray(next(iter(outputs.values())))
        if not m.outputs:
            raise FamilyError("MISSING_OUTPUT", "manifest declares no outputs to select from")
        name = m.outputs[0].name
    if name not in outputs:
        raise FamilyError(
            "MISSING_OUTPUT",
            f"session returned no tensor named {name!r}; got {sorted(outputs)}",
        )
    return np.asarray(outputs[name])


def drop_batch(values: np.ndarray, rank: int, where: str) -> np.ndarray:
    """Remove a leading batch axis so an output has the rank a family reads.

    Args:
        values: The output tensor.
        rank: The rank the family reads, excluding any batch axis.
        where: The tensor name, for the error message.

    Returns:
        The tensor with any leading singleton batch axis removed.

    Raises:
        FamilyError: When the rank is neither ``rank`` nor ``rank + 1``, or when a batch axis
            carries more than one sample.
    """
    if values.ndim == rank:
        return values
    if values.ndim == rank + 1:
        if values.shape[0] != 1:
            raise FamilyError(
                "UNSUPPORTED_BATCH",
                f"{where} carries {values.shape[0]} samples; one image is inferred at a time",
            )
        return values[0]
    raise FamilyError(
        "UNSUPPORTED_OUTPUT_RANK",
        f"{where} has rank {values.ndim}, expected {rank} or {rank + 1}",
    )


@dataclass(frozen=True)
class ResizePlan:
    """The geometry half of ``manifest.preprocess``, resolved once.

    Attributes:
        mode: ``"letterbox"``, ``"stretch"``, ``"centerCrop"``, or ``"none"``.
        width: Model input width in pixels. Zero when ``mode`` is ``"none"``.
        height: Model input height in pixels. Zero when ``mode`` is ``"none"``.
        resample: The Pillow resampling filter.
        pad_color: The ``letterbox`` fill, expressed in the manifest's ``colorOrder``.
        pad_mode: ``"center"`` or ``"topLeft"``, where the shrunk image sits on the canvas.
    """

    mode: str
    width: int
    height: int
    resample: int
    pad_color: tuple
    pad_mode: str


def resize_plan(m: BundleManifest) -> ResizePlan:
    """Resolve the resize geometry a manifest declares.

    Args:
        m: The parsed bundle manifest.

    Returns:
        The resolved :class:`ResizePlan`.

    Raises:
        FamilyError: When the mode, interpolation, pad mode, or target size is unusable.
    """
    block = dict((m.preprocess or {}).get("resize") or {})
    mode = choice(block, "mode", _RESIZE_MODES, "letterbox", "UNSUPPORTED_RESIZE_MODE")
    interpolation = choice(block, "interpolation", _RESAMPLE, "bilinear", "UNSUPPORTED_INTERPOLATION")
    pad_mode = choice(block, "padMode", ("center", "topLeft"), "center", "UNSUPPORTED_PAD_MODE")
    if mode == "none":
        width = height = 0
    else:
        width = positive_int(block, "width", None)
        height = positive_int(block, "height", None)
    raw_pad = block.get("padColor", [0, 0, 0])
    if isinstance(raw_pad, (int, float)) and not isinstance(raw_pad, bool):
        raw_pad = [raw_pad] * 3
    if not isinstance(raw_pad, (list, tuple)) or len(raw_pad) != 3:
        raise FamilyError(
            "INVALID_FAMILY_PARAM",
            f"preprocess.resize.padColor must be a number or three numbers, got {raw_pad!r}",
        )
    pad_color = tuple(float(channel) for channel in raw_pad)
    return ResizePlan(mode, width, height, _RESAMPLE[interpolation], pad_color, pad_mode)


@dataclass(frozen=True)
class SourceMapper:
    """Maps model-input pixel coordinates back onto the source image.

    Every resize mode reduces to one affine relation per axis,
    ``model = source * scale + pad``, so un-letterboxing, un-stretching, and un-cropping are the
    same inversion with different constants. A family that decodes boxes in the model canvas hands
    them here and gets coordinates normalized to the image as it arrived.

    Attributes:
        src_w: Source image width in pixels.
        src_h: Source image height in pixels.
        in_w: Model input width in pixels.
        in_h: Model input height in pixels.
        scale_x: Horizontal source-to-model scale factor.
        scale_y: Vertical source-to-model scale factor.
        pad_x: Horizontal offset of the resized image on the model canvas. Negative when the mode
            crops rather than pads.
        pad_y: Vertical offset of the resized image on the model canvas.
    """

    src_w: int
    src_h: int
    in_w: int
    in_h: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float

    @classmethod
    def build(cls, plan: ResizePlan, image_hw: tuple) -> "SourceMapper":
        """Derive the mapping for one source size under one resize plan.

        Args:
            plan: The resolved resize geometry.
            image_hw: The ``(height, width)`` of the source image.

        Returns:
            The mapper for that pairing.

        Raises:
            FamilyError: When the source image has a zero dimension.
        """
        src_h, src_w = int(image_hw[0]), int(image_hw[1])
        if src_w <= 0 or src_h <= 0:
            raise FamilyError("INVALID_IMAGE", f"source image is {src_w}x{src_h}")
        if plan.mode == "none":
            return cls(src_w, src_h, src_w, src_h, 1.0, 1.0, 0.0, 0.0)
        in_w, in_h = plan.width, plan.height
        if plan.mode == "stretch":
            return cls(src_w, src_h, in_w, in_h, in_w / src_w, in_h / src_h, 0.0, 0.0)
        if plan.mode == "letterbox":
            scale = min(in_w / src_w, in_h / src_h)
            drawn_w, drawn_h = round(src_w * scale), round(src_h * scale)
            if plan.pad_mode == "center":
                pad_x, pad_y = (in_w - drawn_w) // 2, (in_h - drawn_h) // 2
            else:
                pad_x = pad_y = 0
            return cls(src_w, src_h, in_w, in_h, scale, scale, float(pad_x), float(pad_y))
        scale = max(in_w / src_w, in_h / src_h)
        drawn_w, drawn_h = round(src_w * scale), round(src_h * scale)
        return cls(
            src_w, src_h, in_w, in_h, scale, scale,
            float(-((drawn_w - in_w) // 2)), float(-((drawn_h - in_h) // 2)),
        )

    def to_source(self, xy: np.ndarray) -> np.ndarray:
        """Map model-canvas pixel coordinates onto source pixel coordinates.

        Args:
            xy: An array whose last axis is ``(x, y)`` in model-input pixels.

        Returns:
            The same shape, in source-image pixels.
        """
        mapped = np.asarray(xy, dtype=np.float64).copy()
        mapped[..., 0] = (mapped[..., 0] - self.pad_x) / self.scale_x
        mapped[..., 1] = (mapped[..., 1] - self.pad_y) / self.scale_y
        return mapped

    def normalized_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map model-canvas corner boxes onto normalized source ``(x, y, w, h)`` boxes.

        Boxes are clipped to the source image: a detection whose box runs into the letterbox
        padding is reported against the picture that exists, not against the canvas.

        Args:
            boxes_xyxy: An ``(N, 4)`` array of ``(x0, y0, x1, y1)`` in model-input pixels.

        Returns:
            An ``(N, 4)`` array of ``(x, y, w, h)`` in ``[0, 1]`` of the source image.
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        corners = self.to_source(boxes.reshape(-1, 2, 2))
        x0 = np.clip(corners[:, 0, 0] / self.src_w, 0.0, 1.0)
        y0 = np.clip(corners[:, 0, 1] / self.src_h, 0.0, 1.0)
        x1 = np.clip(corners[:, 1, 0] / self.src_w, 0.0, 1.0)
        y1 = np.clip(corners[:, 1, 1] / self.src_h, 0.0, 1.0)
        return np.stack([x0, y0, np.maximum(x1 - x0, 0.0), np.maximum(y1 - y0, 0.0)], axis=1)

    def normalized_region(self, x0: float, y0: float, x1: float, y1: float) -> list:
        """Map one model-canvas corner box onto a normalized source ``[x, y, w, h]`` list.

        Args:
            x0: Left edge in model-input pixels.
            y0: Top edge in model-input pixels.
            x1: Right edge in model-input pixels, exclusive.
            y1: Bottom edge in model-input pixels, exclusive.

        Returns:
            ``[x, y, w, h]`` in ``[0, 1]`` of the source image, as plain floats.
        """
        box = self.normalized_boxes(np.array([[x0, y0, x1, y1]], dtype=np.float64))[0]
        return [float(value) for value in box]


def _channel_triple(block: dict, key: str, default: float) -> np.ndarray:
    """Read a per-channel constant that may be given as one number or three.

    Args:
        block: The ``preprocess`` mapping.
        key: The parameter name, ``"mean"`` or ``"std"``.
        default: The value used when the parameter is absent.

    Returns:
        A ``(3,)`` ``float32`` array.

    Raises:
        FamilyError: When the value is neither a number nor three numbers.
    """
    value = block.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.full(3, float(value), dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.asarray([float(entry) for entry in value], dtype=np.float32)
    raise FamilyError(
        "INVALID_FAMILY_PARAM",
        f"preprocess.{key} must be a number or three numbers, got {value!r}",
    )


def input_binding(m: BundleManifest) -> tuple:
    """Resolve which input tensor to feed and whether it carries a batch axis.

    The batch axis follows the declared input rank: a rank-4 input declares one, a rank-3 input
    does not. ``preprocess.batchAxis`` overrides that when a manifest needs to say so explicitly,
    and ``dynamicBatch`` decides when the rank is unknown. Whether the axis is *dynamic* is a
    scheduling fact (micro-batching, WP4b), not a preprocessing one.

    Args:
        m: The parsed bundle manifest.

    Returns:
        A ``(name, want_batch)`` pair.

    Raises:
        FamilyError: When the manifest declares no inputs, or names one it does not declare.
    """
    block = dict(m.preprocess or {})
    if not m.inputs:
        raise FamilyError("MISSING_INPUT", "manifest declares no inputs")
    name = block.get("inputName") or m.inputs[0].name
    spec = next((entry for entry in m.inputs if entry.name == name), None)
    if spec is None:
        declared = sorted(entry.name for entry in m.inputs)
        raise FamilyError(
            "MISSING_INPUT",
            f"preprocess.inputName={name!r} is not a declared input; has {declared}",
        )
    rank = len(tuple(spec.shape))
    if "batchAxis" in block:
        return name, bool(block["batchAxis"])
    if rank in (3, 4):
        return name, rank == 4
    return name, bool(m.dynamic_batch)


def validate_preprocess(m: BundleManifest) -> None:
    """Refuse a ``manifest.preprocess`` block no family can execute.

    This runs at staging, from every family's ``validate_manifest``, so a bundle whose declared
    transform disagrees with its own graph never reaches a session.

    Args:
        m: The parsed bundle manifest.

    Raises:
        FamilyError: When the geometry, colour order, layout, element type, or normalization is
            unusable, or when the declared input shape contradicts the resize target.
    """
    block = dict(m.preprocess or {})
    plan = resize_plan(m)
    choice(block, "colorOrder", ("RGB", "BGR"), "RGB", "UNSUPPORTED_COLOR_ORDER")
    layout = choice(block, "layout", ("NCHW", "NHWC"), "NCHW", "UNSUPPORTED_LAYOUT")
    dtype = choice(block, "dtype", _DTYPES, "float32", "UNSUPPORTED_DTYPE")
    choice(block, "highBitDepthMode", ("scaleTo8Bit", "raw"), "scaleTo8Bit", "UNSUPPORTED_BIT_DEPTH_MODE")
    scale = number(block, "scale", 1.0, -1e6, 1e6)
    mean = _channel_triple(block, "mean", 0.0)
    std = _channel_triple(block, "std", 1.0)
    if np.any(std == 0):
        raise FamilyError("INVALID_FAMILY_PARAM", "preprocess.std must not contain zero")
    if dtype == "uint8" and (scale != 1.0 or np.any(mean != 0) or np.any(std != 1)):
        raise FamilyError(
            "PREPROCESS_UINT8_NORMALIZATION",
            "preprocess.dtype=uint8 cannot carry scale, mean, or std; the model must take raw samples",
        )

    name, _ = input_binding(m)
    spec = next(entry for entry in m.inputs if entry.name == name)
    rank = len(tuple(spec.shape))
    if rank not in (3, 4):
        raise FamilyError(
            "UNSUPPORTED_INPUT_RANK",
            f"input {name!r} has rank {rank}; an image model takes rank 3 or 4",
        )
    if plan.mode != "none":
        height_axis, width_axis = (-2, -1) if layout == "NCHW" else (-3, -2)
        declared_h, declared_w = static_dim(spec, height_axis), static_dim(spec, width_axis)
        if declared_h is not None and declared_h != plan.height:
            raise FamilyError(
                "INPUT_SHAPE_MISMATCH",
                f"input {name!r} declares height {declared_h} but resize targets {plan.height}",
            )
        if declared_w is not None and declared_w != plan.width:
            raise FamilyError(
                "INPUT_SHAPE_MISMATCH",
                f"input {name!r} declares width {declared_w} but resize targets {plan.width}",
            )


def _resize_channels(work: np.ndarray, width: int, height: int, resample: int) -> np.ndarray:
    """Resample an ``HWC`` float array to an exact pixel size, one channel at a time.

    Resizing in float rather than in 8-bit samples keeps one code path for 8-bit and 16-bit
    sources and keeps the quantization at the end of the transform instead of the middle.

    Args:
        work: An ``(H, W, 3)`` ``float32`` array.
        width: Target width in pixels.
        height: Target height in pixels.
        resample: A Pillow resampling filter.

    Returns:
        A ``(height, width, 3)`` ``float32`` array.
    """
    if work.shape[0] == height and work.shape[1] == width:
        return work
    planes = []
    for channel in range(work.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(work[:, :, channel], dtype=np.float32), mode="F")
        planes.append(np.asarray(plane.resize((width, height), resample), dtype=np.float32))
    return np.stack(planes, axis=2)


def _fit_canvas(work: np.ndarray, plan: ResizePlan, mapper: SourceMapper, pad_scale: float) -> np.ndarray:
    """Place a resampled image on the model canvas, padding or cropping as the plan says.

    Args:
        work: An ``(H, W, 3)`` ``float32`` array in source geometry.
        plan: The resolved resize geometry.
        mapper: The mapping built from ``plan`` and this image's size.
        pad_scale: Factor applied to ``plan.pad_color`` so the same manifest constant means the
            same colour at 8-bit and at 16-bit sample scale.

    Returns:
        A ``(plan.height, plan.width, 3)`` ``float32`` array.
    """
    if plan.mode == "none":
        return work
    if plan.mode == "stretch":
        return _resize_channels(work, plan.width, plan.height, plan.resample)

    src_h, src_w = work.shape[:2]
    drawn_w = int(round(src_w * mapper.scale_x))
    drawn_h = int(round(src_h * mapper.scale_y))
    drawn = _resize_channels(work, drawn_w, drawn_h, plan.resample)
    if plan.mode == "centerCrop":
        left, top = int(-mapper.pad_x), int(-mapper.pad_y)
        return drawn[top : top + plan.height, left : left + plan.width]
    canvas = np.empty((plan.height, plan.width, 3), dtype=np.float32)
    canvas[:, :] = np.asarray(plan.pad_color, dtype=np.float32) * pad_scale
    left, top = int(mapper.pad_x), int(mapper.pad_y)
    canvas[top : top + drawn_h, left : left + drawn_w] = drawn
    return canvas


def preprocess_image(image: np.ndarray, m: BundleManifest) -> dict:
    """Execute ``manifest.preprocess`` on one decoded image.

    The transform runs in this order: colour order, resize and canvas fit, sample scale, per-channel
    mean and standard deviation, tensor layout, batch axis, element type. ``padColor`` is expressed
    in the manifest's ``colorOrder``, so the swap happens first and one constant means one colour.

    A 16-bit source is mapped into the 8-bit sample range before ``scale`` when
    ``highBitDepthMode`` is ``"scaleTo8Bit"``, so an 8-bit manifest serves a 16-bit image without
    change; ``"raw"`` keeps the full 0 to 65535 range and scales ``padColor`` to match.

    Args:
        image: An ``(H, W, 3)`` ``uint8`` or ``uint16`` array from
            :func:`~image_processor.engine.decode.decode_image`.
        m: The parsed bundle manifest.

    Returns:
        A single-entry mapping of input tensor name to the fed array.

    Raises:
        FamilyError: When the image is not ``HWC`` with three channels, or when the manifest's
            preprocess block is unusable.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise FamilyError(
            "INVALID_IMAGE",
            f"expected an (H, W, 3) image, got shape {array.shape}",
        )
    block = dict(m.preprocess or {})
    plan = resize_plan(m)
    color_order = choice(block, "colorOrder", ("RGB", "BGR"), "RGB", "UNSUPPORTED_COLOR_ORDER")
    layout = choice(block, "layout", ("NCHW", "NHWC"), "NCHW", "UNSUPPORTED_LAYOUT")
    dtype_name = choice(block, "dtype", _DTYPES, "float32", "UNSUPPORTED_DTYPE")
    bit_depth_mode = choice(
        block, "highBitDepthMode", ("scaleTo8Bit", "raw"), "scaleTo8Bit", "UNSUPPORTED_BIT_DEPTH_MODE"
    )

    pad_scale = 1.0
    work = array.astype(np.float32)
    if array.dtype == np.uint16:
        if bit_depth_mode == "scaleTo8Bit":
            work = work / _UINT16_TO_UINT8
        else:
            pad_scale = _UINT16_TO_UINT8
    if color_order == "BGR":
        work = work[:, :, ::-1]

    mapper = SourceMapper.build(plan, (array.shape[0], array.shape[1]))
    work = _fit_canvas(np.ascontiguousarray(work), plan, mapper, pad_scale)

    if dtype_name == "uint8":
        tensor = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    else:
        scale = number(block, "scale", 1.0, -1e6, 1e6)
        mean = _channel_triple(block, "mean", 0.0)
        std = _channel_triple(block, "std", 1.0)
        if np.any(std == 0):
            raise FamilyError("INVALID_FAMILY_PARAM", "preprocess.std must not contain zero")
        tensor = (work * np.float32(scale) - mean) / std

    if layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))
    name, want_batch = input_binding(m)
    if want_batch:
        tensor = tensor[None, ...]
    return {name: np.ascontiguousarray(tensor, dtype=_DTYPES[dtype_name])}


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_items: Optional[int] = None,
) -> list:
    """Run class-aware greedy non-maximum suppression.

    Suppression is per class: a high-scoring bolt never suppresses a washer that occupies the same
    pixels, because two different things can be in one place and a line-clearance decision depends
    on seeing both.

    Args:
        boxes_xyxy: An ``(N, 4)`` array of ``(x0, y0, x1, y1)`` in any one consistent space.
        scores: An ``(N,)`` array of confidences.
        classes: An ``(N,)`` array of class indices.
        iou_threshold: Boxes of the same class overlapping a kept box by more than this are
            dropped.
        max_items: Stop after keeping this many boxes, or ``None`` for no cap.

    Returns:
        Indices into the input arrays, highest score first.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(classes).reshape(-1)
    if values.size == 0:
        return []

    order = np.argsort(-values, kind="stable")
    ordered = boxes[order]
    ordered_labels = labels[order]
    widths = np.maximum(ordered[:, 2] - ordered[:, 0], 0.0)
    heights = np.maximum(ordered[:, 3] - ordered[:, 1], 0.0)
    areas = widths * heights

    alive = np.ones(order.size, dtype=bool)
    keep = []
    for index in range(order.size):
        if not alive[index]:
            continue
        keep.append(int(order[index]))
        if max_items is not None and len(keep) >= max_items:
            break
        rest = np.nonzero(alive[index + 1 :])[0] + index + 1
        rest = rest[ordered_labels[rest] == ordered_labels[index]]
        if rest.size == 0:
            continue
        left = np.maximum(ordered[index, 0], ordered[rest, 0])
        top = np.maximum(ordered[index, 1], ordered[rest, 1])
        right = np.minimum(ordered[index, 2], ordered[rest, 2])
        bottom = np.minimum(ordered[index, 3], ordered[rest, 3])
        overlap = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
        union = areas[index] + areas[rest] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        alive[rest[iou > iou_threshold]] = False
    return keep


from image_processor.engine.families.anomaly import AnomalyFamily  # noqa: E402
from image_processor.engine.families.classification import ClassificationFamily  # noqa: E402
from image_processor.engine.families.detection import DetectionFamily  # noqa: E402
from image_processor.engine.families.segmentation import SegmentationFamily  # noqa: E402

#: The four built-in interpreters (DESIGN.md §8.1). There is no fifth, and no bundle adds one.
FAMILIES: dict = {
    Family.CLASSIFICATION: ClassificationFamily(),
    Family.DETECTION: DetectionFamily(),
    Family.SEGMENTATION: SegmentationFamily(),
    Family.ANOMALY: AnomalyFamily(),
}


def family_for(m: BundleManifest) -> TaskFamily:
    """Select the interpreter for a manifest's declared family.

    Args:
        m: The parsed bundle manifest.

    Returns:
        The registered :class:`TaskFamily`.

    Raises:
        FamilyError: When the manifest names a family this component does not implement.
    """
    try:
        family = Family(m.family)
    except ValueError as error:
        raise FamilyError(
            "UNSUPPORTED_FAMILY", f"family {m.family!r} is not a built-in task family"
        ) from error
    return FAMILIES[family]


__all__ = [
    "FAMILIES",
    "AnomalyFamily",
    "ClassificationFamily",
    "DetectionFamily",
    "FamilyError",
    "ResizePlan",
    "SegmentationFamily",
    "SourceMapper",
    "TaskFamily",
    "apply_activation",
    "choice",
    "drop_batch",
    "family_for",
    "family_labels",
    "input_binding",
    "label_for",
    "nms",
    "number",
    "output_spec",
    "pick_output",
    "positive_int",
    "preprocess_image",
    "resize_plan",
    "sigmoid",
    "softmax",
    "static_dim",
    "validate_preprocess",
]
