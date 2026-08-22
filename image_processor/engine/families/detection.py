"""The detection family: two head conventions, one normalized answer (DESIGN.md §8.1).

Detection heads do not agree on what a model returns, so the family supports the two conventions
that cover the reference corpus and most exports beyond it:

``yoloxGrid``
    The anchor-free single-tensor form. One ``[A, 4 + 1 + C]`` block, where ``A`` is every cell of
    every feature map concatenated stride by stride. The first four values are the box as a grid
    offset and a log size, the fifth is objectness, and the rest are class scores. The family
    rebuilds the grid from ``strides`` and the model input size and decodes it.

``decodedBoxes``
    The already-decoded form: separate ``boxes``, ``scores``, and optional ``classes`` tensors, as
    an SSD export produces. Corner order and coordinate space are declared, because exports differ
    on both.

After either decode the path is identical: score floor, class-aware non-maximum suppression, cap
at ``maxResultItems``, and coordinates mapped off the model canvas onto the source image. A
detection that came back from a letterboxed 640x640 canvas is reported in ``[0, 1]`` of the image
the camera actually took.
"""

from __future__ import annotations

import logging

import numpy as np

from image_processor.engine.families import (
    FamilyError,
    SourceMapper,
    apply_activation,
    choice,
    drop_batch,
    family_labels,
    label_for,
    nms,
    number,
    output_spec,
    pick_output,
    positive_int,
    preprocess_image,
    resize_plan,
    static_dim,
    validate_preprocess,
)
from image_processor.types import BundleManifest, Detection, Family, NormalizedOutput

logger = logging.getLogger(__name__)

#: The head conventions this component decodes. A manifest naming anything else is refused.
DECODE_MODES = ("yoloxGrid", "decodedBoxes")

#: Corner orders a ``decodedBoxes`` head may use.
_BOX_FORMATS = ("xyxy", "yxyx", "cxcywh")

#: Default output tensor names for a ``decodedBoxes`` head.
_DEFAULT_OUTPUT_NAMES = {"boxes": "boxes", "scores": "scores", "classes": "classes"}


def _output_names(params: dict) -> dict:
    """Resolve the tensor names of a ``decodedBoxes`` head.

    Args:
        params: The manifest's ``familyParams``.

    Returns:
        A mapping with ``boxes``, ``scores``, ``classes``, and optionally ``count`` names.

    Raises:
        FamilyError: When ``outputNames`` is not a mapping of strings.
    """
    configured = params.get("outputNames", {})
    if not isinstance(configured, dict):
        raise FamilyError("INVALID_FAMILY_PARAM", "familyParams.outputNames must be an object")
    names = dict(_DEFAULT_OUTPUT_NAMES)
    for key, value in configured.items():
        if key not in ("boxes", "scores", "classes", "count") or not isinstance(value, str):
            raise FamilyError(
                "INVALID_FAMILY_PARAM",
                f"familyParams.outputNames.{key} is not a supported tensor name entry",
            )
        names[key] = value
    if "count" in configured:
        names["count"] = configured["count"]
    return names


def _grid(strides: list, in_w: int, in_h: int) -> tuple:
    """Rebuild the anchor-free grid a ``yoloxGrid`` head was exported against.

    Cells are emitted stride by stride, and within a stride row by row, which is the order the
    export concatenates its feature maps in.

    Args:
        strides: The feature-map strides, largest feature map first.
        in_w: Model input width in pixels.
        in_h: Model input height in pixels.

    Returns:
        A ``(grid_x, grid_y, cell_stride)`` triple of ``(A,)`` arrays.

    Raises:
        FamilyError: When a stride does not divide the model input size.
    """
    xs, ys, cell = [], [], []
    for stride in strides:
        if in_w % stride or in_h % stride:
            raise FamilyError(
                "INVALID_FAMILY_PARAM",
                f"stride {stride} does not divide the {in_w}x{in_h} model input",
            )
        columns, rows = in_w // stride, in_h // stride
        grid_y, grid_x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        xs.append(grid_x.reshape(-1))
        ys.append(grid_y.reshape(-1))
        cell.append(np.full(rows * columns, stride, dtype=np.float64))
    return (
        np.concatenate(xs).astype(np.float64),
        np.concatenate(ys).astype(np.float64),
        np.concatenate(cell),
    )


def _to_corners(boxes: np.ndarray, box_format: str) -> np.ndarray:
    """Convert a box block into ``(x0, y0, x1, y1)`` with the corners ordered.

    Args:
        boxes: A ``(K, 4)`` array in ``box_format``.
        box_format: ``"xyxy"``, ``"yxyx"``, or ``"cxcywh"``.

    Returns:
        A ``(K, 4)`` array of ``(x0, y0, x1, y1)`` with ``x0 <= x1`` and ``y0 <= y1``.
    """
    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if box_format == "yxyx":
        values = values[:, [1, 0, 3, 2]]
    elif box_format == "cxcywh":
        half_w, half_h = values[:, 2] / 2.0, values[:, 3] / 2.0
        values = np.stack(
            [
                values[:, 0] - half_w,
                values[:, 1] - half_h,
                values[:, 0] + half_w,
                values[:, 1] + half_h,
            ],
            axis=1,
        )
    return np.stack(
        [
            np.minimum(values[:, 0], values[:, 2]),
            np.minimum(values[:, 1], values[:, 3]),
            np.maximum(values[:, 0], values[:, 2]),
            np.maximum(values[:, 1], values[:, 3]),
        ],
        axis=1,
    )


def _strides(params: dict) -> list:
    """Read and check the ``strides`` parameter of a ``yoloxGrid`` head.

    Args:
        params: The manifest's ``familyParams``.

    Returns:
        The strides as a list of positive integers.

    Raises:
        FamilyError: When the parameter is absent, empty, or holds a non-positive integer.
    """
    strides = params.get("strides")
    if strides is None:
        raise FamilyError("MISSING_FAMILY_PARAM", "familyParams.strides is required for yoloxGrid")
    if not isinstance(strides, (list, tuple)) or not strides:
        raise FamilyError("INVALID_FAMILY_PARAM", "familyParams.strides must be a non-empty list")
    resolved = []
    for stride in strides:
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise FamilyError(
                "INVALID_FAMILY_PARAM", f"familyParams.strides holds {stride!r}, not a positive integer"
            )
        resolved.append(stride)
    return resolved


def _class_index_offset(params: dict) -> int:
    """Read the constant subtracted from a head's raw class ids.

    An export whose class ids start at one because index zero is a background class declares
    ``classIndexOffset: 1``, so the labels list stays the label list.

    Args:
        params: The manifest's ``familyParams``.

    Returns:
        The offset.

    Raises:
        FamilyError: When the value is not an integer.
    """
    value = params.get("classIndexOffset", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FamilyError(
            "INVALID_FAMILY_PARAM", f"familyParams.classIndexOffset must be an integer, got {value!r}"
        )
    return value


class DetectionFamily:
    """Interprets an anchor-free grid head or an already-decoded box head.

    Attributes:
        family: Always :attr:`~image_processor.types.Family.DETECTION`.
    """

    family = Family.DETECTION

    def validate_manifest(self, m: BundleManifest) -> None:
        """Refuse a manifest whose head neither convention reads.

        Args:
            m: The parsed bundle manifest.

        Raises:
            FamilyError: When the family does not match, the decode mode is not one of
                :data:`DECODE_MODES`, the declared outputs do not fit the chosen convention, or
                ``familyParams`` is incomplete or out of range.
        """
        if Family(m.family) is not self.family:
            raise FamilyError("FAMILY_MISMATCH", f"manifest family is {m.family!r}")
        validate_preprocess(m)
        params = dict(m.family_params or {})
        mode = choice(params, "decode", DECODE_MODES, "yoloxGrid", "UNSUPPORTED_DECODE")
        labels = family_labels(m)
        number(params, "scoreThreshold", 0.25, 0.0, 1.0)
        number(params, "iouThreshold", 0.45, 0.0, 1.0)
        positive_int(params, "maxDetections", m.max_result_items or 100)
        if mode == "yoloxGrid":
            self._validate_grid(m, params, labels)
        else:
            self._validate_decoded(m, params, labels)

    def _validate_grid(self, m: BundleManifest, params: dict, labels: list) -> None:
        """Check the manifest of a ``yoloxGrid`` head.

        Args:
            m: The parsed bundle manifest.
            params: The manifest's ``familyParams``.
            labels: The resolved label list.

        Raises:
            FamilyError: When the single output does not match the grid the strides describe.
        """
        if len(m.outputs) != 1:
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_COUNT",
                f"yoloxGrid reads one output, manifest declares {len(m.outputs)}",
            )
        spec = m.outputs[0]
        rank = len(tuple(spec.shape))
        if rank not in (2, 3):
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_RANK",
                f"output {spec.name!r} has rank {rank}; a grid head is rank 2 or 3",
            )
        strides = _strides(params)
        objectness = bool(params.get("objectness", True))
        choice(params, "scoreActivation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION")
        choice(params, "objectnessActivation", ("none", "sigmoid"), "none", "UNSUPPORTED_ACTIVATION")
        expected = 4 + (1 if objectness else 0) + len(labels)
        declared = static_dim(spec, -1)
        if declared is not None and declared != expected:
            raise FamilyError(
                "OUTPUT_DIM_MISMATCH",
                f"output {spec.name!r} has {declared} values per cell, the manifest describes {expected}",
            )
        plan = resize_plan(m)
        if plan.mode != "none":
            anchors = int(_grid(strides, plan.width, plan.height)[0].size)
            declared_anchors = static_dim(spec, -2)
            if declared_anchors is not None and declared_anchors != anchors:
                raise FamilyError(
                    "OUTPUT_DIM_MISMATCH",
                    f"output {spec.name!r} has {declared_anchors} cells, the strides describe {anchors}",
                )

    def _validate_decoded(self, m: BundleManifest, params: dict, labels: list) -> None:
        """Check the manifest of a ``decodedBoxes`` head.

        Args:
            m: The parsed bundle manifest.
            params: The manifest's ``familyParams``.
            labels: The resolved label list.

        Raises:
            FamilyError: When a named tensor is not declared, or the declared shapes do not fit.
        """
        if len(m.outputs) < 2:
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_COUNT",
                f"decodedBoxes reads boxes and scores, manifest declares {len(m.outputs)} output(s)",
            )
        names = _output_names(params)
        layout = choice(
            params, "scoresLayout", ("perBox", "perClass"), "perBox", "UNSUPPORTED_SCORES_LAYOUT"
        )
        choice(params, "boxFormat", _BOX_FORMATS, "xyxy", "UNSUPPORTED_BOX_FORMAT")
        choice(
            params, "boxCoordinates", ("normalized", "pixels"), "normalized", "UNSUPPORTED_BOX_COORDINATES"
        )
        choice(params, "scoreActivation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION")
        _class_index_offset(params)

        boxes = output_spec(m, names["boxes"])
        if static_dim(boxes, -1) not in (None, 4):
            raise FamilyError(
                "OUTPUT_DIM_MISMATCH",
                f"output {names['boxes']!r} does not end in four box values",
            )
        scores = output_spec(m, names["scores"])
        if layout == "perBox":
            output_spec(m, names["classes"])
        else:
            declared = static_dim(scores, -1)
            if declared is not None and declared != len(labels):
                raise FamilyError(
                    "CLASS_DIM_MISMATCH",
                    f"output {names['scores']!r} has {declared} classes but the manifest names {len(labels)}",
                )
        if "count" in params.get("outputNames", {}):
            output_spec(m, names["count"])

    def preprocess(self, image: np.ndarray, m: BundleManifest) -> dict:
        """Build the session feed for one image.

        Args:
            image: An ``(H, W, 3)`` decoded image.
            m: The parsed bundle manifest.

        Returns:
            A mapping of input tensor name to array.
        """
        return preprocess_image(image, m)

    def postprocess(self, outputs: dict, m: BundleManifest, image_hw: tuple) -> NormalizedOutput:
        """Decode, suppress, cap, and map one detection head onto the source image.

        Args:
            outputs: The session outputs, keyed by tensor name.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image.

        Returns:
            The normalized output with ``detections`` populated, highest score first.

        Raises:
            FamilyError: When a tensor is absent or its shape contradicts the validated manifest.
        """
        params = dict(m.family_params or {})
        labels = family_labels(m)
        mode = choice(params, "decode", DECODE_MODES, "yoloxGrid", "UNSUPPORTED_DECODE")
        mapper = SourceMapper.build(resize_plan(m), image_hw)
        if mode == "yoloxGrid":
            corners, scores, classes = self._decode_grid(outputs, m, params, labels, mapper)
            suppress = True
        else:
            corners, scores, classes, suppress = self._decode_boxes(outputs, m, params, labels, mapper)

        floor = number(params, "scoreThreshold", 0.25, 0.0, 1.0)
        iou_threshold = number(params, "iouThreshold", 0.45, 0.0, 1.0)
        cap = positive_int(params, "maxDetections", m.max_result_items or 100)
        if m.max_result_items:
            cap = min(cap, m.max_result_items)

        above = np.nonzero(scores >= floor)[0]
        corners, scores, classes = corners[above], scores[above], classes[above]
        if suppress:
            keep = nms(corners, scores, classes, iou_threshold, cap)
        else:
            keep = [int(index) for index in np.argsort(-scores, kind="stable")[:cap]]

        boxes = mapper.normalized_boxes(corners[keep]) if keep else np.zeros((0, 4))
        detections = [
            Detection(
                label=label_for(labels, int(classes[chosen])),
                index=int(classes[chosen]),
                score=float(scores[chosen]),
                box=tuple(float(value) for value in boxes[position]),
            )
            for position, chosen in enumerate(keep)
        ]
        return NormalizedOutput(
            family=self.family,
            detections=detections,
            raw_shapes={key: tuple(np.asarray(value).shape) for key, value in outputs.items()},
        )

    def _decode_grid(
        self, outputs: dict, m: BundleManifest, params: dict, labels: list, mapper: SourceMapper
    ) -> tuple:
        """Decode an anchor-free grid head into model-canvas corner boxes.

        Args:
            outputs: The session outputs.
            m: The parsed bundle manifest.
            params: The manifest's ``familyParams``.
            labels: The resolved label list.
            mapper: The source mapping, which carries the model input size.

        Returns:
            A ``(corners, scores, classes)`` triple in model-canvas pixels.

        Raises:
            FamilyError: When the tensor does not match the grid the strides describe.
        """
        strides = _strides(params)
        objectness = bool(params.get("objectness", True))
        score_activation = choice(
            params, "scoreActivation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION"
        )
        objectness_activation = choice(
            params, "objectnessActivation", ("none", "sigmoid"), "none", "UNSUPPORTED_ACTIVATION"
        )
        name = params.get("outputName")
        raw = drop_batch(pick_output(outputs, m, name), 2, name or "the detection output")
        grid_x, grid_y, cell = _grid(strides, mapper.in_w, mapper.in_h)
        expected = 4 + (1 if objectness else 0) + len(labels)
        if raw.shape[0] != grid_x.size or raw.shape[1] != expected:
            raise FamilyError(
                "OUTPUT_DIM_MISMATCH",
                f"detection output is {raw.shape}, the manifest describes ({grid_x.size}, {expected})",
            )

        values = raw.astype(np.float64)
        center_x = (values[:, 0] + grid_x) * cell
        center_y = (values[:, 1] + grid_y) * cell
        width = np.exp(np.clip(values[:, 2], -30.0, 30.0)) * cell
        height = np.exp(np.clip(values[:, 3], -30.0, 30.0)) * cell
        if objectness:
            confidence = apply_activation(values[:, 4], objectness_activation)
            class_block = values[:, 5:]
        else:
            confidence = np.ones(values.shape[0], dtype=np.float64)
            class_block = values[:, 4:]
        class_scores = apply_activation(class_block, score_activation, axis=-1) * confidence[:, None]
        classes = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]
        corners = _to_corners(np.stack([center_x, center_y, width, height], axis=1), "cxcywh")
        return corners, scores, classes

    def _decode_boxes(
        self, outputs: dict, m: BundleManifest, params: dict, labels: list, mapper: SourceMapper
    ) -> tuple:
        """Read an already-decoded box head into model-canvas corner boxes.

        Args:
            outputs: The session outputs.
            m: The parsed bundle manifest.
            params: The manifest's ``familyParams``.
            labels: The resolved label list.
            mapper: The source mapping, which carries the model input size.

        Returns:
            A ``(corners, scores, classes, suppress)`` tuple, where ``suppress`` reports whether
            non-maximum suppression still has to run.

        Raises:
            FamilyError: When a tensor is absent or the three blocks disagree on how many boxes
                there are.
        """
        names = _output_names(params)
        layout = choice(
            params, "scoresLayout", ("perBox", "perClass"), "perBox", "UNSUPPORTED_SCORES_LAYOUT"
        )
        box_format = choice(params, "boxFormat", _BOX_FORMATS, "xyxy", "UNSUPPORTED_BOX_FORMAT")
        space = choice(
            params, "boxCoordinates", ("normalized", "pixels"), "normalized", "UNSUPPORTED_BOX_COORDINATES"
        )
        score_activation = choice(
            params, "scoreActivation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION"
        )
        offset = _class_index_offset(params)

        boxes = drop_batch(pick_output(outputs, m, names["boxes"]), 2, names["boxes"])
        if boxes.shape[-1] != 4:
            raise FamilyError(
                "OUTPUT_DIM_MISMATCH", f"output {names['boxes']!r} is {boxes.shape}, expected (K, 4)"
            )
        if layout == "perClass":
            block = drop_batch(pick_output(outputs, m, names["scores"]), 2, names["scores"])
            probabilities = apply_activation(block, score_activation, axis=-1)
            classes = np.argmax(probabilities, axis=1)
            scores = probabilities[np.arange(probabilities.shape[0]), classes]
        else:
            scores = apply_activation(
                drop_batch(pick_output(outputs, m, names["scores"]), 1, names["scores"]),
                score_activation,
            )
            raw_classes = drop_batch(pick_output(outputs, m, names["classes"]), 1, names["classes"])
            classes = np.rint(np.asarray(raw_classes, dtype=np.float64)).astype(np.int64) - offset
        if not (boxes.shape[0] == scores.size == classes.size):
            raise FamilyError(
                "OUTPUT_DIM_MISMATCH",
                f"boxes, scores, and classes disagree: {boxes.shape[0]}, {scores.size}, {classes.size}",
            )

        count_name = names.get("count")
        if count_name is not None and count_name in outputs:
            reported = int(np.asarray(outputs[count_name]).reshape(-1)[0])
            kept = max(0, min(reported, boxes.shape[0]))
            boxes, scores, classes = boxes[:kept], scores[:kept], classes[:kept]

        corners = _to_corners(boxes, box_format)
        if space == "normalized":
            canvas = np.array([mapper.in_w, mapper.in_h, mapper.in_w, mapper.in_h], dtype=np.float64)
            corners = corners * canvas
        suppress = bool(params.get("applyNms", True))
        return corners, np.asarray(scores, dtype=np.float64).reshape(-1), classes, suppress
