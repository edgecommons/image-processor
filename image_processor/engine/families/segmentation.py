"""The segmentation family: pixel counts and regions, never masks (DESIGN.md §8.1).

A segmentation head produces one value per pixel per class, and a full mask is both far larger
than the message budget and useless to a decision rule. So the family reduces the mask to what a
decision is actually made on: how many pixels each class claims, what fraction of the image that
is, and the region those pixels occupy. The mask itself never leaves the executor cell.

Two head shapes are read. ``argmax`` takes the class with the highest value at each pixel, which
is the usual multi-class form. ``threshold`` compares one channel against a constant, which is the
usual binary form. Every class named in the manifest gets an entry, including the ones with no
pixels, so a rule such as "no defect pixels" evaluates on a clean image instead of failing to
resolve its path.
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
    number,
    pick_output,
    preprocess_image,
    resize_plan,
    static_dim,
    validate_preprocess,
    single_output,
)
from image_processor.types import BundleManifest, Family, NormalizedOutput

logger = logging.getLogger(__name__)

#: Head shapes this family reads.
SEGMENTATION_MODES = ("argmax", "threshold")


def _class_axis(m: BundleManifest, layout: str):
    """Locate the class dimension of the declared segmentation output.

    Args:
        m: The parsed bundle manifest.
        layout: ``"NCHW"`` or ``"NHWC"``.

    Returns:
        The declared class count, or ``None`` when the output is rank 2 or the axis is dynamic.
    """
    spec = single_output(m, "segmentation")
    if len(tuple(spec.shape)) < 3:
        return 1
    return static_dim(spec, -3 if layout == "NCHW" else -1)


def _ignore_index(params: dict, classes: int):
    """Read the class index excluded from the reported segments.

    Args:
        params: The manifest's ``familyParams``.
        classes: The number of classes the manifest names.

    Returns:
        The index to skip, or ``None``.

    Raises:
        FamilyError: When the value is neither ``None`` nor an in-range integer.
    """
    value = params.get("ignoreIndex")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < classes:
        raise FamilyError(
            "INVALID_FAMILY_PARAM",
            f"familyParams.ignoreIndex must be a class index below {classes}, got {value!r}",
        )
    return value


def _region(mask: np.ndarray, mapper: SourceMapper) -> tuple:
    """Reduce one boolean mask to a pixel count and a normalized source region.

    Args:
        mask: A ``(gh, gw)`` boolean array in the output grid.
        mapper: The source mapping, which carries the model input size.

    Returns:
        A ``(pixels, bbox)`` pair, where ``bbox`` is ``[x, y, w, h]`` normalized to the source
        image, or ``None`` when the mask is empty.
    """
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return 0, None
    grid_h, grid_w = mask.shape
    step_x, step_y = mapper.in_w / grid_w, mapper.in_h / grid_h
    bbox = mapper.normalized_region(
        float(columns.min()) * step_x,
        float(rows.min()) * step_y,
        float(columns.max() + 1) * step_x,
        float(rows.max() + 1) * step_y,
    )
    return int(rows.size), bbox


class SegmentationFamily:
    """Interprets a per-pixel class head as counts and regions.

    Attributes:
        family: Always :attr:`~image_processor.types.Family.SEGMENTATION`.
    """

    family = Family.SEGMENTATION

    def validate_manifest(self, m: BundleManifest) -> None:
        """Refuse a manifest whose head is not a per-pixel class map.

        Args:
            m: The parsed bundle manifest.

        Raises:
            FamilyError: When the family does not match, the head has the wrong output count or
                rank, the class dimension disagrees with the label set, or ``familyParams`` is
                incomplete or out of range.
        """
        if Family(m.family) is not self.family:
            raise FamilyError("FAMILY_MISMATCH", f"manifest family is {m.family!r}")
        validate_preprocess(m)
        spec = single_output(m, "segmentation")
        rank = len(tuple(spec.shape))
        if rank not in (2, 3, 4):
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_RANK",
                f"output {spec.name!r} has rank {rank}; a class map is rank 2, 3, or 4",
            )
        params = dict(m.family_params or {})
        mode = choice(params, "mode", SEGMENTATION_MODES, "argmax", "UNSUPPORTED_SEGMENTATION_MODE")
        layout = choice(params, "outputLayout", ("NCHW", "NHWC"), "NCHW", "UNSUPPORTED_LAYOUT")
        choice(params, "activation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION")
        number(params, "minPixels", 0, 0, 1e12)
        labels = family_labels(m)
        declared = _class_axis(m, layout)
        if mode == "argmax":
            _ignore_index(params, len(labels))
            if declared is not None and declared != len(labels):
                raise FamilyError(
                    "CLASS_DIM_MISMATCH",
                    f"output {spec.name!r} has {declared} channels but the manifest names {len(labels)}",
                )
        else:
            number(params, "threshold", 0.5, -1e9, 1e9)
            positive = params.get("positiveLabel", labels[-1])
            if not isinstance(positive, str) or not positive:
                raise FamilyError(
                    "INVALID_FAMILY_PARAM", "familyParams.positiveLabel must be a non-empty string"
                )
            if declared is not None and declared != 1:
                raise FamilyError(
                    "UNSUPPORTED_OUTPUT_SHAPE",
                    f"threshold mode reads one channel, output {spec.name!r} declares {declared}",
                )

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
        """Reduce a per-pixel class map to counts, fractions, and regions.

        Args:
            outputs: The session outputs, keyed by tensor name.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image.

        Returns:
            The normalized output with ``segments`` populated. Each entry holds ``pixels``, the
            ``fraction`` of the class map it covers, and a normalized ``bbox`` or ``None``.

        Raises:
            FamilyError: When the tensor is absent or its shape contradicts the manifest.
        """
        params = dict(m.family_params or {})
        labels = family_labels(m)
        mode = choice(params, "mode", SEGMENTATION_MODES, "argmax", "UNSUPPORTED_SEGMENTATION_MODE")
        layout = choice(params, "outputLayout", ("NCHW", "NHWC"), "NCHW", "UNSUPPORTED_LAYOUT")
        activation = choice(
            params, "activation", ("none", "sigmoid", "softmax"), "none", "UNSUPPORTED_ACTIVATION"
        )
        floor = int(number(params, "minPixels", 0, 0, 1e12))

        name = params.get("outputName")
        raw = np.asarray(pick_output(outputs, m, name))
        where = name or "the segmentation output"
        if raw.ndim == 2:
            plane = raw[None, :, :]
        else:
            plane = drop_batch(raw, 3, where)
            if layout == "NHWC":
                plane = np.transpose(plane, (2, 0, 1))
        mapper = SourceMapper.build(resize_plan(m), image_hw)
        total = float(plane.shape[1] * plane.shape[2])

        segments = {}
        if mode == "argmax":
            if plane.shape[0] != len(labels):
                raise FamilyError(
                    "CLASS_DIM_MISMATCH",
                    f"{where} has {plane.shape[0]} channels but the manifest names {len(labels)}",
                )
            skip = _ignore_index(params, len(labels))
            assigned = np.argmax(plane, axis=0)
            for index in range(len(labels)):
                if index == skip:
                    continue
                pixels, bbox = _region(assigned == index, mapper)
                if pixels >= floor:
                    segments[label_for(labels, index)] = {
                        "pixels": pixels,
                        "fraction": pixels / total,
                        "bbox": bbox,
                    }
        else:
            if plane.shape[0] != 1:
                raise FamilyError(
                    "UNSUPPORTED_OUTPUT_SHAPE",
                    f"threshold mode reads one channel, {where} returned {plane.shape[0]}",
                )
            threshold = number(params, "threshold", 0.5, -1e9, 1e9)
            scores = apply_activation(plane[0], activation)
            pixels, bbox = _region(scores >= threshold, mapper)
            if pixels >= floor:
                segments[str(params.get("positiveLabel", labels[-1]))] = {
                    "pixels": pixels,
                    "fraction": pixels / total,
                    "bbox": bbox,
                }
        return NormalizedOutput(
            family=self.family,
            segments=segments,
            raw_shapes={key: tuple(np.asarray(value).shape) for key, value in outputs.items()},
        )
