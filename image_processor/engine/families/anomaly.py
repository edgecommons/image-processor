"""The anomaly family: one score against one threshold (DESIGN.md §8.1).

An anomaly model answers "how unlike the good population is this image". The head is either a
scalar the model computed itself or a per-pixel map the family reduces, and the manifest says
which, because the two are not interchangeable and guessing from a tensor shape would be a guess.

The reported score is comparable across images of the same model: the raw value passes through the
declared activation, then the declared linear normalization, and lands in ``[0, 1]`` when a
normalization is configured. ``direction`` records which end is anomalous, because a model trained
to score similarity is as common as one trained to score distance. A map head also carries a
summary: the extremes, how many pixels crossed the threshold, and the region they occupy. The map
itself is never published.
"""

from __future__ import annotations

import logging

import numpy as np

from image_processor.engine.families import (
    FamilyError,
    SourceMapper,
    apply_activation,
    choice,
    number,
    pick_output,
    preprocess_image,
    resize_plan,
    validate_preprocess,
)
from image_processor.types import BundleManifest, Family, NormalizedOutput

logger = logging.getLogger(__name__)

#: Where the score comes from: the model itself, or a reduction of a per-pixel map.
ANOMALY_SOURCES = ("scalar", "mapMax", "mapMean")


def _threshold(params: dict) -> float:
    """Read the decision threshold, which an anomaly manifest must declare.

    Args:
        params: The manifest's ``familyParams``.

    Returns:
        The threshold.

    Raises:
        FamilyError: When it is absent or is not a number.
    """
    if "threshold" not in params:
        raise FamilyError("MISSING_FAMILY_PARAM", "familyParams.threshold is required for anomaly")
    return number(params, "threshold", 0.0, -1e12, 1e12)


def _normalization(params: dict):
    """Read the optional linear rescale applied before the threshold comparison.

    Args:
        params: The manifest's ``familyParams``.

    Returns:
        A ``(low, high)`` pair, or ``None`` when no rescale is configured.

    Raises:
        FamilyError: When the block is not an object with a numeric ``min`` below its ``max``.
    """
    block = params.get("normalization")
    if block is None:
        return None
    if not isinstance(block, dict) or "min" not in block or "max" not in block:
        raise FamilyError(
            "INVALID_FAMILY_PARAM", "familyParams.normalization must be an object with min and max"
        )
    low = number(block, "min", 0.0, -1e12, 1e12)
    high = number(block, "max", 1.0, -1e12, 1e12)
    if high <= low:
        raise FamilyError(
            "INVALID_FAMILY_PARAM", f"familyParams.normalization.max ({high}) must exceed min ({low})"
        )
    return low, high


def _as_map(values: np.ndarray, where: str) -> np.ndarray:
    """Reduce an anomaly-map tensor to its two spatial axes.

    Args:
        values: The output tensor.
        where: The tensor name, for the error message.

    Returns:
        A ``(gh, gw)`` ``float64`` array.

    Raises:
        FamilyError: When the tensor is not a single-channel spatial map.
    """
    plane = np.asarray(values, dtype=np.float64)
    while plane.ndim > 2 and plane.shape[0] == 1:
        plane = plane[0]
    if plane.ndim == 3:
        if plane.shape[-1] == 1:
            plane = plane[..., 0]
        else:
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_SHAPE", f"{where} is {plane.shape}; an anomaly map has one channel"
            )
    if plane.ndim != 2:
        raise FamilyError(
            "UNSUPPORTED_OUTPUT_SHAPE", f"{where} has rank {plane.ndim}; an anomaly map is a 2-D grid"
        )
    return plane


class AnomalyFamily:
    """Interprets a scalar or per-pixel anomaly head.

    Attributes:
        family: Always :attr:`~image_processor.types.Family.ANOMALY`.
    """

    family = Family.ANOMALY

    def validate_manifest(self, m: BundleManifest) -> None:
        """Refuse a manifest whose head is not one score or one map.

        Args:
            m: The parsed bundle manifest.

        Raises:
            FamilyError: When the family does not match, the head has the wrong output count or
                shape, or ``familyParams`` omits the threshold or holds an unusable normalization.
        """
        if Family(m.family) is not self.family:
            raise FamilyError("FAMILY_MISMATCH", f"manifest family is {m.family!r}")
        validate_preprocess(m)
        if len(m.outputs) != 1:
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_COUNT",
                f"anomaly reads one output, manifest declares {len(m.outputs)}",
            )
        params = dict(m.family_params or {})
        source = choice(params, "source", ANOMALY_SOURCES, "scalar", "UNSUPPORTED_ANOMALY_SOURCE")
        choice(params, "activation", ("none", "sigmoid"), "none", "UNSUPPORTED_ACTIVATION")
        choice(
            params,
            "direction",
            ("higherIsAnomalous", "lowerIsAnomalous"),
            "higherIsAnomalous",
            "UNSUPPORTED_ANOMALY_DIRECTION",
        )
        _threshold(params)
        _normalization(params)

        spec = m.outputs[0]
        shape = tuple(spec.shape)
        rank = len(shape)
        if source == "scalar":
            if rank not in (1, 2):
                raise FamilyError(
                    "UNSUPPORTED_OUTPUT_RANK",
                    f"output {spec.name!r} has rank {rank}; a scalar score is rank 1 or 2",
                )
            trailing = shape[-1]
            if isinstance(trailing, int) and trailing != 1:
                raise FamilyError(
                    "UNSUPPORTED_OUTPUT_SHAPE",
                    f"output {spec.name!r} ends in {trailing} values; a scalar score is one value",
                )
        elif rank not in (2, 3, 4):
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_RANK",
                f"output {spec.name!r} has rank {rank}; an anomaly map is rank 2, 3, or 4",
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
        """Turn one anomaly head into a score, a threshold, and a verdict.

        Args:
            outputs: The session outputs, keyed by tensor name.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image.

        Returns:
            The normalized output whose ``anomaly`` holds ``score``, ``threshold``, ``anomalous``,
            and, for a map head, a ``summary``.

        Raises:
            FamilyError: When the tensor is absent or its shape contradicts the manifest.
        """
        params = dict(m.family_params or {})
        source = choice(params, "source", ANOMALY_SOURCES, "scalar", "UNSUPPORTED_ANOMALY_SOURCE")
        activation = choice(params, "activation", ("none", "sigmoid"), "none", "UNSUPPORTED_ACTIVATION")
        direction = choice(
            params,
            "direction",
            ("higherIsAnomalous", "lowerIsAnomalous"),
            "higherIsAnomalous",
            "UNSUPPORTED_ANOMALY_DIRECTION",
        )
        threshold = _threshold(params)
        rescale = _normalization(params)

        name = params.get("outputName")
        raw = pick_output(outputs, m, name)
        where = name or "the anomaly output"
        summary = {}
        if source == "scalar":
            flat = np.asarray(raw, dtype=np.float64).reshape(-1)
            if flat.size != 1:
                raise FamilyError(
                    "UNSUPPORTED_OUTPUT_SHAPE",
                    f"{where} returned {flat.size} values; a scalar score is one value",
                )
            score = float(self._rescale(apply_activation(flat, activation), rescale)[0])
        else:
            plane = self._rescale(apply_activation(_as_map(raw, where), activation), rescale)
            score = float(plane.max() if source == "mapMax" else plane.mean())
            summary = self._summarize(plane, threshold, direction, m, image_hw)

        anomalous = score >= threshold if direction == "higherIsAnomalous" else score <= threshold
        anomaly = {
            "score": score,
            "threshold": float(threshold),
            "anomalous": bool(anomalous),
            "direction": direction,
        }
        if summary:
            anomaly["summary"] = summary
        return NormalizedOutput(
            family=self.family,
            anomaly=anomaly,
            raw_shapes={key: tuple(np.asarray(value).shape) for key, value in outputs.items()},
        )

    @staticmethod
    def _rescale(values: np.ndarray, rescale) -> np.ndarray:
        """Apply the declared linear normalization, if any.

        Args:
            values: Activated raw values.
            rescale: A ``(low, high)`` pair, or ``None``.

        Returns:
            The values, mapped into ``[0, 1]`` and clipped when a rescale is configured.
        """
        if rescale is None:
            return np.asarray(values, dtype=np.float64)
        low, high = rescale
        return np.clip((np.asarray(values, dtype=np.float64) - low) / (high - low), 0.0, 1.0)

    @staticmethod
    def _summarize(
        plane: np.ndarray, threshold: float, direction: str, m: BundleManifest, image_hw: tuple
    ) -> dict:
        """Describe an anomaly map without publishing it.

        Args:
            plane: The normalized ``(gh, gw)`` map.
            threshold: The decision threshold, in the same normalized units.
            direction: ``"higherIsAnomalous"`` or ``"lowerIsAnomalous"``.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image.

        Returns:
            The bounded summary: extremes, the mean, how many pixels crossed the threshold, the
            fraction of the map that is, and the region they occupy in source coordinates.
        """
        mask = plane >= threshold if direction == "higherIsAnomalous" else plane <= threshold
        rows, columns = np.nonzero(mask)
        bbox = None
        if rows.size:
            mapper = SourceMapper.build(resize_plan(m), image_hw)
            grid_h, grid_w = plane.shape
            step_x, step_y = mapper.in_w / grid_w, mapper.in_h / grid_h
            bbox = mapper.normalized_region(
                float(columns.min()) * step_x,
                float(rows.min()) * step_y,
                float(columns.max() + 1) * step_x,
                float(rows.max() + 1) * step_y,
            )
        return {
            "min": float(plane.min()),
            "max": float(plane.max()),
            "mean": float(plane.mean()),
            "aboveThresholdPixels": int(rows.size),
            "fraction": float(rows.size) / float(plane.size),
            "bbox": bbox,
        }
