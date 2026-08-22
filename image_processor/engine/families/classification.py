"""The classification family: one score vector, ranked and labelled (DESIGN.md §8.1).

A classification head produces one value per class for the whole image. The family turns that into
``classes[]`` of ``{label, index, score}``, highest first: the activation named in the manifest
converts the raw values into comparable scores, ``topK`` bounds how many are reported, and
``maxResultItems`` bounds it again so a 20,000-class head cannot put 20,000 entries on the bus.
"""

from __future__ import annotations

import logging

import numpy as np

from image_processor.engine.families import (
    FamilyError,
    apply_activation,
    choice,
    drop_batch,
    family_labels,
    label_for,
    number,
    pick_output,
    positive_int,
    preprocess_image,
    static_dim,
    validate_preprocess,
)
from image_processor.types import BundleManifest, ClassScore, Family, NormalizedOutput

logger = logging.getLogger(__name__)


class ClassificationFamily:
    """Interprets a single score-vector head.

    Attributes:
        family: Always :attr:`~image_processor.types.Family.CLASSIFICATION`.
    """

    family = Family.CLASSIFICATION

    def validate_manifest(self, m: BundleManifest) -> None:
        """Refuse a manifest whose head is not one score vector.

        Args:
            m: The parsed bundle manifest.

        Raises:
            FamilyError: When the family does not match, the head has the wrong output count or
                rank, the class dimension disagrees with the label set, or ``familyParams`` names
                no classes.
        """
        if Family(m.family) is not self.family:
            raise FamilyError("FAMILY_MISMATCH", f"manifest family is {m.family!r}")
        validate_preprocess(m)
        if len(m.outputs) != 1:
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_COUNT",
                f"classification reads one output, manifest declares {len(m.outputs)}",
            )
        spec = m.outputs[0]
        rank = len(tuple(spec.shape))
        if rank not in (1, 2):
            raise FamilyError(
                "UNSUPPORTED_OUTPUT_RANK",
                f"output {spec.name!r} has rank {rank}; a score vector is rank 1 or 2",
            )
        labels = family_labels(m)
        declared = static_dim(spec, -1)
        if declared is not None and declared != len(labels):
            raise FamilyError(
                "CLASS_DIM_MISMATCH",
                f"output {spec.name!r} has {declared} classes but the manifest names {len(labels)}",
            )
        params = dict(m.family_params or {})
        choice(params, "activation", ("none", "softmax", "sigmoid"), "softmax", "UNSUPPORTED_ACTIVATION")
        positive_int(params, "topK", 5)
        number(params, "scoreThreshold", 0.0, 0.0, 1.0)

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
        """Rank and label one score vector.

        Args:
            outputs: The session outputs, keyed by tensor name.
            m: The parsed bundle manifest.
            image_hw: The ``(height, width)`` of the source image. Unused; a classification result
                has no geometry.

        Returns:
            The normalized output with ``classes`` populated.

        Raises:
            FamilyError: When the output tensor is absent or is not a score vector.
        """
        params = dict(m.family_params or {})
        labels = family_labels(m)
        activation = choice(
            params, "activation", ("none", "softmax", "sigmoid"), "softmax", "UNSUPPORTED_ACTIVATION"
        )
        top_k = positive_int(params, "topK", 5)
        floor = number(params, "scoreThreshold", 0.0, 0.0, 1.0)

        name = params.get("outputName")
        raw = pick_output(outputs, m, name)
        vector = drop_batch(raw, 1, name or "the classification output")
        scores = apply_activation(vector, activation, axis=-1)

        limit = min(top_k, m.max_result_items or top_k, scores.size)
        ranked = np.argsort(-scores, kind="stable")[:limit]
        classes = [
            ClassScore(label=label_for(labels, int(index)), index=int(index), score=float(scores[index]))
            for index in ranked
            if float(scores[index]) >= floor
        ]
        return NormalizedOutput(
            family=self.family,
            classes=classes,
            raw_shapes={key: tuple(np.asarray(value).shape) for key, value in outputs.items()},
        )
