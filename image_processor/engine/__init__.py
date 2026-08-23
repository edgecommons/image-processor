"""Inference engine: image decode, task families, and decision rules (LLD §6).

The engine is the numeric half of the component. It turns bytes into a tensor
(:mod:`~image_processor.engine.decode`), a tensor into a normalized task output
(:mod:`~image_processor.engine.families`), and a normalized task output into a
:class:`~image_processor.types.Decision` (:mod:`~image_processor.engine.decision`).

Nothing in this package imports ``onnxruntime`` or touches a GPU. The families describe what to
feed a session and how to read what comes back; running the session belongs to the executor cell
(WP4b). Keeping the two apart is what lets the whole family surface be tested on plain numpy
arrays, and what lets one implementation serve a CPU parity run and a CUDA run without branching.
"""

from image_processor.engine.decode import (
    DecodeError,
    DecodeLimits,
    decode_image,
    is_high_bit_depth,
)
from image_processor.engine.decision import decide, resolve_path
from image_processor.engine.families import FAMILIES, FamilyError, TaskFamily, family_for

__all__ = [
    "FAMILIES",
    "DecodeError",
    "DecodeLimits",
    "FamilyError",
    "TaskFamily",
    "decide",
    "decode_image",
    "family_for",
    "is_high_bit_depth",
    "resolve_path",
]
