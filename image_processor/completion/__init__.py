"""Completion actions under write-ahead cleanup intents (LLD §5, DESIGN.md §7).

Import :class:`Completer` from here rather than from the submodule.
"""

from image_processor.completion.actions import (
    BUNDLE_MANIFEST_SUFFIX,
    COLLISION_FAIL,
    COLLISION_SUFFIX,
    ERROR_RECORD_SUFFIX,
    CleanupError,
    Completer,
    CompletionPolicy,
    FsOps,
    RealFs,
    coerce_action,
    safe_relative,
    suffixed,
)

__all__ = [
    "BUNDLE_MANIFEST_SUFFIX",
    "COLLISION_FAIL",
    "COLLISION_SUFFIX",
    "CleanupError",
    "Completer",
    "CompletionPolicy",
    "ERROR_RECORD_SUFFIX",
    "FsOps",
    "RealFs",
    "coerce_action",
    "safe_relative",
    "suffixed",
]
