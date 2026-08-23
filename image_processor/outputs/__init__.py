"""Outputs: what the component says about an image once it has an answer (LLD 8).

Four surfaces leave this package, and they are deliberately unequal.

:mod:`~image_processor.outputs.result` builds the authoritative ``app/inference/result`` body and
validates it against ``schemas/inference-result.schema.json``.
:mod:`~image_processor.outputs.sidecar` installs the evidence document beside the image with the
ordering DESIGN.md 7 requires. :mod:`~image_processor.outputs.publisher` drains the ledger's
outbox with positive transport confirmation, which is the only thing that lets cleanup run.
:mod:`~image_processor.outputs.mirror` republishes a few normalized values on the ``data`` class,
best effort, and :mod:`~image_processor.outputs.events` carries bounded operator conditions on
``evt``.

Only the result message gates cleanup (D-IP-6). The mirror and the events are commentary: a
consumer enforcing a safety gate reads the result and nothing else.
"""

from image_processor.outputs.events import EVENT_TYPES, RouteEvents
from image_processor.outputs.mirror import DecisionMirror
from image_processor.outputs.publisher import OutboxPublisher, PublishError
from image_processor.outputs.result import (
    RESULT_CHANNEL,
    RESULT_MESSAGE_NAME,
    RESULT_MESSAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    ResultError,
    ResultLimits,
    body_bytes,
    build_result_body,
    fits_budget,
    validate_result_body,
)
from image_processor.outputs.sidecar import (
    SIDECAR_SUFFIX,
    InstalledSidecar,
    SidecarError,
    sidecar_document,
    sidecar_path_for,
    write_sidecar,
)

__all__ = [
    "EVENT_TYPES",
    "RESULT_CHANNEL",
    "RESULT_MESSAGE_NAME",
    "RESULT_MESSAGE_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SIDECAR_SUFFIX",
    "DecisionMirror",
    "InstalledSidecar",
    "OutboxPublisher",
    "PublishError",
    "ResultError",
    "ResultLimits",
    "RouteEvents",
    "SidecarError",
    "body_bytes",
    "build_result_body",
    "fits_budget",
    "sidecar_document",
    "sidecar_path_for",
    "validate_result_body",
    "write_sidecar",
]
