"""Subscription trigger: images that arrive on the bus rather than in a directory.

A trigger route subscribes to a configured topic filter and accepts two body forms (DESIGN.md
4.2):

* An **inline image**: an opaque binary body, or a structured body whose ``image`` field is bytes.
  The core envelope caps a binary body at 64 KiB (D-IP-5), so an inline image is small by
  construction; anything larger arrives by reference. The bytes are hashed, written into
  processor-owned staging under a digest-derived name, and from that point the job is an ordinary
  file job.
* A **file reference**: ``{"relativePath": ..., "sha256": ..., "bytes": n}``. The path resolves
  under the route's ``fileRoot`` with containment enforced, and the declared size and digest are
  verified against the file before the job is admitted. The verified file is then copied into
  staging, because the producer still owns the original and may delete or rewrite it while the job
  waits for a GPU.

A trigger message carries request correlation. When the envelope names a ``reply_to``, the
identity handed to the application carries it along with the correlation id, so the result
publisher can answer the requester in addition to publishing the normal outputs.

Anything that is neither form is refused through ``events.invalid`` with a stable reason, never
guessed at.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from image_processor.types import SourceIdentity, SourceKind

from image_processor.sources.staging import (
    MAX_INLINE_BYTES,
    PathError,
    SourceError,
    classify_path,
    config_field,
    normalize_relative,
    real_root,
    relative_to_root,
    resolve_under_root,
    sha256_bytes,
    sha256_file,
    stage_bytes,
    stage_copy,
    stat_signature,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from image_processor.sources import SourceEvents

logger = logging.getLogger(__name__)

#: The core's marker key for a bounded binary value carried inside a structured body.
BINARY_BODY_KEY = "_edgecommonsBinary"

#: Content sniffing for the staged file name. The executor decodes by content, never by suffix;
#: the suffix exists so an operator looking at a staging directory can tell what is in it.
_MAGIC = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
    (b"BM", ".bmp"),
    (b"GIF8", ".gif"),
)

#: Suffix for bytes that match no known image magic.
DEFAULT_SUFFIX = ".img"


class TriggerError(SourceError):
    """A trigger route cannot be built, or a message cannot be admitted."""


def suffix_for(data: bytes) -> str:
    """Return the staged file suffix for a byte string, by content."""
    for magic, suffix in _MAGIC:
        if data.startswith(magic):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return DEFAULT_SUFFIX


def message_body(message: Any) -> Any:
    """Return a message body, accepting a core ``Message``, a mapping, or raw bytes."""
    accessor = getattr(message, "get_body", None)
    if callable(accessor):
        return accessor()
    if isinstance(message, Mapping) and "body" in message:
        return message["body"]
    return message


def request_correlation(message: Any) -> tuple:
    """Return ``(correlation_id, reply_to)`` from a message envelope.

    Returns ``(None, None)`` for a message with no header, which is what a plain-body test double
    and a fire-and-forget trigger both look like.
    """
    header = message
    accessor = getattr(message, "get_header", None)
    if callable(accessor):
        header = accessor()
    elif isinstance(message, Mapping):
        header = message.get("header", message)
    if header is None:
        return None, None
    correlation_id = config_field(header, "correlation_id", "correlationId", default=None)
    reply_to = config_field(header, "reply_to", "replyTo", default=None)
    return (
        correlation_id if isinstance(correlation_id, str) else None,
        reply_to if isinstance(reply_to, str) else None,
    )


def decode_binary_marker(value: Any) -> Optional[bytes]:
    """Decode the core's bounded binary marker, when a structured field carries one.

    Returns:
        The decoded bytes, or None when the value is not a binary marker.

    Raises:
        TriggerError: The marker is malformed or declares more than the envelope cap.
    """
    if not isinstance(value, Mapping) or BINARY_BODY_KEY not in value:
        return None
    descriptor = value.get(BINARY_BODY_KEY)
    if not isinstance(descriptor, Mapping):
        raise TriggerError("MALFORMED_BODY", "a binary marker is an object")
    declared = descriptor.get("length")
    if isinstance(declared, int) and declared > MAX_INLINE_BYTES:
        raise TriggerError("INLINE_TOO_LARGE", "a binary body is bounded by the envelope cap")
    encoded = descriptor.get("data")
    if not isinstance(encoded, str):
        raise TriggerError("MALFORMED_BODY", "a binary marker carries base64 data")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise TriggerError("MALFORMED_BODY", "a binary marker carries base64 data") from exc


class TriggerSource:
    """Admits images that arrive as messages on one route's subscribed topics.

    Args:
        route: The route configuration. Its ``source`` carries ``subscribe``, ``fileRoot``,
            ``inlineStaging``, and ``maxInlineBytes``.
        events: The application's sink for discovered inputs and invalid ones.
        staging: The processor-owned staging directory. Defaults to the route's ``inlineStaging``.
        file_root: The root a file reference resolves under. Defaults to the route's ``fileRoot``.
        max_inline_bytes: Inline cap. Clamped to the envelope's 64 KiB, which no configuration
            raises: raising it is a wire-contract change (D-IP-5).

    Raises:
        TriggerError: The route names no id, or no staging directory for inline images.
    """

    def __init__(
        self,
        route: Any,
        events: "SourceEvents",
        staging: Optional[Path] = None,
        *,
        file_root: Optional[Path] = None,
        max_inline_bytes: Optional[int] = None,
    ) -> None:
        try:
            self.route_id = str(config_field(route, "id", "route_id"))
            source = config_field(route, "source")
        except SourceError as exc:
            raise TriggerError("TRIGGER_ROUTE_INCOMPLETE", str(exc)) from exc
        if not self.route_id:
            raise TriggerError("TRIGGER_ROUTE_INCOMPLETE", "a route needs an id")
        staging_value = staging or config_field(source, "inlineStaging", default=None)
        if staging_value is None:
            raise TriggerError(
                "INLINE_STAGING_REQUIRED", "a trigger route stages inline images on disk"
            )
        self.staging = Path(staging_value)
        root_value = file_root or config_field(source, "fileRoot", default=None)
        self.file_root = real_root(Path(root_value)) if root_value is not None else None
        subscribe = config_field(source, "subscribe", default=())
        if isinstance(subscribe, str):
            subscribe = (subscribe,)
        self.subscribe = tuple(str(topic) for topic in subscribe)
        configured = max_inline_bytes
        if configured is None:
            configured = config_field(source, "maxInlineBytes", default=MAX_INLINE_BYTES)
        self.max_inline_bytes = min(int(configured), MAX_INLINE_BYTES)
        if int(configured) > MAX_INLINE_BYTES:
            logger.warning(
                "route %s asks for %s inline bytes; the envelope caps it at %s",
                self.route_id,
                configured,
                MAX_INLINE_BYTES,
            )
        self._events = events
        self.accepted = 0
        self.rejected = 0

    def _invalid(self, relative_path: str, reason: str) -> None:
        """Refuse one message and tell the application why."""
        self.rejected += 1
        logger.warning("route %s refuses a trigger message: %s", self.route_id, reason)
        self._events.invalid(self.route_id, relative_path, reason)

    def _inline_bytes(self, message: Any, body: Any) -> Optional[bytes]:
        """Return the inline image bytes a message carries, or None when it carries none.

        Raises:
            TriggerError: The message carries a binary body larger than the envelope cap.
        """
        accessor = getattr(message, "get_binary_body", None)
        if callable(accessor):
            try:
                raw = accessor()
            except ValueError as exc:
                raise TriggerError("INLINE_TOO_LARGE", str(exc)) from exc
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        if isinstance(body, Mapping):
            image = body.get("image")
            if isinstance(image, (bytes, bytearray)):
                return bytes(image)
            decoded = decode_binary_marker(image)
            if decoded is not None:
                return decoded
        return None

    def on_message(self, message: Any) -> None:
        """Admit one trigger message, or refuse it with a reason.

        A body naming ``relativePath`` is a file reference; anything else that yields bytes is an
        inline image. The two are never guessed between: a body that is neither is refused.
        """
        correlation_id, reply_to = request_correlation(message)
        body = message_body(message)
        if isinstance(body, Mapping) and "relativePath" in body:
            self._admit_reference(body, correlation_id, reply_to)
            return
        try:
            data = self._inline_bytes(message, body)
        except TriggerError as exc:
            self._invalid("", exc.code)
            return
        if data is None:
            self._invalid("", "MALFORMED_BODY")
            return
        self._admit_inline(data, correlation_id, reply_to)

    def _admit_inline(
        self, data: bytes, correlation_id: Optional[str], reply_to: Optional[str]
    ) -> None:
        """Stage inline bytes and announce them."""
        if not data:
            self._invalid("", "EMPTY_BODY")
            return
        if len(data) > self.max_inline_bytes:
            self._invalid("", "INLINE_TOO_LARGE")
            return
        digest = sha256_bytes(data)
        try:
            staged = stage_bytes(data, self.staging, digest, suffix_for(data))
        except (SourceError, OSError) as exc:
            self._invalid("", getattr(exc, "code", "STAGING_FAILED"))
            return
        identity = SourceIdentity(
            kind=SourceKind.INLINE,
            route_id=self.route_id,
            relative_path=relative_to_root(self.staging, staged),
            bytes=len(data),
            sha256=digest,
            correlation_id=correlation_id,
            reply_to=reply_to,
        )
        self.accepted += 1
        self._events.discovered(self.route_id, identity, staged)

    def _admit_reference(
        self, body: Mapping, correlation_id: Optional[str], reply_to: Optional[str]
    ) -> None:
        """Verify a referenced file and announce a staged copy of it."""
        declared_path = body.get("relativePath")
        declared_sha = body.get("sha256")
        declared_bytes = body.get("bytes")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            self._invalid(str(declared_path or ""), "MALFORMED_BODY")
            return
        if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
            self._invalid(str(declared_path or ""), "MALFORMED_BODY")
            return
        if self.file_root is None:
            self._invalid(str(declared_path or ""), "NO_FILE_ROOT")
            return
        try:
            relative_path = normalize_relative(declared_path)
            path = resolve_under_root(self.file_root, relative_path)
        except PathError as exc:
            self._invalid(str(declared_path or ""), exc.code)
            return
        reason = classify_path(path)
        if reason is not None:
            self._invalid(relative_path, reason)
            return
        signature = stat_signature(path)
        if signature is None or signature[0] != declared_bytes:
            self._invalid(relative_path, "SIZE_MISMATCH")
            return
        try:
            digest = sha256_file(path)
        except OSError:
            self._invalid(relative_path, "MISSING")
            return
        if digest != declared_sha.lower():
            self._invalid(relative_path, "DIGEST_MISMATCH")
            return
        if stat_signature(path) != signature:
            self._invalid(relative_path, "CHANGED_DURING_READ")
            return
        try:
            staged = stage_copy(path, self.staging, digest)
        except (SourceError, OSError) as exc:
            self._invalid(relative_path, getattr(exc, "code", "STAGING_FAILED"))
            return
        identity = SourceIdentity(
            kind=SourceKind.REFERENCE,
            route_id=self.route_id,
            relative_path=relative_path,
            bytes=declared_bytes,
            sha256=digest,
            correlation_id=correlation_id,
            reply_to=reply_to,
        )
        self.accepted += 1
        self._events.discovered(self.route_id, identity, staged)
