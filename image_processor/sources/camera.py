"""camera-adapter integration: the capture-status reconciler and the camera topics.

A camera-bound route has two ways to learn that a capture finished, and it uses both. The
``ImageCaptured`` announcement on ``app/image/captured`` is fast and best-effort: it is a hint, it
is not a queue, and a lost hint cannot lose a job (DESIGN.md 4.1). The ``sb/capture-status`` verb
is the authority: the camera's own documentation says a consumer that must not miss an outcome
polls it rather than relying on the announcement.

This module owns the polling half. ``CaptureStatusReconciler`` sweeps the camera's paged
``SUCCEEDED`` list, follows every ``nextCursor`` to the end, deduplicates by ``captureId``,
verifies each record's declared size and digest against the file under the route root, and keeps a
watermark so a restart does not re-emit what it already reconciled.

The camera's request and reply shapes come from ``camera-adapter/docs/reference/messaging-
interface.md`` and ``camera-adapter/src/commands.rs``:

* List mode takes a non-empty ``states``, an optional ``instance``, ``limit`` (1-1000, default
  100), and an opaque ``cursor``, and it replies ``{"jobs": [...], "nextCursor": null | str}``.
* Capture mode takes ``captureId`` alone and replies with one job element, or the error
  ``CAPTURE_NOT_FOUND`` once the record ages past the camera's ``resultRetentionHours`` /
  ``maxResultRecords`` retention.
* A job element is ``{captureId, instance, state, acceptedAtMs, terminalAtMs, captureGroupId,
  errorCode, errorMessage, result}``, where ``result`` is the full terminal body -- the same
  document the announcement carries and the same document the ``<image>.json`` sidecar holds.
* A failed command replies ``{"errorCode": ..., "errorMessage": ...}``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from image_processor.types import SourceIdentity

from image_processor.sources.readiness import identity_from_capture, verify_declared_image
from image_processor.sources.staging import (
    PathError,
    SourceError,
    real_root,
    relative_to_root,
    resolve_under_root,
    stat_signature,
)

logger = logging.getLogger(__name__)

#: The camera command verb this module polls.
CAPTURE_STATUS_VERB = "sb/capture-status"

#: The camera's terminal announcement channel for a successful capture.
IMAGE_CAPTURED_CHANNEL = "app/image/captured"

#: The camera job state that means the image and its catalog record are durable.
SUCCEEDED = "SUCCEEDED"

#: The camera's error code for a capture that is unknown or has aged out of retention.
CAPTURE_NOT_FOUND = "CAPTURE_NOT_FOUND"

#: The camera's default page size. Its accepted range is 1-1000.
DEFAULT_PAGE_LIMIT = 100

#: Ceiling on pages per sweep, so a cursor loop cannot spin forever.
MAX_PAGES_PER_SWEEP = 1000

#: Default sweep interval when configuration does not name one.
DEFAULT_RECONCILE_SECS = 30.0

#: Default request deadline for one page.
DEFAULT_REQUEST_TIMEOUT_SECS = 10.0


class ReconcileError(SourceError):
    """The reconciler could not be built or the camera answered with something unusable."""


def capture_status_topic(device: str, component: str, instance: Optional[str] = None) -> str:
    """Return the command topic that answers ``sb/capture-status``.

    An instance-addressed topic narrows the answer to one camera; the component-scope topic answers
    for the whole component.
    """
    scope = f"{device}/{component}" if not instance else f"{device}/{component}/{instance}"
    return f"ecv1/{scope}/cmd/{CAPTURE_STATUS_VERB}"


def image_captured_topic(device: str, component: str, instance: str) -> str:
    """Return the topic carrying the camera's ``ImageCaptured`` announcements."""
    return f"ecv1/{device}/{component}/{instance}/{IMAGE_CAPTURED_CHANNEL}"


@dataclass(frozen=True)
class CaptureRecord:
    """One capture whose file has been verified against the camera's durable record.

    Attributes:
        capture_id: The camera's durable capture primary key.
        relative_path: The image path under the route root, forward-slashed.
        identity: The verified source identity, carrying capture provenance.
        signature: The ``(size, mtime_ns)`` of the file at the moment it verified.
        terminal_at_ms: When the capture reached its terminal state, from the camera's record.
    """

    capture_id: str
    relative_path: str
    identity: SourceIdentity
    signature: tuple
    terminal_at_ms: Optional[int]


def reply_body(reply: Any) -> Optional[dict]:
    """Return the reply body of a capture-status answer, whatever shape the caller handed back.

    The injected request callable may hand back a decoded body, a core ``Message``, or an object
    exposing one. All three are accepted so the reconciler stays testable without a broker.
    """
    if isinstance(reply, dict):
        return reply
    if reply is None:
        return None
    accessor = getattr(reply, "get_body", None)
    if callable(accessor):
        body = accessor()
        if isinstance(body, dict):
            return body
    body = getattr(reply, "body", None)
    return body if isinstance(body, dict) else None


class CaptureStatusReconciler:
    """Polls ``sb/capture-status`` and turns verified ``SUCCEEDED`` records into readiness proof.

    The reconciler owns no transport. It is handed a ``request`` callable so the sweep is a pure
    function of what the camera answers, and a ``kv_get`` / ``kv_set`` pair so its watermark lives
    in the ledger beside the jobs it produces (WP3 exposes the pair).

    Args:
        route_id: The route these captures belong to.
        root: The spool root the camera writes into. Every ``relativePath`` resolves under it.
        topic: The camera's ``sb/capture-status`` command topic.
        request: ``request(topic, body, timeout_secs) -> reply``.
        kv_get: ``kv_get(key) -> str | None``, reading the persisted watermark.
        kv_set: ``kv_set(key, value)``, persisting the watermark.
        instance: The camera instance to narrow the list to, when the topic is component-scoped.
        interval_secs: Sweep period for the background thread.
        page_limit: Page size, within the camera's 1-1000 range.
        request_timeout_secs: Deadline for one page request.
        on_verified: Called with each newly verified ``CaptureRecord``.
        kv_key: Watermark key. Defaults to one key per route.
        clock: Monotonic clock, injected for tests.
    """

    def __init__(
        self,
        *,
        route_id: str,
        root: Path,
        topic: str,
        request: Callable[..., Any],
        kv_get: Callable[[str], Optional[str]],
        kv_set: Callable[[str, str], None],
        instance: Optional[str] = None,
        interval_secs: float = DEFAULT_RECONCILE_SECS,
        page_limit: int = DEFAULT_PAGE_LIMIT,
        request_timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        on_verified: Optional[Callable[[CaptureRecord], None]] = None,
        kv_key: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= int(page_limit) <= 1000:
            raise ReconcileError("PAGE_LIMIT_OUT_OF_RANGE", "the camera accepts 1-1000 per page")
        self.route_id = route_id
        self.root = real_root(Path(root))
        self.topic = topic
        self.instance = instance
        self.interval_secs = float(interval_secs)
        self.page_limit = int(page_limit)
        self.request_timeout_secs = float(request_timeout_secs)
        self.kv_key = kv_key or f"image-processor/capture-status-watermark/{route_id}"
        self.last_error: Optional[str] = None
        self.pages_read = 0
        self.sweeps = 0
        self.verified_count = 0
        self.rejected_count = 0
        self._request = request
        self._kv_get = kv_get
        self._kv_set = kv_set
        self._on_verified = on_verified
        self._clock = clock
        self._lock = threading.Lock()
        self._records: dict = {}
        self._watermark_ms = 0
        self._watermark_ids: set = set()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._load_watermark()

    # -- watermark ---------------------------------------------------------------------------

    def _load_watermark(self) -> None:
        """Restore the persisted watermark, tolerating an absent or corrupt value."""
        try:
            raw = self._kv_get(self.kv_key)
        except Exception:
            logger.warning("capture-status watermark could not be read", exc_info=True)
            return
        if not raw:
            return
        try:
            document = json.loads(raw)
            self._watermark_ms = int(document.get("terminalAtMs") or 0)
            self._watermark_ids = {
                str(value) for value in document.get("captureIds") or () if value
            }
        except (ValueError, TypeError, AttributeError):
            logger.warning("capture-status watermark is unreadable; starting from the beginning")
            self._watermark_ms = 0
            self._watermark_ids = set()

    def _save_watermark(self) -> None:
        """Persist the watermark: the newest terminal time seen, and the ids at that exact time."""
        payload = json.dumps(
            {
                "terminalAtMs": self._watermark_ms,
                "captureIds": sorted(self._watermark_ids),
            }
        )
        try:
            self._kv_set(self.kv_key, payload)
        except Exception:
            logger.warning("capture-status watermark could not be persisted", exc_info=True)

    def _below_watermark(self, capture_id: str, terminal_at_ms: Optional[int]) -> bool:
        """Report whether a record was already reconciled in an earlier process.

        The camera's list mode has no since-filter, so every sweep walks the whole retained
        ``SUCCEEDED`` window. The watermark is what keeps that walk from re-emitting: anything
        older than the newest terminal time already seen is done, and ties at exactly that time are
        settled by the ids recorded with it.
        """
        if terminal_at_ms is None:
            return False
        if terminal_at_ms < self._watermark_ms:
            return True
        return terminal_at_ms == self._watermark_ms and capture_id in self._watermark_ids

    def _advance_watermark(self, capture_id: str, terminal_at_ms: Optional[int]) -> None:
        """Move the watermark forward to include one reconciled record."""
        if terminal_at_ms is None:
            return
        if terminal_at_ms > self._watermark_ms:
            self._watermark_ms = terminal_at_ms
            self._watermark_ids = {capture_id}
        elif terminal_at_ms == self._watermark_ms:
            self._watermark_ids.add(capture_id)

    # -- record access -----------------------------------------------------------------------

    def lookup(self, relative_path: str) -> Optional[CaptureRecord]:
        """Return the verified record for a relative path, or None.

        This is what the ``cameraStatus`` readiness mode calls.
        """
        with self._lock:
            return self._records.get(relative_path)

    def records(self) -> dict:
        """Return a copy of the verified records, keyed by relative path."""
        with self._lock:
            return dict(self._records)

    # -- polling -----------------------------------------------------------------------------

    def _ask(self, body: dict) -> Optional[dict]:
        """Send one capture-status request and return its reply body, or None on failure."""
        try:
            reply = self._request(self.topic, body, self.request_timeout_secs)
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.warning("capture-status request failed: %s", exc)
            return None
        answer = reply_body(reply)
        if answer is None:
            self.last_error = "UNREADABLE_REPLY"
            logger.warning("capture-status replied with an unreadable body")
        return answer

    def lookup_capture(self, capture_id: str) -> Optional[dict]:
        """Read one capture by id, or return None when the camera no longer holds it.

        A capture that has aged past the camera's retention answers ``CAPTURE_NOT_FOUND``. That is
        an answer, not a fault: the record is gone, so this route learns about the file from the
        spool walk and its sidecar instead.
        """
        answer = self._ask({"captureId": capture_id})
        if answer is None:
            return None
        error = answer.get("errorCode")
        if error == CAPTURE_NOT_FOUND:
            logger.debug("capture %s is no longer retained by the camera", capture_id)
            return None
        if error:
            self.last_error = str(error)
            logger.warning("capture-status refused a lookup: %s", error)
            return None
        return answer

    def poll_once(self) -> int:
        """Run one full sweep of the camera's ``SUCCEEDED`` list.

        Every ``nextCursor`` is followed to the end, because a page boundary is not a natural stop:
        stopping early would leave captures unreconciled until a later sweep happened to page past
        them.

        Returns:
            The number of records newly verified by this sweep.
        """
        self.sweeps += 1
        cursor: Optional[str] = None
        seen_this_sweep: set = set()
        verified = 0
        for _ in range(MAX_PAGES_PER_SWEEP):
            body: dict = {"states": [SUCCEEDED], "limit": self.page_limit}
            if self.instance:
                body["instance"] = self.instance
            if cursor:
                body["cursor"] = cursor
            answer = self._ask(body)
            if answer is None:
                break
            error = answer.get("errorCode")
            if error:
                self.last_error = str(error)
                logger.warning("capture-status refused a page: %s", error)
                break
            self.pages_read += 1
            self.last_error = None
            jobs = answer.get("jobs")
            if isinstance(jobs, list):
                for job in jobs:
                    if self._absorb(job, seen_this_sweep):
                        verified += 1
            next_cursor = answer.get("nextCursor")
            if not next_cursor or not isinstance(next_cursor, str):
                break
            cursor = next_cursor
        else:
            logger.warning("capture-status paging hit the per-sweep page ceiling")
        if verified:
            self._save_watermark()
        return verified

    def _absorb(self, job: Any, seen_this_sweep: set) -> bool:
        """Verify one capture-status job element and record it.

        Returns:
            True when this call added a newly verified record.
        """
        if not isinstance(job, dict):
            return False
        capture_id = job.get("captureId")
        if not isinstance(capture_id, str) or not capture_id:
            return False
        if capture_id in seen_this_sweep:
            return False
        seen_this_sweep.add(capture_id)
        if job.get("state") != SUCCEEDED:
            return False
        terminal_at_ms = job.get("terminalAtMs")
        terminal_at_ms = terminal_at_ms if isinstance(terminal_at_ms, int) else None
        if self._below_watermark(capture_id, terminal_at_ms):
            return False
        result = job.get("result")
        if not isinstance(result, dict):
            self.rejected_count += 1
            logger.debug("capture %s is SUCCEEDED with no terminal body", capture_id)
            return False
        image = result.get("image")
        if not isinstance(image, dict):
            self.rejected_count += 1
            return False
        try:
            path = resolve_under_root(self.root, image.get("relativePath"))
        except PathError as exc:
            self.rejected_count += 1
            logger.warning(
                "capture %s names a path outside the route root: %s", capture_id, exc.code
            )
            return False
        size, digest, reason = verify_declared_image(path, image)
        if reason is not None:
            self.rejected_count += 1
            logger.debug("capture %s did not verify against the file: %s", capture_id, reason)
            return False
        relative_path = relative_to_root(self.root, path)
        record = CaptureRecord(
            capture_id=capture_id,
            relative_path=relative_path,
            identity=identity_from_capture(
                self.route_id, relative_path, result, int(size), str(digest)
            ),
            signature=stat_signature(path),
            terminal_at_ms=terminal_at_ms,
        )
        with self._lock:
            existing = self._records.get(relative_path)
            if existing is not None and existing.capture_id == capture_id:
                return False
            self._records[relative_path] = record
        self._advance_watermark(capture_id, terminal_at_ms)
        self.verified_count += 1
        if self._on_verified is not None:
            try:
                self._on_verified(record)
            except Exception:
                logger.warning("a capture-status subscriber raised", exc_info=True)
        return True

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sweep thread."""
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._loop, name=f"capture-status-{self.route_id}", daemon=True
        )
        self._worker.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the background sweep thread."""
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout_s)

    def _loop(self) -> None:
        """Sweep on the configured interval until stopped."""
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.warning("a capture-status sweep raised", exc_info=True)
            self._stop.wait(self.interval_secs)
