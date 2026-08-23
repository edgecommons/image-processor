"""The confirmed-publish outbox drain (DESIGN.md 7, 12.1, LLD 8).

A committed result is durable before it is published, and it is published from the ledger rather
than from memory. This class is the only thing that turns a ``PUBLISH_PENDING`` row into a
``PUBLISHED`` one, and it does it with the exact bytes ``app().prepare()`` froze: the same
envelope UUID, the same timestamp, the same body, retry after retry. Re-encoding would produce a
logically equivalent message that no consumer could deduplicate against the first attempt.

Confirmed publication means positive transport acceptance -- an MQTT PUBACK at QoS 1, or the
completion of the Greengrass IPC publish -- so a failure here is a real failure and leaves the row
pending. Only after every gating row of a job is confirmed does the job reach ``PUBLISHED``, which
is what lets the completion manager move the input. A broker outage therefore delays cleanup and
never loses it.

A pass stops at the first failure. Every row would fail the same way during an outage, and burning
the retry budget of a hundred rows on one disconnected broker is how an outbox exhausts itself.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from image_processor.types import JobState

logger = logging.getLogger(__name__)

#: How many rows one pass takes.
DEFAULT_BATCH = 32

#: How long the drain thread waits when a pass had nothing to do.
DEFAULT_POLL_INTERVAL_S = 0.5

#: The bound on a stored publication error.
MAX_ERROR_CHARS = 512


class PublishError(Exception):
    """The transport refused a publication, or did not confirm it in time.

    The publisher never raises this: it is the shape a caller supplied ``publish`` callable
    raises, and what the ledger records against the row.
    """


class OutboxPublisher:
    """Drains the ledger outbox with positive transport confirmation.

    Args:
        ledger: The durable ledger.
        publish: ``publish(topic, encoded_bytes, timeout_secs)``. It returns on confirmation and
            raises on anything else.
        timeout_secs: The confirmation deadline for one attempt
            (``publish.confirmationTimeoutSecs``).
        max_attempts: How many attempts one row gets before its job is marked
            ``PUBLISH_EXHAUSTED`` (``publish.maxAttempts``).
        batch: How many rows one pass takes.
        poll_interval_s: How long the thread waits after a pass that published nothing.
        on_published: Called with the job once every gating row of it is confirmed. This is what
            releases cleanup.
        on_exhausted: Called with ``(inference_id, error)`` when a job spends its publish budget.
        on_error: Called with ``(inference_id, error)`` for one failed attempt.
    """

    def __init__(
        self,
        ledger: Any,
        publish: Callable[..., Any],
        *,
        timeout_secs: float = 10.0,
        max_attempts: int = 100,
        batch: int = DEFAULT_BATCH,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        on_published: Optional[Callable[[Any], None]] = None,
        on_exhausted: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Build the publisher over its collaborators."""
        self._ledger = ledger
        self._publish = publish
        self.timeout_secs = float(timeout_secs)
        self.max_attempts = max(1, int(max_attempts))
        self.batch = max(1, int(batch))
        self.poll_interval_s = float(poll_interval_s)
        self._on_published = on_published
        self._on_exhausted = on_exhausted
        self._on_error = on_error
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.counters = {
            "attempted": 0,
            "published": 0,
            "failed": 0,
            "exhausted": 0,
            "confirmationFailures": 0,
        }
        self.last_error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> "OutboxPublisher":
        """Start the drain thread.

        Returns:
            This publisher.
        """
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="image-processor-outbox", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the drain thread and wait for it.

        Args:
            timeout_s: How long to wait for the thread.
        """
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)

    def wake(self) -> None:
        """Ask for a pass now. Repeated calls coalesce into one."""
        self._wake.set()

    def _loop(self) -> None:
        """Drain, then wait for work or for the interval."""
        while not self._stop.is_set():
            try:
                published = self.drain_once()
            except Exception:  # noqa: BLE001 - the thread outlives one bad pass
                logger.warning("the outbox drain failed a pass", exc_info=True)
                published = 0
            if published:
                continue
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()

    # -- the drain -----------------------------------------------------------------------

    def pending(self) -> int:
        """Return how many rows are waiting for confirmation."""
        return len(self._ledger.pending_outbox(self.batch))

    def drain_once(self) -> int:
        """Publish one batch of eligible rows.

        Returns:
            How many rows were confirmed in this pass.
        """
        rows = self._ledger.pending_outbox(self.batch)
        confirmed = 0
        for row in rows:
            if self._stop.is_set():
                break
            if not self._attempt(row):
                break
            confirmed += 1
        return confirmed

    def _attempt(self, row: Any) -> bool:
        """Publish one row and record what happened.

        Args:
            row: The :class:`~image_processor.ledger.ledger.OutboxRow` to publish.

        Returns:
            ``True`` when the row was confirmed, ``False`` when the pass should stop.
        """
        self.counters["attempted"] += 1
        try:
            self._publish(row.topic, row.encoded_bytes, self.timeout_secs)
        except Exception as exc:  # noqa: BLE001 - any transport failure leaves the row pending
            self._record_failure(row, f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS])
            return False
        self._ledger.mark_published(row.id)
        self.counters["published"] += 1
        job = self._ledger.get(row.inference_id)
        if job is not None and job.state is JobState.PUBLISHED and self._on_published is not None:
            try:
                self._on_published(job)
            except Exception:  # noqa: BLE001 - a failed completion is not a failed publish
                logger.exception("the publication callback failed for %s", row.inference_id)
        return True

    def _record_failure(self, row: Any, error: str) -> None:
        """Record one failed attempt, exhausting the job when its budget is spent."""
        self.counters["failed"] += 1
        self.last_error = error
        logger.warning("publishing %s on %s failed: %s", row.inference_id, row.topic, error)
        self._ledger.mark_publish_attempt(row.id, error)
        if self._on_error is not None:
            self._notify(self._on_error, row.inference_id, error)
        if row.attempts + 1 < self.max_attempts:
            return
        try:
            self._ledger.exhaust_publish(row.inference_id)
        except Exception:  # noqa: BLE001 - the job may have moved under us
            logger.warning(
                "could not mark %s publish-exhausted", row.inference_id, exc_info=True
            )
            return
        self.counters["exhausted"] += 1
        logger.error(
            "publication of %s gave up after %d attempts: %s",
            row.inference_id,
            row.attempts + 1,
            error,
        )
        if self._on_exhausted is not None:
            self._notify(self._on_exhausted, row.inference_id, error)

    @staticmethod
    def _notify(callback: Callable[..., Any], inference_id: str, error: str) -> None:
        """Call one publisher callback, keeping the drain alive through a bad one."""
        try:
            callback(inference_id, error)
        except Exception:  # noqa: BLE001 - a reporting failure is not a publication failure
            logger.exception("an outbox callback failed for %s", inference_id)
