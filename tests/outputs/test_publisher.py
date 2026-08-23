"""The outbox drain, against a real ledger and a transport that can fail."""

from __future__ import annotations

import pytest

from image_processor.ledger import Ledger, OutboxRow
from image_processor.outputs.publisher import OutboxPublisher
from image_processor.types import JobState
from tests.outputs.conftest import make_job


class FakeTransport:
    """A transport that publishes, times out, or fails, on command."""

    def __init__(self) -> None:
        self.published: list = []
        self.failures = 0
        self.error: Exception = TimeoutError("no PUBACK within the confirmation timeout")

    def __call__(self, topic: str, encoded: bytes, timeout_secs: float) -> None:
        """Publish, or raise while the test says failures remain."""
        if self.failures:
            self.failures -= 1
            raise self.error
        self.published.append((topic, encoded, timeout_secs))


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    """A real ledger on disk."""
    with Ledger(tmp_path / "state.db", synchronous="OFF") as opened:
        yield opened


def _commit(ledger: Ledger, inference_id: str = "01k5inference0001", gating: bool = True) -> None:
    """Admit, advance, and commit one job with one outbox row."""
    job = make_job(inference_id=inference_id, state=JobState.READY)
    ledger.admit(job, 1024)
    ledger.transition(inference_id, JobState.READY, JobState.CLAIMED)
    ledger.transition(inference_id, JobState.CLAIMED, JobState.WAITING_MODEL)
    ledger.transition(inference_id, JobState.WAITING_MODEL, JobState.INFERENCING)
    ledger.commit_result(
        inference_id,
        b'{"status":"SUCCEEDED"}',
        ("/spool/cap.inference.json", "ab" * 32),
        [
            OutboxRow(
                id=None,
                inference_id=inference_id,
                topic="ecv1/dev/image-processor/clearance-cam-01/app/inference/result",
                encoded_bytes=b"\x01\x02frozen-envelope",
                gating=gating,
            )
        ],
    )


def test_a_confirmed_publication_advances_the_job_and_releases_cleanup(ledger):
    _commit(ledger)
    released: list = []
    transport = FakeTransport()
    publisher = OutboxPublisher(ledger, transport, timeout_secs=2.5, on_published=released.append)

    assert publisher.drain_once() == 1

    assert transport.published == [
        ("ecv1/dev/image-processor/clearance-cam-01/app/inference/result", b"\x01\x02frozen-envelope", 2.5)
    ]
    assert ledger.get("01k5inference0001").state is JobState.PUBLISHED
    assert [job.inference_id for job in released] == ["01k5inference0001"]
    assert publisher.counters["published"] == 1


def test_a_retry_carries_the_same_bytes(ledger):
    _commit(ledger)
    transport = FakeTransport()
    transport.failures = 1
    publisher = OutboxPublisher(ledger, transport)

    assert publisher.drain_once() == 0
    assert ledger.get("01k5inference0001").state is JobState.PUBLISH_PENDING
    assert publisher.drain_once() == 1
    assert transport.published[0][1] == b"\x01\x02frozen-envelope"


def test_a_pass_stops_at_the_first_failure_so_one_outage_costs_one_attempt(ledger):
    _commit(ledger, "01k5inference0001")
    _commit(ledger, "01k5inference0002")
    transport = FakeTransport()
    transport.failures = 5
    publisher = OutboxPublisher(ledger, transport)

    publisher.drain_once()

    rows = ledger.outbox_for("01k5inference0002")
    assert rows[0].attempts == 0, "the second row was not attempted"
    assert ledger.outbox_for("01k5inference0001")[0].attempts == 1


def test_exhausting_the_budget_moves_the_job_and_reports_it(ledger):
    _commit(ledger)
    transport = FakeTransport()
    transport.failures = 10
    exhausted: list = []
    publisher = OutboxPublisher(
        ledger, transport, max_attempts=2, on_exhausted=lambda ident, error: exhausted.append(ident)
    )

    publisher.drain_once()
    assert ledger.get("01k5inference0001").state is JobState.PUBLISH_PENDING
    publisher.drain_once()

    assert ledger.get("01k5inference0001").state is JobState.PUBLISH_EXHAUSTED
    assert exhausted == ["01k5inference0001"]
    assert publisher.counters["exhausted"] == 1
    assert publisher.drain_once() == 0, "an exhausted row is no longer eligible"


def test_an_operator_retry_returns_the_job_to_the_outbox(ledger):
    _commit(ledger)
    transport = FakeTransport()
    transport.failures = 10
    publisher = OutboxPublisher(ledger, transport, max_attempts=1)
    publisher.drain_once()
    assert ledger.get("01k5inference0001").state is JobState.PUBLISH_EXHAUSTED

    ledger.retry_publication("01k5inference0001")
    transport.failures = 0

    assert publisher.drain_once() == 1
    assert ledger.get("01k5inference0001").state is JobState.PUBLISHED


def test_a_failing_publication_callback_never_fails_the_publication(ledger):
    _commit(ledger)

    def _boom(job):
        raise RuntimeError("the completion blew up")

    publisher = OutboxPublisher(ledger, FakeTransport(), on_published=_boom)

    assert publisher.drain_once() == 1
    assert ledger.get("01k5inference0001").state is JobState.PUBLISHED


def test_the_drain_thread_publishes_what_is_committed_while_it_runs(ledger):
    import time

    transport = FakeTransport()
    publisher = OutboxPublisher(ledger, transport, poll_interval_s=0.01).start()
    try:
        _commit(ledger)
        publisher.wake()
        deadline = time.monotonic() + 5
        while not transport.published and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        publisher.stop(timeout_s=5)

    assert len(transport.published) == 1
    assert ledger.get("01k5inference0001").state is JobState.PUBLISHED


def test_a_pass_that_raises_leaves_the_drain_thread_running(ledger):
    import time

    class _Flaky:
        """A ledger whose first read fails and whose later reads work."""

        def __init__(self) -> None:
            self.calls = 0

        def pending_outbox(self, limit):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("the ledger is busy")
            return []

    flaky = _Flaky()
    publisher = OutboxPublisher(flaky, FakeTransport(), poll_interval_s=0.01).start()
    try:
        deadline = time.monotonic() + 5
        while flaky.calls < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        publisher.stop(timeout_s=5)

    assert flaky.calls >= 3, "the thread kept draining after a failed pass"


def test_pending_reports_the_backlog(ledger):
    _commit(ledger)
    publisher = OutboxPublisher(ledger, FakeTransport())

    assert publisher.pending() == 1
    publisher.drain_once()
    assert publisher.pending() == 0
