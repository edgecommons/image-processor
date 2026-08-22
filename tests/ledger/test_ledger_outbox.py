"""commit_result atomicity, outbox ordering and gating, and the publish ladder."""

import hashlib

import pytest
from ledger_support import admitted, row

from image_processor.ledger import IllegalTransition, LedgerConflict, OutboxRow
from image_processor.ledger import ledger as ledger_module
from image_processor.types import JobState

RESULT = b'{"schemaVersion":"1.0","status":"SUCCEEDED"}'
APP_TOPIC = "ecv1/dev/image-processor/cam-01/app/inference/result"
DATA_TOPIC = "ecv1/dev/image-processor/cam-01/data/line-clearance/pass"


def committed(ledger, inference_id="job-1", rows=None, sidecar=("/spool/a.inference.json", "b" * 64)):
    """Admit a job, drive it to INFERENCING, and commit a result with ``rows``."""
    job = admitted(ledger, JobState.INFERENCING, inference_id=inference_id)
    if rows is None:
        rows = [row(inference_id, APP_TOPIC, b"app-bytes", True)]
    ledger.commit_result(job.inference_id, RESULT, sidecar, rows)
    return job


def test_commit_result_stores_everything_and_makes_the_outbox_eligible(ledger):
    committed(ledger)
    job = ledger.get("job-1")
    assert job.state is JobState.PUBLISH_PENDING
    assert ledger.result_bytes("job-1") == RESULT
    stored = ledger._read_one(
        "SELECT result_sha256, sidecar_path, sidecar_sha256 FROM jobs WHERE inference_id = ?",
        ("job-1",),
    )
    assert stored == (hashlib.sha256(RESULT).hexdigest(), "/spool/a.inference.json", "b" * 64)
    pending = ledger.pending_outbox(10)
    assert [r.topic for r in pending] == [APP_TOPIC]
    assert pending[0].encoded_bytes == b"app-bytes"
    assert pending[0].gating is True
    assert pending[0].attempts == 0


def test_commit_result_without_outbox_rows_stops_at_result_committed(ledger):
    committed(ledger, rows=[])
    assert ledger.get("job-1").state is JobState.RESULT_COMMITTED
    assert ledger.pending_outbox(10) == []


def test_outbox_rows_keep_insertion_order(ledger):
    rows = [
        row("job-1", APP_TOPIC, b"app", True),
        row("job-1", DATA_TOPIC, b"pass", False),
        row("job-1", DATA_TOPIC + "/2", b"conf", False),
    ]
    committed(ledger, rows=rows)
    pending = ledger.pending_outbox(10)
    assert [r.topic for r in pending] == [APP_TOPIC, DATA_TOPIC, DATA_TOPIC + "/2"]
    assert [r.id for r in pending] == sorted(r.id for r in pending)
    assert [r.gating for r in pending] == [True, False, False]


def test_commit_result_requires_the_inferencing_state(ledger):
    job = admitted(ledger, JobState.READY, inference_id="job-2")
    with pytest.raises(LedgerConflict):
        ledger.commit_result(job.inference_id, RESULT, None, [row("job-2")])
    with pytest.raises(LedgerConflict):
        ledger.commit_result("nope", RESULT, None, [])


def test_commit_result_is_one_transaction_when_an_outbox_insert_fails(ledger, monkeypatch):
    admitted(ledger, JobState.INFERENCING, inference_id="job-1")
    calls = {"n": 0}
    original = ledger_module.Ledger._insert_outbox_row

    def flaky(self, conn, inference_id, outbox_row):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return original(self, conn, inference_id, outbox_row)

    monkeypatch.setattr(ledger_module.Ledger, "_insert_outbox_row", flaky)
    with pytest.raises(RuntimeError, match="disk full"):
        ledger.commit_result(
            "job-1",
            RESULT,
            ("/spool/a.inference.json", "b" * 64),
            [row("job-1", APP_TOPIC, b"a"), row("job-1", DATA_TOPIC, b"b", False)],
        )
    assert ledger.get("job-1").state is JobState.INFERENCING
    assert ledger.result_bytes("job-1") is None
    assert ledger.outbox_for("job-1") == []
    assert ledger._read_one(
        "SELECT sidecar_path FROM jobs WHERE inference_id = ?", ("job-1",)
    ) == (None,)


def test_commit_result_rolls_back_an_unbindable_payload(ledger):
    admitted(ledger, JobState.INFERENCING, inference_id="job-1")
    bad = OutboxRow(id=None, inference_id="job-1", topic=APP_TOPIC, encoded_bytes=object())
    with pytest.raises(Exception):
        ledger.commit_result("job-1", RESULT, None, [row("job-1"), bad])
    assert ledger.get("job-1").state is JobState.INFERENCING
    assert ledger.outbox_for("job-1") == []
    assert ledger.result_bytes("job-1") is None


def test_the_job_publishes_only_when_every_gating_row_is_confirmed(ledger):
    rows = [
        row("job-1", APP_TOPIC, b"app", True),
        row("job-1", APP_TOPIC + "/2", b"app2", True),
        row("job-1", DATA_TOPIC, b"pass", False),
    ]
    committed(ledger, rows=rows)
    pending = ledger.pending_outbox(10)
    ledger.mark_published(pending[2].id)
    assert ledger.get("job-1").state is JobState.PUBLISH_PENDING
    ledger.mark_published(pending[0].id)
    assert ledger.get("job-1").state is JobState.PUBLISH_PENDING
    ledger.mark_published(pending[1].id)
    assert ledger.get("job-1").state is JobState.PUBLISHED
    assert ledger.pending_outbox(10) == []


def test_a_published_row_is_not_offered_again(ledger):
    committed(ledger)
    first = ledger.pending_outbox(10)[0]
    ledger.mark_published(first.id)
    assert ledger.pending_outbox(10) == []
    assert ledger.outbox_for("job-1")[0].id == first.id


def test_pending_outbox_hides_rows_of_a_job_that_is_not_publish_pending(ledger):
    committed(ledger, rows=[])
    ledger._write(
        lambda conn: conn.execute(
            "INSERT INTO outbox (inference_id, topic, payload) VALUES (?, ?, ?)",
            ("job-1", APP_TOPIC, b"app"),
        )
    )
    assert ledger.pending_outbox(10) == []
    assert len(ledger.outbox_for("job-1")) == 1


def test_pending_outbox_honours_the_limit(ledger):
    committed(
        ledger,
        rows=[row("job-1", f"{APP_TOPIC}/{i}", b"x", i == 0) for i in range(5)],
    )
    assert len(ledger.pending_outbox(2)) == 2
    assert len(ledger.pending_outbox(50)) == 5


def test_mark_publish_attempt_counts_and_records(ledger):
    committed(ledger)
    outbox_id = ledger.pending_outbox(10)[0].id
    ledger.mark_publish_attempt(outbox_id, "no PUBACK")
    ledger.mark_publish_attempt(outbox_id, "broker down")
    again = ledger.pending_outbox(10)[0]
    assert again.attempts == 2
    assert again.last_error == "broker down"
    ledger.mark_published(outbox_id)
    assert ledger.outbox_for("job-1")[0].last_error is None


def test_mark_publish_attempt_and_published_reject_unknown_rows(ledger):
    committed(ledger)
    outbox_id = ledger.pending_outbox(10)[0].id
    ledger.mark_published(outbox_id)
    with pytest.raises(LedgerConflict):
        ledger.mark_publish_attempt(outbox_id, "too late")
    with pytest.raises(LedgerConflict):
        ledger.mark_publish_attempt(9999, "nope")
    with pytest.raises(LedgerConflict):
        ledger.mark_published(9999)


def test_exhaust_and_retry_publication(ledger):
    committed(ledger)
    outbox_id = ledger.pending_outbox(10)[0].id
    ledger.mark_publish_attempt(outbox_id, "broker down")
    assert ledger.exhaust_publish("job-1").state is JobState.PUBLISH_EXHAUSTED
    assert ledger.pending_outbox(10) == []
    assert ledger.retry_publication("job-1").state is JobState.PUBLISH_PENDING
    resumed = ledger.pending_outbox(10)
    assert [r.id for r in resumed] == [outbox_id]
    assert resumed[0].attempts == 0
    assert resumed[0].last_error is None


def test_exhaust_and_retry_are_state_checked(ledger):
    committed(ledger)
    with pytest.raises(LedgerConflict):
        ledger.retry_publication("job-1")
    ledger.exhaust_publish("job-1")
    with pytest.raises(LedgerConflict):
        ledger.exhaust_publish("job-1")


def test_requeue_for_reinference_drops_the_result_and_outbox(ledger):
    committed(ledger)
    job = ledger.requeue_for_reinference("job-1", "sidecar missing")
    assert job.state is JobState.READY
    assert ledger.result_bytes("job-1") is None
    assert ledger.outbox_for("job-1") == []
    assert ledger.last_error("job-1") == "sidecar missing"
    assert ledger._read_one(
        "SELECT sidecar_path, sidecar_sha256, result_sha256 FROM jobs WHERE inference_id = ?",
        ("job-1",),
    ) == (None, None, None)


def test_requeue_for_reinference_refuses_a_job_with_no_committed_result(ledger):
    admitted(ledger, JobState.READY, inference_id="job-9")
    with pytest.raises(IllegalTransition):
        ledger.requeue_for_reinference("job-9", "nothing to drop")
    with pytest.raises(LedgerConflict):
        ledger.requeue_for_reinference("nope", "missing")
