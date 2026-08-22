"""The writer thread's failure handling, the recovery-edge gate, and the wall clock."""

import time

import pytest
from ledger_support import admitted

from image_processor.ledger import IllegalTransition, Ledger
from image_processor.ledger.ledger import _now_ms
from image_processor.types import JobState


class FailingCommitConnection:
    """A write connection whose ``COMMIT`` fails while it is still inside a transaction."""

    def __init__(self, real):
        self._real = real
        self.rolled_back = False

    def execute(self, sql, *args):
        if sql == "COMMIT":
            raise RuntimeError("commit failed")
        if sql == "ROLLBACK":
            self.rolled_back = True
        return self._real.execute(sql, *args)

    @property
    def in_transaction(self):
        return self._real.in_transaction

    def close(self):
        self._real.close()


def test_now_ms_tracks_the_wall_clock():
    before = int(time.time() * 1000)
    value = _now_ms()
    assert before <= value <= int(time.time() * 1000) + 1000


def test_a_failing_commit_rolls_back_and_reports(db_path, clock):
    store = Ledger(db_path, synchronous="NORMAL", clock=clock)
    proxy = FailingCommitConnection(store._write_conn)
    store._write_conn = proxy
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            store.kv_set("a", "b")
        assert proxy.rolled_back is True
        assert proxy.in_transaction is False
    finally:
        store.close()
    reopened = Ledger(db_path, synchronous="NORMAL", clock=clock)
    try:
        assert reopened.kv_get("a") is None
    finally:
        reopened.close()


def test_force_state_refuses_a_non_recovery_edge(ledger):
    admitted(ledger, JobState.READY)
    with pytest.raises(IllegalTransition):
        ledger._write(
            lambda conn: ledger._force_state(
                conn, "job-1", JobState.READY, JobState.COMPLETED
            )
        )
    assert ledger.get("job-1").state is JobState.READY


def test_force_state_refuses_an_unknown_field(ledger):
    admitted(ledger, JobState.INFERENCING)
    with pytest.raises(ValueError):
        ledger._write(
            lambda conn: ledger._force_state(
                conn, "job-1", JobState.INFERENCING, JobState.READY, route_id="other"
            )
        )
    assert ledger.get("job-1").state is JobState.INFERENCING


def test_force_state_sets_declared_fields(ledger):
    admitted(ledger, JobState.INFERENCING)
    job = ledger._write(
        lambda conn: ledger._force_state(
            conn, "job-1", JobState.INFERENCING, JobState.READY, attempts=5
        )
    )
    assert (job.state, job.attempts) == (JobState.READY, 5)
