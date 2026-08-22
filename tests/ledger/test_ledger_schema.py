"""The durable schema and the DESIGN.md §7 transition table."""

import sqlite3

import pytest
from ledger_support import MODEL, build_job

from image_processor.ledger import SCHEMA_VERSION, TRANSITIONS, Ledger, is_legal
from image_processor.ledger import schema as ledger_schema
from image_processor.types import CompletionAction, JobState, SourceKind

#: Every edge of the DESIGN.md §7 state diagram, transcribed independently of the module.
EXPECTED_EDGES = {
    ("DISCOVERED", "READY"),
    ("DISCOVERED", "INPUT_INVALID"),
    ("INPUT_INVALID", "QUARANTINED"),
    ("READY", "CLAIMED"),
    ("CLAIMED", "WAITING_MODEL"),
    ("WAITING_MODEL", "INFERENCING"),
    ("WAITING_MODEL", "BLOCKED_CONFIGURATION"),
    ("WAITING_MODEL", "RETRY_WAIT"),
    ("INFERENCING", "RESULT_COMMITTED"),
    ("INFERENCING", "RETRY_WAIT"),
    ("INFERENCING", "PROCESSING_EXHAUSTED"),
    ("PROCESSING_EXHAUSTED", "RETAINED_FAILED"),
    ("RESULT_COMMITTED", "PUBLISH_PENDING"),
    ("PUBLISH_PENDING", "PUBLISHED"),
    ("PUBLISH_PENDING", "PUBLISH_EXHAUSTED"),
    ("PUBLISH_EXHAUSTED", "PUBLISH_PENDING"),
    ("PUBLISHED", "CLEANUP_PENDING"),
    ("CLEANUP_PENDING", "COMPLETED"),
    ("CLEANUP_PENDING", "CLEANUP_FAILED"),
    ("CLEANUP_FAILED", "CLEANUP_PENDING"),
    ("RETRY_WAIT", "READY"),
}


def test_transition_table_is_the_design_diagram():
    assert {(a.value, b.value) for a, b in TRANSITIONS} == EXPECTED_EDGES


def test_is_legal_agrees_with_the_table():
    for current in JobState:
        for new in JobState:
            assert is_legal(current, new) == ((current, new) in TRANSITIONS)


def test_schema_is_stamped_and_reopening_is_a_no_op(db_path, clock):
    first = Ledger(db_path, clock=clock)
    first.close()
    conn = sqlite3.connect(str(db_path))
    try:
        assert ledger_schema.schema_version(conn) == SCHEMA_VERSION
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()
    assert {
        "meta",
        "kv",
        "jobs",
        "outbox",
        "cleanup_intents",
        "route_generations",
        "reservations",
    } <= tables
    second = Ledger(db_path, clock=clock)
    conn = sqlite3.connect(str(db_path))
    try:
        assert ledger_schema.schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 1
    finally:
        conn.close()
        second.close()


def test_wal_and_pragmas_are_applied(ledger):
    assert ledger._read_one("PRAGMA journal_mode")[0].lower() == "wal"
    assert ledger._read_one("PRAGMA foreign_keys")[0] == 1
    assert ledger._read_one("PRAGMA busy_timeout")[0] == 5000


def test_synchronous_mode_is_validated(db_path):
    with pytest.raises(ValueError):
        Ledger(db_path, synchronous="SOMETIMES")


def test_synchronous_full_is_the_default(db_path, clock):
    store = Ledger(db_path, clock=clock)
    try:
        assert store.synchronous == "FULL"
        assert store._read_one("PRAGMA synchronous")[0] == 2
    finally:
        store.close()


def test_source_and_model_round_trip():
    job = build_job(relative_path="2026/08/22/a b.jpg")
    source = ledger_schema.decode_source(ledger_schema.encode_source(job.source))
    assert source == job.source
    assert source.kind is SourceKind.SPOOL
    assert ledger_schema.decode_model(ledger_schema.encode_model(MODEL)) == MODEL


def test_row_to_intent_decodes_members():
    intent = ledger_schema.row_to_intent(
        ("job-1", CompletionAction.ARCHIVE.value, "/spool/a.jpg", "a" * 64, "/arch/a.jpg",
         '["/spool/a.jpg.json"]')
    )
    assert intent.action is CompletionAction.ARCHIVE
    assert intent.members == ("/spool/a.jpg.json",)
