"""Durable schema and the job-state transition table (LLD §5, DESIGN.md §7).

The tables here are the component's only durable state. Every statement is
``CREATE ... IF NOT EXISTS`` and the ``meta`` table carries ``schema_version``, so opening an
existing database is a no-op and a future migration has a version to branch on.

``TRANSITIONS`` is the machine-readable form of the DESIGN.md §7 state diagram. It is the only
place an edge is declared legal, and :func:`is_legal` is the gate every job write goes through.
``RECOVERY_EDGES`` in :mod:`image_processor.ledger.recovery` is a separate, smaller table used by
restart-time recovery, which moves a job backwards along edges the forward lifecycle never takes.
"""

from __future__ import annotations

import json
from typing import Optional

from image_processor.types import (
    CleanupIntent,
    CompletionAction,
    Job,
    JobState,
    ModelRef,
    SourceIdentity,
    SourceKind,
)

#: Bumped whenever the DDL below changes shape. Stored in the ``meta`` table.
SCHEMA_VERSION = 1

#: States a job may be admitted in. ``[*] --> DISCOVERED`` in DESIGN.md §7; a source that verifies
#: readiness during discovery admits straight at ``READY``.
INITIAL_STATES = frozenset({JobState.DISCOVERED, JobState.READY})

#: The DESIGN.md §7 state diagram, edge for edge. ``transition()`` refuses anything not listed.
TRANSITIONS = frozenset(
    {
        (JobState.DISCOVERED, JobState.READY),
        (JobState.DISCOVERED, JobState.INPUT_INVALID),
        (JobState.INPUT_INVALID, JobState.QUARANTINED),
        (JobState.READY, JobState.CLAIMED),
        (JobState.CLAIMED, JobState.WAITING_MODEL),
        (JobState.WAITING_MODEL, JobState.INFERENCING),
        (JobState.WAITING_MODEL, JobState.BLOCKED_CONFIGURATION),
        (JobState.WAITING_MODEL, JobState.RETRY_WAIT),
        (JobState.INFERENCING, JobState.RESULT_COMMITTED),
        (JobState.INFERENCING, JobState.RETRY_WAIT),
        (JobState.INFERENCING, JobState.PROCESSING_EXHAUSTED),
        (JobState.PROCESSING_EXHAUSTED, JobState.RETAINED_FAILED),
        (JobState.RESULT_COMMITTED, JobState.PUBLISH_PENDING),
        (JobState.PUBLISH_PENDING, JobState.PUBLISHED),
        (JobState.PUBLISH_PENDING, JobState.PUBLISH_EXHAUSTED),
        (JobState.PUBLISH_EXHAUSTED, JobState.PUBLISH_PENDING),
        (JobState.PUBLISHED, JobState.CLEANUP_PENDING),
        (JobState.CLEANUP_PENDING, JobState.COMPLETED),
        (JobState.CLEANUP_PENDING, JobState.CLEANUP_FAILED),
        (JobState.CLEANUP_FAILED, JobState.CLEANUP_PENDING),
        (JobState.RETRY_WAIT, JobState.READY),
    }
)


def is_legal(current: JobState, new: JobState) -> bool:
    """Report whether ``current -> new`` is an edge of the DESIGN.md §7 diagram.

    Args:
        current: The state the job is in.
        new: The state the caller wants to move it to.

    Returns:
        ``True`` when the edge is declared in :data:`TRANSITIONS`.
    """
    return (current, new) in TRANSITIONS


DDL = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kv (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        inference_id       TEXT PRIMARY KEY,
        route_id           TEXT NOT NULL,
        state              TEXT NOT NULL,
        source_json        TEXT NOT NULL,
        model_json         TEXT NOT NULL,
        transform_version  TEXT NOT NULL DEFAULT '',
        attempts           INTEGER NOT NULL DEFAULT 0,
        next_attempt_at_ms INTEGER,
        staged_path        TEXT,
        config_generation  INTEGER NOT NULL DEFAULT 0,
        result_json        BLOB,
        result_sha256      TEXT,
        sidecar_path       TEXT,
        sidecar_sha256     TEXT,
        last_error         TEXT,
        created_at_ms      INTEGER NOT NULL,
        updated_at_ms      INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state, route_id, created_at_ms)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs (state, next_attempt_at_ms, created_at_ms)",
    """
    CREATE TABLE IF NOT EXISTS outbox (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        inference_id    TEXT NOT NULL REFERENCES jobs (inference_id) ON DELETE CASCADE,
        topic           TEXT NOT NULL,
        payload         BLOB NOT NULL,
        gating          INTEGER NOT NULL DEFAULT 1,
        attempts        INTEGER NOT NULL DEFAULT 0,
        last_error      TEXT,
        published_at_ms INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outbox_job ON outbox (inference_id, published_at_ms)",
    "CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox (published_at_ms, id)",
    """
    CREATE TABLE IF NOT EXISTS cleanup_intents (
        inference_id  TEXT PRIMARY KEY REFERENCES jobs (inference_id) ON DELETE CASCADE,
        action        TEXT NOT NULL,
        source_path   TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        target_path   TEXT,
        members_json  TEXT NOT NULL DEFAULT '[]',
        observed      TEXT,
        created_at_ms INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cleanup_open ON cleanup_intents (observed, created_at_ms)",
    """
    CREATE TABLE IF NOT EXISTS route_generations (
        route_id      TEXT PRIMARY KEY,
        desired       TEXT,
        active        TEXT,
        updated_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reservations (
        inference_id TEXT PRIMARY KEY REFERENCES jobs (inference_id) ON DELETE CASCADE,
        bytes        INTEGER NOT NULL
    )
    """,
)

#: The ``jobs`` columns, in the order :func:`row_to_job` expects them.
JOB_COLUMNS = (
    "inference_id",
    "route_id",
    "state",
    "source_json",
    "model_json",
    "transform_version",
    "attempts",
    "next_attempt_at_ms",
    "staged_path",
    "config_generation",
    "result_json",
    "result_sha256",
    "sidecar_path",
    "sidecar_sha256",
    "last_error",
    "created_at_ms",
    "updated_at_ms",
)

JOB_SELECT = "SELECT " + ", ".join(JOB_COLUMNS) + " FROM jobs"

#: Job columns ``transition(**fields)`` may set. Anything else is a programming error.
MUTABLE_JOB_FIELDS = frozenset(
    {
        "attempts",
        "next_attempt_at_ms",
        "staged_path",
        "config_generation",
        "last_error",
    }
)


def apply_schema(conn) -> None:
    """Create every table and index, then stamp the schema version.

    Args:
        conn: An open ``sqlite3.Connection`` positioned outside a transaction.
    """
    for statement in DDL:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?) ON CONFLICT (key) DO NOTHING",
        (str(SCHEMA_VERSION),),
    )


def schema_version(conn) -> Optional[int]:
    """Return the stored schema version, or ``None`` when the database is unstamped.

    Args:
        conn: An open ``sqlite3.Connection``.

    Returns:
        The integer version recorded in ``meta``, or ``None``.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else None


def encode_source(source: SourceIdentity) -> str:
    """Serialize a :class:`~image_processor.types.SourceIdentity` into ``source_json``."""
    return json.dumps(
        {
            "kind": source.kind.value,
            "route_id": source.route_id,
            "relative_path": source.relative_path,
            "bytes": source.bytes,
            "sha256": source.sha256,
            "capture_id": source.capture_id,
            "camera_id": source.camera_id,
            "correlation_id": source.correlation_id,
            "reply_to": source.reply_to,
            "captured_at_ms": source.captured_at_ms,
        },
        sort_keys=True,
    )


def decode_source(blob: str) -> SourceIdentity:
    """Rebuild a :class:`~image_processor.types.SourceIdentity` from ``source_json``."""
    data = json.loads(blob)
    return SourceIdentity(
        kind=SourceKind(data["kind"]),
        route_id=data["route_id"],
        relative_path=data["relative_path"],
        bytes=data["bytes"],
        sha256=data["sha256"],
        capture_id=data.get("capture_id"),
        camera_id=data.get("camera_id"),
        correlation_id=data.get("correlation_id"),
        reply_to=data.get("reply_to"),
        captured_at_ms=data.get("captured_at_ms"),
    )


def encode_model(model: ModelRef) -> str:
    """Serialize a :class:`~image_processor.types.ModelRef` into ``model_json``."""
    return json.dumps(
        {"id": model.id, "version": model.version, "digest": model.digest}, sort_keys=True
    )


def decode_model(blob: str) -> ModelRef:
    """Rebuild a :class:`~image_processor.types.ModelRef` from ``model_json``."""
    data = json.loads(blob)
    return ModelRef(id=data["id"], version=data["version"], digest=data["digest"])


def row_to_job(row) -> Job:
    """Build a :class:`~image_processor.types.Job` from a :data:`JOB_SELECT` row."""
    return Job(
        inference_id=row[0],
        route_id=row[1],
        source=decode_source(row[3]),
        model=decode_model(row[4]),
        transform_version=row[5],
        state=JobState(row[2]),
        attempts=row[6],
        next_attempt_at_ms=row[7],
        staged_path=row[8],
        config_generation=row[9],
    )


def row_to_intent(row) -> CleanupIntent:
    """Build a :class:`~image_processor.types.CleanupIntent` from a ``cleanup_intents`` row."""
    return CleanupIntent(
        inference_id=row[0],
        action=CompletionAction(row[1]),
        source_path=row[2],
        source_sha256=row[3],
        target_path=row[4],
        members=tuple(json.loads(row[5])),
    )
