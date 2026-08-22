"""SQLite job ledger, outbox, cleanup intents, and restart recovery (LLD §5, DESIGN.md §7).

:class:`Ledger` is the component's only durable coordination point. Import it from here rather
than from the submodules.
"""

from image_processor.ledger.ledger import (
    DEFAULT_RESERVE_BUDGET_BYTES,
    SYNCHRONOUS_MODES,
    IllegalTransition,
    Ledger,
    LedgerClosed,
    LedgerConflict,
    LedgerError,
    OutboxRow,
)
from image_processor.ledger.recovery import (
    RECOVERY_EDGES,
    RecoveryMove,
    RecoveryReport,
    SidecarRecord,
    edge_key,
    plan_recovery,
)
from image_processor.ledger.schema import (
    INITIAL_STATES,
    SCHEMA_VERSION,
    TRANSITIONS,
    is_legal,
)

__all__ = [
    "DEFAULT_RESERVE_BUDGET_BYTES",
    "INITIAL_STATES",
    "IllegalTransition",
    "Ledger",
    "LedgerClosed",
    "LedgerConflict",
    "LedgerError",
    "OutboxRow",
    "RECOVERY_EDGES",
    "RecoveryMove",
    "RecoveryReport",
    "SCHEMA_VERSION",
    "SYNCHRONOUS_MODES",
    "SidecarRecord",
    "TRANSITIONS",
    "edge_key",
    "is_legal",
    "plan_recovery",
]
