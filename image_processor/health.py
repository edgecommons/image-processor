"""Readiness and failure rules (DESIGN.md 14, LLD 8).

Ready is a claim about work, not about a process being alive. This component is ready when it can
take an image and produce a durable, confirmed decision: the configuration parsed and left at
least one enabled route, the durable state and the spool roots are writable, the model cache
metadata verified, and -- for a route that requires NVIDIA -- an executor cell is healthy.

Failure is stronger than degradation, and the difference matters operationally. One bad model
degrades its own routes and the component keeps serving the others. The component fails when no
enabled route can execute at all, when durable state is lost, or when the publication backlog has
crossed its fail-safe bound: at that point accepting more images would mean accepting images it
cannot answer for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: How full the outbox may get before readiness fails closed.
DEFAULT_BACKLOG_FRACTION = 1.0

#: How full the outbox may get before the backlog condition is reported.
DEFAULT_BACKLOG_WARNING_FRACTION = 0.8


@dataclass(frozen=True)
class HealthReport:
    """One readiness evaluation.

    Attributes:
        ready: Whether the component is ready to serve.
        failed: Whether the component cannot serve at all, which is the stronger claim.
        reasons: The stable codes explaining a not-ready or failed verdict.
        degraded_routes: The routes that cannot execute right now.
    """

    ready: bool
    failed: bool = False
    reasons: tuple = ()
    degraded_routes: tuple = ()

    def __bool__(self) -> bool:
        """A report is truthy exactly when the component is ready."""
        return self.ready


class Health:
    """Evaluates the DESIGN.md 14 rules and drives ``gg.set_ready``.

    Every input is a callable, so the rules are exercised without a broker, a GPU, or a
    filesystem, and so one evaluation is one instant rather than a walk of live subsystems.

    Args:
        statuses: Returns the current :class:`~image_processor.connectivity.RouteStatus` of every
            configured route.
        state_writable: Returns whether the durable state and its directory accept writes.
        cache_verified: Returns whether the model cache metadata is readable and verified.
        outbox_pending: Returns how many rows are waiting for confirmation.
        outbox_capacity: The configured bound on pending rows.
        requires_executor: Whether any enabled route needs an executor cell.
        executor_healthy: Returns whether an executor cell can serve work.
        backlog_fraction: How full the outbox may get before readiness fails closed.
    """

    def __init__(
        self,
        *,
        statuses: Callable[[], Iterable],
        state_writable: Callable[[], bool],
        cache_verified: Callable[[], bool] = lambda: True,
        outbox_pending: Callable[[], int] = lambda: 0,
        outbox_capacity: int = 100000,
        requires_executor: Callable[[], bool] = lambda: True,
        executor_healthy: Callable[[], bool] = lambda: True,
        backlog_fraction: float = DEFAULT_BACKLOG_FRACTION,
    ) -> None:
        """Build the evaluator over its inputs."""
        self._statuses = statuses
        self._state_writable = state_writable
        self._cache_verified = cache_verified
        self._outbox_pending = outbox_pending
        self.outbox_capacity = max(1, int(outbox_capacity))
        self._requires_executor = requires_executor
        self._executor_healthy = executor_healthy
        self.backlog_fraction = float(backlog_fraction)
        self.last: Optional[HealthReport] = None

    def evaluate(self) -> HealthReport:
        """Decide whether the component is ready, degraded, or failed.

        Returns:
            The report, which is also retained as :attr:`last`.
        """
        reasons: list = []
        failures: list = []
        try:
            statuses = list(self._statuses())
        except Exception as exc:  # noqa: BLE001 - an unreadable configuration is not ready
            logger.warning("route status could not be sampled: %s", exc, exc_info=True)
            statuses = []
            failures.append("ROUTE_STATUS_UNAVAILABLE")

        enabled = [status for status in statuses if status.enabled]
        if not enabled:
            reasons.append("NO_ENABLED_ROUTE")
        degraded = tuple(status.route_id for status in enabled if not status.connected)
        if enabled and len(degraded) == len(enabled):
            failures.append("NO_ROUTE_CAN_EXECUTE")

        self._safe(self._state_writable, "STATE_NOT_WRITABLE", failures)
        self._safe(self._cache_verified, "MODEL_CACHE_UNVERIFIED", reasons)
        if self._requires_executor_now():
            self._safe(self._executor_healthy, "NO_HEALTHY_EXECUTOR", failures)

        pending = self._pending()
        if pending >= self.outbox_capacity * self.backlog_fraction:
            failures.append("PUBLISH_BACKLOG_EXCEEDED")

        ready = not reasons and not failures
        report = HealthReport(
            ready=ready,
            failed=bool(failures),
            reasons=tuple(failures + reasons),
            degraded_routes=degraded,
        )
        self.last = report
        return report

    def _requires_executor_now(self) -> bool:
        """Whether an executor is required for the routes that are enabled."""
        try:
            return bool(self._requires_executor())
        except Exception:  # noqa: BLE001 - assume it is required, which fails closed
            logger.warning("could not read the executor requirement", exc_info=True)
            return True

    def _pending(self) -> int:
        """Read the outbox depth, treating an unreadable ledger as empty."""
        try:
            return int(self._outbox_pending())
        except Exception:  # noqa: BLE001 - the state check already reports an unusable ledger
            logger.warning("the outbox depth could not be read", exc_info=True)
            return 0

    @staticmethod
    def _safe(check: Callable[[], bool], code: str, sink: list) -> bool:
        """Run one check, recording its code when it says no or cannot answer."""
        try:
            answer = bool(check())
        except Exception:  # noqa: BLE001 - a check that cannot run fails closed
            logger.warning("health check %s could not run", code, exc_info=True)
            answer = False
        if not answer:
            sink.append(code)
        return answer

    def apply(self, gg: Any) -> HealthReport:
        """Evaluate and push the verdict into the library readiness flag.

        Args:
            gg: The EdgeCommons handle.

        Returns:
            The report that was applied.
        """
        report = self.evaluate()
        try:
            gg.set_ready(report.ready)
        except Exception:  # noqa: BLE001 - readiness reporting must not kill the caller
            logger.warning("readiness could not be set", exc_info=True)
        if not report.ready:
            logger.warning("not ready: %s", ", ".join(report.reasons))
        return report


__all__ = [
    "DEFAULT_BACKLOG_FRACTION",
    "DEFAULT_BACKLOG_WARNING_FRACTION",
    "Health",
    "HealthReport",
]
