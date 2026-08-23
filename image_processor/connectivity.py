"""Per-route connectivity, on both surfaces at once (DESIGN.md 13, 14; LLD 8).

A route is this component instance: one input source bound to one immutable model generation.
What an operator needs to know about it is the same in a console dashboard and in a ``status``
reply -- is it enabled, can it see its input, is its model active or still staging, and is there
an executor to run it. The library takes one provider and serves both surfaces from it (the
``state`` keepalive pushes it, the built-in ``status`` verb returns it), so a pulled answer can
never disagree with a pushed one.

``connected`` is the normalized flag every consumer reads: it is true only when the route can
actually produce a decision right now. ``state`` is this component vocabulary for why not, and
``attributes`` carries the generations and the queue depth for a console that understands them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from edgecommons.heartbeat.instance_connectivity import InstanceConnectivity

logger = logging.getLogger(__name__)

#: The route conditions this component reports in ``state``.
ONLINE = "ONLINE"
STAGING = "STAGING"
DEGRADED = "DEGRADED"
DISABLED = "DISABLED"


@dataclass(frozen=True)
class RouteStatus:
    """What one route looks like right now.

    Attributes:
        route_id: The route id, which is also its UNS instance token.
        enabled: Whether the route claims new work.
        paused: Whether an operator has paused it.
        source_reachable: Whether its spool root exists, or its trigger subscription is live.
        source_detail: The root or the topic filters, for the operator detail line.
        desired_generation: The model digest configuration asks for.
        active_generation: The model digest the route is running, or ``None`` while it has none.
        executor_healthy: Whether an executor cell can serve it.
        queued: How many jobs are waiting.
        oldest_age_secs: How long the oldest queued job has waited.
        last_error: The most recent condition, bounded, or ``None``.
    """

    route_id: str
    enabled: bool = True
    paused: bool = False
    source_reachable: bool = True
    source_detail: Optional[str] = None
    desired_generation: Optional[str] = None
    active_generation: Optional[str] = None
    executor_healthy: bool = True
    queued: int = 0
    oldest_age_secs: float = 0.0
    last_error: Optional[str] = None

    @property
    def staging(self) -> bool:
        """Whether configuration is ahead of what the route is running (DESIGN.md 9)."""
        if not self.desired_generation:
            return False
        return self.desired_generation != self.active_generation

    @property
    def connected(self) -> bool:
        """Whether the route can produce a decision right now."""
        return bool(
            self.enabled
            and not self.paused
            and self.source_reachable
            and self.active_generation
            and self.executor_healthy
        )

    @property
    def state(self) -> str:
        """The condition token, which says why a route is not connected."""
        if not self.enabled:
            return DISABLED
        if self.staging:
            return STAGING
        if self.connected:
            return ONLINE
        return DEGRADED


def route_connectivity(status: RouteStatus) -> InstanceConnectivity:
    """Render one route status as the library per-instance connectivity element.

    Args:
        status: The route status.

    Returns:
        The element the ``state`` keepalive and the ``status`` verb both carry.
    """
    attributes = {
        "sourceReachable": bool(status.source_reachable),
        "executorHealthy": bool(status.executor_healthy),
        "queued": int(status.queued),
        "oldestAgeSecs": round(float(status.oldest_age_secs), 3),
        "paused": bool(status.paused),
    }
    if status.desired_generation:
        attributes["desiredGeneration"] = status.desired_generation
    if status.active_generation:
        attributes["activeGeneration"] = status.active_generation
    detail = status.last_error or status.source_detail
    element = InstanceConnectivity.of(status.route_id, status.connected, detail)
    return element.with_state(status.state).with_attributes(attributes)


class ConnectivityProvider:
    """The zero-argument callable ``gg.set_instance_connectivity_provider`` takes.

    Args:
        statuses: Returns the current :class:`RouteStatus` of every configured route. It is
            sampled on every keepalive tick, so it must be cheap and must not block.
    """

    def __init__(self, statuses: Callable[[], Iterable[RouteStatus]]) -> None:
        """Build the provider over a status sampler."""
        self._statuses = statuses

    def __call__(self) -> list:
        """Return the current per-route connectivity.

        Returns:
            One element per route. An empty list when the sampler fails, because a keepalive that
            cannot describe the routes must still be published.
        """
        try:
            return [route_connectivity(status) for status in self._statuses()]
        except Exception:  # noqa: BLE001 - a keepalive tick must not fail on a sampling error
            logger.warning("route connectivity could not be sampled", exc_info=True)
            return []


__all__ = [
    "DEGRADED",
    "DISABLED",
    "ONLINE",
    "STAGING",
    "ConnectivityProvider",
    "RouteStatus",
    "route_connectivity",
]
