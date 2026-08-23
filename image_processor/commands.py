"""The component command verbs (DESIGN.md 13, LLD 8).

The library owns liveness, discovery, reload, configuration, and per-instance connectivity
(``ping``, ``describe``, ``reload-config``, ``get-configuration``, ``status``). This module adds
what an operator needs to run an inference component: see what models and what work exist, make a
model ready before a line starts, stop and start claiming work, and repair a job that a crash or
an outage left half-finished.

Three rules shape every verb here.

*A reply is bounded.* Anything that could grow with traffic -- the queue, the model catalog --
is paginated with an opaque ``cursor`` and a ``max``, so a command reply never depends on how
busy the component has been.

*A slow verb defers rather than blocks.* Staging and warming a model takes minutes, and the
inbox dispatch thread cannot wait for it. Those verbs take a guarded deferred-reply token,
return immediately, and settle the token when the work finishes.

*An operator command never fakes an outcome.* A repair verb reports what the ledger and the
filesystem actually show after it ran, including a failure, because the whole point of the repair
surface is to tell an operator the truth about durable state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable, Optional

from edgecommons.command_inbox import (
    CommandException,
    CommandOutcome,
    CommandScope,
)

logger = logging.getLogger(__name__)

#: Error code: the request names something the component does not have.
ERR_NOT_FOUND = "NOT_FOUND"

#: Error code: the request is well-formed but cannot be honoured right now.
ERR_CONFLICT = "CONFLICT"

#: Error code: the request arguments are unusable.
ERR_BAD_ARGS = "BAD_ARGS"

#: Error code: the operation ran and failed.
ERR_FAILED = "OPERATION_FAILED"

#: The default and maximum page size of a paginated reply.
DEFAULT_PAGE = 100
MAX_PAGE = 500

#: How long a deferred repair or preload may take before the core reply window closes.
DEFAULT_DEFER_SECS = 30 * 60.0

#: The verbs this component registers, with the scope each accepts.
VERB_SCOPES = {
    "get-models": CommandScope.COMPONENT,
    "get-queue": CommandScope.BOTH,
    "trigger-rescan": CommandScope.BOTH,
    "preload-model": CommandScope.COMPONENT,
    "evict-model": CommandScope.COMPONENT,
    "reload-model-catalog": CommandScope.COMPONENT,
    "set-route-activation-override": CommandScope.INSTANCE,
    "retry-publication": CommandScope.BOTH,
    "retry-cleanup": CommandScope.BOTH,
    "reconcile": CommandScope.BOTH,
    "pause": CommandScope.BOTH,
    "resume": CommandScope.BOTH,
}


def request_body(request: Any) -> dict:
    """Return a command request body as a mapping.

    Args:
        request: The core ``Message``, a mapping, or ``None``.

    Returns:
        The body, or an empty mapping when the request carries none.
    """
    accessor = getattr(request, "get_body", None)
    body = accessor() if callable(accessor) else request
    if isinstance(body, Mapping) and "body" in body and not _looks_like_body(body):
        body = body["body"]
    return dict(body) if isinstance(body, Mapping) else {}


def _looks_like_body(candidate: Mapping) -> bool:
    """Whether a mapping is already a body rather than a whole envelope."""
    return not any(key in candidate for key in ("header", "identity", "tags"))


def page_size(body: Mapping) -> int:
    """Return the page size a request asks for, bounded.

    Args:
        body: The request body.

    Returns:
        A size between 1 and :data:`MAX_PAGE`.

    Raises:
        CommandException: ``BAD_ARGS`` when ``max`` is not a positive integer.
    """
    raw = body.get("max")
    if raw is None:
        return DEFAULT_PAGE
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise CommandException(ERR_BAD_ARGS, "max is a positive integer")
    return min(int(raw), MAX_PAGE)


def cursor_of(body: Mapping) -> Optional[str]:
    """Return the opaque cursor a request carries, if any.

    Args:
        body: The request body.

    Returns:
        The cursor, or ``None`` for the first page.

    Raises:
        CommandException: ``BAD_ARGS`` when the cursor is not a string.
    """
    raw = body.get("cursor")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise CommandException(ERR_BAD_ARGS, "cursor is the opaque string a previous reply gave")
    return raw


class DeferredApp:
    """A handle to a component that does not exist yet.

    The core registers command verbs while it builds the runtime, before the inbox subscription
    is acknowledged, so that no early request can find a missing verb. The component itself is
    built from the finished runtime, which is the other way round. This closes the circle: the
    verbs are registered against this handle during the build, and the handle is bound to the
    component immediately afterwards, before anything can dispatch to it.
    """

    def __init__(self) -> None:
        """Start unbound."""
        self._app: Any = None

    def bind(self, app: Any) -> Any:
        """Bind the built component.

        Args:
            app: The component every verb acts on.

        Returns:
            The component.
        """
        self._app = app
        return app

    def __getattr__(self, name: str) -> Any:
        """Forward to the bound component, refusing to answer before there is one."""
        app = self.__dict__.get("_app")
        if app is None:
            raise RuntimeError("the component is still starting")
        return getattr(app, name)


class ProcessorCommands:
    """Registers the DESIGN.md 13 verbs against one running component.

    Args:
        app: The component. It supplies every operation the verbs expose; see the module
            docstring for why the two are kept apart.
        defer_secs: How long a deferred repair or preload may take.
    """

    def __init__(self, app: Any, *, defer_secs: float = DEFAULT_DEFER_SECS) -> None:
        """Build the verb set over the running component."""
        self._app = app
        self.defer_secs = float(defer_secs)
        self._inbox: Any = None

    def register(self, inbox: Any) -> Any:
        """Register every verb on a command inbox.

        This is the callable ``EdgeCommonsBuilder.configure_commands`` takes, so the verbs exist
        before the inbox subscription is acknowledged and no early request finds a missing verb.

        Args:
            inbox: The core ``CommandInbox``.

        Returns:
            The inbox.
        """
        self._inbox = inbox
        handlers = {
            "get-models": self.get_models,
            "get-queue": self.get_queue,
            "trigger-rescan": self.trigger_rescan,
            "preload-model": self.preload_model,
            "evict-model": self.evict_model,
            "reload-model-catalog": self.reload_model_catalog,
            "set-route-activation-override": self.set_route_activation_override,
            "retry-publication": self.retry_publication,
            "retry-cleanup": self.retry_cleanup,
            "reconcile": self.reconcile,
            "pause": self.pause,
            "resume": self.resume,
        }
        for verb, handler in handlers.items():
            inbox.register_outcome(verb, VERB_SCOPES[verb], handler)
        logger.info("registered %d component command verb(s)", len(handlers))
        return inbox

    # -- helpers -----------------------------------------------------------------------

    def _route(self, addressed: Optional[str], body: Mapping, *, required: bool) -> Optional[str]:
        """Resolve which route a request addresses.

        The topic instance token wins, then a body ``route``. A request that names none is the
        whole component, except where the verb needs exactly one route: with a single configured
        route that is unambiguous, and with several it is an error rather than a guess.
        """
        named = addressed or body.get("route") or body.get("routeId")
        if named is not None and not isinstance(named, str):
            raise CommandException(ERR_BAD_ARGS, "route is a string")
        routes = list(self._app.route_ids())
        if named:
            if named not in routes:
                raise CommandException(ERR_NOT_FOUND, f"no route '{named}'")
            return named
        if not required:
            return None
        if len(routes) == 1:
            return routes[0]
        raise CommandException(ERR_BAD_ARGS, "this verb addresses one route; name it")

    def _deferred(self, request: Any, work: Callable[[], dict]) -> Any:
        """Accept a slow operation and settle its reply when the work finishes.

        The token is provisioned and activated before the continuation starts, so a request that
        cannot be answered later is refused now rather than accepted and forgotten.
        """
        token = self._inbox.defer(request, self.defer_secs)
        if not token.activate():
            raise CommandException(ERR_CONFLICT, "the reply target is no longer open")

        def _run() -> None:
            try:
                token.settle_success(work())
            except CommandException as exc:
                token.settle_error(exc.code, str(exc))
            except Exception as exc:  # noqa: BLE001 - an operator always gets an answer
                logger.exception("a deferred command failed")
                token.settle_error(ERR_FAILED, f"{type(exc).__name__}: {exc}")

        return CommandOutcome.deferred(token, _run)

    # -- verbs -------------------------------------------------------------------------

    def get_models(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Report the staged, active, and rollback generations, one page at a time."""
        body = request_body(request)
        limit = page_size(body)
        cursor = cursor_of(body)
        models = self._app.list_models()
        start = 0
        if cursor is not None:
            identifiers = [f"{entry['id']}|{entry['version']}" for entry in models]
            if cursor not in identifiers:
                raise CommandException(ERR_BAD_ARGS, "the cursor names no model")
            start = identifiers.index(cursor)
        page = models[start : start + limit]
        following = models[start + limit : start + limit + 1]
        next_cursor = f"{following[0]['id']}|{following[0]['version']}" if following else None
        return CommandOutcome.success(
            {"models": page, "nextCursor": next_cursor, "total": len(models)}
        )

    def get_queue(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Report jobs by state and age, one page at a time."""
        body = request_body(request)
        limit = page_size(body)
        cursor = cursor_of(body)
        route = self._route(addressed_instance, body, required=False)
        states = body.get("states")
        if states is not None and (
            not isinstance(states, (list, tuple))
            or not all(isinstance(state, str) for state in states)
        ):
            raise CommandException(ERR_BAD_ARGS, "states is an array of job state names")
        try:
            jobs, next_cursor = self._app.list_jobs(route, states, cursor, limit)
        except ValueError as exc:
            raise CommandException(ERR_BAD_ARGS, str(exc)) from exc
        return CommandOutcome.success(
            {
                "route": route,
                "jobs": jobs,
                "nextCursor": next_cursor,
                "counts": self._app.job_counts(route),
                "scheduler": self._app.scheduler_summary(),
            }
        )

    def trigger_rescan(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Walk a route root now instead of waiting for the interval."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        found = self._app.rescan(route)
        return CommandOutcome.success({"route": route, "discovered": found})

    def preload_model(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Stage and warm a model generation before a route needs it."""
        body = request_body(request)
        digest = body.get("digest")
        model_id = body.get("id") or body.get("model")
        if digest is not None and not isinstance(digest, str):
            raise CommandException(ERR_BAD_ARGS, "digest is a string")
        if model_id is not None and not isinstance(model_id, str):
            raise CommandException(ERR_BAD_ARGS, "id is a string")
        if not digest and not model_id:
            raise CommandException(ERR_BAD_ARGS, "name the model by id or by digest")
        return self._deferred(request, lambda: self._app.preload_model(model_id, digest))

    def evict_model(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Release an idle resident session. A leased model is refused, not evicted."""
        body = request_body(request)
        digest = body.get("digest")
        if not isinstance(digest, str) or not digest:
            raise CommandException(ERR_BAD_ARGS, "digest names the generation to evict")
        outcome = self._app.evict_model(digest)
        if not outcome.get("evicted"):
            raise CommandException(ERR_CONFLICT, outcome.get("reason", "the model is in use"))
        return CommandOutcome.success(outcome)

    def reload_model_catalog(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Re-evaluate the configured models against the cache, retrying failed generations."""
        return self._deferred(request, self._app.reload_model_catalog)

    def set_route_activation_override(
        self, request: Any, addressed_instance: Optional[str]
    ) -> Any:
        """Persist an operational override of a route configured activation.

        The override is reported beside the configured value and never rewrites configuration: a
        deployment stays the source of truth for what the route is, and this says what an
        operator has done to it in the meantime.
        """
        body = request_body(request)
        route = self._route(addressed_instance, body, required=True)
        if "enabled" not in body:
            raise CommandException(ERR_BAD_ARGS, "enabled is true, false, or null to clear it")
        enabled = body["enabled"]
        if enabled is not None and not isinstance(enabled, bool):
            raise CommandException(ERR_BAD_ARGS, "enabled is true, false, or null to clear it")
        return CommandOutcome.success(self._app.set_activation_override(route, enabled))

    def retry_publication(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Return exhausted publications to the outbox and drain it."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        inference_id = body.get("inferenceId")
        if inference_id is not None and not isinstance(inference_id, str):
            raise CommandException(ERR_BAD_ARGS, "inferenceId is a string")
        return self._deferred(
            request, lambda: self._app.retry_publication(route, inference_id)
        )

    def retry_cleanup(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Retry the completion actions that failed."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        inference_id = body.get("inferenceId")
        if inference_id is not None and not isinstance(inference_id, str):
            raise CommandException(ERR_BAD_ARGS, "inferenceId is a string")
        return self._deferred(request, lambda: self._app.retry_cleanup(route, inference_id))

    def reconcile(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Re-decide every open cleanup intent against observed filesystem state."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        return self._deferred(request, lambda: self._app.reconcile(route))

    def pause(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Stop claiming new work. In-flight jobs finish."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        return CommandOutcome.success(self._app.pause(route))

    def resume(self, request: Any, addressed_instance: Optional[str]) -> Any:
        """Start claiming work again."""
        body = request_body(request)
        route = self._route(addressed_instance, body, required=False)
        return CommandOutcome.success(self._app.resume(route))


__all__ = [
    "DEFAULT_DEFER_SECS",
    "DEFAULT_PAGE",
    "ERR_BAD_ARGS",
    "ERR_CONFLICT",
    "ERR_FAILED",
    "ERR_NOT_FOUND",
    "MAX_PAGE",
    "VERB_SCOPES",
    "DeferredApp",
    "ProcessorCommands",
    "cursor_of",
    "page_size",
    "request_body",
]
