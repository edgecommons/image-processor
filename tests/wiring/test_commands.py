"""Every component verb: what it answers, what it refuses, and how it pages."""

from __future__ import annotations

import pytest
from edgecommons.command_inbox import (
    CommandScope,
    Deferred,
    ImmediateError,
    ImmediateSuccess,
)

from image_processor.commands import (
    MAX_PAGE,
    VERB_SCOPES,
    DeferredApp,
    ProcessorCommands,
    cursor_of,
    page_size,
    request_body,
)
from tests.wiring.conftest import FakeCommandInbox


class FakeApp:
    """A component that records what the verbs asked it to do."""

    def __init__(self) -> None:
        self.calls: list = []
        self.routes = ["clearance-cam-01", "adhoc-inspect"]
        self.models = [
            {"id": f"model-{index}", "version": "1.0.0", "digest": f"sha256:{index:064d}"}
            for index in range(5)
        ]
        self.jobs = ([{"inferenceId": "01k"}], "cursor-2")
        self.evict_result = {"evicted": True, "digest": "sha256:ab", "cells": ["cpu-0"]}

    def route_ids(self):
        """The configured routes."""
        return self.routes

    def list_models(self):
        """The model catalog."""
        return self.models

    def list_jobs(self, route, states, cursor, limit):
        """One page of jobs."""
        self.calls.append(("list_jobs", route, states, cursor, limit))
        if states == ["NOPE"]:
            raise ValueError("'NOPE' is not a job state")
        return self.jobs

    def job_counts(self, route=None):
        """Counts by state."""
        return {"READY": 1}

    def scheduler_summary(self):
        """The scheduler view."""
        return {"queued": 1}

    def rescan(self, route):
        """A walk."""
        self.calls.append(("rescan", route))
        return 3

    def preload_model(self, model_id, digest):
        """A staging request."""
        self.calls.append(("preload", model_id, digest))
        return {"id": model_id or "by-digest", "staged": True}

    def evict_model(self, digest):
        """An eviction request."""
        self.calls.append(("evict", digest))
        return self.evict_result

    def reload_model_catalog(self):
        """A catalog re-evaluation."""
        self.calls.append(("reload-catalog",))
        return {"routesSwitched": 1}

    def set_activation_override(self, route, enabled):
        """An override."""
        self.calls.append(("override", route, enabled))
        return {"route": route, "effective": bool(enabled)}

    def retry_publication(self, route, inference_id):
        """A publication repair."""
        self.calls.append(("retry-publication", route, inference_id))
        return {"returned": ["01k"], "published": 1}

    def retry_cleanup(self, route, inference_id):
        """A cleanup repair."""
        self.calls.append(("retry-cleanup", route, inference_id))
        return {"repaired": ["01k"], "stillFailed": []}

    def reconcile(self, route):
        """A reconciliation."""
        self.calls.append(("reconcile", route))
        return {"reconciled": {}}

    def pause(self, route):
        """A pause."""
        self.calls.append(("pause", route))
        return {"paused": True, "routes": [route] if route else self.routes}

    def resume(self, route):
        """A resume."""
        self.calls.append(("resume", route))
        return {"paused": False, "routes": [route] if route else self.routes}


@pytest.fixture
def inbox() -> FakeCommandInbox:
    """A command inbox with the component verbs registered."""
    return FakeCommandInbox()


@pytest.fixture
def app() -> FakeApp:
    """The component the verbs act on."""
    return FakeApp()


@pytest.fixture
def commands(app, inbox) -> ProcessorCommands:
    """The registered verb set."""
    registered = ProcessorCommands(app)
    registered.register(inbox)
    return registered


def test_every_designed_verb_is_registered_with_its_scope(commands, inbox):
    assert set(inbox.handlers) == set(VERB_SCOPES)
    assert inbox.scopes["get-models"] is CommandScope.COMPONENT
    assert inbox.scopes["get-queue"] is CommandScope.BOTH
    assert inbox.scopes["set-route-activation-override"] is CommandScope.INSTANCE


def test_get_models_pages_with_an_opaque_cursor(commands, inbox):
    first = inbox.dispatch("get-models", {"max": 2})

    assert isinstance(first, ImmediateSuccess)
    assert [entry["id"] for entry in first.result["models"]] == ["model-0", "model-1"]
    assert first.result["nextCursor"] == "model-2|1.0.0"
    assert first.result["total"] == 5

    second = inbox.dispatch("get-models", {"max": 2, "cursor": first.result["nextCursor"]})
    assert [entry["id"] for entry in second.result["models"]] == ["model-2", "model-3"]

    last = inbox.dispatch("get-models", {"cursor": "model-4|1.0.0"})
    assert last.result["nextCursor"] is None


def test_a_cursor_that_names_nothing_is_refused(commands, inbox):
    outcome = inbox.dispatch("get-models", {"cursor": "model-9|9"})

    assert isinstance(outcome, ImmediateError)
    assert outcome.code == "BAD_ARGS"


def test_a_component_verb_addressed_to_an_instance_is_refused(commands, inbox):
    outcome = inbox.dispatch("get-models", {}, instance="clearance-cam-01")

    assert isinstance(outcome, ImmediateError)
    assert outcome.code == "BAD_ARGS"


def test_get_queue_answers_for_one_route_or_the_component(commands, inbox, app):
    scoped = inbox.dispatch("get-queue", {"max": 5}, instance="clearance-cam-01")

    assert scoped.result["route"] == "clearance-cam-01"
    assert scoped.result["jobs"] == [{"inferenceId": "01k"}]
    assert scoped.result["nextCursor"] == "cursor-2"
    assert scoped.result["counts"] == {"READY": 1}
    assert scoped.result["scheduler"] == {"queued": 1}

    whole = inbox.dispatch("get-queue", {})
    assert whole.result["route"] is None


def test_get_queue_refuses_a_state_that_is_not_one(commands, inbox):
    outcome = inbox.dispatch("get-queue", {"states": ["NOPE"]})

    assert isinstance(outcome, ImmediateError)
    assert outcome.code == "BAD_ARGS"


def test_get_queue_refuses_an_unusable_page_size(commands, inbox):
    assert inbox.dispatch("get-queue", {"max": 0}).code == "BAD_ARGS"
    assert inbox.dispatch("get-queue", {"max": True}).code == "BAD_ARGS"
    assert inbox.dispatch("get-queue", {"cursor": 7}).code == "BAD_ARGS"


def test_a_page_size_is_bounded(commands, inbox, app):
    inbox.dispatch("get-queue", {"max": 10000})

    assert app.calls[-1] == ("list_jobs", None, None, None, MAX_PAGE)


def test_an_unknown_route_is_not_found(commands, inbox):
    outcome = inbox.dispatch("get-queue", {"route": "nope"})

    assert isinstance(outcome, ImmediateError)
    assert outcome.code == "NOT_FOUND"


def test_trigger_rescan_walks_now(commands, inbox, app):
    outcome = inbox.dispatch("trigger-rescan", {}, instance="clearance-cam-01")

    assert outcome.result == {"route": "clearance-cam-01", "discovered": 3}
    assert ("rescan", "clearance-cam-01") in app.calls


def test_preload_model_defers_and_settles_with_the_outcome(commands, inbox, app):
    outcome = inbox.dispatch("preload-model", {"id": "model-1"})

    assert isinstance(outcome, Deferred)
    assert outcome.token.state == "SETTLED"
    assert outcome.token.result == {"id": "model-1", "staged": True}
    assert ("preload", "model-1", None) in app.calls


def test_preload_model_needs_something_to_name_the_model(commands, inbox):
    assert inbox.dispatch("preload-model", {}).code == "BAD_ARGS"
    assert inbox.dispatch("preload-model", {"digest": 7}).code == "BAD_ARGS"


def test_a_deferred_verb_that_fails_settles_with_an_error(commands, inbox, app):
    def _boom(model_id, digest):
        raise RuntimeError("the source is unreachable")

    app.preload_model = _boom

    outcome = inbox.dispatch("preload-model", {"digest": "sha256:ab"})

    assert outcome.token.error[0] == "OPERATION_FAILED"


def test_evict_model_refuses_a_leased_generation(commands, inbox, app):
    app.evict_result = {"evicted": False, "reason": "the generation is leased"}

    outcome = inbox.dispatch("evict-model", {"digest": "sha256:ab"})

    assert isinstance(outcome, ImmediateError)
    assert outcome.code == "CONFLICT"
    assert "leased" in str(outcome.message)


def test_evict_model_needs_a_digest(commands, inbox):
    assert inbox.dispatch("evict-model", {}).code == "BAD_ARGS"


def test_evict_model_releases_an_idle_generation(commands, inbox):
    outcome = inbox.dispatch("evict-model", {"digest": "sha256:ab"})

    assert isinstance(outcome, ImmediateSuccess)
    assert outcome.result["cells"] == ["cpu-0"]


def test_reload_model_catalog_defers(commands, inbox, app):
    outcome = inbox.dispatch("reload-model-catalog", {})

    assert outcome.token.result == {"routesSwitched": 1}
    assert ("reload-catalog",) in app.calls


def test_the_activation_override_addresses_exactly_one_route(commands, inbox, app):
    outcome = inbox.dispatch(
        "set-route-activation-override", {"enabled": False}, instance="adhoc-inspect"
    )

    assert outcome.result["effective"] is False
    assert ("override", "adhoc-inspect", False) in app.calls
    assert inbox.dispatch("set-route-activation-override", {"enabled": False}).code == "BAD_ARGS"
    assert (
        inbox.dispatch("set-route-activation-override", {"route": "adhoc-inspect"}).code
        == "BAD_ARGS"
    )
    assert (
        inbox.dispatch(
            "set-route-activation-override", {"route": "adhoc-inspect", "enabled": "yes"}
        ).code
        == "BAD_ARGS"
    )


def test_the_activation_override_clears_with_null(commands, inbox, app):
    inbox.dispatch("set-route-activation-override", {"enabled": None}, instance="adhoc-inspect")

    assert ("override", "adhoc-inspect", None) in app.calls


@pytest.mark.parametrize("verb", ["retry-publication", "retry-cleanup", "reconcile"])
def test_every_repair_verb_defers(commands, inbox, app, verb):
    outcome = inbox.dispatch(verb, {}, instance="clearance-cam-01")

    assert isinstance(outcome, Deferred)
    assert outcome.token.state == "SETTLED"
    assert outcome.token.result is not None


def test_a_repair_verb_refuses_a_non_string_inference_id(commands, inbox):
    assert inbox.dispatch("retry-publication", {"inferenceId": 7}).code == "BAD_ARGS"
    assert inbox.dispatch("retry-cleanup", {"inferenceId": 7}).code == "BAD_ARGS"


def test_pause_and_resume_answer_for_both_scopes(commands, inbox, app):
    scoped = inbox.dispatch("pause", {}, instance="clearance-cam-01")
    assert scoped.result["routes"] == ["clearance-cam-01"]

    whole = inbox.dispatch("resume", {})
    assert whole.result["routes"] == app.routes
    assert ("pause", "clearance-cam-01") in app.calls
    assert ("resume", None) in app.calls


def test_a_single_route_component_needs_no_route_name(app, inbox):
    app.routes = ["only-one"]
    ProcessorCommands(app).register(inbox)

    outcome = inbox.dispatch("set-route-activation-override", {"enabled": True})

    assert outcome.result["route"] == "only-one"


def test_the_deferred_handle_answers_only_once_it_is_bound():
    deferred = DeferredApp()

    with pytest.raises(RuntimeError):
        deferred.route_ids()

    app = FakeApp()
    assert deferred.bind(app) is app
    assert deferred.route_ids() == app.routes


def test_a_request_body_is_read_from_an_envelope_or_a_mapping():
    from edgecommons.messaging.message import Message, MessageHeader

    assert request_body(Message(header=MessageHeader("v", "1.0"), body={"max": 5})) == {"max": 5}
    assert request_body({"body": {"max": 5}, "header": {}}) == {"max": 5}
    assert request_body({"max": 5}) == {"max": 5}
    assert request_body(None) == {}


def test_the_page_helpers_apply_the_defaults():
    assert page_size({}) == 100
    assert page_size({"max": 7}) == 7
    assert cursor_of({}) is None
    assert cursor_of({"cursor": ""}) is None
