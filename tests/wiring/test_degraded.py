"""What the component does when a collaborator misbehaves.

Every path here exists so that one broken thing degrades itself rather than the component: a
vault that will not answer, a broker that refuses a subscription, a filesystem that disappears
mid-flight. They are tested because an untested error path is a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_processor.types import JobState
from tests.wiring.conftest import FakeGg, FakeVault, config_document, write_capture


def _app(document, gg=None):
    """Build a component over a document."""
    from image_processor.ImageProcessor import ImageProcessor

    return ImageProcessor(gg or FakeGg(document))


def test_a_configuration_manager_with_no_generation_starts_at_zero(home, corpus):
    document = config_document(home, corpus, trigger=False)
    gg = FakeGg(document)
    gg.config_manager.get_generation = lambda: (_ for _ in ()).throw(RuntimeError("none"))

    app = _app(document, gg)
    try:
        assert app._generation == 0
    finally:
        app.stop()


def test_a_vault_that_will_not_answer_leaves_the_key_out(home, corpus):
    document = config_document(home, corpus, trigger=False)
    document["component"]["global"]["signing"] = {
        "required": False,
        "trustedKeys": [{"keyId": "publisher-1", "publicKey": {"$secret": "missing/one"}}],
    }
    app = _app(document, FakeGg(document, vault=FakeVault()))
    try:
        assert app._artifacts.trusted_keys == {}
    finally:
        app.stop()


def test_a_route_that_cannot_be_built_degrades_only_itself(home, corpus, monkeypatch):
    document = config_document(home, corpus)
    monkeypatch.setattr(
        "image_processor.ImageProcessor.TriggerSource",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no staging directory")),
    )
    app = _app(document)
    try:
        app._start_routes()

        assert app._routes["adhoc-inspect"].source is None
        assert app._routes["clearance-cam-01"].source is not None
        assert "no staging" in app._routes["adhoc-inspect"].last_error
    finally:
        app.stop()


def test_stopping_survives_a_source_that_will_not_stop(running, gg):
    runtime = running._routes["clearance-cam-01"]
    runtime.source.stop = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("wedged"))
    gg.messaging.unsubscribe = lambda topic: (_ for _ in ()).throw(RuntimeError("gone"))

    running.stop()

    assert running._stopped is True


def test_an_input_for_an_unknown_route_is_reported(running, gg):
    from tests.outputs.conftest import make_source

    running.discovered("nope", make_source(route_id="nope"), None)

    assert running.job_counts() == {}


def test_an_admission_the_ledger_refuses_is_never_a_silent_drop(running, gg):
    from tests.outputs.conftest import make_source

    running._ledger.close()

    running.discovered("clearance-cam-01", make_source(), None)

    assert running._routes["clearance-cam-01"].last_error.startswith("ADMISSION_FAILED")


def test_a_refused_input_whose_path_escapes_is_not_moved(running, gg):
    running.invalid("clearance-cam-01", "../../etc/passwd", "PATH_ESCAPE")

    assert gg.find_event("input-rejected") is not None
    assert running.job_counts() == {}


def test_an_unreadable_cache_entry_leaves_the_transform_unpinned(running, monkeypatch):
    monkeypatch.setattr(
        running._cache, "get", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("gone"))
    )

    assert running._transform_version("sha256:" + "ab" * 32) == ""


def test_a_result_for_a_job_whose_route_vanished_after_publication_is_reported(running, gg):
    from tests.outputs.conftest import make_job

    running._on_published(make_job(route_id="gone"))

    assert gg.messaging.published == []


def test_a_completion_with_no_configured_action_is_reported(running, gg):
    from tests.outputs.conftest import make_job

    route = running._config.route("clearance-cam-01")

    running._complete(make_job(route_id="clearance-cam-01", state=JobState.CLAIMED), route)

    assert gg.find_event("cleanup-failed") is not None


def test_a_reply_that_cannot_be_sent_never_fails_the_job(running, gg, monkeypatch):
    from tests.outputs.conftest import make_job, make_source

    monkeypatch.setattr(
        gg.messaging, "publish", lambda *args: (_ for _ in ()).throw(RuntimeError("no broker"))
    )
    job = make_job(source=make_source(reply_to="ecv1/a/b/c/app/x", correlation_id="corr-1"))

    running._reply(job, running._config.route("clearance-cam-01"), {"status": "SUCCEEDED"})


def test_a_failed_result_that_cannot_be_published_still_completes(running, gg, home, monkeypatch):
    write_capture(home / "spool", "broken.png", b"not an image at all")
    running._source_of("clearance-cam-01").rescan()
    gg.messaging.fail_confirmed = RuntimeError("no broker")

    for _ in range(50):
        running._scheduler.run_once()
        if not running._scheduler.queued():
            break

    assert (home / "failed" / "broken.png").is_file()


def test_the_mirror_error_hook_only_counts(running):
    running._on_mirror_error("clearance-cam-01", "line-clearance/pass", "no broker")


def test_a_rebuilt_route_keeps_its_operator_state(running, gg):
    running.set_activation_override("clearance-cam-01", True)
    document = json.loads(json.dumps(gg.config_manager.document))
    document["component"]["instances"][0]["source"]["include"] = ["**/*.tif"]

    assert gg.config_manager.apply(document) is True

    assert running._routes["clearance-cam-01"].override is True
    assert running._routes["clearance-cam-01"].source.include == ("**/*.tif",)


def test_the_operator_surfaces_answer_for_the_whole_component(running):
    assert running.list_models()[0]["id"] == "synthetic-anomaly-scalar"
    assert "queued" in running.scheduler_summary()
    assert running.rescan(None) == 0
    assert running.retry_publication(None, None) == {"returned": [], "published": 0}
    assert running.retry_cleanup(None, None) == {"repaired": [], "stillFailed": []}
    assert running.retry_cleanup(None, "nope") == {"repaired": [], "stillFailed": []}
    assert running.reconcile("clearance-cam-01")["reconciled"] == {}


def test_preloading_a_model_that_cannot_be_staged_is_an_operation_failure(running, monkeypatch):
    from edgecommons.command_inbox import CommandException

    from image_processor.artifacts import ArtifactError

    monkeypatch.setattr(
        running._artifacts,
        "stage",
        lambda entry: (_ for _ in ()).throw(ArtifactError("DIGEST_MISMATCH", "not that bundle")),
    )

    with pytest.raises(CommandException) as failure:
        running.preload_model(None, running._config.models[0].digest)

    assert failure.value.code == "OPERATION_FAILED"


def test_pausing_a_route_that_never_built_a_source_is_harmless(running):
    running._routes["adhoc-inspect"].source = None

    running.pause("adhoc-inspect")
    running.resume("adhoc-inspect")


def test_pausing_a_trigger_route_drops_its_subscriptions(running, gg):
    runtime = running._routes["adhoc-inspect"]
    assert runtime.topics

    running.pause("adhoc-inspect")
    assert runtime.topics == ()
    assert "ecv1/+/inspection-ui/+/app/inspect/request" not in gg.messaging.subscriptions

    running.resume("adhoc-inspect")
    assert runtime.topics


def test_a_disabled_route_is_not_resumed(running):
    runtime = running._routes["adhoc-inspect"]
    running.pause("adhoc-inspect")
    runtime.override = False
    runtime.paused = False

    running._resume_route(runtime)

    assert runtime.topics == ()


def test_a_camera_route_polls_the_status_verb_when_it_is_configured_to(home, corpus):
    document = config_document(home, corpus, trigger=False)
    document["component"]["instances"][0]["source"]["camera"]["reconcileCaptureStatusSecs"] = 1
    app = _app(document)
    try:
        app._start_routes()
        runtime = app._routes["clearance-cam-01"]

        assert runtime.reconciler is not None
        app.pause("clearance-cam-01")
        app.resume("clearance-cam-01")
    finally:
        app.stop()


def test_an_input_resolves_under_the_root_its_kind_names(running, home):
    from image_processor.types import SourceKind
    from tests.outputs.conftest import make_source

    route = running._config.route("adhoc-inspect")
    reference = make_source(kind=SourceKind.REFERENCE, relative_path="batch/part.png")
    inline = make_source(kind=SourceKind.INLINE, relative_path="ab/abcd.png")
    (home / "inbox" / "batch").mkdir(parents=True, exist_ok=True)
    (home / "inbox" / "batch" / "part.png").write_bytes(b"x")

    assert running._input_path(route, reference) == home / "inbox" / "batch" / "part.png"
    assert running._input_path(route, inline) == home / "staging" / "adhoc" / "ab" / "abcd.png"


def test_evidence_outside_every_root_is_named_by_its_file(running, home):
    route = running._config.route("clearance-cam-01")

    assert running._evidence_relative(route, Path("/elsewhere/cap.inference.json")) == (
        "cap.inference.json"
    )


def test_reconciling_reports_the_state_each_open_intent_settled_into(running, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    for _ in range(50):
        running._scheduler.run_once()
        if not running._scheduler.queued():
            break
    occupied = home / "processed" / "cap.png"
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_bytes(b"an unrelated file")
    running._publisher.drain_once()
    inference_id = running.list_jobs(None, [JobState.CLEANUP_FAILED.value], None, 5)[0][0][
        "inferenceId"
    ]

    occupied.unlink()
    outcome = running.reconcile(None)

    assert outcome["reconciled"] == {inference_id: "COMPLETED"}
    assert running.reconcile("adhoc-inspect")["reconciled"] == {}


# -- a degraded route says why it is degraded (DESIGN.md 12.3) -----------------------------------


def _refuse_warmup(app, message: str) -> None:
    """Make every warmup on this component fail the way a provider-policy refusal does.

    Args:
        app: The component.
        message: The detail the failure carries.
    """
    from image_processor.artifacts import ArtifactError

    def _warm(entry, bundle):
        raise ArtifactError("WARMUP_FAILED", message)

    app._artifacts.warm = _warm


def test_a_route_degraded_by_its_model_carries_the_failure_as_its_reason(app, gg):
    app._metrics.define()
    app._supervisor.start()
    _refuse_warmup(app, "PROVIDER_CPU_ONLY: the session landed on CPUExecutionProvider")
    app._artifacts.reconcile()

    app._tick()

    reason = gg.find_event("route-degraded")["context"]["reason"]
    assert "PROVIDER_CPU_ONLY" in reason
    assert "WARMUP_FAILED" in reason
    status = {entry.route_id: entry for entry in app.route_statuses()}["clearance-cam-01"]
    assert status.last_error == reason
    element = {item.instance: item for item in gg.connectivity_provider()}["clearance-cam-01"]
    assert element.detail == reason


def test_a_warmup_mismatch_reaches_the_operator_verbatim(app, gg):
    app._metrics.define()
    app._supervisor.start()
    _refuse_warmup(app, "WARMUP_MISMATCH: golden sample 1 is outside the declared tolerance")
    app._artifacts.reconcile()
    app._tick()

    assert "WARMUP_MISMATCH" in gg.find_event("route-degraded")["context"]["reason"]


def test_a_successful_activation_clears_the_reason(app, gg):
    app._metrics.define()
    app._supervisor.start()
    warm = app._artifacts.warm
    _refuse_warmup(app, "PROVIDER_CPU_ONLY: the session landed on CPUExecutionProvider")
    app._artifacts.reconcile()
    app._tick()
    assert app._artifacts.route_error("clearance-cam-01")

    app._artifacts.warm = warm
    # force, because the failed generation is inside its retry backoff; this is what
    # reload-model-catalog does for an operator who has fixed the machine
    assert app._artifacts.reconcile(force=True) >= 1

    assert app._artifacts.route_error("clearance-cam-01") is None
    status = {entry.route_id: entry for entry in app.route_statuses()}["clearance-cam-01"]
    assert status.last_error is None
    assert status.connected is True
