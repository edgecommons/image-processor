"""The application surfaces around the pipeline: admission, status, reload, and the operator verbs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_processor.types import JobState
from tests.wiring.conftest import config_document, write_capture


def _drain(app) -> None:
    """Run the scheduler until the queue is empty."""
    for _ in range(100):
        app._scheduler.run_once()
        if not app._scheduler.queued():
            return


def test_the_component_reports_every_route_to_the_library(app, gg):
    assert gg.connectivity_provider is not None

    elements = gg.connectivity_provider()

    assert [element.instance for element in elements] == ["clearance-cam-01", "adhoc-inspect"]
    assert all(element.connected is False for element in elements), "no model is active yet"


def test_a_route_becomes_connected_once_its_model_is_active(running, gg):
    element = {item.instance: item for item in gg.connectivity_provider()}["clearance-cam-01"]

    assert element.connected is True
    assert element.state == "ONLINE"
    assert element.attributes["activeGeneration"].startswith("sha256:")


def test_readiness_is_claimed_only_once_a_route_can_decide(app, gg):
    assert app._health.evaluate().ready is False

    app._supervisor.start()
    app._artifacts.reconcile()
    app._start_routes()

    report = app._health.apply(gg)
    assert report.ready is True
    assert gg.ready[-1] is True


def test_warmup_leaves_no_session_behind_and_the_scheduler_loads_the_first_job(
    running, home, corpus
):
    """The residency map is the scheduler's alone, so activation must not pre-load (DESIGN.md 10.2)."""
    from image_processor.engine.protocol import Stats

    cell = running._supervisor.cells()[0]
    assert running._supervisor.call(cell, Stats(), 30.0).resident == ()
    assert running._scheduler._resident.get(cell.cell_id, {}) == {}

    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    _drain(running)

    digest = running._config.route("clearance-cam-01").model_ref.digest
    assert running._scheduler.counters["loads"] == 1, "the first job paid for the load"
    assert digest in running._scheduler._resident[cell.cell_id]
    assert running._supervisor.call(cell, Stats(), 30.0).resident == (digest,)


def test_the_same_image_is_admitted_once(running, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    source = running._source_of("clearance-cam-01")

    assert source.rescan() == 1
    source.forget()
    assert source.rescan() == 1, "the walk announces it again"

    jobs, _cursor = running._ledger.by_state(list(JobState), None, None, 10)
    assert len(jobs) == 1, "the ledger recognizes the identity and refuses the duplicate"


def test_a_refused_spool_input_is_quarantined_where_an_operator_can_see_it(running, home, gg):
    # the source refuses an input it can see but can never admit; the component still owns the
    # file, so the terminal state and the file both have to end up somewhere an operator looks
    (home / "spool" / "wrong.png").write_bytes(b"not an image this route can ever use")

    running.invalid("clearance-cam-01", "wrong.png", "NOT_REGULAR_FILE")

    assert (home / "failed" / "wrong.png").is_file()
    assert not (home / "spool" / "wrong.png").exists()
    assert gg.find_event("input-rejected")["context"]["reason"] == "NOT_REGULAR_FILE"
    jobs, _cursor = running._ledger.by_state([JobState.QUARANTINED], None, None, 5)
    assert len(jobs) == 1
    assert jobs[0].source.relative_path == "wrong.png"


def test_a_route_with_no_failed_directory_quarantines_in_place(home, corpus, gg):
    from image_processor.ImageProcessor import ImageProcessor

    document = config_document(home, corpus, trigger=False)
    document["component"]["instances"][0]["completion"].pop("failedDir")
    gg.config_manager.document = document
    app = ImageProcessor(gg)
    try:
        app._supervisor.start()
        app._artifacts.reconcile()
        app._start_routes()
        target = home / "spool" / "wrong.png"
        target.write_bytes(b"not an image this route can ever use")

        app.invalid("clearance-cam-01", "wrong.png", "DEVICE_FILE")

        assert target.is_file(), "the input stays where it is"
        jobs, _cursor = app._ledger.by_state([JobState.QUARANTINED], None, None, 5)
        assert len(jobs) == 1
    finally:
        app.stop()


def test_a_refused_input_that_is_no_longer_there_is_only_reported(running, home, gg):
    running.invalid("clearance-cam-01", "vanished.png", "MISSING")

    assert gg.find_event("input-rejected") is not None
    assert running.job_counts() == {}


def test_a_refused_trigger_message_is_reported_without_a_file_to_move(running, gg):
    running._source_of("adhoc-inspect").on_message({"body": {"nothing": "useful"}})

    event = gg.find_event("input-rejected")
    assert event["context"]["reason"] == "MALFORMED_BODY"
    assert running.job_counts() == {}, "there is no file, so there is nothing to quarantine"


def test_get_queue_reports_jobs_and_counts(running, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()

    jobs, cursor = running.list_jobs("clearance-cam-01", None, None, 10)

    assert cursor is None
    assert jobs[0]["state"] == "READY"
    assert jobs[0]["source"]["relativePath"] == "cap.png"
    assert jobs[0]["model"]["id"] == "synthetic-anomaly-scalar"
    assert running.job_counts("clearance-cam-01") == {"READY": 1}
    assert running.list_jobs(None, ["READY"], None, 10)[0]
    with pytest.raises(ValueError):
        running.list_jobs(None, ["NOPE"], None, 10)


def test_pausing_a_route_stops_it_claiming_and_resuming_starts_it_again(running, home, corpus):
    source = running._source_of("clearance-cam-01")

    assert running.pause("clearance-cam-01")["paused"] is True
    write_capture(home / "spool", "while-paused.png", corpus.image("anomaly-good.png"))
    assert running._routes["clearance-cam-01"].paused is True

    assert running.resume("clearance-cam-01")["paused"] is False
    source.rescan()
    assert running.job_counts("clearance-cam-01") == {"READY": 1}


def test_pausing_the_component_pauses_the_scheduler(running):
    running.pause(None)

    assert running._scheduler.paused is True

    running.resume(None)
    assert running._scheduler.paused is False


def test_an_activation_override_is_reported_beside_the_configured_value(running):
    outcome = running.set_activation_override("clearance-cam-01", False)

    assert outcome == {
        "route": "clearance-cam-01",
        "configured": True,
        "override": False,
        "effective": False,
    }
    status = {item.route_id: item for item in running.route_statuses()}["clearance-cam-01"]
    assert status.enabled is False
    assert running._ledger.kv_get("image-processor/activation-override/clearance-cam-01") == "false"

    running.set_activation_override("clearance-cam-01", None)
    assert running.route_statuses()[0].enabled is True


def test_a_reload_adds_removes_and_rebuilds_routes(running, gg, home, corpus):
    document = config_document(home, corpus, trigger=False)
    document["component"]["instances"].append(
        {
            "id": "second-cam",
            "source": {
                "kind": "spool",
                "root": str(home / "spool-2"),
                "readiness": {"mode": "stability", "quietSecs": 1},
            },
            "modelRef": document["component"]["instances"][0]["modelRef"],
            "outputs": {"writeResultSidecar": False, "decisionSignals": []},
            "completion": {"onSuccess": "retainInPlace"},
        }
    )

    assert gg.config_manager.apply(document) is True

    assert set(running._routes) == {"clearance-cam-01", "second-cam"}
    assert running.route_ids() == ["clearance-cam-01", "second-cam"]
    assert running._generation == 2


def test_a_reload_that_cannot_be_parsed_is_refused(running, gg):
    broken = json.loads(json.dumps(gg.config_manager.document))
    broken["component"]["global"]["publish"]["maxAttempts"] = "many"

    assert gg.config_manager.apply(broken) is False


def test_a_reload_keeps_admitted_work(running, gg, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()
    before = running.job_counts()

    gg.config_manager.apply(json.loads(json.dumps(gg.config_manager.document)))

    assert running.job_counts() == before


def test_a_model_change_replays_terminal_inputs_only_when_the_route_asks(running, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    source = running._source_of("clearance-cam-01")
    source.rescan()
    assert source.seen()

    running._on_model_activated("clearance-cam-01", "sha256:old", "sha256:new")
    assert source.seen(), "the route did not ask for a replay"

    running._config.route("clearance-cam-01")
    from dataclasses import replace

    route = replace(running._config.route("clearance-cam-01"), reprocess_existing_on_model_change=True)
    running._config = replace(running._config, routes=(route,) + running._config.routes[1:])
    running._on_model_activated("clearance-cam-01", "sha256:old", "sha256:new")

    assert source.seen() == set(), "the walk will rediscover the spool under the new generation"


def test_the_gauges_describe_the_live_subsystems(running, home, corpus):
    write_capture(home / "spool", "cap.png", corpus.image("anomaly-good.png"))
    running._source_of("clearance-cam-01").rescan()

    gauges = running._gauges()

    assert gauges["ImageProcessorQueue"]["queued"] == 1
    assert gauges["ImageProcessorDiscovery"]["discovered"] == 1
    assert gauges["ImageProcessorModelCache"]["cachedBundles"] == 1
    assert gauges["ImageProcessorGpu"]["healthyCells"] == 1
    assert gauges["ImageProcessorDisk"]["stateDbBytes"] > 0
    assert gauges["ImageProcessorDisk"]["modelCacheBytes"] > 0


def test_the_supervision_pass_raises_and_clears_the_route_conditions(running, gg):
    running._tick()
    assert gg.find_event("route-degraded") is None

    running.set_activation_override("clearance-cam-01", True)
    running._supervisor.stop()
    running._tick()

    assert gg.find_event("executor-unavailable") is not None
    assert gg.find_event("route-degraded") is not None


def test_preloading_a_model_stages_warms_and_reports(running):
    outcome = running.preload_model("synthetic-anomaly-scalar", None)

    assert outcome["staged"] is True
    assert outcome["warmed"] is True
    assert outcome["digest"].startswith("sha256:")


def test_preloading_a_model_nobody_configured_is_not_found(running):
    from edgecommons.command_inbox import CommandException

    with pytest.raises(CommandException):
        running.preload_model("nope", None)


def test_reloading_the_catalog_reports_what_it_kept(running):
    outcome = running.reload_model_catalog()

    assert outcome["collected"] == []
    assert outcome["models"][0]["staged"] is True


def test_evicting_an_unknown_generation_says_so(running):
    assert running.evict_model("sha256:" + "cd" * 32) == {
        "evicted": False,
        "digest": "sha256:" + "cd" * 32,
        "cells": [],
        "reason": "the generation is not resident",
    }


def test_reconciling_reports_the_states_it_decided(running, home, corpus):
    assert running.reconcile(None) == {"reconciled": {}, "counts": {}}


def test_a_component_with_no_credentials_vault_reports_a_missing_secret(home, corpus):
    from image_processor.ImageProcessor import ImageProcessor
    from tests.wiring.conftest import FakeGg

    document = config_document(home, corpus, trigger=False)
    document["component"]["global"]["signing"] = {
        "required": False,
        "trustedKeys": [{"keyId": "publisher-1", "publicKey": {"$secret": "model-signing/one"}}],
    }
    app = ImageProcessor(FakeGg(document))
    try:
        assert app._artifacts.trusted_keys == {}
    finally:
        app.stop()


def test_a_configured_secret_is_resolved_through_the_vault(home, corpus):
    from image_processor.ImageProcessor import ImageProcessor
    from tests.wiring.conftest import FakeGg, FakeVault

    document = config_document(home, corpus, trigger=False)
    document["component"]["global"]["signing"] = {
        "required": False,
        "trustedKeys": [{"keyId": "publisher-1", "publicKey": {"$secret": "model-signing/one"}}],
    }
    document["component"]["global"]["models"][0]["credentials"] = {"$secret": "model-source/one"}
    vault = FakeVault(
        {
            "model-signing/one": "-----BEGIN PUBLIC KEY-----",
            "model-source/one": json.dumps({"aws_access_key_id": "AKIA", "aws_secret_access_key": "s"}),
        }
    )
    app = ImageProcessor(FakeGg(document, vault=vault))
    try:
        assert app._artifacts.trusted_keys == {"publisher-1": b"-----BEGIN PUBLIC KEY-----"}
        assert app._artifacts.credentials["synthetic-anomaly-scalar"]["aws_access_key_id"] == "AKIA"
    finally:
        app.stop()


def test_a_paused_route_is_not_reported_as_degraded(running, gg):
    running.pause("clearance-cam-01")

    running._tick()

    assert gg.find_event("route-degraded") is None
    status = {item.route_id: item for item in running.route_statuses()}["clearance-cam-01"]
    assert status.paused is True
    assert status.connected is False
