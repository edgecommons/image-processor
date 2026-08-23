"""The artifact manager: stage, warm, activate, and what happens when one of those fails."""

from __future__ import annotations

import pytest

from image_processor.artifacts import ArtifactError, ArtifactManager, family_validator
from image_processor.bundles import BundleCache
from image_processor.config import parse_component_config
from image_processor.engine.protocol import LoadFailed, Loaded
from image_processor.ledger import Ledger
from image_processor.outputs.events import RouteEvents
from tests.wiring.conftest import FakeGg, config_document


class FakeCell:
    """A cell handle that answers a load however the test wants."""

    def __init__(self, cell_id: str = "cpu-0", alive: bool = True) -> None:
        self.cell_id = cell_id
        self.device = None
        self._alive = alive
        self.loads: list = []

    def is_alive(self) -> bool:
        """Whether the child is up."""
        return self._alive


class FakeSupervisor:
    """A supervisor whose cells answer with a scripted reply."""

    def __init__(self, replies=None, cells=None) -> None:
        self._cells = cells if cells is not None else [FakeCell()]
        self.replies = list(replies or [])
        self.calls: list = []

    def cells(self) -> list:
        """The cell handles."""
        return list(self._cells)

    def call(self, cell, message, timeout_s=None):
        """Answer one call with the next scripted reply."""
        cell.loads.append(message)
        self.calls.append((cell.cell_id, message))
        reply = self.replies.pop(0) if self.replies else _loaded(message.digest)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _loaded(digest: str) -> Loaded:
    """A successful load."""
    return Loaded(
        digest=digest,
        providers_assigned=("CPUExecutionProvider",),
        load_ms=12.0,
        device_mib=0,
        warmup_samples=2,
    )


@pytest.fixture
def parts(home, corpus, tmp_path):
    """A parsed configuration, a real cache, and a real ledger."""
    document = config_document(home, corpus, trigger=False)
    component = document["component"]
    config = parse_component_config(component["global"], component["instances"])
    cache = BundleCache(config.paths.model_cache)
    with Ledger(config.paths.state_db, synchronous="OFF") as ledger:
        yield config, cache, ledger


def test_staging_verifies_the_bundle_and_promotes_it(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())

    bundle = manager.stage(config.models[0])

    assert bundle.digest == config.models[0].digest
    assert bundle.manifest.model_id == "synthetic-anomaly-scalar"
    assert cache.get(bundle.digest) is not None
    assert manager.counters["staged"] == 1


def test_staging_a_cached_generation_does_not_fetch_it_again(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())
    manager.stage(config.models[0])

    manager.stage(config.models[0])

    assert manager.counters["staged"] == 1


def test_a_bundle_whose_digest_does_not_match_is_refused(parts):
    from dataclasses import replace

    config, cache, ledger = parts
    entry = replace(config.models[0], digest="sha256:" + "cd" * 32)
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())

    with pytest.raises(ArtifactError) as failure:
        manager.stage(entry)

    assert failure.value.code == "DIGEST_MISMATCH"


def test_activation_switches_the_route_generation_only_after_warmup(parts):
    config, cache, ledger = parts
    supervisor = FakeSupervisor()
    activated: list = []
    manager = ArtifactManager(
        config, cache, ledger, supervisor, on_activated=lambda *args: activated.append(args)
    )

    assert manager.reconcile() == 1

    digest = config.models[0].digest
    assert ledger.route_generation("clearance-cam-01") == (digest, digest)
    assert supervisor.calls[0][1].warmup is True
    assert activated == [("clearance-cam-01", None, digest)]
    assert manager.counters["activated"] == 1


def test_a_second_pass_changes_nothing(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())
    manager.reconcile()

    assert manager.reconcile() == 0
    assert manager.counters["activated"] == 1


def test_a_failed_warmup_leaves_the_route_on_its_last_known_good_model(parts, gg):
    config, cache, ledger = parts
    events = RouteEvents(gg)
    good = FakeSupervisor()
    manager = ArtifactManager(config, cache, ledger, good, events=events)
    manager.reconcile()
    digest = config.models[0].digest

    # a new desired generation that will not warm
    from dataclasses import replace

    other = "sha256:" + "ef" * 32
    route = replace(config.routes[0], model_ref=replace(config.routes[0].model_ref, digest=other))
    entry = replace(config.models[0], digest=other)
    manager.adopt(replace(config, routes=(route,), models=(entry,)))
    manager.supervisor = FakeSupervisor(
        [LoadFailed(digest=other, error="the graph is unusable", error_class="permanent", code="LOAD_FAILED")]
    )

    assert manager.reconcile() == 0

    desired, active = ledger.route_generation("clearance-cam-01")
    assert desired == other, "the desire is durable, so the route reports STAGING"
    assert active == digest, "the last known good model keeps serving"
    assert gg.find_event("model-staging-failed") is not None


def test_a_generation_that_warms_but_will_not_load_reports_the_warmup_failure(parts, gg):
    config, cache, ledger = parts
    supervisor = FakeSupervisor(
        [LoadFailed(digest=config.models[0].digest, error="no session", error_class="transient", code="LOAD_FAILED")]
    )
    manager = ArtifactManager(config, cache, ledger, supervisor, events=RouteEvents(gg))

    assert manager.reconcile() == 0

    assert manager.counters["warmupFailures"] == 1
    assert gg.find_event("model-warmup-failed") is not None
    assert ledger.route_generation("clearance-cam-01")[1] is None


def test_a_failed_generation_backs_off_before_it_is_tried_again(parts):
    config, cache, ledger = parts
    supervisor = FakeSupervisor([RuntimeError("the cell is wedged")])
    ticks = iter([0.0, 1.0, 1000.0, 1000.0, 1000.0])
    manager = ArtifactManager(
        config, cache, ledger, supervisor, clock=lambda: next(ticks), retry_backoff_secs=60
    )

    manager.reconcile()
    assert len(supervisor.calls) == 1

    manager.reconcile()
    assert len(supervisor.calls) == 1, "the backoff has not elapsed"

    manager.reconcile()
    assert len(supervisor.calls) == 2, "past the backoff it is tried again"


def test_forcing_a_pass_ignores_the_backoff(parts):
    config, cache, ledger = parts
    supervisor = FakeSupervisor([RuntimeError("wedged")])
    manager = ArtifactManager(config, cache, ledger, supervisor)
    manager.reconcile()

    manager.reconcile(force=True)

    assert len(supervisor.calls) == 2


def test_no_executor_at_all_is_reported_rather_than_assumed(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor(cells=[]))

    with pytest.raises(ArtifactError) as failure:
        manager.warm(config.models[0], manager.stage(config.models[0]))

    assert failure.value.code == "NO_EXECUTOR"


def test_a_dead_cell_is_skipped_and_a_live_one_answers(parts):
    config, cache, ledger = parts
    dead, live = FakeCell("cpu-0", alive=False), FakeCell("cpu-1")
    supervisor = FakeSupervisor(cells=[dead, live])
    manager = ArtifactManager(config, cache, ledger, supervisor)

    manager.warm(config.models[0], manager.stage(config.models[0]))

    assert dead.loads == []
    assert len(live.loads) == 1


def test_the_pins_keep_every_generation_a_route_could_still_need(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())
    manager.reconcile()

    pinned = manager.pinned()

    assert config.models[0].digest in pinned
    assert manager.collect() == ()


def test_the_catalog_reports_what_is_staged_and_what_is_active(parts):
    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor())
    manager.reconcile()

    entry = manager.status()[0]

    assert entry["id"] == "synthetic-anomaly-scalar"
    assert entry["staged"] is True
    assert entry["warmed"] is True
    assert entry["activeRoutes"] == ["clearance-cam-01"]
    assert entry["stagingRoutes"] == []
    assert entry["error"] is None


def test_a_route_naming_an_unconfigured_model_degrades_rather_than_crashes(parts, gg):
    from dataclasses import replace

    config, cache, ledger = parts
    route = replace(
        config.routes[0], model_ref=replace(config.routes[0].model_ref, id="not-configured")
    )
    manager = ArtifactManager(
        replace(config, routes=(route,)), cache, ledger, FakeSupervisor(), events=RouteEvents(gg)
    )

    assert manager.reconcile() == 0
    assert gg.find_event("model-staging-failed")["context"]["error"].startswith(
        "UNRESOLVED_MODEL_REF"
    )


def test_the_family_validator_refuses_a_head_no_family_interprets():
    from dataclasses import replace as dataclass_replace

    from image_processor.engine.families import FamilyError
    from tests.engine.conftest import make_manifest

    with pytest.raises(FamilyError):
        family_validator(make_manifest(family_params={"source": "nonsense"}))


def test_the_background_thread_reconciles_and_stops(parts):
    import time

    config, cache, ledger = parts
    manager = ArtifactManager(config, cache, ledger, FakeSupervisor(), interval_secs=0.01)
    manager.start()
    try:
        deadline = time.monotonic() + 5
        while manager.counters["activated"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        manager.stop(timeout_s=5)

    assert manager.counters["activated"] == 1
