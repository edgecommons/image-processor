"""The authoritative spool walk: what it accepts, what it refuses, and what it announces once."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_processor.types import SourceKind
from image_processor.sources import staging
from image_processor.sources.spool import (
    DEFAULT_INCLUDE,
    SpoolError,
    SpoolSource,
    compile_glob,
)
from tests.sources.conftest import FakeObserver, sha256_of, spool_route, write_capture

DATA = b"one complete image worth of bytes"
OTHER = b"a second image, different bytes entirely"


def source_for(tmp_path, events, **kwargs):
    """Build a spool source over ``tmp_path`` with a camera-sidecar route by default."""
    route_kwargs = {
        key: kwargs.pop(key)
        for key in ("include", "exclude", "mode", "marker_suffix", "quiet_secs", "camera")
        if key in kwargs
    }
    return SpoolSource(spool_route(tmp_path, **route_kwargs), events, **kwargs)


def test_rescan_announces_a_finished_capture_once(tmp_path, events):
    write_capture(tmp_path, "2026/08/22/frame.jpg", DATA)
    source = source_for(tmp_path, events)

    assert source.rescan() == 1
    assert source.rescan() == 0

    route_id, identity, staged_path = events.discovered_calls[0]
    assert route_id == "clearance-cam-01"
    assert staged_path is None
    assert identity.kind is SourceKind.SPOOL
    assert identity.relative_path == "2026/08/22/frame.jpg"
    assert identity.sha256 == sha256_of(DATA)
    assert identity.capture_id == "cap_018f9c2b0001"


def test_the_same_path_rewritten_with_new_bytes_is_a_new_input(tmp_path, events):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    assert source.rescan() == 1

    write_capture(tmp_path, "frame.jpg", OTHER)

    assert source.rescan() == 1
    assert [call[1].sha256 for call in events.discovered_calls] == [
        sha256_of(DATA),
        sha256_of(OTHER),
    ]


def test_priming_from_the_ledger_suppresses_rediscovery(tmp_path, events):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    source.prime([("frame.jpg", sha256_of(DATA))])

    assert source.rescan() == 0
    assert events.discovered_calls == []
    assert source.seen() == {("frame.jpg", sha256_of(DATA))}


def test_the_walk_skips_the_sidecar_and_the_cameras_hidden_partials(tmp_path, events):
    write_capture(tmp_path, "frame.jpg", DATA)
    (tmp_path / ".camera-adapter-cap_1.image.partial").write_bytes(b"half an image")
    (tmp_path / ".camera-adapter-cap_1.sidecar.partial").write_bytes(b"{}")
    (tmp_path / "frame.jpg.inference.json").write_text("{}", encoding="utf-8")
    source = source_for(tmp_path, events, include=("**/*",))

    assert source.rescan() == 1
    assert events.paths == ["frame.jpg"]
    assert events.invalid_calls == []


def test_include_and_exclude_apply_to_the_normalized_relative_path(tmp_path, events):
    for relative in ("keep/a.jpg", "keep/b.png", "drafts/c.jpg", "keep/nested/d.jpg"):
        write_capture(tmp_path, relative, DATA + relative.encode())
    source = source_for(
        tmp_path,
        events,
        include=("**/*.jpg", "**/*.png"),
        exclude=("drafts/**", "**/nested/**"),
    )

    source.rescan()

    assert sorted(events.paths) == ["keep/a.jpg", "keep/b.png"]


@pytest.mark.parametrize(
    "pattern, path, matches",
    [
        ("**/*.jpg", "a.jpg", True),
        ("**/*.jpg", "2026/08/a.jpg", True),
        ("**/*.jpg", "a.png", False),
        ("*.jpg", "2026/a.jpg", False),
        ("2026/**", "2026/08/a.jpg", True),
        ("cam-0?.jpg", "cam-01.jpg", True),
        ("cam-0?.jpg", "cam-011.jpg", False),
        ("cam-[01]*.jpg", "cam-1x.jpg", True),
        ("cam-[!01]*.jpg", "cam-1x.jpg", False),
        ("cam-[01.jpg", "cam-[01.jpg", True),
        ("**/*", "any/thing.tiff", True),
    ],
)
def test_compile_glob_matches_the_configuration_grammar(pattern, path, matches):
    assert bool(compile_glob(pattern).match(path)) is matches


def test_the_walk_refuses_a_file_that_is_not_a_regular_file(tmp_path, events):
    write_capture(tmp_path, "good.jpg", DATA)
    fifo = tmp_path / "pipe.jpg"
    fifo.write_bytes(DATA)
    real_lstat = os.lstat

    def fake_lstat(path):
        result = real_lstat(path)
        if Path(path).name == "pipe.jpg":
            import stat as stat_module

            return SimpleNamespace(
                st_mode=stat_module.S_IFIFO | 0o666,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
            )
        return result

    source = source_for(tmp_path, events)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(staging, "lstat", fake_lstat)
        source.rescan()

    assert events.paths == ["good.jpg"]
    assert events.invalid_calls == [("clearance-cam-01", "pipe.jpg", "NOT_REGULAR_FILE")]


def test_the_walk_refuses_a_reparse_point_without_descending_it(tmp_path, events):
    """A junction is refused through the stat seam, which Windows CI can exercise."""
    write_capture(tmp_path, "good.jpg", DATA)
    (tmp_path / "junction.jpg").write_bytes(DATA)
    real_lstat = os.lstat

    def fake_lstat(path):
        result = real_lstat(path)
        if Path(path).name == "junction.jpg":
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_reparse_tag=0xA0000003,
            )
        return result

    source = source_for(tmp_path, events)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(staging, "lstat", fake_lstat)
        source.rescan()

    assert events.paths == ["good.jpg"]
    assert events.reasons == ["REPARSE_POINT"]


def test_a_refusal_is_reported_once_however_often_the_walk_runs(tmp_path, events):
    (tmp_path / "pipe.jpg").write_bytes(DATA)
    source = source_for(tmp_path, events)
    with pytest.MonkeyPatch.context() as patch:
        import stat as stat_module

        patch.setattr(
            staging,
            "lstat",
            lambda path: SimpleNamespace(st_mode=stat_module.S_IFIFO, st_size=0, st_mtime_ns=0),
        )
        source.rescan()
        source.rescan()

    assert len(events.invalid_calls) == 1
    assert source.rejected_count == 1


def test_the_walk_refuses_a_symlinked_file_and_never_follows_it(tmp_path, events):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"not ours to read")
    root = tmp_path / "spool"
    root.mkdir()
    write_capture(root, "good.jpg", DATA)
    try:
        os.symlink(outside / "secret.jpg", root / "escape.jpg")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account cannot create symlinks: {exc}")
    source = SpoolSource(spool_route(root), events)

    source.rescan()

    assert events.paths == ["good.jpg"]
    assert events.invalid_calls == [("clearance-cam-01", "escape.jpg", "SYMLINK")]


def test_the_walk_does_not_descend_a_symlinked_directory(tmp_path, events):
    outside = tmp_path / "outside"
    outside.mkdir()
    write_capture(outside, "secret.jpg", DATA)
    root = tmp_path / "spool"
    root.mkdir()
    write_capture(root, "good.jpg", OTHER)
    try:
        os.symlink(outside, root / "elsewhere", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account cannot create symlinks: {exc}")
    source = SpoolSource(spool_route(root), events)

    source.rescan()

    assert events.paths == ["good.jpg"]


def test_a_root_that_does_not_exist_yet_is_not_an_error(tmp_path, events):
    source = SpoolSource(spool_route(tmp_path / "not-yet"), events)

    assert source.rescan() == 0
    assert events.discovered_calls == []


def test_a_file_that_changes_while_it_is_hashed_waits_for_the_next_walk(tmp_path, events):
    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)
    (tmp_path / "frame.jpg.done").write_text("", encoding="utf-8")
    source = source_for(tmp_path, events, mode="marker", marker_suffix=".done")
    real_sha256_file = staging.sha256_file

    def hash_then_grow(path, chunk=1 << 20):
        digest = real_sha256_file(path, chunk)
        target.write_bytes(DATA + b" and more arrived mid-read")
        os.utime(target, ns=(0, 0))
        return digest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("image_processor.sources.spool.sha256_file", hash_then_grow)
        assert source.rescan() == 0

    assert events.discovered_calls == []
    assert source.rescan() == 1


def test_a_route_with_no_root_is_refused(events):
    with pytest.raises(SpoolError) as caught:
        SpoolSource(SimpleNamespace(id="r", source=SimpleNamespace(kind="spool")), events)
    assert caught.value.code == "SPOOL_ROUTE_INCOMPLETE"


def test_a_route_with_no_id_is_refused(tmp_path, events):
    with pytest.raises(SpoolError):
        SpoolSource(SimpleNamespace(id="", source=SimpleNamespace(root=str(tmp_path))), events)


def test_a_route_given_as_plain_mappings_works_the_same(tmp_path, events):
    write_capture(tmp_path, "frame.jpg", DATA)
    route = {
        "id": "mapping-route",
        "source": {
            "kind": "spool",
            "root": str(tmp_path),
            "include": ["**/*.jpg"],
            "readiness": {"mode": "cameraSidecar"},
        },
    }

    source = SpoolSource(route, events)

    assert source.rescan() == 1
    assert source.include == ("**/*.jpg",)
    assert events.discovered_calls[0][0] == "mapping-route"


def test_a_route_naming_no_include_takes_everything(tmp_path, events):
    write_capture(tmp_path, "frame.tiff", DATA)
    source = SpoolSource(
        {"id": "r", "source": {"root": str(tmp_path), "readiness": {"mode": "cameraSidecar"}}},
        events,
    )

    assert source.include == DEFAULT_INCLUDE
    assert source.rescan() == 1


# -- the camera hint ---------------------------------------------------------------------------


def test_a_verified_hint_admits_the_image_without_a_walk(tmp_path, events):
    body = write_capture(tmp_path, "2026/08/22/frame.jpg", DATA, sidecar=False)
    source = source_for(tmp_path, events)

    source.on_hint(body)

    assert source.hints_accepted == 1
    assert source.rescans == 0
    identity = events.discovered_calls[0][1]
    assert identity.relative_path == "2026/08/22/frame.jpg"
    assert identity.sha256 == sha256_of(DATA)
    assert identity.capture_id == body["captureId"]
    assert identity.captured_at_ms == 1787393704512


def test_a_hint_and_a_walk_announce_the_same_image_once(tmp_path, events):
    body = write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)

    source.on_hint(body)

    assert source.rescan() == 0
    assert len(events.discovered_calls) == 1


def test_a_hint_never_follows_the_absolute_path_it_carries(tmp_path, events):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "frame.jpg").write_bytes(b"the camera's own copy, not ours")
    root = tmp_path / "spool"
    root.mkdir()
    body = write_capture(root, "frame.jpg", DATA, sidecar=False)
    body["image"]["absolutePath"] = str(elsewhere / "frame.jpg")
    body["image"]["fileUri"] = (elsewhere / "frame.jpg").as_uri()
    source = SpoolSource(spool_route(root), events)

    source.on_hint(body)

    identity = events.discovered_calls[0][1]
    assert identity.sha256 == sha256_of(DATA)
    assert (root / identity.relative_path).read_bytes() == DATA


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["image"].update(bytes=len(DATA) + 1),
        lambda body: body["image"].update(sha256="0" * 64),
    ],
)
def test_a_hint_that_does_not_verify_admits_nothing(tmp_path, events, mutate):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    mutate(body)
    source = source_for(tmp_path, events)

    source.on_hint(body)

    assert events.discovered_calls == []
    assert source.hints_rejected == 1


def test_a_hint_for_a_file_that_has_not_landed_is_simply_early(tmp_path, events):
    from tests.sources.conftest import terminal_body

    source = source_for(tmp_path, events)

    source.on_hint(terminal_body("frame.jpg", DATA, root=tmp_path))

    assert events.discovered_calls == []
    assert events.invalid_calls == []
    assert source.hints_unmapped == 1


def test_a_hint_naming_a_path_outside_the_root_is_refused(tmp_path, events):
    from tests.sources.conftest import terminal_body

    body = terminal_body("../../etc/passwd", DATA, root=tmp_path)
    source = source_for(tmp_path, events)

    source.on_hint(body)

    assert source.hints_rejected == 1
    assert events.reasons == ["PATH_ESCAPE"]


def test_a_terminal_without_an_image_is_not_a_hint(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_hint({"schemaVersion": 1, "captureId": "cap_1", "failure": {"code": "TIMEOUT"}})
    source.on_hint("not a body at all")

    assert events.discovered_calls == []
    assert source.hints_unmapped == 1
    assert source.hints_rejected == 1


# -- the watchdog nudge ------------------------------------------------------------------------


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it holds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_filesystem_notification_makes_the_walk_happen(tmp_path, events):
    observer = FakeObserver()
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.01,
        rescan_interval_secs=3600,
    )
    source.start()
    try:
        assert observer.started
        assert observer.watched == [(str(source.root), True)]

        write_capture(tmp_path, "frame.jpg", DATA)
        observer.fire()

        assert wait_for(lambda: events.paths == ["frame.jpg"])
    finally:
        source.stop()

    assert observer.stopped
    assert source.nudges >= 1


def test_repeated_notifications_coalesce_into_one_walk(tmp_path, events):
    observer = FakeObserver()
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.02,
        rescan_interval_secs=3600,
    )
    source.start()
    try:
        for _ in range(20):
            observer.fire()
        assert wait_for(lambda: events.paths == ["frame.jpg"])
    finally:
        source.stop()

    assert source.nudges == 20
    assert source.rescans < 20


def test_the_periodic_walk_runs_with_no_notification_at_all(tmp_path, events):
    observer = FakeObserver()
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.01,
        rescan_interval_secs=0.0,
    )
    source.start()
    try:
        assert wait_for(lambda: events.paths == ["frame.jpg"])
    finally:
        source.stop()

    assert source.nudges == 0


def test_starting_twice_is_harmless_and_a_failing_observer_does_not_stop_the_walk(
    tmp_path, events
):
    write_capture(tmp_path, "frame.jpg", DATA)

    def refuse():
        raise OSError("no inotify watches left")

    source = source_for(
        tmp_path,
        events,
        observer_factory=refuse,
        debounce_secs=0.01,
        rescan_interval_secs=0.0,
    )
    source.start()
    source.start()
    try:
        assert wait_for(lambda: events.paths == ["frame.jpg"])
    finally:
        source.stop()
        source.stop()


def test_a_failing_walk_does_not_kill_the_watcher_thread(tmp_path, events, monkeypatch):
    observer = FakeObserver()
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.01,
        rescan_interval_secs=3600,
    )
    calls = []

    def explode():
        calls.append(1)
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(source, "rescan", explode)
    source.start()
    try:
        observer.fire()
        assert wait_for(lambda: calls)
        observer.fire()
        assert wait_for(lambda: len(calls) >= 2)
    finally:
        source.stop()


# -- the paths a walk takes when the filesystem does not cooperate -----------------------------


def test_an_image_whose_sidecar_has_not_arrived_is_held_not_refused(tmp_path, events):
    (tmp_path / "frame.jpg").write_bytes(DATA)
    source = source_for(tmp_path, events)

    assert source.rescan() == 0
    assert events.discovered_calls == []
    assert events.invalid_calls == []

    write_capture(tmp_path, "frame.jpg", DATA)
    assert source.rescan() == 1


def test_a_hint_for_an_image_the_walk_already_took_announces_nothing_new(tmp_path, events):
    body = write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    assert source.rescan() == 1

    source.on_hint(body)

    assert len(events.discovered_calls) == 1
    assert source.hints_accepted == 1


def test_a_hint_naming_something_that_is_not_a_regular_file_is_refused(tmp_path, events):
    from tests.sources.conftest import terminal_body

    (tmp_path / "a-directory").mkdir()
    source = source_for(tmp_path, events)

    source.on_hint(terminal_body("a-directory", DATA, root=tmp_path))

    assert source.hints_rejected == 1
    assert events.reasons == ["DIRECTORY"]


def test_a_hint_for_a_file_that_moves_mid_verification_admits_nothing(
    tmp_path, events, monkeypatch
):
    body = write_capture(tmp_path, "frame.jpg", DATA, sidecar=False)
    source = source_for(tmp_path, events)
    import image_processor.sources.spool as spool_module

    real = spool_module.verify_declared_image

    def verify_then_touch(path, image):
        answer = real(path, image)
        os.utime(path, ns=(0, 0))
        return answer

    monkeypatch.setattr(spool_module, "verify_declared_image", verify_then_touch)

    source.on_hint(body)

    assert events.discovered_calls == []
    assert source.hints_rejected == 1


def test_a_directory_the_walk_cannot_list_is_stepped_over(tmp_path, events, monkeypatch):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)

    def refuse(path):
        raise PermissionError("the spool is not readable by this account")

    monkeypatch.setattr("image_processor.sources.spool.os.scandir", refuse)

    assert source.rescan() == 0


def test_a_file_that_vanishes_between_listing_and_stat_is_skipped(tmp_path, events):
    write_capture(tmp_path, "gone.jpg", DATA)
    write_capture(tmp_path, "stays.jpg", OTHER)
    source = source_for(tmp_path, events)
    real_lstat = os.lstat

    def fake_lstat(path):
        if Path(path).name == "gone.jpg":
            raise FileNotFoundError(str(path))
        return real_lstat(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(staging, "lstat", fake_lstat)
        source.rescan()

    assert events.paths == ["stays.jpg"]
    assert events.invalid_calls == []


def test_a_file_that_disappears_before_it_is_judged_is_skipped(tmp_path, events, monkeypatch):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    real = staging.stat_signature

    def vanish(path):
        (tmp_path / "frame.jpg").unlink(missing_ok=True)
        return None

    monkeypatch.setattr("image_processor.sources.spool.stat_signature", vanish)

    assert source.rescan() == 0


def test_a_path_that_escapes_between_the_walk_and_the_read_is_refused(tmp_path, events):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    real = staging.realpath

    def resolve(path):
        if Path(path).name == "frame.jpg":
            return str(tmp_path.parent / "elsewhere" / "frame.jpg")
        return real(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(staging, "realpath", resolve)
        assert source.rescan() == 0

    assert events.reasons == ["PATH_ESCAPE"]


def test_a_file_that_cannot_be_hashed_waits(tmp_path, events, monkeypatch):
    (tmp_path / "frame.jpg").write_bytes(DATA)
    (tmp_path / "frame.jpg.done").write_text("", encoding="utf-8")
    source = source_for(tmp_path, events, mode="marker", marker_suffix=".done")

    def refuse(path, chunk=1 << 20):
        raise PermissionError("another writer holds the file")

    monkeypatch.setattr("image_processor.sources.spool.sha256_file", refuse)

    assert source.rescan() == 0
    assert events.discovered_calls == []


def test_a_stability_route_forgets_the_timer_of_a_file_that_is_gone(tmp_path, events):
    from image_processor.sources.readiness import StabilityReadiness

    target = tmp_path / "frame.jpg"
    target.write_bytes(DATA)

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    source = source_for(tmp_path, events, mode="stability", quiet_secs=1, clock=clock)

    assert source.rescan() == 0
    strategy = source.readiness.strategy
    assert isinstance(strategy, StabilityReadiness)
    assert set(strategy._first_seen) == {"frame.jpg"}

    clock.now = 5
    assert source.rescan() == 1

    target.unlink()
    source.rescan()

    assert strategy._first_seen == {}


def test_an_observer_that_refuses_to_stop_does_not_break_shutdown(tmp_path, events):
    class Stubborn(FakeObserver):
        def stop(self):
            raise RuntimeError("the observer thread is wedged")

    observer = Stubborn()
    source = source_for(
        tmp_path, events, observer_factory=lambda: observer, debounce_secs=0.01
    )
    source.start()

    source.stop()

    assert source._observer is None


def test_an_entry_the_walk_cannot_place_under_the_root_is_skipped(tmp_path, events, monkeypatch):
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(tmp_path, events)
    import image_processor.sources.spool as spool_module

    real = spool_module.relative_to_root

    def refuse(root, path):
        if Path(path).name == "frame.jpg":
            raise ValueError("not under the root")
        return real(root, path)

    monkeypatch.setattr(spool_module, "relative_to_root", refuse)

    assert source.rescan() == 0


def test_the_watcher_does_not_walk_while_it_is_neither_nudged_nor_due(tmp_path, events):
    observer = FakeObserver()
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.01,
        rescan_interval_secs=3600,
    )
    source.start()
    try:
        time.sleep(0.15)
        assert source.rescans == 0
        assert events.discovered_calls == []
    finally:
        source.stop()


def test_notifications_arriving_during_the_debounce_keep_extending_it(tmp_path, events):
    import threading

    observer = FakeObserver()
    write_capture(tmp_path, "frame.jpg", DATA)
    source = source_for(
        tmp_path,
        events,
        observer_factory=lambda: observer,
        debounce_secs=0.05,
        rescan_interval_secs=3600,
    )
    stop_firing = threading.Event()

    def fire_continuously():
        while not stop_firing.is_set():
            observer.fire()
            time.sleep(0.005)

    firing = threading.Thread(target=fire_continuously, daemon=True)
    source.start()
    firing.start()
    try:
        time.sleep(0.3)
    finally:
        stop_firing.set()
        firing.join(2)
        source.stop()

    assert source.nudges > 10
    assert source.rescans <= source.nudges
