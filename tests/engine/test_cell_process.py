"""The parent-side handle against a real subprocess, and the cell's own request loop (LLD §6).

One spawned cell does the whole round trip -- load, infer, unload, shutdown -- on
``CPUExecutionProvider``, because ``spawn`` on Windows is the case that breaks first: the child
re-imports everything, so a message that is not picklable or an entry point that is not importable
shows up here and nowhere else.
"""

import hashlib

import pytest

from image_processor.engine.cell import (
    CellDead,
    CellError,
    CellTimeout,
    ExecutorCell,
)
from image_processor.engine.cell_main import CellConfig, CellState, serve
from image_processor.engine.protocol import (
    CPU_PROVIDER,
    CellStats,
    Infer,
    LoadFailed,
    LoadModel,
    Loaded,
    Shutdown,
    Stats,
    Unload,
    Unloaded,
)
from tests.engine.executor_support import dying_entrypoint, sleeping_entrypoint

DIGEST = "sha256:" + "e" * 64


@pytest.fixture()
def spawned():
    """A real executor cell, stopped at the end of the test."""
    cell = ExecutorCell("cpu-0", None, (CPU_PROVIDER,), call_timeout_s=120.0)
    try:
        yield cell
    finally:
        cell.stop(10.0)


def test_a_spawned_cell_loads_infers_unloads_and_shuts_down(spawned, corpus):
    bundle = corpus.expected["bundles"]["synthetic-classification-1.0.0"]
    case = bundle["cases"][0]
    image = corpus.path(case["image"])

    spawned.start()
    assert spawned.is_alive() and spawned.pid
    assert spawned.uptime_s() >= 0.0

    loaded = spawned.call(
        LoadModel(
            digest=DIGEST,
            bundle_root=str(corpus.path(bundle["path"])),
            providers=(CPU_PROVIDER,),
            allow_cpu_only=True,
        ),
        timeout_s=120.0,
    )
    assert isinstance(loaded, Loaded)
    assert loaded.providers_assigned == (CPU_PROVIDER,)

    result = spawned.call(
        Infer(
            inference_id="job-1",
            staged_path=str(image),
            sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            digest=DIGEST,
            transform_version="1",
            queue_ms=3.0,
        ),
        timeout_s=120.0,
    )
    assert result.status == "SUCCEEDED", result.error
    assert result.decision.outcome.value == case["decision"]["outcome"]
    assert result.providers == [CPU_PROVIDER]

    stats = spawned.call(Stats(), timeout_s=60.0)
    assert isinstance(stats, CellStats) and stats.resident == (DIGEST,)

    unloaded = spawned.call(Unload(DIGEST), timeout_s=60.0)
    assert isinstance(unloaded, Unloaded) and unloaded.was_resident is True

    spawned.stop(10.0)
    assert spawned.is_alive() is False
    assert spawned.pid is None


def test_a_cell_that_never_answers_is_marked_broken_at_its_deadline():
    cell = ExecutorCell("cpu-0", None, (CPU_PROVIDER,), entrypoint=sleeping_entrypoint)
    try:
        cell.start()
        with pytest.raises(CellTimeout):
            cell.call(Stats(), timeout_s=0.3)
        assert cell.broken is not None
        assert cell.is_alive() is False
        with pytest.raises(CellDead):
            cell.call(Stats(), timeout_s=0.3)
    finally:
        cell.stop(5.0)


def test_a_cell_that_dies_mid_request_raises_rather_than_hanging():
    cell = ExecutorCell("cpu-0", None, (CPU_PROVIDER,), entrypoint=dying_entrypoint)
    try:
        cell.start()
        with pytest.raises(CellDead):
            cell.call(Stats(), timeout_s=30.0)
        assert cell.is_alive() is False
    finally:
        cell.stop(5.0)


def test_a_handle_that_was_never_started_refuses_requests():
    cell = ExecutorCell("cpu-0", None, (CPU_PROVIDER,))
    with pytest.raises(CellDead):
        cell.send(Stats())
    with pytest.raises(CellError):
        cell.receive(0.1)
    cell.stop(1.0)
    assert cell.is_alive() is False
    assert repr(cell).startswith("<ExecutorCell cpu-0")


def test_a_cell_takes_one_request_at_a_time(spawned):
    spawned.start()
    spawned.send(Stats())
    with pytest.raises(CellError):
        spawned.send(Stats())
    assert isinstance(spawned.receive(60.0), CellStats)
    spawned.start()


def test_starting_a_running_cell_is_a_no_op(spawned):
    spawned.start()
    pid = spawned.pid
    spawned.start()
    assert spawned.pid == pid


def test_a_broken_handle_refuses_before_it_touches_the_pipe(spawned):
    spawned.start()
    spawned.mark_broken("a test said so")
    spawned.mark_broken("and again")
    assert spawned.broken == "a test said so"
    with pytest.raises(CellDead):
        spawned.send(Stats())


class FakeConnection:
    """A pipe end that hands over scripted requests and records the replies."""

    def __init__(self, requests, fail_on_send=False) -> None:
        """Initialize the connection.

        Args:
            requests: The requests to hand over, in order. An ``EOFError`` instance is raised
                instead of being returned.
            fail_on_send: Whether sending raises, the way a closed pipe does.
        """
        self.requests = list(requests)
        self.sent = []
        self.closed = False
        self.fail_on_send = fail_on_send

    def recv(self):
        """Return the next scripted request."""
        if not self.requests:
            raise EOFError("the parent is gone")
        message = self.requests.pop(0)
        if isinstance(message, BaseException):
            raise message
        return message

    def send(self, message) -> None:
        """Record one reply."""
        if self.fail_on_send:
            raise OSError("the pipe is closed")
        self.sent.append(message)

    def close(self) -> None:
        """Close the connection."""
        self.closed = True


def cpu_state() -> CellState:
    """A CPU cell state for the serve loop."""
    return CellState(CellConfig(cell_id="cpu-0", device_id=None, providers=(CPU_PROVIDER,)))


def test_the_request_loop_answers_until_it_is_asked_to_stop():
    connection = FakeConnection([Stats(), Unload("sha256:absent"), Shutdown(), Stats()])
    state = cpu_state()
    serve(state, connection)
    assert [type(reply).__name__ for reply in connection.sent] == ["CellStats", "Unloaded"]
    assert connection.closed is True


def test_the_request_loop_stops_when_the_parent_goes_away():
    connection = FakeConnection([Stats()])
    serve(cpu_state(), connection)
    assert len(connection.sent) == 1


def test_an_unknown_message_is_answered_rather_than_crashing_the_cell():
    connection = FakeConnection(["not a message", Shutdown()])
    serve(cpu_state(), connection)
    assert isinstance(connection.sent[0], LoadFailed)
    assert connection.sent[0].code == "UNKNOWN_MESSAGE"


def test_a_pipe_that_cannot_carry_the_reply_ends_the_loop():
    connection = FakeConnection([Stats(), Stats()], fail_on_send=True)
    serve(cpu_state(), connection)
    assert connection.sent == []


def test_a_pipe_error_while_reading_ends_the_loop():
    connection = FakeConnection([OSError("broken pipe")])
    serve(cpu_state(), connection)
    assert connection.sent == []


def test_closing_a_pipe_that_refuses_to_close_is_survivable():
    class Stubborn(FakeConnection):
        """A connection whose close raises."""

        def close(self):
            raise OSError("no")

    serve(cpu_state(), Stubborn([Shutdown()]))


class StubProcess:
    """A process handle that is alive until a test says otherwise."""

    def __init__(self, alive: bool = True) -> None:
        """Initialize the stub."""
        self.alive = alive
        self.pid = 1234

    def is_alive(self) -> bool:
        """Report whether the imagined child is running."""
        return self.alive


class RefusingConnection:
    """A pipe end that fails whichever operation a test points it at."""

    def __init__(self, on_send=None, on_poll=None, on_recv=None) -> None:
        """Initialize the connection with the errors it should raise."""
        self.on_send = on_send
        self.on_poll = on_poll
        self.on_recv = on_recv

    def send(self, message):
        """Send, or raise what the test asked for."""
        if self.on_send:
            raise self.on_send

    def poll(self, timeout=None):
        """Poll, or raise what the test asked for."""
        if self.on_poll:
            raise self.on_poll
        return self.on_recv is not None

    def recv(self):
        """Receive, or raise what the test asked for."""
        if self.on_recv:
            raise self.on_recv
        return None

    def close(self):
        """Close the connection."""


def wired(cell: ExecutorCell, process, connection) -> ExecutorCell:
    """Attach a stub process and pipe to a handle that was never spawned.

    Args:
        cell: The handle.
        process: The stub process.
        connection: The stub pipe end.

    Returns:
        The handle.
    """
    cell._process = process
    cell._conn = connection
    return cell


def test_sending_to_a_child_that_has_exited_is_a_dead_cell():
    cell = wired(ExecutorCell("cpu-0"), StubProcess(alive=False), RefusingConnection())
    with pytest.raises(CellDead):
        cell.send(Stats())
    assert cell.broken == "the child exited"


def test_a_pipe_that_refuses_the_request_is_a_dead_cell():
    cell = wired(
        ExecutorCell("cpu-0"), StubProcess(), RefusingConnection(on_send=OSError("closed"))
    )
    with pytest.raises(CellDead):
        cell.send(Stats())
    assert "rejected" in cell.broken


def test_a_pipe_that_fails_while_waiting_is_a_dead_cell():
    connection = RefusingConnection(on_poll=OSError("closed"))
    cell = wired(ExecutorCell("cpu-0"), StubProcess(), connection)
    cell._in_flight = Stats()
    with pytest.raises(CellDead):
        cell.receive(1.0)
    assert cell.broken is not None


def test_a_child_that_dies_mid_reply_is_a_dead_cell():
    connection = RefusingConnection(on_recv=EOFError("gone"))
    cell = wired(ExecutorCell("cpu-0"), StubProcess(), connection)
    cell._in_flight = Stats()
    with pytest.raises(CellDead):
        cell.receive(1.0)


def test_the_child_configuration_carries_the_decode_limits_it_was_given():
    from image_processor.engine.cell import _cell_config
    from image_processor.engine.decode import DecodeLimits

    limits = DecodeLimits(max_bytes=1024, max_pixels=16, max_dim=4)
    config = _cell_config("gpu1-0", "1", (CPU_PROVIDER,), limits, 250, "DEBUG")
    assert config.device_id == 1
    assert config.decode_limits == limits
    assert config.settle_ms == 250
    assert config.log_level == "DEBUG"
    assert _cell_config("cpu-0", None, (CPU_PROVIDER,), None, 0, "INFO").device_id is None


def test_the_parent_side_modules_do_not_pull_the_runtime_in():
    import subprocess
    import sys as system
    from pathlib import Path as FilePath

    root = FilePath(__file__).resolve().parents[2]
    probe = (
        "import sys; "
        "import image_processor.engine.cell, image_processor.engine.supervisor, "
        "image_processor.engine.scheduler, image_processor.engine.protocol, "
        "image_processor.engine.residency; "
        "print('onnxruntime' in sys.modules)"
    )
    finished = subprocess.run(
        [system.executable, "-c", probe], cwd=str(root), capture_output=True, text=True, timeout=120
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "False", finished.stdout
