"""Supervision: what comes up, what comes back, and what is left down (DESIGN.md §6.2, §10.4, §14).

The cells here are fakes, because what is under test is the supervisor's judgment rather than a
subprocess: that a configuration with no GPU and no development override is refused rather than
quietly served on the CPU, that a dead or poisoned cell is drained and replaced, and that a cell
which keeps dying eventually stays down and takes ``healthy()`` with it.
"""

import pytest

from image_processor.engine.cell import CellDead, CellTimeout
from image_processor.engine.protocol import CPU_PROVIDER, CUDA_PROVIDER, Stats
from image_processor.engine.supervisor import Supervisor, SupervisorError
from tests.engine.executor_support import FakeCell


class SecondsClock:
    """A monotonic seconds clock a test moves by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now

    def advance(self, seconds: float) -> float:
        """Move the clock forward.

        Args:
            seconds: How far.

        Returns:
            The new time.
        """
        self.now += seconds
        return self.now


def build(**kwargs) -> Supervisor:
    """Build a supervisor over fake cells.

    Args:
        **kwargs: Overrides for :class:`Supervisor`.

    Returns:
        The supervisor, not started.
    """
    made = []

    def factory(cell_id, device, providers):
        cell = FakeCell(cell_id=cell_id, device=device, providers=providers)
        made.append(cell)
        return cell

    fields = {
        "runtime": {"providers": [CUDA_PROVIDER], "requiredProvider": CUDA_PROVIDER},
        "gpu": {"devices": ["0"]},
        "cell_factory": factory,
    }
    fields.update(kwargs)
    supervisor = Supervisor(**fields)
    supervisor.made = made
    return supervisor


def test_a_configuration_with_no_device_and_no_override_is_refused():
    with pytest.raises(SupervisorError) as raised:
        build(gpu={"devices": []}).start()
    assert raised.value.code == "NO_GPU_DEVICES"


def test_a_development_host_gets_one_cpu_cell():
    supervisor = build(
        runtime={"providers": [CPU_PROVIDER], "allowCpuOnly": True}, gpu={"devices": []}
    ).start()
    assert [cell.cell_id for cell in supervisor.cells()] == ["cpu-0"]
    assert supervisor.cells()[0].device is None
    assert supervisor.healthy() is True
    supervisor.stop()


def test_one_cell_comes_up_per_device_and_per_configured_cell():
    supervisor = build(
        gpu={"devices": ["0", "1"]},
        runtime={"providers": [CUDA_PROVIDER], "executorCellsPerGpu": 2},
    ).start()
    assert [cell.cell_id for cell in supervisor.cells()] == [
        "gpu0-0", "gpu0-1", "gpu1-0", "gpu1-1",
    ]
    assert [cell.cell_id for cell in supervisor.cells_for("1")] == ["gpu1-0", "gpu1-1"]
    assert supervisor.cell("gpu0-0") is supervisor.cells()[0]
    assert supervisor.cell("nothing") is None
    supervisor.stop()


def test_a_runtime_that_names_no_provider_or_no_cell_is_refused():
    with pytest.raises(SupervisorError) as providers:
        build(runtime={"providers": []}).start()
    assert providers.value.code == "NO_PROVIDERS"

    with pytest.raises(SupervisorError) as cells:
        build(runtime={"providers": [CUDA_PROVIDER], "executorCellsPerGpu": 0}).start()
    assert cells.value.code == "NO_EXECUTOR_CELLS"


def test_starting_twice_changes_nothing():
    supervisor = build().start()
    supervisor.start()
    assert len(supervisor.cells()) == 1
    supervisor.stop()
    assert supervisor.cells() == []
    assert supervisor.healthy() is False


def test_a_cell_that_died_is_noticed_and_replaced():
    supervisor = build().start()
    cell = supervisor.cells()[0]
    cell.die()
    assert supervisor.healthy() is False
    assert supervisor.check() == 1
    assert supervisor.recycle_count == 1
    assert supervisor.healthy() is True
    assert cell.started == 2
    assert supervisor.check() == 0
    supervisor.stop()


def test_a_recycle_drains_what_was_in_flight_and_hands_it_back():
    supervisor = build().start()
    cell = supervisor.cells()[0]
    cell.send(Stats())
    drained = supervisor.recycle(cell, "a job poisoned the cell")
    assert isinstance(drained, Stats)
    assert cell.stopped == 1 and cell.started == 2
    assert cell.resident == {}
    assert supervisor.recycle_count == 1
    supervisor.stop()


def test_recycling_a_cell_this_supervisor_does_not_own_is_refused():
    supervisor = build().start()
    with pytest.raises(SupervisorError) as raised:
        supervisor.recycle("nothing", "why")
    assert raised.value.code == "NO_SUCH_CELL"
    supervisor.stop()


def test_a_cell_that_keeps_dying_is_left_down_and_the_boundary_is_unhealthy():
    clock = SecondsClock()
    supervisor = build(max_restarts=2, restart_window_s=600.0, clock=clock).start()
    cell = supervisor.cells()[0]

    assert supervisor.restarts_remaining("gpu0-0") == 2
    supervisor.recycle(cell, "first")
    supervisor.recycle(cell, "second")
    assert supervisor.restarts_remaining("gpu0-0") == 0

    with pytest.raises(SupervisorError) as raised:
        supervisor.recycle(cell, "third")
    assert raised.value.code == "RESTART_BUDGET_EXHAUSTED"
    assert supervisor.healthy() is False
    assert cell.is_alive() is False
    assert supervisor.check() == 0
    supervisor.stop()


def test_the_restart_budget_refills_as_the_window_moves():
    clock = SecondsClock()
    supervisor = build(max_restarts=1, restart_window_s=60.0, clock=clock).start()
    cell = supervisor.cells()[0]
    supervisor.recycle(cell, "first")
    assert supervisor.restarts_remaining("gpu0-0") == 0
    clock.advance(61.0)
    assert supervisor.restarts_remaining("gpu0-0") == 1
    supervisor.recycle(cell, "second")
    assert supervisor.recycle_count == 2
    supervisor.stop()


def test_a_call_that_fails_recycles_the_cell_before_it_raises():
    supervisor = build().start()
    cell = supervisor.cells()[0]

    def refuse(message):
        raise CellDead("the child exited")

    cell.on_load = refuse
    cell.die()
    with pytest.raises(CellDead):
        supervisor.call(cell, Stats(), timeout_s=1.0)
    assert supervisor.recycle_count == 1
    assert cell.is_alive() is True
    supervisor.stop()


def test_a_call_that_times_out_recycles_the_cell_too():
    supervisor = build().start()
    cell = supervisor.cells()[0]

    def never(message):
        raise CellTimeout("no answer")

    cell.on_load = never
    from image_processor.engine.protocol import LoadModel

    with pytest.raises(CellTimeout):
        supervisor.call(cell, LoadModel(digest="sha256:aa", bundle_root="/x"), timeout_s=0.1)
    assert supervisor.recycle_count == 1
    assert supervisor.cells()[0].started == 2
    supervisor.stop()


def test_a_call_that_fails_when_the_budget_is_spent_still_raises():
    supervisor = build(max_restarts=0).start()
    cell = supervisor.cells()[0]
    cell.die()
    with pytest.raises(CellDead):
        supervisor.call(cell, Stats(), timeout_s=1.0)
    assert supervisor.healthy() is False
    supervisor.stop()


def test_a_call_to_a_cell_this_supervisor_does_not_own_is_refused():
    supervisor = build().start()
    with pytest.raises(SupervisorError):
        supervisor.call("nothing", Stats())
    supervisor.stop()


def test_the_status_summary_names_every_cell_and_the_recycle_count():
    supervisor = build(gpu={"devices": ["0", "1"]}).start()
    supervisor.recycle(supervisor.cells()[0], "a test said so")
    status = supervisor.status()
    assert status["healthy"] is True
    assert status["recycleCount"] == 1
    assert [entry["cellId"] for entry in status["cells"]] == ["gpu0-0", "gpu1-0"]
    assert status["cells"][0]["device"] == "0"
    assert status["cells"][0]["restartsRemaining"] == 4
    assert status["cells"][0]["alive"] is True
    supervisor.stop()


def test_the_real_cell_factory_builds_a_handle_bound_to_its_device():
    supervisor = Supervisor(
        runtime={"providers": [CUDA_PROVIDER], "loadConcurrencyPerGpu": 2}, gpu={"devices": ["1"]}
    )
    cell = supervisor._spawn_cell("gpu1-0", "1", (CUDA_PROVIDER,))
    assert (cell.cell_id, cell.device, cell.providers) == ("gpu1-0", "1", (CUDA_PROVIDER,))
    assert cell.is_alive() is False
    assert supervisor.load_concurrency_per_gpu == 2


def test_a_cell_that_will_not_stop_cleanly_does_not_stop_the_shutdown():
    supervisor = build().start()

    def refuse(timeout_s=None):
        raise RuntimeError("the child is wedged")

    supervisor.cells()[0].stop = refuse
    supervisor.stop()
    assert supervisor.cells() == []


def test_a_dead_cell_whose_budget_is_spent_is_reported_rather_than_restarted():
    supervisor = build(max_restarts=1).start()
    cell = supervisor.cells()[0]
    supervisor.recycle(cell, "first")
    cell.die()
    assert supervisor.check() == 0
    assert supervisor.healthy() is False
    assert cell.is_alive() is False
    supervisor.stop()
