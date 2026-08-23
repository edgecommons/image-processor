"""Supervision of the executor cells (DESIGN.md §6.2, §10.4, §14, LLD §6).

One cell per configured GPU, or one CPU cell on a development host that has set ``allowCpuOnly``.
The supervisor owns their lifetimes and nothing else: it starts them, notices when one dies or is
poisoned, drains whatever was in flight, and starts a replacement under the same identity. It does
not know what a job is, which is why the scheduler can requeue work without the supervisor having
an opinion about it.

Restarting is bounded. A cell that keeps dying is a machine problem -- a driver that fell over, a
device that is gone, a model that poisons every context it touches -- and restarting it forever
would turn that into an invisible loop. After ``max_restarts`` inside ``restart_window_s`` the cell
is left down and :meth:`Supervisor.healthy` answers ``False``, which is the reading DESIGN.md §14
turns into a degraded route and, when no route can execute, a failed component.

There is no CPU fallback here. A configuration that names no device and does not set
``allowCpuOnly`` is refused at :meth:`Supervisor.start`, because a component that quietly runs
image inference on the CPU is reporting decisions nobody signed off on (DESIGN.md §10.1).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from image_processor.engine.cell import CellDead, CellError, CellTimeout, ExecutorCell
from image_processor.engine.protocol import CPU_PROVIDER, CUDA_PROVIDER
from image_processor.engine.residency import setting

logger = logging.getLogger(__name__)

#: How many times one cell may be recycled inside the window before it is left down.
DEFAULT_MAX_RESTARTS = 5

#: The window the restart budget is measured over, in seconds.
DEFAULT_RESTART_WINDOW_S = 600.0


class SupervisorError(RuntimeError):
    """The executor boundary cannot be brought up, or cannot be kept up.

    Attributes:
        code: Stable SCREAMING_SNAKE code, safe for events and metrics.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Supervisor:
    """Owns the executor cells and their restarts.

    Args:
        runtime: The ``runtime`` configuration block (DESIGN.md §11): ``providers``,
            ``requiredProvider``, ``allowCpuOnly``, ``executorCellsPerGpu``,
            ``loadConcurrencyPerGpu``. Any object or mapping carrying those names, in either
            spelling.
        gpu: The ``gpu`` configuration block: ``devices``.
        devices: Overrides the configured device list.
        providers: Overrides the configured provider list.
        required_provider: Overrides the configured ``requiredProvider``.
        allow_cpu_only: Overrides the configured ``allowCpuOnly``.
        executor_cells_per_gpu: Overrides the configured cells per device.
        max_restarts: Recycles allowed per cell inside ``restart_window_s``.
        restart_window_s: The window the restart budget is measured over.
        cell_factory: Builds one :class:`~image_processor.engine.cell.ExecutorCell`. Injected by
            tests to stand a fake cell in place of a subprocess.
        stop_timeout_s: How long a stop waits for a clean exit.
        call_timeout_s: The default per-call deadline given to each cell.
        clock: Returns a monotonic time in seconds. Injected by tests.
    """

    def __init__(
        self,
        runtime=None,
        gpu=None,
        devices=None,
        providers=None,
        required_provider=None,
        allow_cpu_only=None,
        executor_cells_per_gpu=None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        restart_window_s: float = DEFAULT_RESTART_WINDOW_S,
        cell_factory=None,
        stop_timeout_s: float = 10.0,
        call_timeout_s: float = 300.0,
        clock=time.monotonic,
    ) -> None:
        """Read the configuration. Nothing is spawned until :meth:`start`."""
        configured_devices = devices if devices is not None else setting(gpu, "devices", default=())
        self.devices = tuple(str(device) for device in configured_devices or ())
        configured_providers = (
            providers
            if providers is not None
            else setting(runtime, "providers", default=(CUDA_PROVIDER,))
        )
        self.providers = tuple(str(name) for name in configured_providers or ())
        self.required_provider = (
            required_provider
            if required_provider is not None
            else setting(runtime, "requiredProvider", default=None)
        )
        self.allow_cpu_only = bool(
            allow_cpu_only
            if allow_cpu_only is not None
            else setting(runtime, "allowCpuOnly", default=False)
        )
        self.cells_per_gpu = int(
            executor_cells_per_gpu
            if executor_cells_per_gpu is not None
            else setting(runtime, "executorCellsPerGpu", default=1)
        )
        self.load_concurrency_per_gpu = int(
            setting(runtime, "loadConcurrencyPerGpu", default=1) or 1
        )
        self.max_restarts = int(max_restarts)
        self.restart_window_s = float(restart_window_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self._cell_factory = cell_factory or self._spawn_cell
        self._clock = clock
        self._cells = {}
        self._restarts = {}
        self._exhausted = set()
        self.recycle_count = 0
        self._started = False

    def _spawn_cell(self, cell_id: str, device: Optional[str], providers) -> ExecutorCell:
        """Build one real executor cell.

        Args:
            cell_id: The cell identity.
            device: The device ordinal as a string, or ``None`` for a CPU cell.
            providers: The providers the cell requests.

        Returns:
            The handle, not yet started.
        """
        return ExecutorCell(
            cell_id,
            device,
            providers,
            call_timeout_s=self.call_timeout_s,
        )

    def _plan(self) -> list:
        """Work out which cells this configuration calls for.

        Returns:
            A list of ``(cell_id, device, providers)``.

        Raises:
            SupervisorError: ``NO_GPU_DEVICES`` when no device is configured and ``allowCpuOnly``
                is not set, or ``NO_PROVIDERS`` when the runtime names no provider at all.
        """
        if not self.providers:
            raise SupervisorError("NO_PROVIDERS", "runtime.providers names no execution provider")
        if self.cells_per_gpu < 1:
            raise SupervisorError(
                "NO_EXECUTOR_CELLS", "runtime.executorCellsPerGpu must be at least 1"
            )
        if not self.devices:
            if not self.allow_cpu_only:
                raise SupervisorError(
                    "NO_GPU_DEVICES",
                    "gpu.devices is empty and runtime.allowCpuOnly is not set, so there is no "
                    "executor this component may run inference on",
                )
            return [("cpu-0", None, tuple(self.providers) or (CPU_PROVIDER,))]
        plan = []
        for device in self.devices:
            for index in range(self.cells_per_gpu):
                plan.append((f"gpu{device}-{index}", device, tuple(self.providers)))
        return plan

    def start(self) -> "Supervisor":
        """Bring every configured cell up.

        Returns:
            This supervisor.

        Raises:
            SupervisorError: The configuration names no executor this component may use.
        """
        if self._started:
            return self
        for cell_id, device, providers in self._plan():
            cell = self._cell_factory(cell_id, device, providers)
            cell.start()
            self._cells[cell_id] = cell
        self._started = True
        logger.info(
            "executor boundary up: %d cell(s) across device(s) %s",
            len(self._cells),
            list(self.devices) or ["cpu"],
        )
        return self

    def stop(self) -> None:
        """Stop every cell and forget them."""
        for cell in list(self._cells.values()):
            try:
                cell.stop(self.stop_timeout_s)
            except Exception as exc:  # pragma: no cover - a stop that itself fails
                logger.warning("cell %s did not stop cleanly: %s", cell.cell_id, exc)
        self._cells.clear()
        self._started = False

    def cells(self) -> list:
        """Return the current cell handles, in configuration order.

        Returns:
            The handles, including any that are down. A caller that needs only usable cells filters
            on :meth:`~image_processor.engine.cell.ExecutorCell.is_alive`.
        """
        return list(self._cells.values())

    def cell(self, cell_id: str):
        """Return one cell by identity.

        Args:
            cell_id: The identity given at start.

        Returns:
            The handle, or ``None`` when this supervisor has no such cell.
        """
        return self._cells.get(cell_id)

    def cells_for(self, device) -> list:
        """Return the cells bound to one device.

        Args:
            device: The device ordinal as a string, or ``None`` for the CPU cell.

        Returns:
            The handles on that device.
        """
        wanted = None if device is None else str(device)
        return [cell for cell in self._cells.values() if cell.device == wanted]

    def healthy(self) -> bool:
        """Report whether the executor boundary can serve work (DESIGN.md §14).

        Returns:
            ``True`` when the supervisor is started, every configured cell is alive, and no cell
            has spent its restart budget.
        """
        if not self._started or not self._cells:
            return False
        if self._exhausted:
            return False
        return all(cell.is_alive() for cell in self._cells.values())

    def restarts_remaining(self, cell_id: str) -> int:
        """Return how many recycles one cell has left in the current window.

        Args:
            cell_id: The cell identity.

        Returns:
            The remaining budget, never below zero.
        """
        self._prune(cell_id)
        return max(self.max_restarts - len(self._restarts.get(cell_id, [])), 0)

    def _prune(self, cell_id: str) -> None:
        """Drop restart timestamps that have left the window.

        Args:
            cell_id: The cell identity.
        """
        history = self._restarts.get(cell_id)
        if not history:
            return
        cutoff = self._clock() - self.restart_window_s
        self._restarts[cell_id] = [stamp for stamp in history if stamp >= cutoff]

    def recycle(self, cell, reason: str):
        """Drain and restart one cell (DESIGN.md §6.2, §10.4).

        The handle is marked broken first, so anything that tries to use it while the restart runs
        is refused rather than queued against a dying process. Whatever was in flight is dropped
        and returned, because the parent -- not the supervisor -- knows how to put that work back:
        a job that did not commit a result returns to retry at the same attempt with the same
        ``inferenceId``.

        Args:
            cell: The cell to recycle, or its identity.
            reason: Why, for the log and the recycle metric.

        Returns:
            The request that was in flight when the cell was drained, or ``None``.

        Raises:
            SupervisorError: ``RESTART_BUDGET_EXHAUSTED`` when this cell has already been recycled
                ``max_restarts`` times inside the window. The cell is left down and
                :meth:`healthy` answers ``False``.
        """
        handle = self._cells.get(cell) if isinstance(cell, str) else cell
        if handle is None:
            raise SupervisorError("NO_SUCH_CELL", f"this supervisor does not own {cell!r}")
        cell_id = handle.cell_id
        drained = getattr(handle, "_in_flight", None)
        handle.mark_broken(reason)
        handle.stop(self.stop_timeout_s)
        self.recycle_count += 1

        self._prune(cell_id)
        history = self._restarts.setdefault(cell_id, [])
        if len(history) >= self.max_restarts:
            self._exhausted.add(cell_id)
            logger.error(
                "cell %s has been recycled %d times in %.0fs and is left down: %s",
                cell_id,
                len(history),
                self.restart_window_s,
                reason,
            )
            raise SupervisorError(
                "RESTART_BUDGET_EXHAUSTED",
                f"cell {cell_id} was recycled {len(history)} times inside "
                f"{self.restart_window_s:.0f}s; it is left down",
            )
        history.append(self._clock())
        logger.warning("recycling cell %s: %s", cell_id, reason)
        handle.start()
        return drained

    def check(self) -> int:
        """Restart any cell whose child has died.

        A cell can die without a request in flight -- a driver reset, an out-of-memory kill -- and
        nothing would notice until the next dispatch. This is the poll that notices.

        Returns:
            The number of cells recycled by this call.
        """
        recycled = 0
        for cell in list(self._cells.values()):
            if cell.cell_id in self._exhausted or cell.is_alive():
                continue
            try:
                self.recycle(cell, "the child is not running")
                recycled += 1
            except SupervisorError as exc:
                logger.error("cell %s cannot be restarted: %s", cell.cell_id, exc.message)
        return recycled

    def call(self, cell, message, timeout_s=None):
        """Send one request to a cell and recycle the cell if it cannot answer.

        Args:
            cell: The cell, or its identity.
            message: The request.
            timeout_s: The deadline in seconds, or ``None`` for the cell's default.

        Returns:
            The reply.

        Raises:
            CellDead: The cell died. It has been recycled before this is raised, so the caller can
                requeue and try again on the next pass.
            CellTimeout: The cell missed its deadline. It has been recycled too.
        """
        handle = self._cells.get(cell) if isinstance(cell, str) else cell
        if handle is None:
            raise SupervisorError("NO_SUCH_CELL", f"this supervisor does not own {cell!r}")
        try:
            return handle.call(message, timeout_s)
        except (CellDead, CellTimeout) as exc:
            try:
                self.recycle(handle, f"a call failed: {exc}")
            except SupervisorError as budget:
                logger.error("cell %s stays down: %s", handle.cell_id, budget.message)
            raise

    def status(self) -> dict:
        """Summarize the executor boundary for health and the ``status`` command.

        Returns:
            The per-cell state, the recycle count, and whether the boundary is healthy.
        """
        return {
            "healthy": self.healthy(),
            "recycleCount": self.recycle_count,
            "cells": [
                {
                    "cellId": cell.cell_id,
                    "device": cell.device,
                    "alive": cell.is_alive(),
                    "pid": cell.pid,
                    "uptimeSecs": round(cell.uptime_s(), 3),
                    "restartsRemaining": self.restarts_remaining(cell.cell_id),
                    "broken": cell.broken,
                }
                for cell in self._cells.values()
            ],
        }


__all__ = [
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_RESTART_WINDOW_S",
    "CellDead",
    "CellError",
    "CellTimeout",
    "Supervisor",
    "SupervisorError",
]
