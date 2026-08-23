"""The parent-side handle on one executor cell (LLD §6, DESIGN.md §6.2).

This module is the whole of what the parent knows about a cell: a process, a pipe, and a deadline.
It never imports ``onnxruntime`` and never imports it transitively -- the subprocess entry point is
resolved inside :meth:`ExecutorCell.start`, and the module that holds it defers its own runtime
import -- so the parent process stays free of CUDA initialization, which is what lets a cell die
without taking the durability plane with it.

Two failure modes matter and are distinguished:

* The child died. Any call in flight raises :class:`CellDead`, and so does the next one, because
  the process behind the pipe is gone.
* The child did not answer in time. The call raises :class:`CellTimeout` and the handle is marked
  broken. A late reply to a call the parent has given up on would be read as the answer to the
  *next* request, so a cell that misses a deadline is never spoken to again; the supervisor
  recycles it.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from typing import Optional

from image_processor.engine.protocol import CPU_PROVIDER, Shutdown

logger = logging.getLogger(__name__)

#: How long :meth:`ExecutorCell.stop` waits for a clean exit before terminating the child.
DEFAULT_STOP_TIMEOUT_S = 10.0

#: How long one request waits for its reply by default.
DEFAULT_CALL_TIMEOUT_S = 300.0

#: How often a waiting call re-checks that the child is still alive.
POLL_INTERVAL_S = 0.05


class CellError(RuntimeError):
    """A cell could not serve a request."""


class CellDead(CellError):
    """The cell process is gone, or was never started."""


class CellTimeout(CellError, TimeoutError):
    """The cell did not answer inside the call deadline."""


def _entrypoint():
    """Resolve the subprocess entry point.

    The import happens here rather than at module scope so that a parent holding this handle never
    loads the cell's runtime imports.

    Returns:
        The callable the child process runs.
    """
    from image_processor.engine.cell_main import cell_entrypoint

    return cell_entrypoint


def _cell_config(cell_id, device, providers, decode_limits, settle_ms, log_level):
    """Build the child's configuration.

    Args:
        cell_id: The cell identity.
        device: The device ordinal as a string, or ``None`` for a CPU cell.
        providers: The providers the cell requests by default.
        decode_limits: The decode bounds, or ``None`` for the defaults.
        settle_ms: The unload settle period.
        log_level: The child's log level.

    Returns:
        The ``CellConfig`` for the child.
    """
    from image_processor.engine.cell_main import CellConfig

    fields = {
        "cell_id": cell_id,
        "device_id": None if device is None else int(device),
        "providers": tuple(providers),
        "settle_ms": int(settle_ms),
        "log_level": str(log_level),
    }
    if decode_limits is not None:
        fields["decode_limits"] = decode_limits
    return CellConfig(**fields)


class ExecutorCell:
    """One supervised executor subprocess and the pipe to it.

    Args:
        cell_id: The identity the supervisor gave this cell.
        device: The device ordinal as a string, or ``None`` for a CPU cell.
        providers: The execution providers the cell requests by default.
        decode_limits: The decode bounds the cell applies to every input, or ``None`` for the
            defaults.
        settle_ms: How long an unload waits before sampling device memory.
        log_level: The child's log level.
        start_timeout_s: How long :meth:`start` waits for the child to come up.
        call_timeout_s: The default per-call deadline.
        context: The ``multiprocessing`` context. Defaults to ``spawn``, which is the only
            context that behaves the same on Windows and Linux and the only one that gives the
            child a clean CUDA state.
        entrypoint: The callable the child runs. Injected by tests.
    """

    def __init__(
        self,
        cell_id: str,
        device: Optional[str] = None,
        providers=(CPU_PROVIDER,),
        decode_limits=None,
        settle_ms: int = 0,
        log_level: str = "INFO",
        start_timeout_s: float = 60.0,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        context=None,
        entrypoint=None,
    ) -> None:
        """Build the handle. Nothing is spawned until :meth:`start`."""
        self.cell_id = str(cell_id)
        self.device = None if device is None else str(device)
        self.providers = tuple(providers)
        self.decode_limits = decode_limits
        self.settle_ms = int(settle_ms)
        self.log_level = log_level
        self.start_timeout_s = float(start_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self._context = context or multiprocessing.get_context("spawn")
        self._entrypoint = entrypoint
        self._lock = threading.RLock()
        self._process = None
        self._conn = None
        self._broken = None
        self._in_flight = None
        self._started_at = None

    def __repr__(self) -> str:
        """Return a short identification for logs."""
        return f"<ExecutorCell {self.cell_id} device={self.device} pid={self.pid}>"

    @property
    def pid(self):
        """The child's process id, or ``None`` when it is not running."""
        process = self._process
        return None if process is None else process.pid

    @property
    def broken(self):
        """Why this handle refuses further calls, or ``None`` when it is usable."""
        return self._broken

    def uptime_s(self) -> float:
        """Return how long the child has been running, in seconds."""
        if self._started_at is None:
            return 0.0
        return max(time.monotonic() - self._started_at, 0.0)

    def is_alive(self) -> bool:
        """Report whether the child process is running.

        Returns:
            ``True`` only when a process exists, is alive, and the handle is not broken.
        """
        process = self._process
        return bool(process is not None and process.is_alive() and self._broken is None)

    def start(self) -> "ExecutorCell":
        """Spawn the child and open the pipe.

        Starting an already-running cell is a no-op, so a supervisor may call it defensively.

        Returns:
            This handle.
        """
        with self._lock:
            if self.is_alive():
                return self
            self._close_conn()
            entrypoint = self._entrypoint or _entrypoint()
            config = _cell_config(
                self.cell_id,
                self.device,
                self.providers,
                self.decode_limits,
                self.settle_ms,
                self.log_level,
            )
            parent_conn, child_conn = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=entrypoint,
                args=(config, child_conn),
                name=f"image-processor-cell-{self.cell_id}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._process = process
            self._conn = parent_conn
            self._broken = None
            self._in_flight = None
            self._started_at = time.monotonic()
            logger.info(
                "started cell %s (pid %s) for device %s", self.cell_id, process.pid, self.device
            )
            return self

    def stop(self, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        """Ask the child to exit, then make sure it did.

        The child is asked politely first so that it releases its sessions and its CUDA context;
        a child that does not exit inside ``timeout_s`` is terminated, and one that survives that
        is killed. The handle is left ready to :meth:`start` again, which is how a recycle keeps
        the cell's identity.

        Args:
            timeout_s: How long to wait for a clean exit before terminating.
        """
        with self._lock:
            process, conn = self._process, self._conn
            if process is None:
                self._close_conn()
                return
            if conn is not None and process.is_alive():
                try:
                    conn.send(Shutdown())
                except (OSError, EOFError, ValueError) as exc:
                    logger.debug("cell %s did not accept the shutdown: %s", self.cell_id, exc)
            process.join(timeout_s)
            if process.is_alive():
                logger.warning("cell %s did not exit, terminating it", self.cell_id)
                process.terminate()
                process.join(timeout_s)
            if process.is_alive():  # pragma: no cover - only a wedged child reaches this
                logger.error("cell %s ignored termination, killing it", self.cell_id)
                process.kill()
                process.join(timeout_s)
            self._close_conn()
            self._process = None
            self._in_flight = None
            self._started_at = None

    def mark_broken(self, reason: str) -> None:
        """Refuse further calls on this handle.

        Args:
            reason: Why, for the log and for :attr:`broken`.
        """
        with self._lock:
            if self._broken is None:
                self._broken = reason
                logger.warning("cell %s is no longer usable: %s", self.cell_id, reason)

    def _close_conn(self) -> None:
        """Close the parent end of the pipe, if it is open."""
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError as exc:  # pragma: no cover - closing a closed pipe
                logger.debug("cell %s pipe did not close: %s", self.cell_id, exc)

    def send(self, message) -> None:
        """Send one request without waiting for its reply.

        Splitting send from receive is what lets one scheduling pass keep every cell busy at once
        while still applying the replies in a fixed order.

        Args:
            message: The request.

        Raises:
            CellDead: The cell is not running or is broken.
            CellError: A request is already in flight on this cell.
        """
        with self._lock:
            if self._broken is not None:
                raise CellDead(f"cell {self.cell_id} is broken: {self._broken}")
            if self._process is None or self._conn is None:
                raise CellDead(f"cell {self.cell_id} is not started")
            if self._in_flight is not None:
                raise CellError(
                    f"cell {self.cell_id} already has a {type(self._in_flight).__name__} in flight"
                )
            if not self._process.is_alive():
                self.mark_broken("the child exited")
                raise CellDead(f"cell {self.cell_id} is not running")
            try:
                self._conn.send(message)
            except (OSError, EOFError, ValueError) as exc:
                self.mark_broken(f"the pipe rejected a request: {exc}")
                raise CellDead(f"cell {self.cell_id} could not be sent a request: {exc}") from exc
            self._in_flight = message

    def receive(self, timeout_s: Optional[float] = None):
        """Wait for the reply to the request in flight.

        Args:
            timeout_s: The deadline in seconds, or ``None`` for the cell's default.

        Returns:
            The reply.

        Raises:
            CellDead: The child exited before it answered.
            CellTimeout: The deadline passed. The handle is marked broken, because a reply that
                arrives later cannot be told apart from the answer to the next request.
            CellError: Nothing is in flight.
        """
        with self._lock:
            if self._in_flight is None:
                raise CellError(f"cell {self.cell_id} has no request in flight")
            conn, process = self._conn, self._process
        deadline = time.monotonic() + (
            self.call_timeout_s if timeout_s is None else float(timeout_s)
        )
        while True:
            try:
                ready = conn.poll(POLL_INTERVAL_S)
            except (OSError, EOFError, ValueError) as exc:
                self._fail(f"the pipe closed while waiting: {exc}")
                raise CellDead(f"cell {self.cell_id} lost its pipe: {exc}") from exc
            if ready:
                try:
                    reply = conn.recv()
                except (EOFError, OSError, ValueError) as exc:
                    self._fail(f"the child died mid-reply: {exc}")
                    raise CellDead(f"cell {self.cell_id} died mid-reply: {exc}") from exc
                with self._lock:
                    self._in_flight = None
                return reply
            if not process.is_alive():
                self._fail("the child exited before it answered")
                raise CellDead(f"cell {self.cell_id} exited before it answered")
            if time.monotonic() >= deadline:
                self._fail("a call passed its deadline")
                raise CellTimeout(f"cell {self.cell_id} did not answer in time")

    def _fail(self, reason: str) -> None:
        """Mark the handle broken and drop the in-flight request.

        Args:
            reason: Why the handle is no longer usable.
        """
        self.mark_broken(reason)
        with self._lock:
            self._in_flight = None

    def call(self, message, timeout_s: Optional[float] = None):
        """Send one request and wait for its reply.

        Args:
            message: The request.
            timeout_s: The deadline in seconds, or ``None`` for the cell's default.

        Returns:
            The reply.

        Raises:
            CellDead: The cell is not running, or died before answering.
            CellTimeout: The deadline passed.
        """
        self.send(message)
        return self.receive(timeout_s)


__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_STOP_TIMEOUT_S",
    "POLL_INTERVAL_S",
    "CellDead",
    "CellError",
    "CellTimeout",
    "ExecutorCell",
]
