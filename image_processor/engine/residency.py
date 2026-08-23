"""Device-memory admission and cost-aware eviction (DESIGN.md §10.2, §10.3, LLD §6).

A GPU holds a handful of models and a site has hundreds. What is resident at any moment is
therefore a decision, and this module is where it is made. It answers two questions and nothing
else, so both are testable without a GPU:

* **Admission.** Does this model fit right now? The answer combines the manifest estimate, the
  peak this digest actually took the last time it loaded, the device's current free memory, the
  runtime safety reserve, the allowance for processes this component does not own, the resident
  budget percentage, and the transient activation peak a session pays while it initializes.
* **Eviction.** If it does not fit, what leaves? Sessions are priced, not aged out: queued work,
  observed reuse, the measured cost of loading this digest again, the route priority, and recency
  make up a retained value, and the lowest value per byte leaves first.

Two rules bound the second answer. A leased session is never a victim -- a lease is what a job
in flight and a freshly loaded model draining its burst hold. And a session younger than
``minResidencySecs`` is never a victim either, which is the hysteresis that stops two hot models
from evicting each other on alternating passes. When neither rule leaves enough to evict, the load
waits: DESIGN.md §10.3 is explicit that accepted jobs are never dropped, so back-pressure is
latency, never loss.

Device memory is read through :class:`DeviceMemoryProbe`. The NVML implementation is behind the
optional ``nvml`` extra, because a CPU development host has no NVML to import and must still run
the identical scheduling code.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

#: ``gpu.residentMemoryBudgetPercent`` default (DESIGN.md §11).
DEFAULT_BUDGET_PERCENT = 80.0

#: ``gpu.reserveMiB`` default (DESIGN.md §11).
DEFAULT_RESERVE_MIB = 2048

#: ``scheduler.minResidencySecs`` default (DESIGN.md §11).
DEFAULT_MIN_RESIDENCY_SECS = 15.0

#: ``scheduler.hotTtlSecs`` default (DESIGN.md §11).
DEFAULT_HOT_TTL_SECS = 120.0

#: Multiplier applied to a model's steady-state footprint to cover the transient peak a session
#: pays while it initializes its provider arena and copies weights to the device.
DEFAULT_ACTIVATION_PEAK = 1.25

#: How much a repeated out-of-memory failure inflates the next estimate for that digest, so a
#: retry asks for more headroom instead of failing the same way.
DEFAULT_PRESSURE_GROWTH = 1.25


def _now_ms() -> int:
    """Return the current wall clock in milliseconds."""
    return int(time.time() * 1000)


def setting(source, *names, default=None):
    """Read one configuration value from an object or a mapping, by any of its spellings.

    WP1's configuration dataclasses are not merged yet, so every WP4b entry point takes whatever
    carries the DESIGN.md §11 fields: the eventual ``GpuConfig``/``SchedulerConfig``/
    ``RuntimeConfig``, a plain mapping of the JSON, or a stand-in in a test. Spelling is not part
    of the contract: ``residentMemoryBudgetPercent``, ``resident_memory_budget_percent``, and
    ``reserveMiB`` against ``reserve_mib`` all resolve, because names are compared with their
    underscores removed and their case dropped.

    Args:
        source: The configuration object, mapping, or ``None``.
        *names: Accepted names, highest priority first.
        default: The value used when the source carries none of the names.

    Returns:
        The first value found, or ``default``.
    """
    if source is None:
        return default
    if isinstance(source, dict):
        available = dict(source)
    else:
        available = {key: getattr(source, key) for key in dir(source) if not key.startswith("_")}
    normalized = {}
    for key, value in available.items():
        normalized.setdefault(_normal(key), value)
    for name in names:
        value = normalized.get(_normal(name))
        if value is not None and not callable(value):
            return value
    return default


def _normal(name: str) -> str:
    """Return the spelling-insensitive form of a configuration name.

    Args:
        name: The name to normalize.

    Returns:
        The name without underscores, in lower case.
    """
    return str(name).replace("_", "").lower()


# -- device memory -------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceMemory:
    """One reading of a device's memory.

    Attributes:
        total_mib: Memory installed on the device, or ``0`` when unknown.
        free_mib: Memory currently free, or ``0`` when unknown.
        used_mib: Memory currently in use, or ``0`` when unknown.
        device_class: The device name, such as ``NVIDIA GeForce RTX 5080``, or ``None``.
    """

    total_mib: int = 0
    free_mib: int = 0
    used_mib: int = 0
    device_class: Optional[str] = None

    @property
    def known(self) -> bool:
        """Whether this reading carries device accounting at all."""
        return self.total_mib > 0 or self.free_mib > 0


class DeviceMemoryProbe(Protocol):
    """Reads device memory. The seam that keeps NVML optional."""

    def snapshot(self, device_id=None) -> DeviceMemory:
        """Read one device.

        Args:
            device_id: The device ordinal, or ``None`` for a cell with no device.

        Returns:
            The reading. A probe that cannot read the device returns an empty
            :class:`DeviceMemory` rather than raising, so a missing NVML degrades the accounting
            and never the inference.
        """


class StaticMemoryProbe:
    """A probe with numbers a caller sets.

    It serves two roles: the reading for a CPU cell, which has no device and reports nothing, and
    the controllable device in a test, where ``allocate`` and ``release`` move free memory the way
    a real load and unload would.

    Attributes:
        total_mib: Memory the imagined device has installed.
        free_mib: Memory currently free.
        device_class: The name reported as the GPU class.
    """

    def __init__(
        self, total_mib: int = 0, free_mib: int = 0, device_class: Optional[str] = None
    ) -> None:
        """Initialize the probe.

        Args:
            total_mib: Memory installed. ``0`` means no device accounting at all.
            free_mib: Memory free at the start.
            device_class: The device name to report.
        """
        self.total_mib = int(total_mib)
        self.free_mib = int(free_mib)
        self.device_class = device_class

    def snapshot(self, device_id=None) -> DeviceMemory:
        """Read the imagined device.

        Args:
            device_id: Ignored; this probe has one device.

        Returns:
            The current reading.
        """
        return DeviceMemory(
            total_mib=self.total_mib,
            free_mib=self.free_mib,
            used_mib=max(self.total_mib - self.free_mib, 0),
            device_class=self.device_class,
        )

    def allocate(self, mib: int) -> None:
        """Take memory, the way loading a session does.

        Args:
            mib: Mebibytes to take.
        """
        self.free_mib = max(self.free_mib - int(mib), 0)

    def release(self, mib: int) -> None:
        """Give memory back, the way unloading a session does.

        Args:
            mib: Mebibytes to return.
        """
        self.free_mib = min(self.free_mib + int(mib), self.total_mib)


class NvmlProbe:
    """A probe backed by NVML through ``nvidia-ml-py`` (the ``nvml`` extra).

    NVML is initialized on the first reading and kept open, because a cell reads it around every
    load and unload. A device that cannot be read reports an empty :class:`DeviceMemory`: the
    accounting degrades to the manifest estimate, which is what the residency policy already falls
    back to.
    """

    def __init__(self) -> None:
        """Initialize the probe without touching NVML."""
        self._nvml = None
        self._failed = False

    def _library(self):
        """Import and initialize NVML once.

        Returns:
            The ``pynvml`` module, or ``None`` when it is unavailable.
        """
        if self._nvml is not None or self._failed:
            return self._nvml
        try:
            import pynvml
        except ImportError as exc:
            logger.debug("NVML is not installed, device accounting is unavailable: %s", exc)
            self._failed = True
            return None
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # pragma: no cover - live NVML seam
            logger.warning("NVML failed to initialize, device accounting is unavailable: %s", exc)
            self._failed = True
            return None
        self._nvml = pynvml
        return pynvml

    def snapshot(self, device_id=None) -> DeviceMemory:
        """Read one device through NVML.

        Args:
            device_id: The device ordinal, or ``None`` for a cell with no device.

        Returns:
            The reading, empty when NVML or the device is unavailable.
        """
        if device_id is None:
            return DeviceMemory()
        pynvml = self._library()
        if pynvml is None:
            return DeviceMemory()
        try:  # pragma: no cover - live NVML seam
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            return DeviceMemory(
                total_mib=int(memory.total) // (1 << 20),
                free_mib=int(memory.free) // (1 << 20),
                used_mib=int(memory.used) // (1 << 20),
                device_class=str(name),
            )
        except Exception as exc:  # pragma: no cover - live NVML seam
            logger.warning("NVML could not read device %s: %s", device_id, exc)
            return DeviceMemory()


def probe_for(device_id=None) -> DeviceMemoryProbe:
    """Choose the probe for one cell.

    Args:
        device_id: The device ordinal the cell owns, or ``None`` for a CPU cell.

    Returns:
        An :class:`NvmlProbe` for a device, a :class:`StaticMemoryProbe` reporting nothing for a
        CPU cell.
    """
    if device_id is None:
        return StaticMemoryProbe()
    return NvmlProbe()


# -- policy --------------------------------------------------------------------------------------


@dataclass
class ResidencyStats:
    """What the policy knows about one resident session.

    Attributes:
        digest: The bundle digest the session serves.
        size_mib: The session's measured device footprint. Falls back to ``estimate_mib`` when the
            device could not be measured.
        estimate_mib: The manifest's ``estimatedDeviceMib``.
        measured_load_peak_mib: The peak measured the last time this digest loaded, or ``None``.
        load_ms: The measured cost of loading this digest again.
        queued_jobs: Jobs waiting on this session right now.
        priority: The highest route priority among the work bound to it.
        last_used_ms: When the session last answered a job.
        loaded_at_ms: When the session became resident. The minimum-residency clock.
        hits: Jobs this session has answered, the observed reuse.
        leased: Whether the session is pinned by in-flight work or an unfinished burst.
    """

    digest: str
    size_mib: int = 0
    estimate_mib: int = 0
    measured_load_peak_mib: Optional[int] = None
    load_ms: float = 0.0
    queued_jobs: int = 0
    priority: int = 100
    last_used_ms: int = 0
    loaded_at_ms: int = 0
    hits: int = 0
    leased: bool = False

    @property
    def footprint_mib(self) -> int:
        """The bytes an eviction of this session actually returns."""
        return max(int(self.size_mib or 0), int(self.estimate_mib or 0), 0)


@dataclass(frozen=True)
class Admission:
    """The answer to "does this model fit right now".

    The object is truthy exactly when the load is admitted, so a caller that only wants the
    decision reads it as a boolean and a caller that must free memory reads the shortfall.

    Attributes:
        admitted: Whether the load may proceed.
        required_mib: Device memory the load is expected to need at its peak.
        shortfall_mib: Memory that must be freed first. ``0`` when admitted.
        reason: ``ADMITTED``, ``NEEDS_EVICTION``, ``OVER_BUDGET``, or ``NO_DEVICE_ACCOUNTING``.
    """

    admitted: bool
    required_mib: int
    shortfall_mib: int
    reason: str

    def __bool__(self) -> bool:
        """Return whether the load is admitted."""
        return self.admitted


class ResidencyPolicy:
    """Decides what may become resident and what leaves to make room.

    Args:
        gpu: The ``gpu`` configuration block (DESIGN.md §11): ``devices``,
            ``residentMemoryBudgetPercent``, ``reserveMiB``. Any object or mapping carrying those
            names, in either spelling.
        scheduler: The ``scheduler`` configuration block: ``minResidencySecs``, ``hotTtlSecs``.
        resident_memory_budget_percent: Overrides the configured budget percentage.
        reserve_mib: Overrides the configured runtime safety reserve.
        min_residency_secs: Overrides the configured minimum residency.
        hot_ttl_secs: Overrides the configured hot window, over which recency decays.
        activation_peak_factor: Multiplier covering the transient peak of a session's activation.
        colocated_allowance_mib: Memory left for processes this component does not own.
        queued_weight: Weight of queued work in a session's retained value.
        reuse_weight: Weight of observed reuse.
        reload_weight: Weight of the measured reload cost, per second.
        priority_weight: Weight of route priority, normalized to a priority of 100.
        recency_weight: Weight of recency inside the hot window.
        pressure_growth: How much a repeated out-of-memory inflates the next estimate.
        clock: Returns the current wall clock in milliseconds. Injected by tests.
    """

    def __init__(
        self,
        gpu=None,
        scheduler=None,
        resident_memory_budget_percent=None,
        reserve_mib=None,
        min_residency_secs=None,
        hot_ttl_secs=None,
        activation_peak_factor: float = DEFAULT_ACTIVATION_PEAK,
        colocated_allowance_mib: int = 0,
        queued_weight: float = 10.0,
        reuse_weight: float = 1.0,
        reload_weight: float = 2.0,
        priority_weight: float = 1.0,
        recency_weight: float = 5.0,
        pressure_growth: float = DEFAULT_PRESSURE_GROWTH,
        clock=_now_ms,
    ) -> None:
        """Build a policy from the configuration blocks and the overrides."""
        self.resident_memory_budget_percent = float(
            resident_memory_budget_percent
            if resident_memory_budget_percent is not None
            else setting(
                gpu,
                "residentMemoryBudgetPercent",
                default=DEFAULT_BUDGET_PERCENT,
            )
        )
        self.reserve_mib = int(
            reserve_mib
            if reserve_mib is not None
            else setting(gpu, "reserveMiB", "reserveMib", default=DEFAULT_RESERVE_MIB)
        )
        self.min_residency_secs = float(
            min_residency_secs
            if min_residency_secs is not None
            else setting(scheduler, "minResidencySecs", default=DEFAULT_MIN_RESIDENCY_SECS)
        )
        self.hot_ttl_secs = float(
            hot_ttl_secs
            if hot_ttl_secs is not None
            else setting(scheduler, "hotTtlSecs", default=DEFAULT_HOT_TTL_SECS)
        )
        self.devices = tuple(str(device) for device in setting(gpu, "devices", default=()) or ())
        self.activation_peak_factor = float(activation_peak_factor)
        self.colocated_allowance_mib = int(colocated_allowance_mib)
        self.queued_weight = float(queued_weight)
        self.reuse_weight = float(reuse_weight)
        self.reload_weight = float(reload_weight)
        self.priority_weight = float(priority_weight)
        self.recency_weight = float(recency_weight)
        self.pressure_growth = float(pressure_growth)
        self._clock = clock
        self._measured: dict = {}
        self._load_ms: dict = {}
        self._pressure: dict = {}

    # -- measurements ----------------------------------------------------------------------

    def record_load(self, digest: str, peak_mib=None, load_ms=None) -> None:
        """Remember what a load actually cost.

        Args:
            digest: The bundle digest that loaded.
            peak_mib: Device memory the load consumed, or ``None``/``0`` when unmeasured.
            load_ms: Wall-clock milliseconds the load took, or ``None``.
        """
        if peak_mib:
            self._measured[digest] = max(int(peak_mib), self._measured.get(digest, 0))
        if load_ms:
            self._load_ms[digest] = float(load_ms)

    def record_memory_pressure(self, digest: str) -> None:
        """Remember that this digest ran the device out of memory.

        The next admission asks for :attr:`pressure_growth` more headroom per recorded failure, so
        a retry after an out-of-memory is not the identical request that just failed.

        Args:
            digest: The bundle digest whose load or inference failed for memory.
        """
        self._pressure[digest] = self._pressure.get(digest, 0) + 1

    def measured_load_peak(self, digest: str):
        """Return the measured device peak for a digest.

        Args:
            digest: The bundle digest.

        Returns:
            The measured peak in MiB, or ``None`` when this digest has never been measured.
        """
        return self._measured.get(digest)

    def measured_load_ms(self, digest: str) -> float:
        """Return the measured load cost for a digest.

        Args:
            digest: The bundle digest.

        Returns:
            The measured load time in milliseconds, or ``0.0`` when it has never loaded.
        """
        return self._load_ms.get(digest, 0.0)

    def forget(self, digest: str) -> None:
        """Drop the measurements for a digest that is no longer referenced.

        Args:
            digest: The bundle digest.
        """
        self._measured.pop(digest, None)
        self._load_ms.pop(digest, None)
        self._pressure.pop(digest, None)

    # -- admission -------------------------------------------------------------------------

    def required_mib(self, digest: str, estimate_mib) -> int:
        """Return the device memory a load of this digest is expected to need at its peak.

        The manifest estimate is a floor, the measured peak from a previous load supersedes it
        when it is larger, the activation factor covers the transient cost of initializing the
        session, and every recorded out-of-memory for this digest inflates the result again.

        Args:
            digest: The bundle digest.
            estimate_mib: The manifest's ``estimatedDeviceMib``.

        Returns:
            The expected peak in MiB.
        """
        base = max(int(estimate_mib or 0), int(self._measured.get(digest, 0)))
        inflated = base * self.activation_peak_factor
        pressure = self._pressure.get(digest, 0)
        if pressure:
            inflated *= self.pressure_growth**pressure
        return int(math.ceil(inflated))

    def admit(
        self,
        digest: str,
        estimate_mib,
        free_mib,
        reserve_mib=None,
        budget_pct=None,
        total_mib=None,
        resident_mib: int = 0,
    ) -> Admission:
        """Decide whether one model may become resident right now (DESIGN.md §10.2).

        Args:
            digest: The bundle digest to load.
            estimate_mib: The manifest's ``estimatedDeviceMib``.
            free_mib: Device memory currently free, from the probe.
            reserve_mib: The runtime safety reserve. Defaults to the configured ``reserveMiB``.
            budget_pct: The resident budget percentage. Defaults to the configured
                ``residentMemoryBudgetPercent``.
            total_mib: Device memory installed, from the probe. ``0`` or ``None`` means the
                budget percentage cannot be applied.
            resident_mib: What this device already holds for this component.

        Returns:
            The :class:`Admission`. It is truthy when the load may proceed, and carries the
            shortfall the caller must evict when it may not.
        """
        required = self.required_mib(digest, estimate_mib)
        reserve = self.reserve_mib if reserve_mib is None else int(reserve_mib)
        percent = (
            self.resident_memory_budget_percent if budget_pct is None else float(budget_pct)
        )
        total = int(total_mib or 0)
        free = int(free_mib or 0)

        if total <= 0 and free <= 0:
            return Admission(True, required, 0, "NO_DEVICE_ACCOUNTING")

        headroom = free - reserve - self.colocated_allowance_mib
        shortfall = max(required - headroom, 0)

        if total > 0:
            budget = int(total * percent / 100.0)
            if required > budget:
                return Admission(
                    False,
                    required,
                    max(required - budget, 0),
                    "OVER_BUDGET",
                )
            shortfall = max(shortfall, int(resident_mib) + required - budget)

        if shortfall > 0:
            return Admission(False, required, shortfall, "NEEDS_EVICTION")
        return Admission(True, required, 0, "ADMITTED")

    # -- eviction --------------------------------------------------------------------------

    def value(self, stats: ResidencyStats, now_ms=None) -> float:
        """Return the retained value of one resident session.

        The terms are the ones DESIGN.md §10.3 names: queued work, predicted reuse, the measured
        reload cost, route priority, and recency. Size is not one of them, because size divides the
        value rather than joining it (:meth:`value_per_byte`).

        Args:
            stats: What the scheduler knows about the session.
            now_ms: The current wall clock in milliseconds, or ``None`` to read the policy clock.

        Returns:
            The retained value. Larger means more expensive to lose.
        """
        now = self._clock() if now_ms is None else int(now_ms)
        idle_secs = max(now - int(stats.last_used_ms or 0), 0) / 1000.0
        if self.hot_ttl_secs > 0:
            recency = max(0.0, 1.0 - idle_secs / self.hot_ttl_secs)
        else:
            recency = 0.0
        reload_ms = stats.load_ms or self._load_ms.get(stats.digest, 0.0)
        return (
            self.queued_weight * float(stats.queued_jobs)
            + self.reuse_weight * float(stats.hits)
            + self.reload_weight * (float(reload_ms) / 1000.0)
            + self.priority_weight * (float(stats.priority) / 100.0)
            + self.recency_weight * recency
        )

    def value_per_byte(self, stats: ResidencyStats, now_ms=None) -> float:
        """Return a session's retained value per MiB it occupies.

        Args:
            stats: What the scheduler knows about the session.
            now_ms: The current wall clock in milliseconds, or ``None`` to read the policy clock.

        Returns:
            The value density. The lowest leaves first.
        """
        return self.value(stats, now_ms) / float(max(stats.footprint_mib, 1))

    def evictable(self, stats: ResidencyStats, leased=(), now_ms=None) -> bool:
        """Report whether one session may be evicted at all.

        Args:
            stats: What the scheduler knows about the session.
            leased: Digests pinned by in-flight work or an unfinished burst.
            now_ms: The current wall clock in milliseconds, or ``None`` to read the policy clock.

        Returns:
            ``False`` for a leased session and for one still inside ``minResidencySecs``.
        """
        if stats.leased or stats.digest in set(leased):
            return False
        now = self._clock() if now_ms is None else int(now_ms)
        if self.min_residency_secs > 0:
            age_secs = (now - int(stats.loaded_at_ms or 0)) / 1000.0
            if age_secs < self.min_residency_secs:
                return False
        return True

    def victims(self, needed_mib, resident, leased=()) -> list:
        """Choose the sessions to evict to free ``needed_mib`` (DESIGN.md §10.3).

        Args:
            needed_mib: How much device memory must be freed.
            resident: The resident sessions, keyed by digest.
            leased: Digests pinned by in-flight work or an unfinished burst. Never victims.

        Returns:
            The digests to unload, in the order they should go: lowest retained value per byte
            first, ties broken by least recently used and then by digest so a pass is
            reproducible. The list is short when the evictable sessions do not add up to
            ``needed_mib``; the caller waits rather than dropping the job.
        """
        need = int(needed_mib or 0)
        if need <= 0:
            return []
        now = self._clock()
        locked = set(leased)
        ranked = []
        for digest, stats in resident.items():
            if not self.evictable(stats, locked, now):
                continue
            ranked.append((self.value_per_byte(stats, now), int(stats.last_used_ms or 0), digest))
        ranked.sort()

        chosen = []
        freed = 0
        for _, _, digest in ranked:
            if freed >= need:
                break
            chosen.append(digest)
            freed += resident[digest].footprint_mib
        if freed < need:
            logger.info(
                "eviction can free only %d MiB of the %d MiB needed; the load waits", freed, need
            )
        return chosen


__all__ = [
    "DEFAULT_ACTIVATION_PEAK",
    "DEFAULT_BUDGET_PERCENT",
    "DEFAULT_HOT_TTL_SECS",
    "DEFAULT_MIN_RESIDENCY_SECS",
    "DEFAULT_PRESSURE_GROWTH",
    "DEFAULT_RESERVE_MIB",
    "Admission",
    "DeviceMemory",
    "DeviceMemoryProbe",
    "NvmlProbe",
    "ResidencyPolicy",
    "ResidencyStats",
    "StaticMemoryProbe",
    "probe_for",
    "setting",
]
