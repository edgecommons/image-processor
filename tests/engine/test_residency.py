"""Admission and eviction, as tables rather than as anecdotes (DESIGN.md §10.2, §10.3).

Every number here is arithmetic a reader can redo: a budget percentage of a total, a reserve
subtracted from free memory, an activation factor over an estimate. The eviction tests assert the
ordering rather than a single victim, because the ordering is the policy.
"""

import pytest

from image_processor.engine.residency import (
    DEFAULT_ACTIVATION_PEAK,
    DeviceMemory,
    NvmlProbe,
    ResidencyPolicy,
    ResidencyStats,
    StaticMemoryProbe,
    probe_for,
    setting,
)

NOW = 1_000_000


def policy(**kwargs) -> ResidencyPolicy:
    """Build a policy whose clock stands still.

    Args:
        **kwargs: Overrides passed to :class:`ResidencyPolicy`.

    Returns:
        The policy.
    """
    defaults = {
        "resident_memory_budget_percent": 80,
        "reserve_mib": 1024,
        "min_residency_secs": 0,
        "hot_ttl_secs": 120,
        "clock": lambda: NOW,
    }
    defaults.update(kwargs)
    return ResidencyPolicy(**defaults)


def stats(digest: str, **kwargs) -> ResidencyStats:
    """Build a residency record.

    Args:
        digest: The digest it describes.
        **kwargs: Any :class:`ResidencyStats` field.

    Returns:
        The record.
    """
    base = {"size_mib": 1000, "last_used_ms": NOW, "loaded_at_ms": NOW}
    base.update(kwargs)
    return ResidencyStats(digest=digest, **base)


def test_configuration_is_read_in_either_spelling():
    assert setting({"reserveMiB": 512}, "reserveMiB") == 512
    assert setting({"reserve_mib": 512}, "reserveMiB") == 512

    class Block:
        resident_memory_budget_percent = 70

    assert setting(Block(), "residentMemoryBudgetPercent") == 70
    assert setting(None, "residentMemoryBudgetPercent", default=80) == 80
    assert setting({}, "missing", default="fallback") == "fallback"


def test_the_configuration_blocks_supply_the_defaults():
    built = ResidencyPolicy(
        gpu={"devices": ["0", "1"], "residentMemoryBudgetPercent": 70, "reserveMiB": 512},
        scheduler={"minResidencySecs": 30, "hotTtlSecs": 60},
    )
    assert built.resident_memory_budget_percent == 70
    assert built.reserve_mib == 512
    assert built.min_residency_secs == 30
    assert built.hot_ttl_secs == 60
    assert built.devices == ("0", "1")


def test_the_required_size_is_the_estimate_times_the_activation_peak():
    assert policy().required_mib("a", 1000) == int(1000 * DEFAULT_ACTIVATION_PEAK)


def test_a_measured_load_supersedes_a_low_estimate():
    subject = policy()
    subject.record_load("a", peak_mib=2000, load_ms=900.0)
    assert subject.measured_load_peak("a") == 2000
    assert subject.measured_load_ms("a") == 900.0
    assert subject.required_mib("a", 1000) == int(2000 * DEFAULT_ACTIVATION_PEAK)


def test_a_repeated_out_of_memory_asks_for_more_headroom_next_time():
    subject = policy()
    first = subject.required_mib("a", 1000)
    subject.record_memory_pressure("a")
    second = subject.required_mib("a", 1000)
    subject.record_memory_pressure("a")
    third = subject.required_mib("a", 1000)
    assert first < second < third


def test_forgetting_a_digest_drops_its_measurements():
    subject = policy()
    subject.record_load("a", 2000, 100.0)
    subject.record_memory_pressure("a")
    subject.forget("a")
    assert subject.measured_load_peak("a") is None
    assert subject.required_mib("a", 1000) == int(1000 * DEFAULT_ACTIVATION_PEAK)


ADMISSIONS = [
    # free, total, resident, estimate, admitted, reason
    (8000, 16000, 0, 1000, True, "ADMITTED"),
    (1500, 16000, 0, 1000, False, "NEEDS_EVICTION"),
    (16000, 16000, 0, 13000, False, "OVER_BUDGET"),
    (16000, 16000, 12000, 1000, False, "NEEDS_EVICTION"),
    (0, 0, 0, 1000, True, "NO_DEVICE_ACCOUNTING"),
]


@pytest.mark.parametrize("free,total,resident,estimate,admitted,reason", ADMISSIONS)
def test_admission_reads_free_memory_the_reserve_and_the_budget(
    free, total, resident, estimate, admitted, reason
):
    decision = policy().admit(
        "a", estimate, free, total_mib=total, resident_mib=resident
    )
    assert bool(decision) is admitted
    assert decision.reason == reason
    assert (decision.shortfall_mib > 0) is (not admitted)


def test_the_reserve_and_the_budget_can_be_overridden_per_call():
    subject = policy(reserve_mib=8000)
    assert not subject.admit("a", 1000, 2000, total_mib=16000)
    assert subject.admit("a", 1000, 2000, reserve_mib=0, budget_pct=100, total_mib=16000)


def test_a_colocated_allowance_is_taken_off_the_headroom():
    tight = policy(reserve_mib=0, colocated_allowance_mib=2000)
    assert not tight.admit("a", 1000, 2500, total_mib=16000)
    assert tight.admit("a", 1000, 4000, total_mib=16000)


def test_the_cheapest_session_per_byte_leaves_first():
    subject = policy()
    resident = {
        "busy": stats("busy", queued_jobs=4),
        "idle": stats("idle", queued_jobs=0, last_used_ms=NOW - 600_000),
        "small": stats("small", size_mib=100, queued_jobs=0, last_used_ms=NOW - 600_000),
    }
    assert subject.victims(1000, resident) == ["idle"]


def test_queued_work_reuse_and_reload_cost_all_raise_the_retained_value():
    subject = policy()
    plain = stats("plain", last_used_ms=NOW - 600_000)
    assert subject.value(stats("queued", queued_jobs=3, last_used_ms=NOW - 600_000)) > subject.value(plain)
    assert subject.value(stats("reused", hits=20, last_used_ms=NOW - 600_000)) > subject.value(plain)
    assert subject.value(stats("costly", load_ms=30_000, last_used_ms=NOW - 600_000)) > subject.value(plain)
    assert subject.value(stats("important", priority=900, last_used_ms=NOW - 600_000)) > subject.value(plain)
    assert subject.value(stats("hot", last_used_ms=NOW)) > subject.value(plain)


def test_a_leased_session_is_never_a_victim():
    subject = policy()
    resident = {"a": stats("a", queued_jobs=0), "b": stats("b", queued_jobs=0)}
    assert subject.victims(1000, resident, leased={"a"}) == ["b"]
    resident["b"].leased = True
    assert subject.victims(1000, resident, leased={"a"}) == []


def test_minimum_residency_protects_a_session_that_has_just_loaded():
    subject = policy(min_residency_secs=15)
    resident = {"fresh": stats("fresh", loaded_at_ms=NOW - 5_000)}
    assert subject.victims(1000, resident) == []
    resident["fresh"].loaded_at_ms = NOW - 20_000
    assert subject.victims(1000, resident) == ["fresh"]


def test_eviction_stops_once_it_has_freed_enough():
    subject = policy()
    resident = {
        "one": stats("one", size_mib=600, last_used_ms=NOW - 500_000),
        "two": stats("two", size_mib=600, last_used_ms=NOW - 400_000),
        "three": stats("three", size_mib=600, last_used_ms=NOW - 300_000),
    }
    assert subject.victims(1000, resident) == ["one", "two"]


def test_a_shortfall_nothing_can_cover_returns_what_there_is_and_the_load_waits():
    subject = policy()
    resident = {"one": stats("one", size_mib=100, last_used_ms=NOW - 500_000)}
    assert subject.victims(100_000, resident) == ["one"]
    assert subject.victims(0, resident) == []


def test_a_records_footprint_falls_back_to_the_manifest_estimate():
    assert ResidencyStats("a", size_mib=0, estimate_mib=700).footprint_mib == 700


def test_a_static_probe_reads_back_what_a_load_and_an_unload_do():
    probe = StaticMemoryProbe(total_mib=8192, free_mib=8192, device_class="NVIDIA Test GPU")
    probe.allocate(1024)
    reading = probe.snapshot(0)
    assert (reading.total_mib, reading.free_mib, reading.used_mib) == (8192, 7168, 1024)
    assert reading.device_class == "NVIDIA Test GPU"
    assert reading.known is True
    probe.release(4096)
    assert probe.snapshot(0).free_mib == 8192


def test_a_cell_without_a_device_reads_nothing():
    reading = StaticMemoryProbe().snapshot(None)
    assert (reading.total_mib, reading.free_mib, reading.device_class) == (0, 0, None)
    assert reading.known is False
    assert DeviceMemory().known is False


def test_the_probe_for_a_cpu_cell_is_the_one_that_reads_nothing():
    assert isinstance(probe_for(None), StaticMemoryProbe)
    assert isinstance(probe_for(0), NvmlProbe)


def test_nvml_degrades_to_no_accounting_rather_than_failing_the_load(monkeypatch):
    probe = NvmlProbe()
    assert probe.snapshot(None).known is False

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def refuse(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no nvidia-ml-py here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)
    assert probe.snapshot(0).known is False
    assert probe.snapshot(0).known is False
