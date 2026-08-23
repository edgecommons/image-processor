"""The tier-3 residency and burst gates (DESIGN.md section 16.1 tier 3, section 10.2, section 10.3).

The corpus is several times the size of the card, so every pattern here forces the scheduler to
choose: what stays resident, what is evicted, what is reloaded, and in which order the lanes are
served. The gates are the ones DESIGN.md sets for this tier -- zero out-of-memory, bounded executor
recycles, every admitted job terminal, eviction and reload per the residency policy, thrash
measured -- and the numbers each pattern produces are written to
``tests/nvidia/results/<gpu-class>-<date>.json``.
"""

from __future__ import annotations

import pytest

from tests.nvidia import patterns, require_nvidia, setting

require_nvidia()

#: The arrival patterns this run drives.
PATTERNS = patterns()

#: Codes that mean the device ran out of memory rather than the model being wrong.
MEMORY_CODES = ("MEMORY", "OOM", "ALLOC")


def test_the_corpus_is_larger_than_the_card(harness, results):
    """The rig only measures residency if the configured model set cannot all be resident."""
    total = float(sum(route.tier_mib for route in harness.routes))
    budget = harness.budget_mib()
    assert budget > 0, "the device reported no memory, so no budget could be computed"
    assert total > budget * 1.5, (
        f"the corpus is {total:.0f} MiB against a {budget} MiB residency budget; that fits, so "
        "nothing would ever be evicted. Build more or larger bundles with tools/synth_corpus.py"
    )


@pytest.mark.parametrize("pattern", PATTERNS)
def test_arrival_pattern(pattern, harness, results):
    """Drive one arrival pattern, record what it cost, and hold it to the tier-3 gates."""
    record = harness.run_pattern(
        pattern,
        arrivals=setting("ARRIVALS", 96),
        rate=setting("RATE", 8.0),
        drain_timeout=setting("DRAIN", 1200.0),
    )
    results.add(record)

    assert record["arrivals"] > 0, "the pattern wrote nothing"
    assert record["drained"], (
        f"{pattern}: the queue did not drain inside the budget; "
        f"{harness.outstanding()} jobs are still not terminal, states {record['ledgerStates']}"
    )
    assert record["oom"]["loadPressure"] == 0, (
        f"{pattern}: the runtime reported device-memory exhaustion "
        f"{record['oom']['loadPressure']} time(s); admission is supposed to prevent that"
    )
    offending = [
        code
        for code in record["loads"]["failureCodes"]
        if any(token in code.upper() for token in MEMORY_CODES)
    ]
    assert not offending, f"{pattern}: loads failed for memory: {offending}"
    assert record["recycles"] <= 1, (
        f"{pattern}: the executor was recycled {record['recycles']} times; "
        "DESIGN.md section 6.2 makes repeated recycles a release-gate failure"
    )
    assert record["jobs"]["admissionFailures"] == [], record["jobs"]["admissionFailures"]
    assert record["jobs"]["completionFailures"] == [], record["jobs"]["completionFailures"]
    assert record["jobs"]["invalidInputs"] == 0, (
        f"{pattern}: {record['jobs']['invalidInputs']} capture(s) were refused as invalid"
    )
    assert record["jobs"]["failed"] == 0, (
        f"{pattern}: {record['jobs']['failed']} job(s) failed: {record['oom']['jobErrors']}"
    )
    assert record["jobs"]["results"] == record["arrivals"], (
        f"{pattern}: {record['arrivals']} captures arrived but "
        f"{record['jobs']['results']} results came back; accepted jobs are never dropped"
    )
    assert record["device"]["peakResidentMiB"] <= record["device"]["budgetMiB"], (
        f"{pattern}: the residency map peaked at {record['device']['peakResidentMiB']} MiB "
        f"against a {record['device']['budgetMiB']} MiB budget"
    )


def test_every_admitted_job_reached_a_terminal_state(harness, results):
    """Nothing is left in flight, blocked, or waiting once every pattern has run."""
    assert results.patterns, "no arrival pattern ran"
    states = harness.ledger.counts_by_state()
    stuck = {
        state.value: count
        for state, count in states.items()
        if state.value not in ("COMPLETED", "RETAINED_FAILED", "QUARANTINED")
    }
    assert not stuck, f"jobs did not reach a terminal state: {stuck}"


def test_residency_evicted_and_reloaded_under_pressure(harness, results):
    """Pressure produced evictions, the reloads that follow them, and a measured reload cost."""
    evictions = sum(entry["evictions"]["unloads"] for entry in results.patterns)
    reloads = sum(entry["loads"]["reload"] for entry in results.patterns)
    assert evictions > 0, (
        "no session was ever evicted, so the run never reached the residency budget; "
        "raise EC_NVIDIA_ARRIVALS or build a larger corpus"
    )
    assert reloads > 0, "nothing was reloaded, so eviction cost was never paid back"
    measured = [
        digest
        for digest in harness._ever_loaded
        if harness.policy.measured_load_ms(digest) > 0 or harness.policy.measured_load_peak(digest)
    ]
    assert measured, "the residency policy recorded no load measurement to price eviction against"


def test_thrash_and_residency_are_reported(harness, results):
    """The operator surfaces carry what tier 3 measures: loads, evictions, recycles, lanes."""
    status = harness.scheduler.status()
    assert set(status) >= {"lanes", "cells", "recycleCount", "counters", "queued"}
    for entry in results.patterns:
        assert set(entry["schedulerStatus"]) >= {"lanes", "cells", "recycleCount", "counters"}
    counters = status["counters"]
    assert set(counters) >= {"loads", "evictions", "deferred", "recycles", "dispatched"}
    assert counters["loads"] > 0 and counters["evictions"] > 0
    assert status["recycleCount"] == harness.supervisor.recycle_count
    for cell in status["cells"]:
        assert cell["alive"] is True
        assert set(cell) >= {"resident", "residentMib", "leased", "contextMib"}
    assert all(entry["device"]["contextMiB"] >= 0 for entry in results.patterns), (
        "the CUDA context is recorded apart from the models it holds (DESIGN.md section 10.2)"
    )


def test_every_result_ran_on_the_cuda_provider(harness, results):
    """No job was quietly served on CPU: the provider policy fails closed (DESIGN.md 10.1)."""
    assert harness.supervisor.required_provider == "CUDAExecutionProvider"
    assert harness.supervisor.allow_cpu_only is False
    succeeded = sum(entry["jobs"]["succeeded"] for entry in results.patterns)
    assert succeeded > 0, "no job succeeded, so nothing proves the provider assignment"
    for entry in results.patterns:
        # The recorded assignment is what the session actually got, which is CUDA first and the
        # CPU provider behind it: ONNX Runtime always appends CPU as the fallback for a node no
        # other provider claims. What the policy forbids is CPU *alone* (DESIGN.md section 10.1).
        for assignment in entry["providers"]:
            first = assignment.split(",")[0]
            assert first == "CUDAExecutionProvider", (
                f"{entry['pattern']}: a result recorded the provider assignment {assignment!r}"
            )
        assert entry["gpuDevices"] == [str(harness.device)], entry["gpuDevices"]
