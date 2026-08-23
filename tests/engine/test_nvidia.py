"""The NVIDIA leg of the executor boundary (D-IP-13, D-IP-14, DESIGN.md §16.1 tier 3).

Everything here needs a real CUDA device and ``onnxruntime-gpu``, so the suite runs only when
``EC_NVIDIA`` is set: the desktop RTX 5080 under WSL2 and ``lab-5950x``'s RTX 2080 Super. What it
proves is what a CPU host cannot: that the session actually lands on ``CUDAExecutionProvider``,
that the device accounting is real and separates the CUDA context from the models it holds, that a
CUDA session and a CPU session agree on the same image, and that the whole parent-to-cell path
works through a spawned subprocess holding a CUDA context.
"""

import hashlib
import os

import pytest

from image_processor.engine.cell import ExecutorCell
from image_processor.engine.cell_main import CellConfig, CellState, handle_infer, handle_load, handle_unload
from image_processor.engine.protocol import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    REQUIRE_LISTED,
    Infer,
    LoadModel,
    Loaded,
    Unload,
)
from image_processor.engine.residency import NvmlProbe, ResidencyPolicy

pytestmark = pytest.mark.skipif(
    not os.environ.get("EC_NVIDIA"),
    reason="the NVIDIA suite needs a CUDA device and onnxruntime-gpu; set EC_NVIDIA=1 to run it",
)

#: The device this suite uses.
DEVICE = int(os.environ.get("EC_NVIDIA_DEVICE", "0"))

#: The bundle every case runs.
BUNDLE = "synthetic-classification-1.0.0"

#: The digest the cell keys the session by.
DIGEST = "sha256:" + "f" * 64


def bundle_root(corpus):
    """Return the corpus bundle directory."""
    return str(corpus.path(corpus.expected["bundles"][BUNDLE]["path"]))


def image_and_digest(corpus):
    """Return one corpus image and its digest."""
    case = corpus.expected["bundles"][BUNDLE]["cases"][0]
    path = corpus.path(case["image"])
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), case


def cuda_state() -> CellState:
    """A cell bound to the CUDA device this suite runs on."""
    return CellState(
        CellConfig(cell_id=f"gpu{DEVICE}-0", device_id=DEVICE, providers=(CUDA_PROVIDER, CPU_PROVIDER))
    )


def test_a_session_actually_lands_on_the_cuda_provider(corpus):
    state = cuda_state()
    reply = handle_load(
        state,
        LoadModel(
            digest=DIGEST,
            bundle_root=bundle_root(corpus),
            providers=(CUDA_PROVIDER, CPU_PROVIDER),
            provider_policy=REQUIRE_LISTED,
            providers_permitted=(CUDA_PROVIDER, CPU_PROVIDER),
            required_provider=CUDA_PROVIDER,
            allow_cpu_only=False,
            gpu_mem_limit_mib=2048,
        ),
    )
    assert isinstance(reply, Loaded), getattr(reply, "error", "")
    assert reply.providers_assigned[0] == CUDA_PROVIDER
    assert reply.gpu_device == str(DEVICE)
    assert reply.gpu_class
    # the first CUDA load establishes the device context, which is hundreds of MiB on both cards,
    # and reports it apart from the model. The model itself is a synthetic graph, so its own
    # footprint may round to nothing at NVML's granularity -- which is the point: it is no longer
    # the context (DESIGN.md §10.2).
    assert reply.context_mib > 0
    assert reply.device_mib >= 0
    assert state.context_mib == reply.context_mib

    loaded = state.sessions[DIGEST]
    assert loaded.use_io_binding is True

    path, digest, case = image_and_digest(corpus)
    result = handle_infer(state, Infer("gpu-job", str(path), digest, DIGEST, "1", 2.0))
    assert result.status == "SUCCEEDED", result.error
    assert result.providers[0] == CUDA_PROVIDER
    assert result.gpu_device == str(DEVICE)
    assert result.memory_high_water_mib and result.memory_high_water_mib > 0
    assert result.decision.outcome.value == case["decision"]["outcome"]

    freed = handle_unload(state, Unload(DIGEST))
    assert freed.was_resident is True
    assert freed.expected_mib == reply.device_mib, (
        "an unload is judged against the model's own footprint, never against the context"
    )


def test_a_cuda_session_and_a_cpu_session_agree_on_the_same_image(corpus):
    path, digest, _ = image_and_digest(corpus)

    gpu = cuda_state()
    handle_load(
        gpu,
        LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus),
                  providers=(CUDA_PROVIDER, CPU_PROVIDER), required_provider=CUDA_PROVIDER),
    )
    on_gpu = handle_infer(gpu, Infer("parity", str(path), digest, DIGEST, "1"))

    cpu = CellState(CellConfig(cell_id="cpu-0", device_id=None, providers=(CPU_PROVIDER,)))
    handle_load(
        cpu,
        LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus), providers=(CPU_PROVIDER,),
                  allow_cpu_only=True),
    )
    on_cpu = handle_infer(cpu, Infer("parity", str(path), digest, DIGEST, "1"))

    assert on_gpu.status == on_cpu.status == "SUCCEEDED"
    assert [entry.label for entry in on_gpu.normalized.classes] == [
        entry.label for entry in on_cpu.normalized.classes
    ]
    for left, right in zip(on_gpu.normalized.classes, on_cpu.normalized.classes):
        assert left.score == pytest.approx(right.score, abs=1e-4)
    assert on_gpu.decision.outcome == on_cpu.decision.outcome


def test_a_cpu_only_machine_policy_is_refused_on_a_gpu_route(corpus):
    state = CellState(CellConfig(cell_id="cpu-0", device_id=None, providers=(CPU_PROVIDER,)))
    reply = handle_load(
        state,
        LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus), providers=(CPU_PROVIDER,),
                  required_provider=CUDA_PROVIDER, allow_cpu_only=True),
    )
    assert reply.code == "PROVIDER_REQUIRED_MISSING"


def test_a_second_load_does_not_pay_for_the_context_again(corpus):
    state = cuda_state()
    first = handle_load(
        state,
        LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus),
                  providers=(CUDA_PROVIDER, CPU_PROVIDER), required_provider=CUDA_PROVIDER),
    )
    assert isinstance(first, Loaded), getattr(first, "error", "")
    handle_unload(state, Unload(DIGEST))

    second = handle_load(
        state,
        LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus),
                  providers=(CUDA_PROVIDER, CPU_PROVIDER), required_provider=CUDA_PROVIDER),
    )

    assert isinstance(second, Loaded), getattr(second, "error", "")
    assert second.context_mib == 0, "the context was established once and stays"
    assert state.context_mib == first.context_mib


def test_nvml_reports_the_device_the_cell_runs_on():
    reading = NvmlProbe().snapshot(DEVICE)
    assert reading.total_mib > 0
    assert reading.free_mib > 0
    assert reading.device_class

    policy = ResidencyPolicy(reserve_mib=256, min_residency_secs=0)
    assert bool(policy.admit(DIGEST, 64, reading.free_mib, total_mib=reading.total_mib))


def test_a_spawned_cuda_cell_serves_a_job_end_to_end(corpus):
    path, digest, case = image_and_digest(corpus)
    cell = ExecutorCell(f"gpu{DEVICE}-0", str(DEVICE), (CUDA_PROVIDER, CPU_PROVIDER), call_timeout_s=600.0)
    try:
        cell.start()
        loaded = cell.call(
            LoadModel(digest=DIGEST, bundle_root=bundle_root(corpus),
                      providers=(CUDA_PROVIDER, CPU_PROVIDER), required_provider=CUDA_PROVIDER,
                      gpu_mem_limit_mib=2048),
            timeout_s=600.0,
        )
        assert isinstance(loaded, Loaded), getattr(loaded, "error", "")
        assert loaded.providers_assigned[0] == CUDA_PROVIDER
        assert loaded.context_mib > 0, "a spawned cell measures its own context too"

        result = cell.call(Infer("spawned", str(path), digest, DIGEST, "1"), timeout_s=600.0)
        assert result.status == "SUCCEEDED", result.error
        assert result.decision.outcome.value == case["decision"]["outcome"]
        assert result.gpu_class
    finally:
        cell.stop(30.0)
