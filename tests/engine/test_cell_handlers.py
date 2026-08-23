"""The executor cell, in process, against the synthetic corpus (DESIGN.md §16.1 tier 1, D-IP-14).

The handlers are functions over a :class:`CellState`, so the whole cell behavior runs here without
a subprocess: the same code the spawned child executes, on ``CPUExecutionProvider``, against the
seven synthetic bundles and their hand-computed answers. What this suite proves is that a result
leaving the cell is the corpus oracle's answer -- the decision included -- and that everything the
cell refuses, it refuses with the classification the scheduler acts on.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from image_processor.bundles.archive import BundleError
from image_processor.engine.cell_main import (
    CellConfig,
    CellState,
    dispatch,
    handle_infer,
    handle_load,
    handle_stats,
    handle_unload,
    member_path,
    provider_options,
    read_bundle_manifest,
    read_member,
    resolve_shape,
    static_shapes,
)
from image_processor.engine.protocol import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    PERMANENT,
    REQUIRE_LISTED,
    TRANSIENT,
    Infer,
    LoadFailed,
    LoadModel,
    Loaded,
    Shutdown,
    Stats,
    Unload,
)
from image_processor.engine.residency import StaticMemoryProbe

#: Tolerance for a score a float32 graph and a float64 oracle both compute.
SCORE_TOLERANCE = 1e-5

#: Tolerance for a coordinate, which is exact arithmetic on both sides.
GEOMETRY_TOLERANCE = 1e-6


def digest_of(bundle_key: str) -> str:
    """Return the bundle digest a test uses for one corpus bundle.

    Args:
        bundle_key: The oracle's bundle key.

    Returns:
        A ``sha256:`` digest derived from the key, stable across runs.
    """
    return "sha256:" + hashlib.sha256(bundle_key.encode("utf-8")).hexdigest()


@pytest.fixture()
def cell() -> CellState:
    """A CPU cell with no device accounting."""
    return CellState(CellConfig(cell_id="cpu-0", device_id=None, providers=(CPU_PROVIDER,)))


def load(cell_state: CellState, corpus, bundle_key: str, **kwargs) -> Loaded:
    """Make one corpus bundle resident.

    Args:
        cell_state: The cell.
        corpus: The corpus fixture.
        bundle_key: The oracle's bundle key.
        **kwargs: Overrides for the :class:`LoadModel` message.

    Returns:
        The reply.
    """
    fields = {
        "digest": digest_of(bundle_key),
        "bundle_root": str(corpus.path(corpus.expected["bundles"][bundle_key]["path"])),
        "providers": (CPU_PROVIDER,),
        "allow_cpu_only": True,
    }
    fields.update(kwargs)
    return handle_load(cell_state, LoadModel(**fields))


def infer(cell_state: CellState, corpus, bundle_key: str, image_relative: str, **kwargs):
    """Run one corpus image through one resident bundle.

    Args:
        cell_state: The cell.
        corpus: The corpus fixture.
        bundle_key: The oracle's bundle key.
        image_relative: The image path in the corpus.
        **kwargs: Overrides for the :class:`Infer` message.

    Returns:
        The result.
    """
    path = corpus.path(image_relative)
    fields = {
        "inference_id": "job-" + image_relative,
        "staged_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "digest": digest_of(bundle_key),
        "transform_version": "1",
        "queue_ms": 7.0,
    }
    fields.update(kwargs)
    return handle_infer(cell_state, Infer(**fields))


def assert_classes(actual, expected, where):
    """Compare the classification block against the oracle."""
    assert [entry.label for entry in actual] == [entry["label"] for entry in expected], where
    for got, want in zip(actual, expected):
        assert got.index == want["index"], where
        assert got.score == pytest.approx(want["score"], abs=SCORE_TOLERANCE), where


def assert_detections(actual, expected, where):
    """Compare the detection block against the oracle."""
    assert [entry.label for entry in actual] == [entry["label"] for entry in expected], where
    for got, want in zip(actual, expected):
        assert got.score == pytest.approx(want["score"], abs=SCORE_TOLERANCE), where
        assert list(got.box) == pytest.approx(want["box"], abs=GEOMETRY_TOLERANCE), where


def assert_segments(actual, expected, where):
    """Compare the segmentation block against the oracle."""
    assert set(actual) == set(expected), where
    for label, want in expected.items():
        assert actual[label]["pixels"] == want["pixels"], f"{where}:{label}"


def assert_anomaly(actual, expected, where):
    """Compare the anomaly block against the oracle."""
    assert actual["score"] == pytest.approx(expected["score"], abs=SCORE_TOLERANCE), where
    assert actual["anomalous"] is expected["anomalous"], where


ASSERTIONS = {
    "classes": assert_classes,
    "detections": assert_detections,
    "segments": assert_segments,
    "anomaly": assert_anomaly,
}


def test_every_synthetic_bundle_loads_and_answers_as_the_oracle_says(cell, corpus):
    checked = 0
    for key, bundle in corpus.expected["bundles"].items():
        reply = load(cell, corpus, key)
        assert isinstance(reply, Loaded), getattr(reply, "error", "")
        assert reply.providers_assigned == (CPU_PROVIDER,)
        assert reply.load_ms > 0
        assert reply.warmup_samples == 0
        for case in bundle["cases"]:
            where = f"{key}:{case['image']}"
            result = infer(cell, corpus, key, case["image"])
            assert result.status == "SUCCEEDED", result.error
            assert result.error is None and result.error_class is None, where
            for field, want in case["expected"].items():
                ASSERTIONS[field](getattr(result.normalized, field), want, where)
            assert result.decision.outcome.value == case["decision"]["outcome"], where
            assert result.decision.passed is case["decision"]["passed"], where
            assert result.providers == [CPU_PROVIDER], where
            assert result.gpu_device is None and result.gpu_class is None, where
            assert result.memory_high_water_mib is None, where
            assert result.timings.queue_ms == 7.0, where
            assert result.timings.total_ms >= result.timings.inference_ms, where
            checked += 1
    assert checked >= 7


def test_the_corpus_covers_the_seven_bundles_and_all_four_families(corpus):
    bundles = corpus.expected["bundles"]
    assert len(bundles) == 7
    assert {bundle["family"] for bundle in bundles.values()} == {
        "classification",
        "detection",
        "segmentation",
        "anomaly",
    }


def test_a_failed_inference_never_carries_a_decision(cell, corpus):
    load(cell, corpus, "synthetic-classification-1.0.0")
    result = infer(
        cell, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png", sha256="0" * 64
    )
    assert result.status == "FAILED"
    assert result.decision is None and result.normalized is None
    assert result.error_class == PERMANENT
    assert "INPUT_DIGEST_MISMATCH" in result.error


def build_warmup_bundle(tmp_path: Path, corpus, bundle_key: str, perturb: float = 0.0) -> Path:
    """Copy one corpus bundle and give it a golden warmup sample.

    The sample's answer comes from the model itself, so the comparison the cell runs is the real
    one; ``perturb`` moves the recorded answer away from it to prove the comparison bites.

    Args:
        tmp_path: The temporary directory to build in.
        corpus: The corpus fixture.
        bundle_key: The oracle's bundle key.
        perturb: How far to move every expected value.

    Returns:
        The new bundle root.
    """
    import shutil

    from image_processor.engine.cell_main import run_session

    source = corpus.path(corpus.expected["bundles"][bundle_key]["path"])
    root = tmp_path / ("warm-" + bundle_key)
    shutil.copytree(source, root)

    state = CellState(CellConfig(cell_id="warm", device_id=None, providers=(CPU_PROVIDER,)))
    assert isinstance(
        handle_load(
            state,
            LoadModel(digest="sha256:warm", bundle_root=str(root), providers=(CPU_PROVIDER,),
                      allow_cpu_only=True),
        ),
        Loaded,
    )
    loaded = state.sessions["sha256:warm"]
    spec = loaded.manifest.inputs[0]
    shape = resolve_shape(spec.shape)
    sample = (
        np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape).copy()
    )
    outputs = run_session(loaded, {spec.name: sample})
    state.close()

    (root / "warmup").mkdir()
    (root / "warmup" / "input-01.bin").write_bytes(sample.tobytes())
    expected = {
        name: {"shape": list(value.shape), "values": (np.asarray(value, dtype=np.float64) + perturb).ravel().tolist()}
        for name, value in outputs.items()
    }
    (root / "warmup" / "expected-01.json").write_text(json.dumps(expected), encoding="utf-8")

    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative in ("warmup/input-01.bin", "warmup/expected-01.json"):
        data = (root / relative).read_bytes()
        document["files"][relative] = hashlib.sha256(data).hexdigest()
    document["warmup"] = [
        {
            "id": "01",
            "inputs": {spec.name: "warmup/input-01.bin"},
            "expected": "warmup/expected-01.json",
        }
    ]
    document["tolerances"] = {"absolute": 1e-4, "relative": 1e-5}
    (root / "manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    return root


def test_a_bundle_with_golden_samples_is_warmed_before_it_serves(cell, corpus, tmp_path):
    root = build_warmup_bundle(tmp_path, corpus, "synthetic-classification-1.0.0")
    reply = handle_load(
        cell,
        LoadModel(digest="sha256:warm", bundle_root=str(root), providers=(CPU_PROVIDER,),
                  allow_cpu_only=True, warmup=False),
    )
    assert isinstance(reply, Loaded)
    assert reply.warmup_samples == 1


def test_a_golden_answer_that_no_longer_reproduces_refuses_the_model(cell, corpus, tmp_path):
    root = build_warmup_bundle(tmp_path, corpus, "synthetic-classification-1.0.0", perturb=0.5)
    reply = handle_load(
        cell,
        LoadModel(digest="sha256:warm", bundle_root=str(root), providers=(CPU_PROVIDER,),
                  allow_cpu_only=True),
    )
    assert isinstance(reply, LoadFailed)
    assert reply.error_class == PERMANENT
    assert reply.code == "WARMUP_MISMATCH"
    assert "sha256:warm" not in cell.sessions


def test_a_cpu_only_session_is_refused_when_the_route_does_not_allow_it(cell, corpus):
    reply = load(cell, corpus, "synthetic-classification-1.0.0", allow_cpu_only=False)
    assert isinstance(reply, LoadFailed)
    assert reply.error_class == PERMANENT
    assert reply.code == "PROVIDER_CPU_ONLY"
    assert cell.sessions == {}


def test_a_require_listed_cuda_manifest_is_refused_on_a_cpu_machine(cell, corpus):
    reply = load(
        cell,
        corpus,
        "synthetic-classification-1.0.0",
        provider_policy=REQUIRE_LISTED,
        providers_permitted=(CUDA_PROVIDER,),
        allow_cpu_only=True,
    )
    assert isinstance(reply, LoadFailed)
    assert reply.error_class == PERMANENT
    assert reply.code == "PROVIDER_NOT_PERMITTED"
    assert CPU_PROVIDER in reply.error and CUDA_PROVIDER in reply.error


def test_a_required_provider_the_machine_lacks_is_refused(cell, corpus):
    reply = load(
        cell, corpus, "synthetic-classification-1.0.0", required_provider=CUDA_PROVIDER
    )
    assert isinstance(reply, LoadFailed)
    assert reply.code == "PROVIDER_REQUIRED_MISSING"


def test_loading_a_resident_digest_again_returns_the_session_it_already_has(cell, corpus):
    first = load(cell, corpus, "synthetic-classification-1.0.0")
    second = load(cell, corpus, "synthetic-classification-1.0.0")
    assert isinstance(second, Loaded)
    assert second.load_ms == first.load_ms
    assert len(cell.sessions) == 1


def test_a_job_pinned_to_another_transform_generation_is_refused(cell, corpus):
    load(cell, corpus, "synthetic-classification-1.0.0")
    result = infer(
        cell, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png",
        transform_version="99",
    )
    assert result.status == "FAILED"
    assert result.error_class == PERMANENT
    assert "TRANSFORM_VERSION_MISMATCH" in result.error


def test_a_job_for_a_model_that_is_not_resident_is_a_retry(cell, corpus):
    result = infer(cell, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png")
    assert result.status == "FAILED"
    assert result.error_class == TRANSIENT
    assert "MODEL_NOT_RESIDENT" in result.error


def test_an_undecodable_image_is_permanent(cell, corpus, tmp_path):
    load(cell, corpus, "synthetic-classification-1.0.0")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    result = handle_infer(
        cell,
        Infer(
            inference_id="job-broken",
            staged_path=str(broken),
            sha256=hashlib.sha256(broken.read_bytes()).hexdigest(),
            digest=digest_of("synthetic-classification-1.0.0"),
            transform_version="1",
        ),
    )
    assert result.status == "FAILED"
    assert result.error_class == PERMANENT


def test_a_missing_staged_file_is_permanent(cell, corpus, tmp_path):
    load(cell, corpus, "synthetic-classification-1.0.0")
    result = handle_infer(
        cell,
        Infer("job-gone", str(tmp_path / "gone.png"), "0" * 64,
              digest_of("synthetic-classification-1.0.0"), "1"),
    )
    assert result.status == "FAILED"
    assert result.error_class == PERMANENT


def test_an_oversized_staged_file_is_refused_before_it_is_read(cell, corpus, tmp_path):
    small = replace(cell.config, decode_limits=replace(cell.config.decode_limits, max_bytes=16))
    bounded = CellState(small)
    load(bounded, corpus, "synthetic-classification-1.0.0")
    result = infer(bounded, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png")
    assert result.status == "FAILED"
    assert "INPUT_TOO_LARGE" in result.error
    assert result.error_class == PERMANENT


def test_the_device_delta_and_the_high_water_come_from_the_probe(corpus):
    probe = StaticMemoryProbe(total_mib=8192, free_mib=8192, device_class="NVIDIA Test GPU")

    class Measuring(CellState):
        """A cell whose imagined device loses memory when a session is built."""

        def snapshot(self):
            return probe.snapshot(0)

    state = Measuring(CellConfig(cell_id="gpu0-0", device_id=0, providers=(CPU_PROVIDER,)), probe=probe)
    factory = state.session_factory

    def taking(model_path, providers, options):
        probe.allocate(512)
        return factory(model_path, providers, options)

    state.session_factory = taking
    reply = load(state, corpus, "synthetic-classification-1.0.0")
    assert isinstance(reply, Loaded)
    assert reply.device_mib == 512
    assert reply.gpu_device == "0"
    assert reply.gpu_class == "NVIDIA Test GPU"

    result = infer(state, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png")
    assert result.status == "SUCCEEDED"
    assert result.memory_high_water_mib == 512

    freed = handle_unload(state, Unload(digest_of("synthetic-classification-1.0.0")))
    assert freed.was_resident is True
    assert freed.expected_mib == 512


def test_unloading_a_digest_the_cell_never_held_is_not_an_error(cell):
    reply = handle_unload(cell, Unload("sha256:absent"))
    assert reply.was_resident is False
    assert reply.freed_mib == 0


def test_stats_report_what_is_resident(cell, corpus):
    load(cell, corpus, "synthetic-classification-1.0.0")
    infer(cell, corpus, "synthetic-classification-1.0.0", "images/quadrant-red.png")
    reply = handle_stats(cell, Stats())
    assert reply.resident == (digest_of("synthetic-classification-1.0.0"),)
    assert reply.cell_id == "cpu-0"
    assert reply.inferences == 1
    assert reply.uptime_s >= 0.0
    assert reply.device_total_mib == 0


def test_dispatch_routes_every_message_and_refuses_anything_else(cell, corpus):
    assert isinstance(
        dispatch(
            cell,
            LoadModel(
                digest=digest_of("synthetic-classification-1.0.0"),
                bundle_root=str(
                    corpus.path(
                        corpus.expected["bundles"]["synthetic-classification-1.0.0"]["path"]
                    )
                ),
                providers=(CPU_PROVIDER,),
                allow_cpu_only=True,
            ),
        ),
        Loaded,
    )
    assert dispatch(cell, Stats()).resident
    assert dispatch(cell, Unload(digest_of("synthetic-classification-1.0.0"))).was_resident
    assert dispatch(cell, Shutdown()) is None
    with pytest.raises(Exception) as raised:
        dispatch(cell, "not a message")
    assert "UNKNOWN_MESSAGE" in str(raised.value)


def test_the_io_binding_path_answers_exactly_as_the_plain_run_does(cell, corpus):
    key = "synthetic-detection-grid-1.0.0"
    load(cell, corpus, key)
    plain = infer(cell, corpus, key, "images/detect-scene.png")
    cell.sessions[digest_of(key)].use_io_binding = True
    bound = infer(cell, corpus, key, "images/detect-scene.png")
    assert bound.status == "SUCCEEDED", bound.error
    assert [entry.label for entry in bound.normalized.detections] == [
        entry.label for entry in plain.normalized.detections
    ]
    for left, right in zip(bound.normalized.detections, plain.normalized.detections):
        assert left.score == pytest.approx(right.score, abs=SCORE_TOLERANCE)
        assert list(left.box) == pytest.approx(list(right.box), abs=GEOMETRY_TOLERANCE)


def bundle_root(corpus, bundle_key: str = "synthetic-classification-1.0.0") -> Path:
    """Return one corpus bundle's directory."""
    return corpus.path(corpus.expected["bundles"][bundle_key]["path"])


def test_a_member_that_leaves_the_bundle_is_refused(corpus):
    root = bundle_root(corpus)
    assert member_path(root, "model.onnx").name == "model.onnx"
    for bad in ("../escape.json", "/etc/passwd", "", "sub/../../out.json"):
        with pytest.raises(BundleError) as raised:
            member_path(root, bad)
        assert raised.value.code == "MEMBER_PATH_INVALID"


def test_a_member_is_verified_against_the_digest_the_manifest_declares(corpus, tmp_path):
    import shutil

    root = tmp_path / "bundle"
    shutil.copytree(bundle_root(corpus), root)
    manifest = read_bundle_manifest(root)
    assert read_member(root, manifest, "labels.json")

    (root / "labels.json").write_bytes(b'["tampered"]')
    with pytest.raises(BundleError) as raised:
        read_member(root, manifest, "labels.json")
    assert raised.value.code == "FILE_DIGEST_MISMATCH"


def test_an_undeclared_or_missing_member_is_refused(corpus, tmp_path):
    root = bundle_root(corpus)
    manifest = read_bundle_manifest(root)
    with pytest.raises(BundleError) as missing:
        read_member(root, manifest, "warmup/nothing.bin")
    assert missing.value.code == "FILE_MISSING"

    from image_processor.engine.cell_main import verify_declared_file

    with pytest.raises(BundleError) as undeclared:
        verify_declared_file(root, manifest, "manifest.json")
    assert undeclared.value.code == "FILE_UNDECLARED"


def test_an_oversized_member_is_refused(corpus):
    root = bundle_root(corpus)
    manifest = read_bundle_manifest(root)
    with pytest.raises(BundleError) as raised:
        read_member(root, manifest, "labels.json", max_bytes=1)
    assert raised.value.code == "MEMBER_TOO_LARGE"


def test_a_bundle_without_a_readable_manifest_is_refused(tmp_path):
    with pytest.raises(BundleError) as missing:
        read_bundle_manifest(tmp_path)
    assert missing.value.code == "MANIFEST_MISSING"

    (tmp_path / "manifest.json").write_bytes(b"{not json")
    with pytest.raises(BundleError) as invalid:
        read_bundle_manifest(tmp_path)
    assert invalid.value.code == "MANIFEST_INVALID"


def test_a_load_that_raises_is_classified_rather_than_escaping(cell, corpus):
    def explode(model_path, providers, options):
        raise RuntimeError("CUDA failure 2: out of memory")

    cell.session_factory = explode
    reply = load(cell, corpus, "synthetic-classification-1.0.0")
    assert isinstance(reply, LoadFailed)
    assert (reply.error_class, reply.code, reply.memory_pressure) == (TRANSIENT, "CUDA_OOM", True)


def test_a_load_that_poisons_the_context_is_reported_as_contaminating(cell, corpus):
    def explode(model_path, providers, options):
        raise RuntimeError("CUDA failure 700: an illegal memory access was encountered")

    cell.session_factory = explode
    reply = load(cell, corpus, "synthetic-classification-1.0.0")
    assert reply.error_class == "contaminating"


def test_the_cuda_provider_options_carry_the_device_and_the_arena_bound():
    options = provider_options((CUDA_PROVIDER, CPU_PROVIDER), device_id=1, gpu_mem_limit_mib=2048)
    assert options[0]["device_id"] == 1
    assert options[0]["arena_extend_strategy"] == "kSameAsRequested"
    assert options[0]["gpu_mem_limit"] == 2048 * (1 << 20)
    assert options[1] == {}
    assert "gpu_mem_limit" not in provider_options((CUDA_PROVIDER,), 0, None)[0]


def test_shape_helpers_read_a_dynamic_axis_as_one(corpus):
    assert resolve_shape((1, 3, 64, 64)) == (1, 3, 64, 64)
    assert resolve_shape(("N", 3, "H", "W")) == (1, 3, 1, 1)
    manifest = read_bundle_manifest(bundle_root(corpus))
    assert static_shapes(manifest) is True
    assert static_shapes(replace(manifest, dynamic_batch=True)) is False
    assert static_shapes(replace(manifest, inputs=[replace(manifest.inputs[0], shape=("N", 3))])) is False


def test_a_warmup_sample_may_describe_its_own_tensor(corpus, tmp_path):
    import shutil

    from image_processor.engine.cell_main import tolerances, warmup_expected, warmup_feed

    root = tmp_path / "bundle"
    shutil.copytree(bundle_root(corpus), root)
    manifest = read_bundle_manifest(root)
    (root / "warmup").mkdir()
    values = np.zeros((1, 3, 64, 64), dtype=np.float32)
    (root / "warmup" / "input-01.bin").write_bytes(values.tobytes())

    named = warmup_feed(
        root,
        manifest,
        {"inputs": {manifest.inputs[0].name: {"path": "warmup/input-01.bin",
                                              "dtype": "float32", "shape": [1, 3, 64, 64]}}},
    )
    assert named[manifest.inputs[0].name].shape == (1, 3, 64, 64)

    shorthand = warmup_feed(root, manifest, {"input": "warmup/input-01.bin"})
    assert shorthand[manifest.inputs[0].name].shape == (1, 3, 64, 64)
    assert warmup_feed(root, manifest, {}) == {}

    assert tolerances(manifest) == (1e-5, 0.0)
    assert warmup_expected(root, manifest, {}) == {}
    inline = warmup_expected(root, manifest, {"expected": {"outputs": {"logits": [1.0, 2.0]}}})
    assert list(inline["logits"]) == [1.0, 2.0]


def test_a_warmup_sample_the_cell_cannot_read_is_a_permanent_refusal(corpus, tmp_path):
    import shutil

    from image_processor.engine.cell_main import warmup_expected, warmup_feed
    from image_processor.engine.protocol import ProtocolError

    root = tmp_path / "bundle"
    shutil.copytree(bundle_root(corpus), root)
    manifest = read_bundle_manifest(root)
    (root / "warmup").mkdir()
    (root / "warmup" / "short.bin").write_bytes(b"\x00\x00\x00\x00")
    (root / "warmup" / "bad.json").write_bytes(b"{not json")

    for sample, code in [
        ({"inputs": {"nowhere": {"dtype": "float32", "shape": [1]}}}, "WARMUP_INVALID"),
        ({"inputs": {manifest.inputs[0].name: {}}}, "WARMUP_INVALID"),
        ({"inputs": {manifest.inputs[0].name: "warmup/short.bin"}}, "WARMUP_INVALID"),
    ]:
        with pytest.raises(ProtocolError) as raised:
            warmup_feed(root, manifest, sample)
        assert raised.value.code == code

    with pytest.raises(ProtocolError):
        warmup_feed(
            root, manifest,
            {"inputs": {manifest.inputs[0].name: {"path": "warmup/short.bin", "dtype": "float128",
                                                  "shape": [1]}}},
        )
    with pytest.raises(ProtocolError):
        warmup_expected(root, manifest, {"expected": "warmup/bad.json"})
    with pytest.raises(ProtocolError):
        warmup_expected(root, manifest, {"expected": [1, 2, 3]})
    with pytest.raises(ProtocolError):
        warmup_expected(root, manifest, {"expected": {"logits": {"shape": [2]}}})


def test_a_golden_comparison_reports_a_missing_or_mis_sized_output():
    from image_processor.engine.cell_main import compare_warmup
    from image_processor.engine.protocol import ProtocolError

    with pytest.raises(ProtocolError) as missing:
        compare_warmup({}, {"logits": np.zeros(2)}, 1e-5, 0.0, "sample 01")
    assert missing.value.code == "WARMUP_MISMATCH"

    with pytest.raises(ProtocolError) as sized:
        compare_warmup({"logits": np.zeros(3)}, {"logits": np.zeros(2)}, 1e-5, 0.0, "sample 01")
    assert "values" in sized.value.message

    compare_warmup({"logits": np.zeros(2)}, {"logits": np.zeros(2)}, 1e-5, 0.0, "sample 01")


def test_priming_a_session_that_refuses_zeros_is_not_a_load_failure(cell, corpus):
    from image_processor.engine.cell_main import prime_session

    load(cell, corpus, "synthetic-classification-1.0.0")
    loaded = cell.sessions[digest_of("synthetic-classification-1.0.0")]

    class Refusing:
        """A session that raises whatever it is fed."""

        def run(self, names, feed):
            raise RuntimeError("no")

    loaded.session = Refusing()
    prime_session(loaded)

    loaded.manifest = replace(loaded.manifest, inputs=[])
    prime_session(loaded)
