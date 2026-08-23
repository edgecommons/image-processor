"""Tier-1 cover for the tier-3 corpus generator (tools/synth_corpus.py).

The generator itself is only ever run on a GPU host against 23 GiB of real architectures, so this
suite runs it the other way round: a few one-megabyte bundles grown from the tier-1 synthetic
classification graph, on ``CPUExecutionProvider``, with no network and no fetched asset. That is
enough to assert everything that can go wrong silently -- the plan, the padding arithmetic, the
manifest, the signature, the staging, the determinism of a seed, and that the padded graph still
loads and reproduces its own golden warmup sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_processor.bundles import BundleCache, validate_document
from image_processor.engine.cell_main import CellConfig, CellState, handle_infer, handle_load
from image_processor.engine.protocol import CPU_PROVIDER, Infer, LoadModel, Loaded
from tests.fixtures.build import classification_graph, quadrant_image
from tools import synth_corpus

#: The labels the tier-1 classification graph answers with.
LABELS = ["red", "green", "blue", "other"]

#: A tier small enough that a whole corpus fits in a temporary directory.
TINY_MIB = 1

#: The tier-1 classification graph's largest initializer holds twelve elements, so the corpus
#: default floor of sixteen would leave every weight untouched. The generator is told so.
TINY_MIN_ELEMENTS = 4


def _document() -> dict:
    """Build the authored manifest of the tier-1 classification graph.

    Returns:
        The manifest document, in the shape ``synth_corpus.manifest_document`` copies from.
    """
    return {
        "schemaVersion": 1,
        "modelId": "base",
        "version": "0.0.0",
        "minOnnxRuntime": "1.17.0",
        "providersPermitted": ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "providerPolicy": "preferListed",
        "family": "classification",
        "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 64, 64]}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 4]}],
        "dynamicBatch": False,
        "familyParams": {"labels": LABELS, "activation": "softmax", "topK": 4},
        "preprocess": {
            "colorOrder": "RGB",
            "resize": {"mode": "stretch", "width": 64, "height": 64, "interpolation": "bilinear"},
            "scale": 1.0 / 255.0,
            "mean": 0.0,
            "std": 1.0,
            "layout": "NCHW",
            "dtype": "float32",
            "inputName": "images",
        },
        "decisionRules": {
            "pass": {"path": "$.classes[0].score", "op": ">=", "value": 0.25},
            "confidence": "$.classes[0].score",
            "threshold": 0.25,
            "outcomeOnPass": "CLEAR",
            "outcomeOnFail": "HOLD",
        },
        "maxResultItems": 8,
        "estimatedDeviceMiB": 8,
        "warmup": [],
        "tolerances": {"absolute": 1e-5},
        "compatibilityKeys": {},
        "provenance": {"publisher": "tier-1"},
        "keyId": None,
        "transformVersion": "1",
    }


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> Path:
    """One encoded image the generated bundles warm on.

    Returns:
        The PNG file.
    """
    from PIL import Image

    path = tmp_path_factory.mktemp("synth-image") / "quadrant-red.png"
    Image.fromarray(quadrant_image((255, 0, 0), 64)).save(path, format="PNG")
    return path


@pytest.fixture(scope="module")
def graph(tmp_path_factory) -> Path:
    """The tier-1 classification graph, on disk.

    Returns:
        The ONNX file.
    """
    path = tmp_path_factory.mktemp("synth-graph") / "model.onnx"
    path.write_bytes(classification_graph())
    return path


@pytest.fixture
def bases(graph, image) -> dict:
    """One base architecture built from the tier-1 graph.

    Returns:
        The architectures by key.
    """
    return {
        "tiny": synth_corpus.BaseArchitecture(
            key="tiny",
            graph=graph,
            document=_document(),
            warmup_image=image,
            input_name="images",
            activation_headroom_mib=8,
        )
    }


def test_a_small_corpus_builds_signs_and_stages(tmp_path, bases):
    """Every planned bundle is packed, signed, staged, and described in the index."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=2, tiers=[TINY_MIB], seed=7,
        min_elements=TINY_MIN_ELEMENTS,
    )
    assert len(index["bundles"]) == 2
    assert index["tiersMiB"] == [TINY_MIB]
    assert index["seed"] == 7
    assert index["totalModelMiB"] == pytest.approx(2 * TINY_MIB, abs=0.1)
    assert (tmp_path / "corpus" / "corpus.json").is_file()
    assert (tmp_path / "corpus" / "keys" / "tier3.pem").is_file()
    assert not (tmp_path / "corpus" / "src").exists()

    cache = BundleCache(tmp_path / "corpus" / "cache", schema_path=synth_corpus.SCHEMA_PATH)
    digests = set()
    for entry in index["bundles"]:
        archive = tmp_path / "corpus" / entry["archive"]
        assert archive.is_file() and archive.stat().st_size == entry["archiveBytes"]
        assert entry["modelBytes"] == pytest.approx(TINY_MIB * 2**20, rel=0.01)
        assert entry["estimatedDeviceMiB"] == TINY_MIB + 8
        assert entry["initializersPerturbed"] > 0
        assert entry["padElements"] > 0
        digests.add(entry["digest"])
        bundle = cache.get(entry["digest"], verify=True)
        assert bundle is not None
        assert bundle.manifest.model_id == entry["modelId"]
        assert bundle.manifest.key_id == synth_corpus.KEY_ID
        assert len(bundle.manifest.warmup) == 1
    assert len(digests) == 2, "two bundles of the same architecture shared a digest"


def test_the_generated_manifest_satisfies_the_shipped_schema(tmp_path, bases):
    """A manifest the generator writes is one the component's own schema accepts."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=1, tiers=[TINY_MIB], seed=11, stage=False
    )
    entry = index["bundles"][0]
    import tarfile

    with tarfile.open(tmp_path / "corpus" / entry["archive"]) as archive:
        document = json.loads(archive.extractfile("manifest.json").read().decode("utf-8"))
    validate_document(document, synth_corpus.SCHEMA_PATH)
    assert document["modelId"] == entry["modelId"]
    assert document["tolerances"]["absolute"] == synth_corpus.WARMUP_ABSOLUTE_TOLERANCE
    assert document["warmup"][0]["inputName"] == "images"
    assert set(document["files"]) >= {
        "model.onnx",
        "labels.json",
        "transforms.json",
        "warmup/input-01.bin",
        "warmup/expected-01.json",
    }


def test_a_corpus_can_be_built_without_staging(tmp_path, bases):
    """--no-stage packs and signs but leaves the cache alone."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=1, tiers=[TINY_MIB], seed=3, stage=False
    )
    assert index["staged"] is False
    assert index["bundles"][0]["staged"] is False
    assert not (tmp_path / "corpus" / "cache").exists()


def test_the_same_seed_produces_the_same_bundles(tmp_path, bases):
    """The corpus is reproducible: same seed, same digests, byte for byte."""
    first = synth_corpus.build_corpus(
        tmp_path / "a", bases, count=2, tiers=[TINY_MIB], seed=99, stage=False,
        min_elements=TINY_MIN_ELEMENTS,
    )
    second = synth_corpus.build_corpus(
        tmp_path / "b", bases, count=2, tiers=[TINY_MIB], seed=99, stage=False,
        min_elements=TINY_MIN_ELEMENTS,
    )
    assert [entry["digest"] for entry in first["bundles"]] == [
        entry["digest"] for entry in second["bundles"]
    ]


def test_a_different_seed_produces_different_bundles(tmp_path, bases):
    """A different seed perturbs different weights, so nothing is shared."""
    first = synth_corpus.build_corpus(
        tmp_path / "a", bases, count=1, tiers=[TINY_MIB], seed=1, stage=False,
        min_elements=TINY_MIN_ELEMENTS,
    )
    second = synth_corpus.build_corpus(
        tmp_path / "b", bases, count=1, tiers=[TINY_MIB], seed=2, stage=False,
        min_elements=TINY_MIN_ELEMENTS,
    )
    assert first["bundles"][0]["digest"] != second["bundles"][0]["digest"]


def test_the_padded_graph_loads_and_reproduces_its_golden(tmp_path, bases, image):
    """The pad survives graph optimization, and the CPU-made golden warmup passes on load."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=1, tiers=[TINY_MIB], seed=5
    )
    entry = index["bundles"][0]
    cache = BundleCache(tmp_path / "corpus" / "cache", schema_path=synth_corpus.SCHEMA_PATH)
    bundle = cache.get(entry["digest"])
    state = CellState(CellConfig(cell_id="cpu-0", device_id=None, providers=(CPU_PROVIDER,)))
    reply = handle_load(
        state,
        LoadModel(
            digest=entry["digest"],
            bundle_root=str(bundle.root),
            providers=(CPU_PROVIDER,),
            warmup=True,
            allow_cpu_only=True,
        ),
    )
    assert isinstance(reply, Loaded), getattr(reply, "error", "")
    assert reply.warmup_samples == 1, "the golden warmup sample did not run"

    loaded = state.sessions[entry["digest"]]
    assert synth_corpus.PAD_OUTPUT in loaded.output_names, "the pad probe was pruned"

    import hashlib

    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    result = handle_infer(state, Infer("tier1", str(image), digest, entry["digest"], "1", 0.0))
    assert result.status == "SUCCEEDED", result.error
    assert result.normalized.classes, "the padded graph produced no classification"


def test_the_probe_carries_the_bundle_ordinal(tmp_path, bases):
    """Each bundle's pad answers with its own ordinal, so a mixed-up bundle fails its warmup."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=3, tiers=[TINY_MIB], seed=13
    )
    assert [entry["probeValue"] for entry in index["bundles"]] == [
        synth_corpus.PROBE_SCALE * ordinal for ordinal in (1, 2, 3)
    ]
    cache = BundleCache(tmp_path / "corpus" / "cache", schema_path=synth_corpus.SCHEMA_PATH)
    for entry in index["bundles"]:
        bundle = cache.get(entry["digest"])
        golden = json.loads(
            (bundle.root / "warmup" / "expected-01.json").read_text(encoding="utf-8")
        )
        assert golden[synth_corpus.PAD_OUTPUT]["values"] == [float(entry["probeValue"])]


def test_the_plan_cycles_tiers_and_balances_architectures():
    """The tier cycles fastest and the architecture changes once per full cycle."""
    specs = synth_corpus.plan(8, [50, 200], ["a", "b"], seed=100)
    assert [spec.tier_mib for spec in specs] == [50, 200, 50, 200, 50, 200, 50, 200]
    assert [spec.base for spec in specs] == ["a", "a", "b", "b", "a", "a", "b", "b"]
    assert [spec.seed for spec in specs] == list(range(100, 108))
    assert specs[0].key == "synth-a-t00050-000"
    assert specs[0].model_id == specs[0].key


@pytest.mark.parametrize(
    "count,tiers,bases",
    [(0, [50], ["a"]), (2, [], ["a"]), (2, [50], [])],
)
def test_an_empty_plan_is_refused(count, tiers, bases):
    """A corpus with no bundles, no tiers, or no architectures is refused loudly."""
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.plan(count, tiers, bases, seed=1)
    assert failure.value.code == "PLAN_EMPTY"


def test_the_pad_is_never_empty():
    """A tier smaller than the graph still leaves one element for the probe to read."""
    assert synth_corpus.pad_elements_for(1024, 4_000_000) == 1
    assert synth_corpus.pad_elements_for(8 * 2**20, 0) == (8 * 2**20 - 1024) // 4


def test_the_pad_refuses_an_input_the_graph_does_not_declare(graph):
    """Padding names the input the index is derived from; a wrong name is a build error."""
    import onnx

    model = onnx.load(str(graph))
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.pad_model(model, 16, 1.0, "not-an-input")
    assert failure.value.code == "PAD_INPUT_UNKNOWN"


def test_a_tier_larger_than_a_protobuf_is_refused(tmp_path, bases):
    """A single ONNX protobuf cannot hold two gigabytes, so the tier is refused up front."""
    spec = synth_corpus.BundleSpec(index=0, base="tiny", tier_mib=4096, seed=1)
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.write_source(spec, bases["tiny"], tmp_path / "src")
    assert failure.value.code == "TIER_TOO_LARGE"


def test_small_initializers_are_left_alone(graph):
    """Biases and scalars are not perturbed: it buys no distinctness and destabilizes a graph."""
    import onnx

    model = onnx.load(str(graph))
    before = {
        entry.name: onnx.numpy_helper.to_array(entry).copy() for entry in model.graph.initializer
    }
    touched = synth_corpus.perturb_weights(model, seed=4, epsilon=0.05, min_elements=8)
    after = {entry.name: onnx.numpy_helper.to_array(entry) for entry in model.graph.initializer}
    assert touched == sum(1 for value in before.values() if value.size >= 8)
    for name, value in before.items():
        if value.size >= 8:
            assert not np.array_equal(value, after[name]), f"{name} was not perturbed"
        else:
            assert np.array_equal(value, after[name]), f"{name} should have been left alone"


def test_the_signing_key_is_a_function_of_the_seed(tmp_path):
    """The key is derived, not generated: a fresh key would change every bundle digest."""
    first = synth_corpus.signing_key(42)
    assert first == synth_corpus.signing_key(42)
    assert first != synth_corpus.signing_key(43)
    assert first[0].startswith(b"-----BEGIN PRIVATE KEY-----")
    assert len(first[1]) == 32


def test_a_missing_corpus_is_reported(tmp_path):
    """Reading an index that was never built says so rather than raising a file error."""
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.load_index(tmp_path)
    assert failure.value.code == "CORPUS_MISSING"


def test_an_existing_corpus_index_round_trips(tmp_path, bases):
    """The written index is the one load_index reads back."""
    index = synth_corpus.build_corpus(
        tmp_path / "corpus", bases, count=1, tiers=[TINY_MIB], seed=21, stage=False
    )
    assert synth_corpus.load_index(tmp_path / "corpus") == index


def test_a_missing_tier2_asset_is_reported(tmp_path):
    """Without the fetched tier-2 corpus the generator names the tool that fetches it."""
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.tier2_bases(tmp_path / "empty-cache")
    assert failure.value.code == "ASSET_MISSING"
    assert "fetch_test_assets" in failure.value.message


def test_an_empty_warmup_directory_is_reported(tmp_path, monkeypatch, graph):
    """An asset directory that exists but holds nothing is refused, not silently skipped."""
    cache = tmp_path / "cache"
    (cache / "labels-imagenet-synset").mkdir(parents=True)
    (cache / "labels-imagenet-synset" / "synset.txt").write_text(
        "".join(f"n{i:08d} class-{i}" + chr(10) for i in range(1000)), encoding="utf-8"
    )
    (cache / "model-mobilenetv2-12").mkdir(parents=True)
    (cache / "model-mobilenetv2-12" / "mobilenetv2-12.onnx").write_bytes(graph.read_bytes())
    (cache / "dataset-imagenette2-160" / "extracted" / "imagenette2-160" / "val").mkdir(parents=True)
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.tier2_bases(cache)
    assert failure.value.code == "ASSET_MISSING"
    assert "warmup image" in failure.value.message


def test_a_warmup_that_needs_two_inputs_is_refused(tmp_path, bases, monkeypatch):
    """The single-input warmup form cannot describe a two-tensor feed, so it is refused."""

    class TwoInputFamily:
        """A family whose preprocessing produces two tensors."""

        def preprocess(self, image, manifest):
            """Return two tensors."""
            return {"images": np.zeros((1, 3, 64, 64), np.float32), "extra": np.zeros(1, np.float32)}

    monkeypatch.setattr(synth_corpus, "family_for", lambda manifest: TwoInputFamily())
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(bases["tiny"].graph.read_bytes())
    with pytest.raises(synth_corpus.CorpusError) as failure:
        synth_corpus.golden_warmup(
            directory, _document(), bases["tiny"].warmup_image.read_bytes()
        )
    assert failure.value.code == "WARMUP_MULTI_INPUT"


def test_the_command_line_builds_a_corpus(tmp_path, bases, monkeypatch, capsys):
    """The entry point wires the arguments to the builder and reports what it wrote."""
    monkeypatch.setattr(synth_corpus, "tier2_bases", lambda cache, **kwargs: bases)
    code = synth_corpus.main(
        ["--out", str(tmp_path / "corpus"), "--count", "2", "--tiers", str(TINY_MIB), "--seed", "6"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "2 bundles" in printed
    index = synth_corpus.load_index(tmp_path / "corpus")
    assert len(index["bundles"]) == 2


def test_the_command_line_reports_a_bad_tier_list(tmp_path, capsys):
    """A tier list that is not numbers is a usage error, not a traceback."""
    code = synth_corpus.main(["--out", str(tmp_path / "corpus"), "--tiers", "small"])
    assert code == 2
    assert "TIERS_INVALID" in capsys.readouterr().err


def test_the_command_line_reports_a_missing_asset(tmp_path, capsys):
    """Without the tier-2 cache the entry point exits with the code, not a stack trace."""
    code = synth_corpus.main(
        ["--out", str(tmp_path / "corpus"), "--cache", str(tmp_path / "nothing")]
    )
    assert code == 2
    assert "ASSET_MISSING" in capsys.readouterr().err
