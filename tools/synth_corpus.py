"""Synthesize the tier-3 residency corpus (DESIGN.md section 16.1 tier 3, D-IP-17).

Tier 3 measures residency, eviction, and burst behaviour on a real device, and that needs more
distinct model generations than fit in device memory. Real models cannot supply them: the tier-2
corpus is seven graphs totalling under 300 MB, so a 16 GB card holds all of them at once and
nothing is ever evicted. This tool manufactures the pressure instead. It takes the two tier-2
architectures DESIGN.md names -- MobileNetV2 and YOLOX-S -- and produces N bundles that are
distinct in every way the component keys residency by:

* **Perturbed weights.** Every float initializer of at least ``--min-elements`` entries is scaled
  by ``1 + epsilon * N(0, 1)`` from a seeded generator, so no two bundles share a byte of weight
  data, each has its own SHA-256, and none is a copy the runtime or the page cache can share.
* **A padded initializer.** One ``float32`` initializer sized so that ``model.onnx`` reaches the
  bundle's target tier. This is what makes a 14 MB MobileNetV2 occupy 1.5 GB of device memory.

The padding trick, stated honestly
----------------------------------
The pad costs device memory and one host-to-device copy per load. It costs no meaningful compute.
It is wired into the graph as::

    ec_pad_min    = ReduceMin(<model input>)           # one reduction over the input tensor
    ec_pad_zeroed = Mul(ec_pad_min, 0.0)
    ec_pad_idx    = Cast(ec_pad_zeroed, INT64)         # always 0, but not constant-foldable
    ec_pad_probe  = Gather(ec_pad, ec_pad_idx, axis=0) # one element out of the pad

``ec_pad_probe`` is an extra graph output the manifest does not declare, exactly as the tier-2
FCN-ResNet50 bundle leaves the auxiliary head undeclared: the task family reads
``familyParams.outputName`` and never looks at it. The index is derived from the input tensor at
run time, so ONNX Runtime cannot constant-fold the ``Gather`` away and cannot prune the pad as an
unused initializer, which a plain unused initializer would be. Measured on an RTX 5080, a 512 MiB
pad on MobileNetV2 raised the session's device footprint by 512 MiB and left warm inference at
1.9 ms, the same as unpadded.

So a tier-3 bundle is honest about **load cost, host-to-device transfer, device occupancy,
eviction cost, and reload cost**, which is what tier 3 measures. It is *not* a model whose
inference time reflects a 1.5 GB network: the arithmetic is still MobileNetV2's or YOLOX-S's, plus
one reduction over the input. Read the inference latencies in the results as "the latency of the
base architecture, measured under residency pressure".

Warmup goldens and their tolerance
----------------------------------
Every bundle carries one golden warmup sample -- the raw preprocessed input tensor and the raw
session outputs for it -- produced on ``CPUExecutionProvider``, so the corpus is reproducible on
any machine and a bundle is not tied to the card that generated it. The declared tolerance is
therefore a **provider** tolerance: at load time the sample is replayed on whatever provider the
session got, and on CUDA that is a different set of kernels.

The numbers it is set from were measured, not guessed. Over a whole 40-bundle corpus on an
RTX 5080, the largest element-wise difference between the CPU golden and the CUDA session was:

===============  ===  ========  ========  ========
architecture       n  median    p95       max
===============  ===  ========  ========  ========
MobileNetV2       20  0.019     0.026     0.030
YOLOX-S           20  0.098     0.252     0.318
===============  ===  ========  ========  ========

YOLOX-S is the wide one because its head leaves the box sizes in log space and puts objectness and
the class scores through a sigmoid inside the graph, so a float32 kernel difference is amplified
before it reaches the output. :data:`WARMUP_ABSOLUTE_TOLERANCE` is set to 1.5, about five times the
worst case, which leaves room for a second GPU class without letting a grossly wrong graph through.

Bundle **identity** is not left to that tolerance. ``ec_pad_probe`` answers with the bundle's
ordinal multiplied by :data:`PROBE_SCALE`, a value a ``Gather`` reproduces exactly on both
providers (measured difference: zero), so consecutive bundles are 1000 apart and loading the wrong
bundle fails its warmup on the probe no matter how wide the head tolerance is.

Examples:
    Build the default corpus (40 bundles, about 23 GiB) on a local disk::

        python tools/synth_corpus.py --out /home/me/ip-corpus

    Build a small corpus of 300 MiB bundles::

        python tools/synth_corpus.py --out /tmp/corpus --count 4 --tiers 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.bundles import (  # noqa: E402  (path bootstrap must run first)
    BundleCache,
    stage_bundle,
)
from image_processor.engine.decode import DecodeLimits, decode_image  # noqa: E402
from image_processor.engine.families import family_for  # noqa: E402
from tools.make_bundle import make_bundle  # noqa: E402

logger = logging.getLogger("synth_corpus")

#: The bundle-manifest schema every generated manifest is validated against.
SCHEMA_PATH = _REPO_ROOT / "schemas" / "model-bundle-manifest.schema.json"

#: Target ``model.onnx`` sizes in MiB, in the order DESIGN.md section 16.1 lists them.
DEFAULT_TIERS_MIB = (50, 200, 600, 1500)

#: How many bundles the default corpus holds. Ten per tier is about 23 GiB.
DEFAULT_COUNT = 40

#: The seed every pseudo-random choice derives from, so two runs produce identical digests.
DEFAULT_SEED = 20260823

#: Relative standard deviation of the multiplicative weight perturbation.
DEFAULT_EPSILON = 0.02

#: Initializers smaller than this are left alone: they are biases, shapes, and scalars whose
#: perturbation buys no distinctness and can destabilize a normalization.
DEFAULT_MIN_ELEMENTS = 16

#: The signing key id every tier-3 bundle is signed with.
KEY_ID = "tier3-synth-publisher"

#: Absolute warmup tolerance, set from the measured CPU-versus-CUDA deltas documented above.
WARMUP_ABSOLUTE_TOLERANCE = 1.5

#: Relative warmup tolerance, which carries the large-magnitude outputs.
WARMUP_RELATIVE_TOLERANCE = 1e-3

#: Decimal places the golden outputs are rounded to. Far inside the tolerance, and it keeps a
#: YOLOX-S golden (714,000 numbers) near 6 MiB rather than 18 MiB.
GOLDEN_DECIMALS = 6

#: A single ONNX protobuf cannot exceed 2 GiB; refuse a tier that would not serialize.
MAX_MODEL_BYTES = 1900 * 2**20

#: The padded initializer.
PAD_INITIALIZER = "ec_pad"

#: The zero constant the run-time index is multiplied by.
PAD_ZERO = "ec_pad_zero"

#: The undeclared extra graph output that keeps the pad alive.
PAD_OUTPUT = "ec_pad_probe"

#: Bytes of protobuf framing the pad and its nodes add beyond the raw tensor.
PAD_OVERHEAD_BYTES = 1024

#: What a bundle's ordinal is multiplied by to give its probe value. It puts consecutive bundles
#: far enough apart that the head's provider tolerance can never mask a swapped bundle.
PROBE_SCALE = 1000

#: The element type of each raw warmup tensor, keyed by the manifest spelling.
WARMUP_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "uint8": np.uint8,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
}


class CorpusError(RuntimeError):
    """A corpus cannot be built as asked.

    Attributes:
        code: Stable SCREAMING_SNAKE token.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE token.
            message: Operator-readable detail.
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class BaseArchitecture:
    """One architecture the corpus is generated from.

    Attributes:
        key: Short name, which prefixes every bundle derived from it.
        graph: The ONNX file to perturb and pad.
        document: The authored manifest document -- family, family parameters, preprocessing,
            decision rules, inputs, and outputs -- without ``files``, ``warmup``, ``modelId``, or
            ``version``.
        warmup_image: The encoded image the golden warmup sample is taken from.
        input_name: The graph input the pad index is derived from.
        activation_headroom_mib: Device memory this architecture needs beyond its weights, added
            to the tier to give ``estimatedDeviceMiB``.
    """

    key: str
    graph: Path
    document: Dict[str, Any]
    warmup_image: Path
    input_name: str
    activation_headroom_mib: int = 384


@dataclass(frozen=True)
class BundleSpec:
    """One bundle the corpus plan calls for.

    Attributes:
        index: The bundle's ordinal in the corpus, which seeds it and identifies it in its probe.
        base: The architecture key it derives from.
        tier_mib: The target ``model.onnx`` size in MiB.
        seed: The generator seed for this bundle's weight perturbation.
    """

    index: int
    base: str
    tier_mib: int
    seed: int

    @property
    def key(self) -> str:
        """Return the bundle's corpus-unique name."""
        return f"synth-{self.base}-t{self.tier_mib:05d}-{self.index:03d}"

    @property
    def model_id(self) -> str:
        """Return the ``modelId`` the manifest declares."""
        return self.key


def signing_key(seed: int) -> tuple:
    """Derive the corpus signing keypair from its seed.

    A fresh key per run would make the corpus irreproducible: ``manifest.sig`` is a member of the
    tarball, so a different key gives a different bundle digest for identical model bytes. The key
    is derived from the seed instead, which keeps a signing key out of the repository, keeps the
    whole corpus a function of ``--seed``, and is safe because nothing outside a test ever trusts
    it. It is not a key any real bundle is signed with.

    Args:
        seed: The corpus seed.

    Returns:
        A ``(private_pem, raw_public_key)`` pair.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    material = hashlib.sha256(f"edgecommons/image-processor tier-3 corpus {seed}".encode()).digest()
    private = ed25519.Ed25519PrivateKey.from_private_bytes(material)
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return private_pem, public_raw


def _onnx():
    """Import ``onnx`` on first use.

    The runtime never imports it (LLD section 1); only this generator and the fixture builder do.

    Returns:
        The ``onnx`` module.
    """
    import onnx

    return onnx


def plan(count: int, tiers: Sequence[int], bases: Sequence[str], seed: int) -> List[BundleSpec]:
    """Lay out which bundle is which architecture at which size.

    The tier cycles fastest, so any prefix of the plan already spans every size; the architecture
    changes once per full cycle of tiers, so the architectures stay balanced.

    Args:
        count: How many bundles to plan.
        tiers: The target sizes in MiB.
        bases: The architecture keys, in order.
        seed: The corpus seed.

    Returns:
        The plan, in build order.

    Raises:
        CorpusError: ``PLAN_EMPTY`` when the count, the tiers, or the architectures are empty.
    """
    if count <= 0:
        raise CorpusError("PLAN_EMPTY", "a corpus needs at least one bundle")
    if not tiers:
        raise CorpusError("PLAN_EMPTY", "a corpus needs at least one tier")
    if not bases:
        raise CorpusError("PLAN_EMPTY", "a corpus needs at least one base architecture")
    specs = []
    for index in range(count):
        tier = int(tiers[index % len(tiers)])
        base = bases[(index // len(tiers)) % len(bases)]
        specs.append(BundleSpec(index=index, base=base, tier_mib=tier, seed=seed + index))
    return specs


def perturb_weights(
    model, seed: int, epsilon: float = DEFAULT_EPSILON, min_elements: int = DEFAULT_MIN_ELEMENTS
) -> int:
    """Scale every substantial float initializer by seeded multiplicative noise.

    Multiplicative noise keeps each tensor's scale, which keeps batch normalization stable and the
    outputs finite, while making the bytes -- and so the digest -- unique per seed.

    Args:
        model: The ``onnx.ModelProto`` to modify in place.
        seed: The generator seed.
        epsilon: Relative standard deviation of the noise.
        min_elements: Initializers with fewer elements are left alone.

    Returns:
        How many initializers were perturbed.
    """
    onnx = _onnx()
    rng = np.random.default_rng(seed)
    touched = 0
    for initializer in model.graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        array = onnx.numpy_helper.to_array(initializer)
        if array.size < min_elements:
            continue
        noise = rng.standard_normal(array.shape).astype(np.float32)
        perturbed = (array * (1.0 + np.float32(epsilon) * noise)).astype(np.float32)
        initializer.CopyFrom(onnx.numpy_helper.from_array(perturbed, initializer.name))
        touched += 1
    return touched


def pad_elements_for(target_bytes: int, graph_bytes: int) -> int:
    """Return how many ``float32`` pad elements bring a graph up to a target size.

    Args:
        target_bytes: The size ``model.onnx`` should reach.
        graph_bytes: The serialized size of the graph before padding.

    Returns:
        The element count, at least one so the ``Gather`` always has something to read.
    """
    remaining = int(target_bytes) - int(graph_bytes) - PAD_OVERHEAD_BYTES
    return max(1, remaining // 4)


def pad_model(model, elements: int, probe_value: float, input_name: str) -> None:
    """Add the padded initializer and the run-time-gated probe that keeps it alive.

    Args:
        model: The ``onnx.ModelProto`` to modify in place.
        elements: How many ``float32`` entries the pad holds.
        probe_value: The value stored at index zero, which the warmup golden records. It is the
            bundle's identity, so it is spaced far enough apart to survive the head tolerance.
        input_name: The graph input the pad index is derived from.

    Raises:
        CorpusError: ``PAD_INPUT_UNKNOWN`` when the graph declares no such input.
    """
    onnx = _onnx()
    if input_name not in {entry.name for entry in model.graph.input}:
        raise CorpusError("PAD_INPUT_UNKNOWN", f"the graph declares no input named {input_name!r}")
    pad = np.zeros(int(elements), dtype=np.float32)
    pad[0] = np.float32(probe_value)
    graph = model.graph
    graph.initializer.append(onnx.numpy_helper.from_array(pad, PAD_INITIALIZER))
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(0.0, dtype=np.float32), PAD_ZERO)
    )
    graph.node.append(
        onnx.helper.make_node(
            "ReduceMin", [input_name], ["ec_pad_min"], keepdims=0, name="ec_pad_min_n"
        )
    )
    graph.node.append(
        onnx.helper.make_node(
            "Mul", ["ec_pad_min", PAD_ZERO], ["ec_pad_zeroed"], name="ec_pad_mul_n"
        )
    )
    graph.node.append(
        onnx.helper.make_node(
            "Cast",
            ["ec_pad_zeroed"],
            ["ec_pad_idx"],
            to=onnx.TensorProto.INT64,
            name="ec_pad_cast_n",
        )
    )
    graph.node.append(
        onnx.helper.make_node(
            "Gather",
            [PAD_INITIALIZER, "ec_pad_idx"],
            [PAD_OUTPUT],
            axis=0,
            name="ec_pad_gather_n",
        )
    )
    graph.output.append(onnx.helper.make_tensor_value_info(PAD_OUTPUT, onnx.TensorProto.FLOAT, []))


def manifest_document(spec: BundleSpec, base: BaseArchitecture, tier_bytes: int) -> Dict[str, Any]:
    """Build one bundle's manifest document from its architecture's.

    Every field that describes the model -- family, family parameters, preprocessing, decision
    rules, tensors -- is the tier-2 manifest verbatim. What changes is the identity, the memory
    estimate, the provenance, and the warmup tolerances.

    Args:
        spec: The bundle being built.
        base: The architecture it derives from.
        tier_bytes: The target ``model.onnx`` size in bytes.

    Returns:
        The manifest document, without ``files`` and ``warmup``, which are filled in later.
    """
    document = json.loads(json.dumps(base.document))
    document["modelId"] = spec.model_id
    document["version"] = "1.0.0"
    document["estimatedDeviceMiB"] = int(round(tier_bytes / 2**20) + base.activation_headroom_mib)
    document["keyId"] = KEY_ID
    document["tolerances"] = {
        "absolute": WARMUP_ABSOLUTE_TOLERANCE,
        "relative": WARMUP_RELATIVE_TOLERANCE,
    }
    document["provenance"] = {
        "publisher": "edgecommons/image-processor tier-3 residency corpus",
        "notes": (
            f"tools/synth_corpus.py: {base.key} with seed {spec.seed}, "
            f"padded to {tier_bytes // 2**20} MiB"
        ),
    }
    return document


def golden_warmup(directory: Path, document: Dict[str, Any], image: bytes) -> Dict[str, Any]:
    """Write one golden warmup sample and return its manifest entry.

    The sample is produced the way the component produces one (DESIGN.md section 16.1): the image
    is decoded, the bundle's own task family preprocesses it, and the bundle's own graph answers on
    ``CPUExecutionProvider``. What lands on disk is the raw little-endian input tensor and the
    session's raw outputs, which is the pair ``engine.cell_main.run_warmup`` compares.

    Args:
        directory: The bundle directory, which already holds ``model.onnx``.
        document: The manifest document so far, without its ``warmup`` entry.
        image: The encoded warmup image.

    Returns:
        One ``manifest.warmup`` entry, in the schema's single-input form.

    Raises:
        CorpusError: ``WARMUP_MULTI_INPUT`` when the preprocessing produces more than one input
            tensor, which the single-input form cannot describe.
    """
    import onnxruntime as ort

    from tests.fixtures.build import manifest_from_document

    # ``files`` is filled in by ``make_bundle`` when the directory is packed; the family only
    # needs the tensor and preprocessing declarations to build the feed.
    manifest = manifest_from_document(dict(document, files=document.get("files", {})))
    family = family_for(manifest)
    decoded = decode_image(image, DecodeLimits())
    feed = family.preprocess(decoded, manifest)
    if len(feed) != 1:
        raise CorpusError(
            "WARMUP_MULTI_INPUT",
            f"{document['modelId']} preprocesses to {len(feed)} inputs, not one",
        )
    name, tensor = next(iter(feed.items()))

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(names, session.run(names, feed)))
    del session

    dtype = next(spec["dtype"] for spec in document["inputs"] if spec["name"] == name)
    array = np.ascontiguousarray(tensor, dtype=WARMUP_DTYPES[dtype])
    (directory / "warmup").mkdir(exist_ok=True)
    (directory / "warmup" / "input-01.bin").write_bytes(
        array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
    )
    expected = {
        output: {
            "shape": [int(value) for value in np.shape(result)],
            "values": np.round(
                np.asarray(result, dtype=np.float64).ravel(), GOLDEN_DECIMALS
            ).tolist(),
        }
        for output, result in outputs.items()
    }
    (directory / "warmup" / "expected-01.json").write_bytes(
        (json.dumps(expected, separators=(",", ":")) + "\n").encode("utf-8")
    )
    return {
        "input": "warmup/input-01.bin",
        "expected": "warmup/expected-01.json",
        "inputName": name,
        "dtype": dtype,
        "shape": [int(value) for value in array.shape],
    }


def write_source(
    spec: BundleSpec,
    base: BaseArchitecture,
    src_dir: Path,
    epsilon: float = DEFAULT_EPSILON,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
) -> Dict[str, Any]:
    """Lay out one bundle source directory: the padded graph, labels, transforms, and manifest.

    Args:
        spec: The bundle being built.
        base: The architecture it derives from.
        src_dir: The directory to write, which is created.
        epsilon: Relative standard deviation of the weight perturbation.
        min_elements: Initializers with fewer elements are left alone.

    Returns:
        A record of what was written: ``initializersPerturbed``, ``padElements``, ``modelBytes``,
        and the manifest document under ``manifest``.

    Raises:
        CorpusError: ``TIER_TOO_LARGE`` when the tier would not serialize as one protobuf.
    """
    onnx = _onnx()
    tier_bytes = spec.tier_mib * 2**20
    if tier_bytes > MAX_MODEL_BYTES:
        raise CorpusError(
            "TIER_TOO_LARGE",
            f"{spec.tier_mib} MiB exceeds the {MAX_MODEL_BYTES // 2**20} MiB a single ONNX "
            "protobuf can hold",
        )
    src_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(base.graph))
    touched = perturb_weights(model, spec.seed, epsilon, min_elements)
    elements = pad_elements_for(tier_bytes, model.ByteSize())
    pad_model(model, elements, float((spec.index + 1) * PROBE_SCALE), base.input_name)
    model_path = src_dir / "model.onnx"
    onnx.save(model, str(model_path))
    del model

    document = manifest_document(spec, base, tier_bytes)
    params = document.get("familyParams", {})
    payload = params.get("labels") or {"classes": params.get("numClasses", 1)}
    (src_dir / "labels.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (src_dir / "transforms.json").write_text(
        json.dumps(
            {
                "transformVersion": document["transformVersion"],
                "preprocess": document["preprocess"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    document["warmup"] = [golden_warmup(src_dir, document, base.warmup_image.read_bytes())]
    (src_dir / "manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "initializersPerturbed": touched,
        "padElements": elements,
        "modelBytes": model_path.stat().st_size,
        "manifest": document,
    }


def build_bundle(
    spec: BundleSpec,
    base: BaseArchitecture,
    out: Path,
    private_key: bytes,
    public_key: bytes,
    epsilon: float = DEFAULT_EPSILON,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    stage: bool = True,
) -> Dict[str, Any]:
    """Build, sign, and optionally stage one bundle of the corpus.

    Staging runs the component's own path -- tarball digest, Ed25519 signature, bounded extraction,
    per-file digests, manifest schema, family validation, atomic promotion -- so a corpus that
    builds is a corpus the component accepts.

    Args:
        spec: The bundle to build.
        base: The architecture it derives from.
        out: The corpus root.
        private_key: The Ed25519 private key PEM the manifest is signed with.
        public_key: The raw public key the staging call trusts under :data:`KEY_ID`.
        epsilon: Relative standard deviation of the weight perturbation.
        min_elements: Initializers with fewer elements are left alone.
        stage: Whether to stage the finished tarball into ``<out>/cache``.

    Returns:
        The bundle's record for the corpus index.
    """
    started = time.perf_counter()
    src_dir = out / "src" / spec.key
    written = write_source(spec, base, src_dir, epsilon, min_elements)
    archive = out / "archives" / f"{spec.key}.tar"
    digest = make_bundle(
        src_dir=src_dir,
        out_path=archive,
        key=private_key,
        key_id=KEY_ID,
        compress=False,
        schema_path=SCHEMA_PATH,
    )
    if stage:
        stage_bundle(
            uri=str(archive),
            digest=digest,
            staging_root=out / "staging",
            cache=BundleCache(out / "cache", schema_path=SCHEMA_PATH),
            signing_required=True,
            trusted_keys={KEY_ID: public_key},
            schema_path=SCHEMA_PATH,
            model_id=spec.model_id,
            version="1.0.0",
            available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
            validators=[lambda manifest: family_for(manifest).validate_manifest(manifest)],
        )
    shutil.rmtree(src_dir, ignore_errors=True)
    document = written["manifest"]
    return {
        "key": spec.key,
        "index": spec.index,
        "modelId": spec.model_id,
        "version": "1.0.0",
        "digest": digest,
        "base": spec.base,
        "family": document["family"],
        "tierMiB": spec.tier_mib,
        "seed": spec.seed,
        "padElements": written["padElements"],
        "padMiB": round(written["padElements"] * 4 / 2**20, 1),
        "initializersPerturbed": written["initializersPerturbed"],
        "modelBytes": written["modelBytes"],
        "archive": archive.relative_to(out).as_posix(),
        "archiveBytes": archive.stat().st_size,
        "estimatedDeviceMiB": document["estimatedDeviceMiB"],
        "probeValue": (spec.index + 1) * PROBE_SCALE,
        "warmupImage": base.warmup_image.name,
        "staged": bool(stage),
        "buildSecs": round(time.perf_counter() - started, 2),
    }


def build_corpus(
    out: Path,
    bases: Dict[str, BaseArchitecture],
    count: int = DEFAULT_COUNT,
    tiers: Sequence[int] = DEFAULT_TIERS_MIB,
    seed: int = DEFAULT_SEED,
    epsilon: float = DEFAULT_EPSILON,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    stage: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Build the whole corpus and write its index.

    Args:
        out: The corpus root, created if missing.
        bases: The architectures to derive from, by key.
        count: How many bundles to build.
        tiers: The target ``model.onnx`` sizes in MiB.
        seed: The corpus seed.
        epsilon: Relative standard deviation of the weight perturbation.
        min_elements: Initializers with fewer elements are left alone.
        stage: Whether to stage each bundle into ``<out>/cache``.
        progress: Called with one line per finished bundle.

    Returns:
        The corpus index, which is also written to ``<out>/corpus.json``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = plan(count, tiers, sorted(bases), seed)
    private_key, public_key = signing_key(seed)
    keys = out / "keys"
    keys.mkdir(exist_ok=True)
    (keys / "tier3.pem").write_bytes(private_key)
    (keys / "tier3.pub").write_bytes(public_key)

    records = []
    for spec in specs:
        record = build_bundle(
            spec, bases[spec.base], out, private_key, public_key, epsilon, min_elements, stage
        )
        records.append(record)
        if progress is not None:
            progress(
                f"[{spec.index + 1}/{len(specs)}] {record['key']} "
                f"{record['modelBytes'] / 2**20:.0f} MiB {record['digest'][:19]} "
                f"in {record['buildSecs']:.1f}s"
            )
    shutil.rmtree(out / "staging", ignore_errors=True)
    shutil.rmtree(out / "src", ignore_errors=True)

    index = {
        "schemaVersion": 1,
        "generator": "tools/synth_corpus.py",
        "seed": seed,
        "epsilon": epsilon,
        "minElements": min_elements,
        "tiersMiB": [int(tier) for tier in tiers],
        "keyId": KEY_ID,
        "publicKey": public_key.hex(),
        "staged": bool(stage),
        "cache": "cache",
        "bases": {
            key: {
                "graph": base.graph.name,
                "family": base.document["family"],
                "inputName": base.input_name,
                "warmupImage": base.warmup_image.name,
                "activationHeadroomMiB": base.activation_headroom_mib,
            }
            for key, base in sorted(bases.items())
        },
        "totalModelMiB": round(sum(entry["modelBytes"] for entry in records) / 2**20, 1),
        "bundles": records,
    }
    (out / "corpus.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def load_index(out: Path) -> Dict[str, Any]:
    """Read the index of an existing corpus.

    Args:
        out: The corpus root.

    Returns:
        The parsed ``corpus.json``.

    Raises:
        CorpusError: ``CORPUS_MISSING`` when the root holds no index.
    """
    path = Path(out) / "corpus.json"
    if not path.is_file():
        raise CorpusError("CORPUS_MISSING", f"{path} does not exist; build the corpus first")
    return json.loads(path.read_text(encoding="utf-8"))


def tier2_bases(
    cache_root: Path, warmup_images: Optional[Dict[str, Path]] = None
) -> Dict[str, BaseArchitecture]:
    """Describe the two tier-2 architectures the corpus derives from.

    The manifests are the tier-2 ones verbatim (``tests/live_models/bundles.py``), so a tier-3
    bundle exercises the same preprocessing, the same head decoding, and the same decision rules as
    the real model it was grown from.

    Args:
        cache_root: The tier-2 asset cache, normally ``tests/.cache``.
        warmup_images: Overrides for the warmup image of each architecture, by key.

    Returns:
        The architectures by key.

    Raises:
        CorpusError: ``ASSET_MISSING`` when a graph, the ImageNet synset, or a warmup image is not
            in the cache.
    """
    from tests.live_models import bundles as tier2
    from tests.live_models import labels as label_sets

    cache_root = Path(cache_root)
    overrides = dict(warmup_images or {})

    def _require(*parts: str) -> Path:
        path = cache_root.joinpath(*parts)
        if not path.exists():
            raise CorpusError(
                "ASSET_MISSING", f"{path} is missing; run python tools/fetch_test_assets.py"
            )
        return path

    def _image(key: str, *parts: str) -> Path:
        if key in overrides:
            return Path(overrides[key])
        directory = _require(*parts)
        chosen = sorted(path for path in directory.rglob("*") if path.is_file())
        if not chosen:
            raise CorpusError("ASSET_MISSING", f"{directory} holds no warmup image")
        return chosen[0]

    labels = label_sets.imagenet_1000(_require("labels-imagenet-synset", "synset.txt"))
    return {
        "mobilenetv2": BaseArchitecture(
            key="mobilenetv2",
            graph=_require("model-mobilenetv2-12", "mobilenetv2-12.onnx"),
            document=tier2.classifier_document(
                "mobilenetv2", "input", "output", "batch_size", labels
            ),
            warmup_image=_image(
                "mobilenetv2", "dataset-imagenette2-160", "extracted", "imagenette2-160", "val"
            ),
            input_name="input",
            activation_headroom_mib=256,
        ),
        "yoloxs": BaseArchitecture(
            key="yoloxs",
            graph=_require("model-yolox-s", "yolox_s.onnx"),
            document=tier2.yolox_document("yolox-s", 640, 8400),
            warmup_image=_image("yoloxs", "dataset-coco-val2017-slice"),
            input_name="images",
            activation_headroom_mib=512,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="synth_corpus",
        description="Synthesize the tier-3 residency corpus (DESIGN.md section 16.1).",
    )
    parser.add_argument("--out", required=True, help="corpus root; use a local disk, not a mount")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"bundles to build (default {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--tiers",
        default=",".join(str(tier) for tier in DEFAULT_TIERS_MIB),
        help="comma-separated target model.onnx sizes in MiB",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="corpus seed")
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON, help="relative weight perturbation"
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=DEFAULT_MIN_ELEMENTS,
        help="smallest initializer that gets perturbed",
    )
    parser.add_argument(
        "--cache",
        default=str(_REPO_ROOT / "tests" / ".cache"),
        help="the tier-2 asset cache the base architectures come from",
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="pack and sign only; do not stage the bundles into <out>/cache",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` when the corpus cannot be built.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    try:
        tiers = [int(value) for value in str(args.tiers).split(",") if value.strip()]
        index = build_corpus(
            out=Path(args.out),
            bases=tier2_bases(Path(args.cache)),
            count=args.count,
            tiers=tiers,
            seed=args.seed,
            epsilon=args.epsilon,
            min_elements=args.min_elements,
            stage=not args.no_stage,
            progress=print,
        )
    except CorpusError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"TIERS_INVALID: {exc}", file=sys.stderr)
        return 2
    print(
        f"{len(index['bundles'])} bundles, {index['totalModelMiB']:.0f} MiB of model.onnx, "
        f"in {time.perf_counter() - started:.0f}s -> {Path(args.out) / 'corpus.json'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
