"""The executor cell: the only place an ONNX Runtime session exists (DESIGN.md §6.2, LLD §6).

A cell is one supervised subprocess owning one CUDA context and a cache of sessions keyed by
bundle digest. The parent sends a job descriptor -- a path, the digest that file must have, the
model digest, and the transform version -- and the cell does the rest: it reads the immutable
staged file itself, verifies the digest, decodes and preprocesses on the CPU, runs the session,
postprocesses through the task family, applies the decision rules, and answers with a bounded
typed result. Pixels never travel back to the parent, and the parent never initializes CUDA.

Three properties shape the module:

* **``onnxruntime`` is imported lazily, inside :func:`create_session`.** Importing this module
  therefore costs nothing and pulls no CUDA runtime in, which is what lets the parent-side handle
  in :mod:`image_processor.engine.cell` name the subprocess entry point without ever loading the
  runtime itself.
* **The handlers are functions over a :class:`CellState`.** ``handle_load``, ``handle_infer``,
  ``handle_unload``, and ``handle_stats`` take state and a message and return a reply. Nothing in
  them knows about pipes, so the whole cell behavior runs in-process in a test, against the same
  synthetic bundles the real subprocess serves.
* **A session is verified before it is trusted and after it is built.** The declared digest of the
  ONNX graph is checked before the session is created, the golden warmup samples are compared
  against the manifest tolerances before the session serves a job, and the provider assignment is
  checked against the policy on what the session reports, not on what was requested.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

import numpy as np

from image_processor.bundles.archive import BundleError, sha256_file
from image_processor.bundles.manifest import MANIFEST_NAME, parse_manifest, resolve_model_path
from image_processor.engine.decision import decide
from image_processor.engine.decode import DecodeLimits, decode_image
from image_processor.engine.families import family_for
from image_processor.engine.protocol import (
    CONTAMINATING,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    PERMANENT,
    TRANSIENT,
    CellStats,
    Infer,
    LoadFailed,
    LoadModel,
    Loaded,
    ProtocolError,
    Shutdown,
    Stats,
    Unload,
    Unloaded,
    classify_error,
    verify_provider_assignment,
)
from image_processor.engine.residency import DeviceMemory, probe_for
from image_processor.types import InferenceResult, Timings

logger = logging.getLogger(__name__)

#: Element types a warmup tensor may declare, mapped to numpy.
NUMPY_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "uint8": np.uint8,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}

#: Default absolute tolerance for a golden warmup comparison when the manifest declares none.
DEFAULT_ABSOLUTE_TOLERANCE = 1e-5

#: Default relative tolerance for a golden warmup comparison when the manifest declares none.
DEFAULT_RELATIVE_TOLERANCE = 0.0

#: Largest warmup or expected-output member the cell reads.
MAX_WARMUP_BYTES = 64 * 2**20

#: Cached ``onnxruntime`` module, imported on first use.
_ORT = None


def onnxruntime_module():
    """Import ``onnxruntime`` on first use.

    The import is deferred so that merely importing this module -- which the parent does to name
    the subprocess entry point -- neither loads the runtime nor initializes a CUDA context.

    Returns:
        The ``onnxruntime`` module.
    """
    global _ORT
    if _ORT is None:
        import onnxruntime

        _ORT = onnxruntime
    return _ORT


@dataclass(frozen=True)
class CellConfig:
    """How one cell is built. Pickled to the child, so every field is a plain value.

    Attributes:
        cell_id: The identity the supervisor gave this cell. Echoed in stats and logs.
        device_id: The CUDA device ordinal this cell owns, or ``None`` for a CPU cell.
        providers: The execution providers to request by default, highest priority first. A
            :class:`~image_processor.engine.protocol.LoadModel` may name its own.
        decode_limits: The bounds every input image is decoded under (DESIGN.md §15).
        settle_ms: How long an unload waits before it samples device memory (DESIGN.md §10.4).
        log_level: The level the child's root logger is configured at.
    """

    cell_id: str = "cell-0"
    device_id: Optional[int] = None
    providers: tuple = (CPU_PROVIDER,)
    decode_limits: DecodeLimits = DecodeLimits()
    settle_ms: int = 0
    log_level: str = "INFO"


@dataclass
class LoadedSession:
    """One resident model generation.

    Attributes:
        digest: The bundle digest the session serves.
        session: The ONNX Runtime session.
        manifest: The bundle manifest the session was built from.
        family: The task family that interprets this head.
        bundle_root: The cached bundle directory.
        providers_assigned: The session's actual provider assignment.
        load_ms: Wall-clock milliseconds the load and its warmup took.
        device_mib: Device memory the load consumed, or ``0`` when unmeasured.
        input_names: The session's input tensor names.
        output_names: The session's output tensor names.
        use_io_binding: Whether to run through an I/O binding rather than ``run``.
        warmup_samples: Golden samples compared at load.
        hits: Jobs this session has answered.
    """

    digest: str
    session: object
    manifest: object
    family: object
    bundle_root: Path
    providers_assigned: tuple
    load_ms: float = 0.0
    device_mib: int = 0
    input_names: tuple = ()
    output_names: tuple = ()
    use_io_binding: bool = False
    warmup_samples: int = 0
    hits: int = 0


class CellState:
    """Everything one cell holds between messages.

    Args:
        config: How this cell is built.
        probe: The device-memory probe. Defaults to NVML for a device cell and to a reading that
            reports nothing for a CPU cell.
        session_factory: Builds an ONNX Runtime session. Defaults to :func:`create_session`;
            a test substitutes a stand-in to drive a failure the hardware will not produce.
        clock: Returns a monotonic time in seconds. Injected by tests.
    """

    def __init__(self, config: CellConfig, probe=None, session_factory=None, clock=None) -> None:
        """Build the state and its device probe."""
        self.config = config
        self.probe = probe if probe is not None else probe_for(config.device_id)
        self.session_factory = session_factory or create_session
        self.sessions: dict = {}
        self.inferences = 0
        self._clock = clock or time.monotonic
        self.started_at = self._clock()
        self._device_class = None

    @property
    def device(self) -> Optional[str]:
        """The device ordinal as a string, or ``None`` for a CPU cell."""
        return None if self.config.device_id is None else str(self.config.device_id)

    def snapshot(self) -> DeviceMemory:
        """Read this cell's device memory.

        Returns:
            The reading, empty when the cell has no device or no probe is available.
        """
        return self.probe.snapshot(self.config.device_id)

    def device_class(self, providers=()) -> Optional[str]:
        """Return the GPU class this cell runs on.

        NVML names the device exactly; without NVML, a session that actually holds a CUDA
        assignment still reports what ONNX Runtime says it is running on, so the result carries a
        device identity either way.

        Args:
            providers: The session's actual provider assignment.

        Returns:
            The device name, or ``None`` on a CPU cell.
        """
        if self._device_class is None:
            self._device_class = self.snapshot().device_class
        if self._device_class:
            return self._device_class
        if CUDA_PROVIDER in tuple(providers):
            try:
                return str(onnxruntime_module().get_device())
            except Exception as exc:  # pragma: no cover - live runtime seam
                logger.debug("the runtime did not report a device: %s", exc)
        return None

    def release(self, digest: str) -> Optional[LoadedSession]:
        """Drop one session and let the runtime free its buffers.

        Args:
            digest: The bundle digest to release.

        Returns:
            The released session, or ``None`` when it was not resident.
        """
        loaded = self.sessions.pop(digest, None)
        if loaded is None:
            return None
        loaded.session = None
        gc.collect()
        return loaded

    def close(self) -> None:
        """Release every session this cell holds."""
        for digest in list(self.sessions):
            self.release(digest)


# -- bundle access -------------------------------------------------------------------------------


def read_bundle_manifest(bundle_root: Path):
    """Read the manifest of a cached bundle.

    The bundle reached the cache through ``bundles.stage_bundle``, which is where DESIGN.md §9
    step 3 puts schema validation and the full per-file digest sweep. The cell re-reads the
    manifest because it needs the tensors, the family, the preprocessing, and the decision rules,
    and it re-verifies the digest of every file it actually opens (:func:`read_member`,
    :func:`verify_declared_file`) so nothing it loads is trusted on the strength of an earlier
    check alone.

    Args:
        bundle_root: The cached bundle directory.

    Returns:
        The parsed :class:`~image_processor.types.BundleManifest`.

    Raises:
        BundleError: ``MANIFEST_MISSING`` or ``MANIFEST_INVALID``.
    """
    path = Path(bundle_root) / MANIFEST_NAME
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BundleError("MANIFEST_MISSING", f"{path} cannot be read: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError("MANIFEST_INVALID", f"{path} is not valid JSON: {exc}") from exc
    return parse_manifest(document)


def member_path(bundle_root: Path, relative: str) -> Path:
    """Resolve one bundle member, refusing anything that leaves the bundle.

    Args:
        bundle_root: The cached bundle directory.
        relative: The member's manifest-relative POSIX path.

    Returns:
        The absolute path of the member.

    Raises:
        BundleError: ``MEMBER_PATH_INVALID`` when the name is absolute, walks up, or resolves
            outside the bundle.
    """
    posix = PurePosixPath(str(relative).replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise BundleError("MEMBER_PATH_INVALID", f"{relative!r} is not a bundle-relative path")
    root = Path(bundle_root).resolve()
    resolved = (root / Path(*posix.parts)).resolve()
    if root != resolved and root not in resolved.parents:
        raise BundleError("MEMBER_PATH_INVALID", f"{relative!r} resolves outside the bundle")
    return resolved


def verify_declared_file(bundle_root: Path, manifest, relative: str) -> Path:
    """Verify one member against the digest the manifest declares for it.

    Args:
        bundle_root: The cached bundle directory.
        manifest: The parsed manifest.
        relative: The member's manifest-relative POSIX path.

    Returns:
        The verified member's path.

    Raises:
        BundleError: ``FILE_UNDECLARED``, ``FILE_MISSING``, or ``FILE_DIGEST_MISMATCH``.
    """
    declared = manifest.files.get(relative)
    if declared is None:
        raise BundleError("FILE_UNDECLARED", f"the manifest does not declare {relative!r}")
    path = member_path(bundle_root, relative)
    if not path.is_file():
        raise BundleError("FILE_MISSING", f"the bundle does not contain {relative!r}")
    actual = sha256_file(path)
    if actual != str(declared).lower():
        raise BundleError(
            "FILE_DIGEST_MISMATCH",
            f"{relative!r} hashes to {actual}, the manifest declares {declared}",
        )
    return path


def read_member(bundle_root: Path, manifest, relative: str, max_bytes: int = MAX_WARMUP_BYTES):
    """Read one bundle member, verifying its declared digest when the manifest declares one.

    Args:
        bundle_root: The cached bundle directory.
        manifest: The parsed manifest.
        relative: The member's manifest-relative POSIX path.
        max_bytes: Largest member this reads.

    Returns:
        The member's bytes.

    Raises:
        BundleError: The member is missing, oversized, outside the bundle, or does not match its
            declared digest.
    """
    if relative in manifest.files:
        path = verify_declared_file(bundle_root, manifest, relative)
    else:
        path = member_path(bundle_root, relative)
        if not path.is_file():
            raise BundleError("FILE_MISSING", f"the bundle does not contain {relative!r}")
    size = path.stat().st_size
    if size > max_bytes:
        raise BundleError(
            "MEMBER_TOO_LARGE", f"{relative!r} is {size} bytes, over the {max_bytes} byte bound"
        )
    return path.read_bytes()


# -- sessions ------------------------------------------------------------------------------------


def provider_options(providers, device_id=None, gpu_mem_limit_mib=None) -> list:
    """Build the per-provider options for a session.

    CUDA gets the three that matter to a shared, long-lived device: the ordinal this cell owns,
    the arena bound the residency budget allows, and ``kSameAsRequested``, which stops the arena
    from doubling itself and stranding memory another model was admitted against
    (DESIGN.md §10.2).

    Args:
        providers: The providers being requested, highest priority first.
        device_id: The CUDA device ordinal this cell owns.
        gpu_mem_limit_mib: The arena bound in MiB, or ``None`` for the provider default.

    Returns:
        One options mapping per provider, in the same order.
    """
    options = []
    for name in providers:
        if name == CUDA_PROVIDER:
            entry = {
                "device_id": int(device_id or 0),
                "arena_extend_strategy": "kSameAsRequested",
            }
            if gpu_mem_limit_mib:
                entry["gpu_mem_limit"] = int(gpu_mem_limit_mib) * (1 << 20)
            options.append(entry)
        else:
            options.append({})
    return options


def create_session(model_path, providers, options):
    """Create the ONNX Runtime session. The only such call in the component.

    Args:
        model_path: The verified ONNX graph.
        providers: The providers to request, highest priority first.
        options: One options mapping per provider, from :func:`provider_options`.

    Returns:
        The session.
    """
    ort = onnxruntime_module()
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=list(providers),
        provider_options=[dict(entry) for entry in options],
    )


def static_shapes(manifest) -> bool:
    """Report whether every declared input has a fully static shape.

    Args:
        manifest: The parsed manifest.

    Returns:
        ``True`` when no input carries a symbolic dimension and no dynamic batch axis is declared.
    """
    if manifest.dynamic_batch:
        return False
    for spec in manifest.inputs:
        for dimension in spec.shape:
            if not isinstance(dimension, int) or dimension <= 0:
                return False
    return True


def run_session(loaded: LoadedSession, feed: dict) -> dict:
    """Run one session over one feed.

    An I/O binding is used when the session holds a CUDA assignment and the model's shapes are
    static, which is the case where binding pays: the input is copied to the device once and the
    output buffers are reused rather than reallocated per call. Everything else goes through
    ``run``, which is also what the CPU parity suite exercises (D-IP-14).

    Args:
        loaded: The resident session.
        feed: Input tensor name to array, from the task family's preprocessing.

    Returns:
        Output tensor name to array.
    """
    names = list(loaded.output_names)
    if loaded.use_io_binding:
        binding = loaded.session.io_binding()
        for name, array in feed.items():
            binding.bind_cpu_input(name, np.ascontiguousarray(array))
        for name in names:
            binding.bind_output(name)
        loaded.session.run_with_iobinding(binding)
        return dict(zip(names, binding.copy_outputs_to_cpu()))
    return dict(zip(names, loaded.session.run(names, feed)))


# -- warmup --------------------------------------------------------------------------------------


def tolerances(manifest) -> tuple:
    """Read the golden-warmup tolerances a manifest declares.

    Args:
        manifest: The parsed manifest.

    Returns:
        ``(absolute, relative)``.
    """
    declared = manifest.tolerances or {}
    absolute = float(declared.get("absolute", DEFAULT_ABSOLUTE_TOLERANCE))
    relative = float(declared.get("relative", DEFAULT_RELATIVE_TOLERANCE))
    return absolute, relative


def resolve_shape(shape) -> tuple:
    """Turn a declared shape into a concrete one.

    Args:
        shape: The declared dimensions. A symbolic or non-positive dimension, such as the ``"N"``
            of a dynamic batch axis, becomes ``1``.

    Returns:
        The concrete shape.
    """
    return tuple(
        dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in shape
    )


def _tensor_from_bytes(data: bytes, dtype: str, shape) -> np.ndarray:
    """Read one raw tensor file into an array.

    Args:
        data: The raw little-endian buffer, as ``warmup/*.bin`` stores it.
        dtype: The element type name.
        shape: The declared shape.

    Returns:
        The array.

    Raises:
        ProtocolError: ``WARMUP_INVALID`` when the type is unknown or the buffer does not fill the
            shape.
    """
    element = NUMPY_DTYPES.get(str(dtype))
    if element is None:
        raise ProtocolError("WARMUP_INVALID", f"warmup dtype {dtype!r} is not supported")
    concrete = resolve_shape(shape)
    try:
        array = np.frombuffer(data, dtype=element)
    except ValueError as exc:
        raise ProtocolError("WARMUP_INVALID", f"warmup buffer is not {dtype}: {exc}") from exc
    expected = int(np.prod(concrete)) if concrete else array.size
    if array.size != expected:
        raise ProtocolError(
            "WARMUP_INVALID",
            f"warmup buffer holds {array.size} elements, shape {list(concrete)} needs {expected}",
        )
    return array.reshape(concrete).copy()


def warmup_feed(bundle_root: Path, manifest, sample: dict) -> dict:
    """Build the input feed of one golden warmup sample.

    A sample names its inputs either as ``{"inputs": {"<tensor>": "warmup/input-01.bin"}}`` or, in
    the bundle schema's single-input form, as ``{"input": "warmup/input-01.bin", "inputName": ...,
    "dtype": ..., "shape": ...}`` where ``inputName`` defaults to the manifest's first input and
    ``dtype``/``shape`` default to that input's declaration. Each ``inputs`` entry is the member
    path of a raw tensor, or a mapping carrying ``path`` and optionally ``dtype`` and ``shape``.

    Args:
        bundle_root: The cached bundle directory.
        manifest: The parsed manifest.
        sample: One entry of ``manifest.warmup``.

    Returns:
        Input tensor name to array.

    Raises:
        ProtocolError: ``WARMUP_INVALID`` when the sample names a tensor the manifest does not
            declare and does not describe it itself.
    """
    declared = {spec.name: spec for spec in manifest.inputs}
    entries = sample.get("inputs", sample.get("input"))
    if entries is None:
        return {}
    if isinstance(entries, str):
        # The schema's single-input form: ``input`` names the file; ``inputName`` (else the
        # manifest's first input) names the tensor; ``dtype``/``shape`` at the sample level
        # override the manifest's declaration for this sample.
        name = sample.get("inputName")
        if not name:
            if not manifest.inputs:
                raise ProtocolError("WARMUP_INVALID", "the manifest declares no input to warm")
            name = manifest.inputs[0].name
        entry = {"path": entries}
        if sample.get("dtype") is not None:
            entry["dtype"] = sample["dtype"]
        if sample.get("shape") is not None:
            entry["shape"] = sample["shape"]
        entries = {name: entry}

    feed = {}
    for name, entry in entries.items():
        spec = declared.get(name)
        if isinstance(entry, str):
            relative, dtype, shape = entry, None, None
        else:
            relative = entry.get("path")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
        if relative is None:
            raise ProtocolError("WARMUP_INVALID", f"warmup input {name!r} names no file")
        if dtype is None or shape is None:
            if spec is None:
                raise ProtocolError(
                    "WARMUP_INVALID",
                    f"warmup input {name!r} is not declared by the manifest and carries no "
                    "dtype and shape of its own",
                )
            dtype = dtype or spec.dtype
            shape = shape if shape is not None else spec.shape
        feed[name] = _tensor_from_bytes(read_member(bundle_root, manifest, relative), dtype, shape)
    return feed


def warmup_expected(bundle_root: Path, manifest, sample: dict) -> dict:
    """Read the expected outputs of one golden warmup sample.

    The expectation is either the member path of an ``expected-*.json`` document or the document
    itself. The document maps output tensor names to values, either directly or under an
    ``outputs`` key, and a value is either a nested list or a mapping carrying ``values`` and an
    optional ``shape``.

    Args:
        bundle_root: The cached bundle directory.
        manifest: The parsed manifest.
        sample: One entry of ``manifest.warmup``.

    Returns:
        Output tensor name to the expected array. Empty when the sample declares no expectation,
        which makes it a priming sample rather than a golden one.

    Raises:
        ProtocolError: ``WARMUP_INVALID`` when the document is not readable as expectations.
    """
    expected = sample.get("expected", sample.get("outputs"))
    if expected is None:
        return {}
    if isinstance(expected, str):
        try:
            document = json.loads(read_member(bundle_root, manifest, expected).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProtocolError(
                "WARMUP_INVALID", f"{expected!r} is not a valid expectation document: {exc}"
            ) from exc
    else:
        document = expected
    if not isinstance(document, dict):
        raise ProtocolError("WARMUP_INVALID", "a warmup expectation must be a JSON object")
    if "outputs" in document and isinstance(document["outputs"], dict):
        document = document["outputs"]

    values = {}
    for name, entry in document.items():
        if isinstance(entry, dict):
            raw = entry.get("values")
            shape = entry.get("shape")
        else:
            raw = entry
            shape = None
        if raw is None:
            raise ProtocolError("WARMUP_INVALID", f"expected output {name!r} carries no values")
        array = np.asarray(raw, dtype=np.float64)
        if shape is not None:
            array = array.reshape(resolve_shape(shape))
        values[name] = array
    return values


def compare_warmup(actual: dict, expected: dict, absolute: float, relative: float, where: str):
    """Compare one warmup run against its golden answer.

    Args:
        actual: The session outputs.
        expected: The expected outputs.
        absolute: Absolute tolerance.
        relative: Relative tolerance.
        where: The sample identity, for the message.

    Raises:
        ProtocolError: ``WARMUP_MISMATCH`` when an output is missing, is a different size, or
            falls outside the tolerances. A golden mismatch is permanent: the same bundle on the
            same provider reproduces it, so the model is refused rather than retried.
    """
    for name, want in expected.items():
        if name not in actual:
            raise ProtocolError(
                "WARMUP_MISMATCH", f"{where}: the session produced no output named {name!r}"
            )
        got = np.asarray(actual[name], dtype=np.float64).ravel()
        reference = np.asarray(want, dtype=np.float64).ravel()
        if got.size != reference.size:
            raise ProtocolError(
                "WARMUP_MISMATCH",
                f"{where}: {name} has {got.size} values, the golden answer has {reference.size}",
            )
        if not np.allclose(got, reference, atol=absolute, rtol=relative):
            worst = float(np.max(np.abs(got - reference))) if got.size else 0.0
            raise ProtocolError(
                "WARMUP_MISMATCH",
                f"{where}: {name} differs from the golden answer by {worst:g}, over the declared "
                f"tolerance (absolute {absolute:g}, relative {relative:g})",
            )


def prime_session(loaded: LoadedSession) -> None:
    """Run one zero-filled pass so the first real job does not pay the provider's setup.

    A bundle without golden samples still benefits from a pass that allocates the arena and
    compiles the kernels. It proves nothing about the model, so a failure here is logged and left
    for the first real job to raise and classify.

    Args:
        loaded: The freshly built session.
    """
    manifest = loaded.manifest
    feed = {}
    for spec in manifest.inputs:
        element = NUMPY_DTYPES.get(str(spec.dtype), np.float32)
        feed[spec.name] = np.zeros(resolve_shape(spec.shape), dtype=element)
    if not feed:
        return
    try:
        run_session(loaded, feed)
    except Exception as exc:
        logger.debug("priming %s did not complete: %s", loaded.digest, exc)


def run_warmup(loaded: LoadedSession) -> int:
    """Warm a session before it serves work (DESIGN.md §9 step 5).

    Args:
        loaded: The freshly built session.

    Returns:
        The number of golden samples compared. ``0`` means the bundle carries none and the session
        was primed instead.

    Raises:
        ProtocolError: ``WARMUP_MISMATCH`` when a golden sample does not reproduce.
    """
    manifest = loaded.manifest
    samples = list(manifest.warmup or [])
    if not samples:
        prime_session(loaded)
        return 0
    absolute, relative = tolerances(manifest)
    compared = 0
    for index, sample in enumerate(samples):
        where = f"warmup sample {sample.get('id', index)}"
        feed = warmup_feed(loaded.bundle_root, manifest, sample)
        if not feed:
            raise ProtocolError("WARMUP_INVALID", f"{where} names no input")
        outputs = run_session(loaded, feed)
        expected = warmup_expected(loaded.bundle_root, manifest, sample)
        if not expected:
            continue
        compare_warmup(outputs, expected, absolute, relative, where)
        compared += 1
    return compared


# -- handlers ------------------------------------------------------------------------------------


def handle_load(state: CellState, message: LoadModel):
    """Make one model generation resident (DESIGN.md §9 steps 4-5, §10.1).

    The order is the one that makes each step meaningful: the manifest is parsed, the family that
    will read this head accepts it, the ONNX graph is verified against its declared digest, the
    session is built with the requested providers, the assignment the session actually got is
    checked against the policy, and only then is the session warmed and published to the cache. A
    session that fails any of those never becomes resident.

    Args:
        state: The cell state.
        message: The load request.

    Returns:
        :class:`~image_processor.engine.protocol.Loaded` on success, otherwise
        :class:`~image_processor.engine.protocol.LoadFailed` carrying the classification.
    """
    resident = state.sessions.get(message.digest)
    if resident is not None:
        return Loaded(
            digest=message.digest,
            providers_assigned=resident.providers_assigned,
            load_ms=resident.load_ms,
            device_mib=resident.device_mib,
            warmup_samples=resident.warmup_samples,
            gpu_device=state.device,
            gpu_class=state.device_class(resident.providers_assigned),
        )

    started = time.perf_counter()
    before = state.snapshot()
    loaded = None
    try:
        root = Path(message.bundle_root)
        manifest = read_bundle_manifest(root)
        family = family_for(manifest)
        family.validate_manifest(manifest)

        model_path = resolve_model_path(root, manifest)
        verify_declared_file(root, manifest, model_path.relative_to(root).as_posix())

        providers = tuple(message.providers or state.config.providers)
        session = state.session_factory(
            model_path,
            providers,
            provider_options(providers, state.config.device_id, message.gpu_mem_limit_mib),
        )
        assigned = verify_provider_assignment(
            session.get_providers(),
            tuple(message.providers_permitted or manifest.providers_permitted or ()),
            message.provider_policy or manifest.provider_policy,
            message.required_provider,
            message.allow_cpu_only,
        )
        loaded = LoadedSession(
            digest=message.digest,
            session=session,
            manifest=manifest,
            family=family,
            bundle_root=root,
            providers_assigned=assigned,
            input_names=tuple(tensor.name for tensor in session.get_inputs()),
            output_names=tuple(tensor.name for tensor in session.get_outputs()),
            use_io_binding=CUDA_PROVIDER in assigned and static_shapes(manifest),
        )
        if message.warmup or manifest.warmup:
            loaded.warmup_samples = run_warmup(loaded)
    except Exception as exc:
        if loaded is not None:
            loaded.session = None
            gc.collect()
        info = classify_error(exc)
        logger.warning(
            "cell %s could not load %s: %s (%s, %s)",
            state.config.cell_id,
            message.digest,
            info.message,
            info.code,
            info.error_class,
        )
        return LoadFailed(
            digest=message.digest,
            error=info.message,
            error_class=info.error_class,
            code=info.code,
            memory_pressure=info.memory_pressure,
        )

    after = state.snapshot()
    loaded.load_ms = (time.perf_counter() - started) * 1000.0
    loaded.device_mib = max(before.free_mib - after.free_mib, 0) if after.known else 0
    state.sessions[message.digest] = loaded
    logger.info(
        "cell %s loaded %s on %s in %.1f ms (%d MiB, %d golden samples)",
        state.config.cell_id,
        message.digest,
        list(loaded.providers_assigned),
        loaded.load_ms,
        loaded.device_mib,
        loaded.warmup_samples,
    )
    return Loaded(
        digest=message.digest,
        providers_assigned=loaded.providers_assigned,
        load_ms=loaded.load_ms,
        device_mib=loaded.device_mib,
        warmup_samples=loaded.warmup_samples,
        gpu_device=state.device,
        gpu_class=state.device_class(loaded.providers_assigned),
    )


def read_staged(path, limits: DecodeLimits) -> bytes:
    """Read the immutable staged file the parent named.

    Args:
        path: The staged file.
        limits: The decode bounds. The byte count is checked before the file is read, so an
            oversized input never allocates.

    Returns:
        The file bytes.

    Raises:
        ProtocolError: ``INPUT_TOO_LARGE`` when the file is over ``max_bytes``.
        OSError: When the file cannot be read, which classification turns into ``FILE_MISSING``
            or ``IO_ERROR``.
    """
    staged = Path(path)
    size = staged.stat().st_size
    if size > limits.max_bytes:
        raise ProtocolError(
            "INPUT_TOO_LARGE", f"{staged} is {size} bytes, over the {limits.max_bytes} byte bound"
        )
    return staged.read_bytes()


def failure_result(
    message: Infer,
    timings: Timings,
    code: str,
    error_class: str,
    error: str,
    providers=(),
    gpu_device=None,
    gpu_class=None,
    memory_high_water_mib=None,
) -> InferenceResult:
    """Build the ``FAILED`` answer for one job.

    Args:
        message: The request being answered.
        timings: What was measured before the failure.
        code: Stable SCREAMING_SNAKE code.
        error_class: One of the protocol error classes.
        error: The bounded operator-readable detail.
        providers: The session provider assignment, when a session was involved.
        gpu_device: The device ordinal, when known.
        gpu_class: The device name, when known.
        memory_high_water_mib: Device memory in use, when known.

    Returns:
        The failed result. It carries no normalized output and no decision, which is what makes a
        failed inference a ``HOLD`` for every consumer downstream (DESIGN.md §15).
    """
    return InferenceResult(
        inference_id=message.inference_id,
        status="FAILED",
        normalized=None,
        decision=None,
        providers=list(providers),
        gpu_device=gpu_device,
        gpu_class=gpu_class,
        timings=timings,
        memory_high_water_mib=memory_high_water_mib,
        error=f"{code}: {error}",
        error_class=error_class,
    )


def handle_infer(state: CellState, message: Infer) -> InferenceResult:
    """Run one job (DESIGN.md §6.2).

    Args:
        state: The cell state.
        message: The job descriptor.

    Returns:
        The result. A failure is a ``FAILED`` result carrying the classification, never an
        exception across the pipe: the parent must be able to record an answer for every job it
        dispatched.
    """
    started = time.perf_counter()
    queue_ms = float(message.queue_ms or 0.0)

    def measured(load=0.0, pre=0.0, run=0.0, post=0.0) -> Timings:
        elapsed = (time.perf_counter() - started) * 1000.0
        return Timings(
            queue_ms=queue_ms,
            model_load_ms=load,
            preprocess_ms=pre,
            inference_ms=run,
            postprocess_ms=post,
            total_ms=queue_ms + elapsed,
        )

    loaded = state.sessions.get(message.digest)
    if loaded is None:
        return failure_result(
            message,
            measured(),
            "MODEL_NOT_RESIDENT",
            TRANSIENT,
            f"{message.digest} is not resident in cell {state.config.cell_id}",
            gpu_device=state.device,
        )

    providers = loaded.providers_assigned
    gpu_class = state.device_class(providers)
    before = state.snapshot()

    if message.transform_version and message.transform_version != loaded.manifest.transform_version:
        return failure_result(
            message,
            measured(),
            "TRANSFORM_VERSION_MISMATCH",
            PERMANENT,
            f"the job is pinned to transform {message.transform_version} and the resident bundle "
            f"declares {loaded.manifest.transform_version}",
            providers,
            state.device,
            gpu_class,
        )

    load_ms = loaded.load_ms if loaded.hits == 0 else 0.0
    try:
        data = read_staged(message.staged_path, state.config.decode_limits)
        actual = hashlib.sha256(data).hexdigest()
        expected = str(message.sha256 or "").split(":")[-1].lower()
        if expected and actual != expected:
            return failure_result(
                message,
                measured(load=load_ms),
                "INPUT_DIGEST_MISMATCH",
                PERMANENT,
                f"the staged file hashes to {actual}, the job expects {expected}",
                providers,
                state.device,
                gpu_class,
            )

        mark = time.perf_counter()
        image = decode_image(data, state.config.decode_limits)
        feed = loaded.family.preprocess(image, loaded.manifest)
        preprocess_ms = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        outputs = run_session(loaded, feed)
        inference_ms = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        normalized = loaded.family.postprocess(
            outputs, loaded.manifest, (int(image.shape[0]), int(image.shape[1]))
        )
        decision = decide(normalized, loaded.manifest.decision_rules)
        postprocess_ms = (time.perf_counter() - mark) * 1000.0
    except Exception as exc:
        info = classify_error(exc)
        if info.error_class == CONTAMINATING:
            logger.error(
                "cell %s hit a contaminating failure on %s: %s",
                state.config.cell_id,
                message.inference_id,
                info.message,
            )
        return failure_result(
            message,
            measured(load=load_ms),
            info.code,
            info.error_class,
            info.message,
            providers,
            state.device,
            gpu_class,
            _high_water(before, state.snapshot()),
        )

    loaded.hits += 1
    state.inferences += 1
    return InferenceResult(
        inference_id=message.inference_id,
        status="SUCCEEDED",
        normalized=normalized,
        decision=decision,
        providers=list(providers),
        gpu_device=state.device,
        gpu_class=gpu_class,
        timings=measured(load_ms, preprocess_ms, inference_ms, postprocess_ms),
        memory_high_water_mib=_high_water(before, state.snapshot()),
    )


def _high_water(before: DeviceMemory, after: DeviceMemory):
    """Return the highest device memory in use observed around one job.

    Args:
        before: The reading taken before the job ran.
        after: The reading taken after it.

    Returns:
        The larger of the two readings in MiB, or ``None`` when the cell has no device accounting.
    """
    if not before.known and not after.known:
        return None
    return max(before.used_mib, after.used_mib)


def handle_unload(state: CellState, message: Unload) -> Unloaded:
    """Release one resident session and report what came back (DESIGN.md §10.4).

    The session and its buffers are dropped, the collector runs, the cell waits out the configured
    settle period, and only then is device memory sampled. Memory that does not come back is
    reported rather than assumed: the parent compares ``freed_mib`` against ``expected_mib`` and
    recycles the cell when the difference says the arena is fragmented or the runtime is stuck.

    Args:
        state: The cell state.
        message: The unload request.

    Returns:
        The reply, carrying what was freed and what was expected.
    """
    before = state.snapshot()
    loaded = state.release(message.digest)
    if loaded is None:
        return Unloaded(digest=message.digest, freed_mib=0, was_resident=False, expected_mib=0)
    if state.config.settle_ms:
        time.sleep(state.config.settle_ms / 1000.0)
    after = state.snapshot()
    freed = max(after.free_mib - before.free_mib, 0) if after.known else 0
    logger.info(
        "cell %s unloaded %s, %d MiB of an expected %d MiB came back",
        state.config.cell_id,
        message.digest,
        freed,
        loaded.device_mib,
    )
    return Unloaded(
        digest=message.digest,
        freed_mib=freed,
        was_resident=True,
        expected_mib=loaded.device_mib,
    )


def handle_stats(state: CellState, message: Stats) -> CellStats:
    """Report what the cell is holding.

    Args:
        state: The cell state.
        message: The request.

    Returns:
        The reply, which is what the scheduler's residency accounting and the ``get-models``
        command read.
    """
    reading = state.snapshot()
    return CellStats(
        resident=tuple(state.sessions),
        device_free_mib=reading.free_mib,
        device_total_mib=reading.total_mib,
        uptime_s=max(state._clock() - state.started_at, 0.0),
        cell_id=state.config.cell_id,
        gpu_device=state.device,
        gpu_class=reading.device_class,
        resident_mib={digest: entry.device_mib for digest, entry in state.sessions.items()},
        inferences=state.inferences,
    )


def dispatch(state: CellState, message):
    """Route one request to its handler.

    Args:
        state: The cell state.
        message: The request.

    Returns:
        The reply, or ``None`` for :class:`~image_processor.engine.protocol.Shutdown`, which has
        no reply.

    Raises:
        ProtocolError: ``UNKNOWN_MESSAGE`` when the request is not part of the protocol.
    """
    if isinstance(message, LoadModel):
        return handle_load(state, message)
    if isinstance(message, Infer):
        return handle_infer(state, message)
    if isinstance(message, Unload):
        return handle_unload(state, message)
    if isinstance(message, Stats):
        return handle_stats(state, message)
    if isinstance(message, Shutdown):
        return None
    raise ProtocolError(
        "UNKNOWN_MESSAGE", f"a cell cannot act on a {type(message).__name__} message"
    )


def serve(state: CellState, connection) -> None:
    """Answer requests until the parent asks the cell to stop or the pipe closes.

    Args:
        state: The cell state.
        connection: The pipe end the parent talks over. Anything with ``recv``, ``send``, and
            ``close`` serves, which is how the loop is tested without a subprocess.
    """
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError) as exc:
            logger.info("cell %s lost its parent: %s", state.config.cell_id, exc)
            break
        if isinstance(message, Shutdown):
            break
        try:
            reply = dispatch(state, message)
        except Exception as exc:
            info = classify_error(exc)
            reply = LoadFailed(
                digest=str(getattr(message, "digest", "")),
                error=info.message,
                error_class=info.error_class,
                code=info.code,
            )
        try:
            connection.send(reply)
        except (EOFError, OSError) as exc:
            logger.info("cell %s could not answer: %s", state.config.cell_id, exc)
            break
    state.close()
    try:
        connection.close()
    except OSError as exc:
        logger.debug("cell %s could not close its pipe: %s", state.config.cell_id, exc)


def cell_entrypoint(config: CellConfig, connection) -> None:  # pragma: no cover - subprocess entry: this is the spawned child's first frame, so no in-process test executes it; every handler it calls is covered by tests/engine/test_cell_handlers.py and the loop by tests/engine/test_cell_serve.py
    """Run one executor cell. The target of the spawned process.

    Args:
        config: How this cell is built.
        connection: The child end of the pipe to the parent.
    """
    logging.basicConfig(
        level=getattr(logging, str(config.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state = CellState(config)
    logger.info(
        "cell %s started for device %s with providers %s",
        config.cell_id,
        config.device_id,
        list(config.providers),
    )
    serve(state, connection)


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "MAX_WARMUP_BYTES",
    "NUMPY_DTYPES",
    "CellConfig",
    "CellState",
    "LoadedSession",
    "cell_entrypoint",
    "compare_warmup",
    "create_session",
    "dispatch",
    "failure_result",
    "handle_infer",
    "handle_load",
    "handle_stats",
    "handle_unload",
    "member_path",
    "onnxruntime_module",
    "prime_session",
    "provider_options",
    "read_bundle_manifest",
    "read_member",
    "read_staged",
    "resolve_shape",
    "run_session",
    "run_warmup",
    "serve",
    "static_shapes",
    "tolerances",
    "verify_declared_file",
    "warmup_expected",
    "warmup_feed",
]
