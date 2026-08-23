"""The parent-to-cell protocol and the runtime-error vocabulary (LLD §6, DESIGN.md §10).

Everything that crosses the executor boundary is declared here: five request messages, their
replies, and the classification that decides what a failure means. The module imports nothing but
the standard library and :mod:`image_processor.types`, so both sides of the pipe can hold it -- the
parent, which must never import ``onnxruntime``, and the cell, which runs it in a subprocess.

The messages are plain frozen dataclasses. They are pickled onto a ``multiprocessing`` pipe, so
every field is a primitive, a tuple, a dict, or a dataclass built from those: no ``Path``, no open
handle, no callable, nothing whose unpickling would import a module the other side does not have.

A failure is one of three things, and the difference is what the scheduler does next
(DESIGN.md §7, §6.2):

* ``transient`` -- the same work may succeed later: a device that was out of memory, a busy GPU, a
  session that is no longer resident. The job goes to ``RETRY_WAIT`` with backoff until its retry
  budget is spent.
* ``permanent`` -- the same work fails the same way every time: an unreadable image, a head no task
  family serves, a provider policy this machine cannot satisfy. The job goes to
  ``PROCESSING_EXHAUSTED``, and a model load goes to ``BLOCKED_CONFIGURATION``.
* ``contaminating`` -- the CUDA context is no longer trustworthy: an illegal address, a failed
  launch, a destroyed context, an ECC fault. The cell is recycled and the job runs again at the
  same attempt.

Classification reads the exception type name and the message rather than importing the types it
recognizes. The ``onnxruntime`` exceptions live in a module the parent must not import, and the
component's own errors live in packages that sit above this one; matching on names keeps the table
in one place and keeps this module free of both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from image_processor.types import InferenceResult

#: A failure that a later attempt may survive.
TRANSIENT = "transient"

#: A failure that every later attempt reproduces.
PERMANENT = "permanent"

#: A failure that leaves the CUDA context untrustworthy, so the cell is recycled.
CONTAMINATING = "contaminating"

#: The whole vocabulary, in escalation order.
ERROR_CLASSES = (TRANSIENT, PERMANENT, CONTAMINATING)

#: Provider policy demanding every listed provider, in the listed order (WP1 vocabulary).
REQUIRE_LISTED = "requireListed"

#: Provider policy taking the first available listed provider (WP1 vocabulary).
PREFER_LISTED = "preferListed"

#: Manifest spellings accepted for each policy. The WP1 vocabulary is ``requireListed`` and
#: ``preferListed``; the shorter spellings in circulation map onto the same two behaviors, so a
#: bundle written against either reads the same.
_POLICY_ALIASES = {
    "requirelisted": REQUIRE_LISTED,
    "require": REQUIRE_LISTED,
    "required": REQUIRE_LISTED,
    "preferlisted": PREFER_LISTED,
    "prefer": PREFER_LISTED,
    "preferred": PREFER_LISTED,
}

#: The provider that means "this ran on the CPU".
CPU_PROVIDER = "CPUExecutionProvider"

#: The provider Phase 1 targets (D-IP-15).
CUDA_PROVIDER = "CUDAExecutionProvider"


class ProtocolError(Exception):
    """A message the receiving side cannot act on.

    Attributes:
        code: Stable SCREAMING_SNAKE code, safe for metrics and the bus.
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


class ProviderPolicyError(ProtocolError):
    """The session actual provider assignment does not satisfy the policy.

    This is the "no silent CPU fallback" gate. It is raised after the session exists and has
    reported what it actually runs on, never from a guess made before the session was built.
    """


# -- messages ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadModel:
    """Ask a cell to make one model generation resident.

    Attributes:
        digest: The bundle digest, ``sha256:<hex>``. The session cache is keyed by it.
        bundle_root: The cached bundle directory, as a string, holding ``manifest.json`` and the
            ONNX graph.
        providers: The execution providers to request, highest priority first.
        provider_policy: ``requireListed`` or ``preferListed``, from the bundle manifest.
        providers_permitted: The manifest ``providersPermitted``, which the policy is about. An
            empty tuple means the cell reads the list from the manifest it loads.
        warmup: Whether to warm the session before it serves work. A bundle carrying golden warmup
            samples is warmed whether or not this is set.
        required_provider: The provider ``runtime.requiredProvider`` demands in the actual
            assignment, or ``None`` when the route names none.
        allow_cpu_only: Whether a CPU-only assignment is acceptable. Development only
            (``runtime.allowCpuOnly``); a production route fails closed instead.
        gpu_mem_limit_mib: The CUDA provider arena bound, from the residency budget, or ``None``
            to leave the provider default in place.
    """

    digest: str
    bundle_root: str
    providers: tuple = (CPU_PROVIDER,)
    provider_policy: str = PREFER_LISTED
    providers_permitted: tuple = ()
    warmup: bool = True
    required_provider: Optional[str] = None
    allow_cpu_only: bool = False
    gpu_mem_limit_mib: Optional[int] = None


@dataclass(frozen=True)
class Loaded:
    """A model generation is resident and warmed.

    Attributes:
        digest: The bundle digest that is now resident.
        providers_assigned: The session actual provider assignment, highest priority first.
        load_ms: Wall-clock milliseconds the load and its warmup took. This is the measured reload
            cost the residency policy prices an eviction against.
        device_mib: Device memory the load consumed, measured across it, or ``0`` when no device
            probe is available.
        warmup_samples: Golden warmup samples compared against the manifest tolerances.
        gpu_device: The device ordinal as a string, or ``None`` for a CPU cell.
        gpu_class: The device name reported by NVML, or ``None`` when unknown.
    """

    digest: str
    providers_assigned: tuple
    load_ms: float
    device_mib: int
    warmup_samples: int = 0
    gpu_device: Optional[str] = None
    gpu_class: Optional[str] = None


@dataclass(frozen=True)
class LoadFailed:
    """A model generation could not be made resident.

    Attributes:
        digest: The bundle digest that failed to load.
        error: Operator-readable detail, already bounded.
        error_class: One of :data:`ERROR_CLASSES`.
        code: Stable SCREAMING_SNAKE code for metrics and events.
        memory_pressure: Whether the failure was device-memory exhaustion, which the residency
            policy treats as a measurement rather than only as a retry.
    """

    digest: str
    error: str
    error_class: str = TRANSIENT
    code: str = "UNCLASSIFIED"
    memory_pressure: bool = False


@dataclass(frozen=True)
class Infer:
    """Ask a cell to run one job.

    The cell reads the file itself. The parent sends a path and the digest it expects, never
    pixels (DESIGN.md §6.2).

    Attributes:
        inference_id: The durable job identity, echoed on the result.
        staged_path: The immutable file the cell reads, as a string.
        sha256: The digest that file must have, hex with or without the ``sha256:`` prefix.
        digest: The bundle digest whose session must serve this job.
        transform_version: The transform generation the job was pinned to. A session whose
            manifest declares another one refuses the job rather than answering with different
            preprocessing.
        queue_ms: How long the job waited before dispatch, measured by the parent and stamped into
            the result timings.
    """

    inference_id: str
    staged_path: str
    sha256: str
    digest: str
    transform_version: str
    queue_ms: float = 0.0


@dataclass(frozen=True)
class Unload:
    """Ask a cell to release one resident session.

    Attributes:
        digest: The bundle digest to release.
    """

    digest: str


@dataclass(frozen=True)
class Unloaded:
    """A session has been released.

    Attributes:
        digest: The bundle digest that was released.
        freed_mib: Device memory reclaimed, sampled after the settle period, or ``0`` when no
            device probe is available.
        was_resident: Whether the digest was resident at all. Unloading an absent digest is not an
            error; it is what a racing eviction looks like.
        expected_mib: What the session was measured to occupy, so the caller can see that an
            unload did not reclaim it. DESIGN.md §10.4 recycles the cell on that reading.
    """

    digest: str
    freed_mib: int
    was_resident: bool = True
    expected_mib: int = 0


@dataclass(frozen=True)
class Stats:
    """Ask a cell what it is holding."""


@dataclass(frozen=True)
class CellStats:
    """What a cell is holding.

    Attributes:
        resident: The resident bundle digests, in load order.
        device_free_mib: Device memory currently free, or ``0`` when no device probe is available.
        device_total_mib: Device memory installed, or ``0`` when no device probe is available.
        uptime_s: Seconds since the cell started.
        cell_id: The cell identity, as the supervisor named it.
        gpu_device: The device ordinal as a string, or ``None`` for a CPU cell.
        gpu_class: The device name reported by NVML, or ``None`` when unknown.
        resident_mib: Measured device memory per resident digest.
        inferences: Jobs this cell has answered since it started.
    """

    resident: tuple = ()
    device_free_mib: int = 0
    device_total_mib: int = 0
    uptime_s: float = 0.0
    cell_id: str = ""
    gpu_device: Optional[str] = None
    gpu_class: Optional[str] = None
    resident_mib: dict = field(default_factory=dict)
    inferences: int = 0


@dataclass(frozen=True)
class Shutdown:
    """Ask a cell to release everything and exit."""


#: Every request the cell dispatch loop accepts.
REQUESTS = (LoadModel, Infer, Unload, Stats, Shutdown)

#: Every reply the parent may receive. :class:`~image_processor.types.InferenceResult` answers
#: :class:`Infer`.
REPLIES = (Loaded, LoadFailed, InferenceResult, Unloaded, CellStats)


# -- provider policy -----------------------------------------------------------------------------


def normalize_policy(policy: Optional[str]) -> str:
    """Resolve a manifest ``providerPolicy`` to the WP1 vocabulary.

    Args:
        policy: The manifest spelling, or ``None`` for the default.

    Returns:
        :data:`REQUIRE_LISTED` or :data:`PREFER_LISTED`.

    Raises:
        ProviderPolicyError: ``PROVIDER_POLICY_UNKNOWN`` when the spelling is neither.
    """
    if policy is None:
        return PREFER_LISTED
    resolved = _POLICY_ALIASES.get(str(policy).strip().lower())
    if resolved is None:
        raise ProviderPolicyError(
            "PROVIDER_POLICY_UNKNOWN",
            f"providerPolicy {policy!r} is not {REQUIRE_LISTED} or {PREFER_LISTED}",
        )
    return resolved


def verify_provider_assignment(
    assigned,
    permitted=(),
    policy: Optional[str] = None,
    required_provider: Optional[str] = None,
    allow_cpu_only: bool = False,
) -> tuple:
    """Check a session actual provider assignment against the route and bundle policy.

    The check runs on what the session reports, not on what was asked for, because those differ
    exactly when it matters: ONNX Runtime falls back to the CPU silently, and a component that
    reports a CUDA route while running on a CPU is reporting a decision nobody made
    (DESIGN.md §10.1).

    Args:
        assigned: The session provider assignment, highest priority first.
        permitted: The manifest ``providersPermitted``. Empty places no restriction.
        policy: The manifest ``providerPolicy``. ``requireListed`` demands every permitted
            provider, in the listed order; ``preferListed`` demands that the highest-priority
            assigned provider is one of them.
        required_provider: The provider ``runtime.requiredProvider`` demands, or ``None``.
        allow_cpu_only: Whether a CPU-only assignment is acceptable.

    Returns:
        The assignment as a tuple, unchanged, so the caller records what it verified.

    Raises:
        ProviderPolicyError: ``PROVIDER_NONE``, ``PROVIDER_CPU_ONLY``,
            ``PROVIDER_REQUIRED_MISSING``, ``PROVIDER_NOT_PERMITTED``, or ``PROVIDER_ORDER`` when
            the assignment does not satisfy the policy.
    """
    actual = tuple(str(name) for name in assigned)
    resolved = normalize_policy(policy)
    listed = tuple(str(name) for name in permitted)

    if not actual:
        raise ProviderPolicyError("PROVIDER_NONE", "the session reported no execution provider")

    if set(actual) <= {CPU_PROVIDER} and not allow_cpu_only:
        raise ProviderPolicyError(
            "PROVIDER_CPU_ONLY",
            f"the session runs on {list(actual)} and allowCpuOnly is not set",
        )

    if required_provider and required_provider not in actual:
        raise ProviderPolicyError(
            "PROVIDER_REQUIRED_MISSING",
            f"requiredProvider {required_provider} is not in the assignment {list(actual)}",
        )

    if listed:
        if resolved == REQUIRE_LISTED:
            missing = [name for name in listed if name not in actual]
            if missing:
                raise ProviderPolicyError(
                    "PROVIDER_NOT_PERMITTED",
                    f"{REQUIRE_LISTED} demands {list(listed)}; the assignment is {list(actual)} "
                    f"and lacks {missing}",
                )
            order = [name for name in actual if name in listed]
            if order != list(listed):
                raise ProviderPolicyError(
                    "PROVIDER_ORDER",
                    f"{REQUIRE_LISTED} demands the order {list(listed)}; the assignment orders "
                    f"them {order}",
                )
        elif actual[0] not in listed:
            raise ProviderPolicyError(
                "PROVIDER_NOT_PERMITTED",
                f"{PREFER_LISTED} permits {list(listed)}; the session chose {actual[0]}",
            )

    return actual


# -- error classification ------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorInfo:
    """What one failure means.

    Attributes:
        error_class: One of :data:`ERROR_CLASSES`.
        code: Stable SCREAMING_SNAKE code, safe for metrics and events.
        message: The bounded operator-readable detail.
        memory_pressure: Whether the failure is evidence that the device ran out of memory. The
            residency policy treats it as a measurement, not only as a retry.
    """

    error_class: str
    code: str
    message: str
    memory_pressure: bool = False


#: Longest message kept on a result or an event. Model runtimes emit multi-kilobyte diagnostics,
#: and DESIGN.md §12.1 bounds every collection that reaches the bus.
MAX_ERROR_CHARS = 1024

#: Contaminating signatures. A CUDA context that produced one of these is not reusable, so the
#: cell is recycled rather than retried in place (DESIGN.md §6.2, §10.4).
_CONTAMINATING_PATTERNS = (
    (
        "CUDA_ILLEGAL_ADDRESS",
        (
            "an illegal memory access",
            "illegal address",
            "cudaerrorillegaladdress",
            "misaligned address",
            "device-side assert",
        ),
    ),
    (
        "CUDA_LAUNCH_FAILURE",
        (
            "unspecified launch failure",
            "cudaerrorlaunchfailure",
            "launch timed out",
            "cudaerrorlaunchtimeout",
        ),
    ),
    (
        "CUDA_CONTEXT_LOST",
        (
            "context is destroyed",
            "cudaerrorcontextisdestroyed",
            "cuda_error_invalid_context",
            "invalid device context",
            "cuda_error_context_is_destroyed",
            "cudaerrorcudartunloading",
        ),
    ),
    (
        "CUDA_HARDWARE_FAULT",
        (
            "uncorrectable ecc",
            "ecc uncorrectable",
            "cudaerroreccuncorrectable",
            "cudaerrorhardwarestackerror",
            "has fallen off the bus",
        ),
    ),
)

#: Device-memory exhaustion. Transient, and a measurement the residency policy uses.
_MEMORY_PATTERNS = (
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "out of memory",
    "failed to allocate memory",
    "failed to allocate a buffer",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "bad_alloc",
    "outofmemory",
)

#: Signatures that reproduce on every attempt.
_PERMANENT_PATTERNS = (
    (
        "PROVIDER_UNAVAILABLE",
        (
            "no cuda-capable device",
            "cudaerrornodevice",
            "cuda driver version is insufficient",
            "cudaerrorinsufficientdriver",
            "no kernel image is available",
            "cannot open shared object",
            "the specified module could not be found",
            "libcudnn",
            "libcublas",
        ),
    ),
    (
        "MODEL_INVALID",
        (
            "invalid model",
            "invalid protobuf",
            "protobuf parsing failed",
            "failed to load model",
            "load model from",
            "invalid graph",
            "no graph was found",
            "unsupported model ir version",
        ),
    ),
    (
        "UNSUPPORTED_MODEL",
        (
            "not implemented",
            "unsupported op",
            "no opset import",
            "invalid opset",
            "is not a registered function/op",
        ),
    ),
    (
        "SHAPE_MISMATCH",
        (
            "got invalid dimensions",
            "invalid rank",
            "shape mismatch",
            "invalid input shape",
            "static input shape",
            "index out of range",
        ),
    ),
    (
        "INPUT_MISMATCH",
        (
            "invalid feed input name",
            "required inputs",
            "missing input",
            "unexpected input",
            "invalid input name",
            "unexpected type",
        ),
    ),
    (
        "FILE_MISSING",
        (
            "no such file or directory",
            "system cannot find the file",
            "system cannot find the path",
        ),
    ),
)

#: Signatures a later attempt may survive without recycling anything.
_TRANSIENT_PATTERNS = (
    (
        "DEVICE_BUSY",
        (
            "device is busy",
            "all cuda-capable devices are busy",
            "cudaerrordevicealreadyinuse",
            "resource temporarily unavailable",
        ),
    ),
    ("TIMEOUT", ("timed out", "timeout")),
)

#: Exception type names classified without reading the message. The component's own errors are
#: matched by name so this module imports neither the packages above it nor ``onnxruntime``.
_TYPE_CLASSES = {
    "DecodeError": (PERMANENT, "IMAGE_UNDECODABLE", False),
    "FamilyError": (PERMANENT, "FAMILY_REFUSED", False),
    "BundleError": (PERMANENT, "BUNDLE_INVALID", False),
    "ProviderPolicyError": (PERMANENT, "PROVIDER_POLICY", False),
    "ProtocolError": (PERMANENT, "PROTOCOL", False),
    "MemoryError": (TRANSIENT, "HOST_OOM", True),
    "OutOfMemory": (TRANSIENT, "CUDA_OOM", True),
    "InvalidArgument": (PERMANENT, "MODEL_ARGUMENT_INVALID", False),
    "InvalidProtobuf": (PERMANENT, "MODEL_INVALID", False),
    "InvalidGraph": (PERMANENT, "MODEL_INVALID", False),
    "NoSuchFile": (PERMANENT, "FILE_MISSING", False),
    "NoModel": (PERMANENT, "MODEL_INVALID", False),
    "NotImplemented": (PERMANENT, "UNSUPPORTED_MODEL", False),
    "ModelLoadCanceled": (TRANSIENT, "MODEL_LOAD_CANCELED", False),
    "FileNotFoundError": (PERMANENT, "FILE_MISSING", False),
    "IsADirectoryError": (PERMANENT, "FILE_MISSING", False),
    "PermissionError": (TRANSIENT, "IO_ERROR", False),
    "TimeoutError": (TRANSIENT, "TIMEOUT", False),
    "OSError": (TRANSIENT, "IO_ERROR", False),
}

#: The component's own error types, whose ``code`` attribute is already the stable code.
_OWN_ERRORS = frozenset(
    {"DecodeError", "FamilyError", "BundleError", "ProviderPolicyError", "ProtocolError"}
)

#: Collapses the multi-line diagnostics model runtimes emit.
_WHITESPACE = re.compile(r"\s+")


def bound_message(text, limit: int = MAX_ERROR_CHARS) -> str:
    """Collapse and truncate a runtime message so it is safe to publish.

    Args:
        text: The raw message.
        limit: Longest result, in characters.

    Returns:
        The message on one line, truncated with an ellipsis when it was longer than ``limit``.
    """
    collapsed = _WHITESPACE.sub(" ", str(text)).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def classify_message(text) -> Optional[ErrorInfo]:
    """Classify a runtime message by signature.

    The order is fixed: a context that may be poisoned is recognized before a memory failure,
    memory before a permanent diagnosis, and permanent before a retry, so a message carrying more
    than one signature is read as the most serious one.

    Args:
        text: The exception message.

    Returns:
        The classification, or ``None`` when no signature matches.
    """
    lowered = str(text).lower()
    for code, patterns in _CONTAMINATING_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return ErrorInfo(CONTAMINATING, code, bound_message(text))
    if any(pattern in lowered for pattern in _MEMORY_PATTERNS):
        return ErrorInfo(TRANSIENT, "CUDA_OOM", bound_message(text), memory_pressure=True)
    for code, patterns in _PERMANENT_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return ErrorInfo(PERMANENT, code, bound_message(text))
    for code, patterns in _TRANSIENT_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return ErrorInfo(TRANSIENT, code, bound_message(text))
    return None


def classify_error(exc: BaseException) -> ErrorInfo:
    """Classify one exception raised while loading a model or running a job.

    Args:
        exc: The exception.

    Returns:
        The classification. An exception no rule recognizes is ``transient`` with the code
        ``UNCLASSIFIED``: a failure the component cannot explain is retried under its budget and
        then exhausted, never dropped and never silently called permanent.
    """
    names = [type(exc).__name__] + [base.__name__ for base in type(exc).__mro__[1:]]
    text = str(exc) or type(exc).__name__
    code_attribute = getattr(exc, "code", None)

    if names[0] in _OWN_ERRORS:
        error_class, code, memory = _TYPE_CLASSES[names[0]]
        if isinstance(code_attribute, str) and code_attribute:
            code = code_attribute
        return ErrorInfo(error_class, code, bound_message(text), memory_pressure=memory)

    from_message = classify_message(text)
    if from_message is not None:
        return from_message

    for name in names:
        known = _TYPE_CLASSES.get(name)
        if known is not None:
            error_class, code, memory = known
            return ErrorInfo(error_class, code, bound_message(text), memory_pressure=memory)

    return ErrorInfo(TRANSIENT, "UNCLASSIFIED", bound_message(text))


__all__ = [
    "CONTAMINATING",
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "ERROR_CLASSES",
    "MAX_ERROR_CHARS",
    "PERMANENT",
    "PREFER_LISTED",
    "REPLIES",
    "REQUESTS",
    "REQUIRE_LISTED",
    "TRANSIENT",
    "CellStats",
    "ErrorInfo",
    "Infer",
    "LoadFailed",
    "LoadModel",
    "Loaded",
    "ProtocolError",
    "ProviderPolicyError",
    "Shutdown",
    "Stats",
    "Unload",
    "Unloaded",
    "bound_message",
    "classify_error",
    "classify_message",
    "normalize_policy",
    "verify_provider_assignment",
]
