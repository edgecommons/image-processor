"""The parent-to-cell protocol: message shape, provider policy, and error classification (LLD §6).

Classification is the table the whole retry story rests on, so it is asserted signature by
signature rather than by example: an out-of-memory must be a retry with a memory signal, an illegal
address must recycle the cell, and a model the runtime refuses must never come back as a retry.
"""

import pickle

import pytest

from image_processor.engine import protocol
from image_processor.engine.protocol import (
    CONTAMINATING,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    PERMANENT,
    PREFER_LISTED,
    REQUIRE_LISTED,
    TRANSIENT,
    CellStats,
    Infer,
    LoadFailed,
    LoadModel,
    Loaded,
    ProtocolError,
    ProviderPolicyError,
    Shutdown,
    Stats,
    Unload,
    Unloaded,
    bound_message,
    classify_error,
    classify_message,
    normalize_policy,
    verify_provider_assignment,
)

MESSAGES = [
    LoadModel(digest="sha256:aa", bundle_root="/cache/aa"),
    Loaded(digest="sha256:aa", providers_assigned=(CUDA_PROVIDER,), load_ms=1.0, device_mib=8),
    LoadFailed(digest="sha256:aa", error="boom"),
    Infer("job-1", "/staged/a.jpg", "d" * 64, "sha256:aa", "t1"),
    Unload("sha256:aa"),
    Unloaded("sha256:aa", 8),
    Stats(),
    CellStats(resident=("sha256:aa",)),
    Shutdown(),
]


@pytest.mark.parametrize("message", MESSAGES, ids=lambda m: type(m).__name__)
def test_every_message_survives_the_pipe(message):
    assert pickle.loads(pickle.dumps(message)) == message


def test_the_request_and_reply_sets_are_declared():
    assert protocol.REQUESTS == (LoadModel, Infer, Unload, Stats, Shutdown)
    assert Loaded in protocol.REPLIES and CellStats in protocol.REPLIES


def test_the_error_classes_are_the_three_the_scheduler_acts_on():
    assert protocol.ERROR_CLASSES == (TRANSIENT, PERMANENT, CONTAMINATING)


CLASSIFICATIONS = [
    ("CUDA failure 700: an illegal memory access was encountered", CONTAMINATING, "CUDA_ILLEGAL_ADDRESS", False),
    ("cudaErrorIllegalAddress: misaligned address", CONTAMINATING, "CUDA_ILLEGAL_ADDRESS", False),
    ("CUDA error: device-side assert triggered", CONTAMINATING, "CUDA_ILLEGAL_ADDRESS", False),
    ("unspecified launch failure", CONTAMINATING, "CUDA_LAUNCH_FAILURE", False),
    ("CUDA_ERROR_LAUNCH_TIMEOUT: the launch timed out and was terminated", CONTAMINATING, "CUDA_LAUNCH_FAILURE", False),
    ("cudaErrorContextIsDestroyed: context is destroyed", CONTAMINATING, "CUDA_CONTEXT_LOST", False),
    ("CUDA_ERROR_INVALID_CONTEXT", CONTAMINATING, "CUDA_CONTEXT_LOST", False),
    ("Uncorrectable ECC error encountered", CONTAMINATING, "CUDA_HARDWARE_FAULT", False),
    ("CUDA failure 2: out of memory", TRANSIENT, "CUDA_OOM", True),
    ("CUDA_ERROR_OUT_OF_MEMORY", TRANSIENT, "CUDA_OOM", True),
    ("Failed to allocate memory for requested buffer of size 4294967296", TRANSIENT, "CUDA_OOM", True),
    ("CUDNN_STATUS_ALLOC_FAILED", TRANSIENT, "CUDA_OOM", True),
    ("std::bad_alloc", TRANSIENT, "CUDA_OOM", True),
    ("No CUDA-capable device is detected", PERMANENT, "PROVIDER_UNAVAILABLE", False),
    ("CUDA driver version is insufficient for CUDA runtime version", PERMANENT, "PROVIDER_UNAVAILABLE", False),
    ("no kernel image is available for execution on the device", PERMANENT, "PROVIDER_UNAVAILABLE", False),
    ("libcudnn.so.8: cannot open shared object file", PERMANENT, "PROVIDER_UNAVAILABLE", False),
    ("Load model from /cache/aa/model.onnx failed: Protobuf parsing failed", PERMANENT, "MODEL_INVALID", False),
    ("Invalid Model: no graph was found in the protobuf", PERMANENT, "MODEL_INVALID", False),
    ("Unsupported model IR version 12", PERMANENT, "MODEL_INVALID", False),
    ("Fatal error: Foo(1) is not a registered function/op", PERMANENT, "UNSUPPORTED_MODEL", False),
    ("Could not find an implementation: not implemented", PERMANENT, "UNSUPPORTED_MODEL", False),
    ("Got invalid dimensions for input: images for the following indices", PERMANENT, "SHAPE_MISMATCH", False),
    ("Invalid rank for input: images Got: 3 Expected: 4", PERMANENT, "SHAPE_MISMATCH", False),
    ("Invalid Feed Input Name: pixel_values", PERMANENT, "INPUT_MISMATCH", False),
    ("[Errno 2] No such file or directory: '/staged/a.jpg'", PERMANENT, "FILE_MISSING", False),
    ("all CUDA-capable devices are busy or unavailable", TRANSIENT, "DEVICE_BUSY", False),
    ("the operation timed out", TRANSIENT, "TIMEOUT", False),
]


@pytest.mark.parametrize(
    "text,error_class,code,memory", CLASSIFICATIONS, ids=[row[2] + ":" + row[0][:28] for row in CLASSIFICATIONS]
)
def test_a_runtime_message_is_classified_by_signature(text, error_class, code, memory):
    info = classify_message(text)
    assert info is not None
    assert (info.error_class, info.code, info.memory_pressure) == (error_class, code, memory)


def test_a_message_with_no_signature_is_not_guessed_at():
    assert classify_message("something the table has never seen") is None


def test_a_poisoned_context_outranks_an_out_of_memory_in_the_same_message():
    info = classify_message("out of memory after an illegal memory access was encountered")
    assert info.error_class == CONTAMINATING


class DecodeError(Exception):
    """Stands in for the engine's decode error, which is matched by type name."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class OutOfMemory(Exception):
    """Stands in for the onnxruntime exception of the same name."""


class InvalidArgument(Exception):
    """Stands in for the onnxruntime exception of the same name."""


def test_the_components_own_errors_keep_their_code():
    info = classify_error(DecodeError("IMAGE_TRUNCATED", "the file ends mid-scan"))
    assert (info.error_class, info.code) == (PERMANENT, "IMAGE_TRUNCATED")


def test_a_protocol_error_is_permanent_and_keeps_its_code():
    info = classify_error(ProtocolError("WARMUP_MISMATCH", "the golden answer moved"))
    assert (info.error_class, info.code) == (PERMANENT, "WARMUP_MISMATCH")


def test_a_runtime_exception_type_is_classified_when_its_message_says_nothing():
    assert classify_error(OutOfMemory("")).error_class == TRANSIENT
    assert classify_error(OutOfMemory("")).memory_pressure is True
    assert classify_error(InvalidArgument("unreadable")).error_class == PERMANENT


def test_a_host_memory_error_is_a_retry_with_a_memory_signal():
    info = classify_error(MemoryError())
    assert (info.error_class, info.code, info.memory_pressure) == (TRANSIENT, "HOST_OOM", True)


def test_a_missing_staged_file_is_permanent_and_a_permission_problem_is_not():
    assert classify_error(FileNotFoundError(2, "nope")).error_class == PERMANENT
    assert classify_error(PermissionError(13, "locked")).error_class == TRANSIENT


def test_an_unrecognized_failure_is_retried_rather_than_declared_permanent():
    info = classify_error(RuntimeError("a failure nobody has a rule for"))
    assert (info.error_class, info.code) == (TRANSIENT, "UNCLASSIFIED")


def test_an_exception_with_no_message_still_carries_its_type():
    assert classify_error(RuntimeError()).message == "RuntimeError"


def test_a_runtime_message_is_collapsed_and_bounded():
    assert bound_message("a\n  b\tc") == "a b c"
    assert len(bound_message("x" * 5000)) == protocol.MAX_ERROR_CHARS
    assert bound_message("x" * 5000).endswith("...")


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("requireListed", REQUIRE_LISTED),
        ("require", REQUIRE_LISTED),
        ("preferListed", PREFER_LISTED),
        ("preferred", PREFER_LISTED),
        (None, PREFER_LISTED),
    ],
)
def test_the_provider_policy_vocabulary_is_normalized(spelling, expected):
    assert normalize_policy(spelling) == expected


def test_an_unknown_provider_policy_is_refused():
    with pytest.raises(ProviderPolicyError) as raised:
        normalize_policy("whateverListed")
    assert raised.value.code == "PROVIDER_POLICY_UNKNOWN"


def test_a_cpu_only_assignment_is_refused_unless_development_allows_it():
    with pytest.raises(ProviderPolicyError) as raised:
        verify_provider_assignment([CPU_PROVIDER], [CPU_PROVIDER], PREFER_LISTED)
    assert raised.value.code == "PROVIDER_CPU_ONLY"
    assert verify_provider_assignment(
        [CPU_PROVIDER], [CPU_PROVIDER], PREFER_LISTED, allow_cpu_only=True
    ) == (CPU_PROVIDER,)


def test_a_session_with_no_provider_is_refused():
    with pytest.raises(ProviderPolicyError) as raised:
        verify_provider_assignment([])
    assert raised.value.code == "PROVIDER_NONE"


def test_the_required_provider_must_be_in_the_actual_assignment():
    with pytest.raises(ProviderPolicyError) as raised:
        verify_provider_assignment(
            [CPU_PROVIDER], [CPU_PROVIDER], PREFER_LISTED,
            required_provider=CUDA_PROVIDER, allow_cpu_only=True,
        )
    assert raised.value.code == "PROVIDER_REQUIRED_MISSING"


def test_require_listed_demands_every_listed_provider_in_order():
    permitted = [CUDA_PROVIDER, CPU_PROVIDER]
    assert verify_provider_assignment(
        [CUDA_PROVIDER, CPU_PROVIDER], permitted, REQUIRE_LISTED
    ) == (CUDA_PROVIDER, CPU_PROVIDER)

    with pytest.raises(ProviderPolicyError) as missing:
        verify_provider_assignment([CPU_PROVIDER], permitted, REQUIRE_LISTED, allow_cpu_only=True)
    assert missing.value.code == "PROVIDER_NOT_PERMITTED"

    with pytest.raises(ProviderPolicyError) as order:
        verify_provider_assignment(
            [CPU_PROVIDER, CUDA_PROVIDER], permitted, REQUIRE_LISTED, allow_cpu_only=True
        )
    assert order.value.code == "PROVIDER_ORDER"


def test_prefer_listed_demands_only_that_the_chosen_provider_is_permitted():
    assert verify_provider_assignment(
        [CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER], PREFER_LISTED
    ) == (CUDA_PROVIDER, CPU_PROVIDER)
    with pytest.raises(ProviderPolicyError) as raised:
        verify_provider_assignment(
            ["TensorrtExecutionProvider", CPU_PROVIDER], [CUDA_PROVIDER], PREFER_LISTED
        )
    assert raised.value.code == "PROVIDER_NOT_PERMITTED"


def test_an_empty_permitted_list_places_no_restriction():
    assert verify_provider_assignment([CUDA_PROVIDER], (), REQUIRE_LISTED) == (CUDA_PROVIDER,)
