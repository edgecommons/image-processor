"""Trigger admission: inline images within the envelope cap, and verified file references."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgecommons.messaging.message import Message, MessageHeader

from image_processor.types import SourceKind
from image_processor.sources.staging import MAX_INLINE_BYTES, sha256_bytes, staged_path_for
from image_processor.sources.trigger import (
    BINARY_BODY_KEY,
    TriggerError,
    TriggerSource,
    request_correlation,
    suffix_for,
)
from tests.sources.conftest import trigger_route

JPEG = b"\xff\xd8\xff\xe0" + b"a small jpeg body" * 4
PNG = b"\x89PNG\r\n\x1a\x0a" + b"a small png body"


def binary_marker(data: bytes, length: int | None = None) -> dict:
    """Build the core's bounded binary marker for a structured body field."""
    return {
        BINARY_BODY_KEY: {
            "encoding": "base64",
            "length": len(data) if length is None else length,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def source_for(tmp_path, events, **kwargs):
    """Build a trigger source staging under ``tmp_path``."""
    kwargs.setdefault("inline_staging", tmp_path / "staging")
    return TriggerSource(trigger_route(**kwargs), events)


def test_an_opaque_binary_body_is_staged_and_announced(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message(Message(header=MessageHeader("InspectRequest", "1.0"), body=JPEG))

    route_id, identity, staged = events.discovered_calls[0]
    assert route_id == "adhoc-inspect"
    assert identity.kind is SourceKind.INLINE
    assert identity.bytes == len(JPEG)
    assert identity.sha256 == sha256_bytes(JPEG)
    assert staged == staged_path_for(tmp_path / "staging", sha256_bytes(JPEG), ".jpg")
    assert staged.read_bytes() == JPEG
    assert identity.relative_path == "/".join(staged.parts[-2:])


def test_a_structured_body_carrying_image_bytes_is_staged(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message({"body": {"image": PNG, "note": "from the inspection UI"}})

    identity = events.discovered_calls[0][1]
    assert identity.kind is SourceKind.INLINE
    assert identity.sha256 == sha256_bytes(PNG)


def test_a_structured_body_carrying_the_core_binary_marker_is_staged(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message({"body": {"image": binary_marker(JPEG)}})

    assert events.discovered_calls[0][1].sha256 == sha256_bytes(JPEG)


def test_the_same_inline_image_twice_stages_one_file(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message(Message(body=JPEG))
    source.on_message(Message(body=JPEG))

    first, second = (call[2] for call in events.discovered_calls)
    assert first == second
    assert source.accepted == 2


def test_request_correlation_travels_with_the_identity(tmp_path, events):
    header = MessageHeader("InspectRequest", "1.0", correlation_id="corr-42")
    header.make_request("ecv1/dallas-01/inspection-ui/ui-1/app/inspect/reply")
    source = source_for(tmp_path, events)

    source.on_message(Message(header=header, body=JPEG))

    identity = events.discovered_calls[0][1]
    assert identity.correlation_id == "corr-42"
    assert identity.reply_to == "ecv1/dallas-01/inspection-ui/ui-1/app/inspect/reply"


def test_an_inline_image_over_the_envelope_cap_is_refused(tmp_path, events):
    source = source_for(tmp_path, events)
    oversized = b"x" * (MAX_INLINE_BYTES + 1)

    source.on_message({"body": {"image": oversized}})

    assert events.discovered_calls == []
    assert events.reasons == ["INLINE_TOO_LARGE"]


def test_a_configured_cap_above_the_envelope_cap_is_clamped(tmp_path, events):
    source = source_for(tmp_path, events, max_inline_bytes=10 * 1024 * 1024)

    assert source.max_inline_bytes == MAX_INLINE_BYTES

    source.on_message({"body": {"image": b"x" * (MAX_INLINE_BYTES + 1)}})

    assert events.reasons == ["INLINE_TOO_LARGE"]


def test_a_configured_cap_below_the_envelope_cap_is_honored(tmp_path, events):
    source = source_for(tmp_path, events, max_inline_bytes=16)

    source.on_message({"body": {"image": b"x" * 17}})
    source.on_message({"body": {"image": b"x" * 16}})

    assert events.reasons == ["INLINE_TOO_LARGE"]
    assert len(events.discovered_calls) == 1


def test_a_core_message_that_refuses_its_own_binary_body_is_refused_here(tmp_path, events):
    class Oversized:
        def get_binary_body(self):
            raise ValueError("Binary message body exceeds 65536 bytes")

        def get_body(self):
            return None

    source = source_for(tmp_path, events)

    source.on_message(Oversized())

    assert events.reasons == ["INLINE_TOO_LARGE"]


def test_a_binary_marker_declaring_more_than_the_cap_is_refused(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message({"body": {"image": binary_marker(JPEG, length=MAX_INLINE_BYTES + 1)}})

    assert events.reasons == ["INLINE_TOO_LARGE"]


@pytest.mark.parametrize(
    "body",
    [
        {"image": {BINARY_BODY_KEY: "not an object"}},
        {"image": {BINARY_BODY_KEY: {"encoding": "base64", "length": 3}}},
        {"image": {BINARY_BODY_KEY: {"encoding": "base64", "length": 3, "data": "!!!"}}},
    ],
)
def test_a_malformed_binary_marker_is_refused(tmp_path, events, body):
    source = source_for(tmp_path, events)

    source.on_message({"body": body})

    assert events.reasons == ["MALFORMED_BODY"]


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"note": "no image here"}, "MALFORMED_BODY"),
        ("a string body", "MALFORMED_BODY"),
        (None, "MALFORMED_BODY"),
        ({"image": b""}, "EMPTY_BODY"),
    ],
)
def test_a_body_that_is_neither_form_is_refused(tmp_path, events, body, reason):
    source = source_for(tmp_path, events)

    source.on_message({"body": body})

    assert events.discovered_calls == []
    assert events.reasons == [reason]


def test_an_empty_binary_body_is_refused(tmp_path, events):
    class Empty:
        def get_binary_body(self):
            return b""

        def get_body(self):
            return b""

    source = source_for(tmp_path, events)

    source.on_message(Empty())

    assert events.reasons == ["EMPTY_BODY"]


@pytest.mark.parametrize(
    "data, suffix",
    [
        (JPEG, ".jpg"),
        (PNG, ".png"),
        (b"II*\x00rest", ".tif"),
        (b"MM\x00*rest", ".tif"),
        (b"BM bitmap", ".bmp"),
        (b"GIF89a", ".gif"),
        (b"RIFF" + b"size" + b"WEBP", ".webp"),
        (b"anything else at all", ".img"),
    ],
)
def test_the_staged_name_says_what_the_bytes_are(data, suffix):
    assert suffix_for(data) == suffix


# -- file references ---------------------------------------------------------------------------


def reference_source(tmp_path, events, **kwargs):
    """Build a trigger source with a file root as well as staging."""
    file_root = tmp_path / "inspection"
    file_root.mkdir(exist_ok=True)
    kwargs.setdefault("file_root", file_root)
    kwargs.setdefault("inline_staging", tmp_path / "staging")
    return TriggerSource(trigger_route(**kwargs), events), file_root


def test_a_verified_reference_is_staged_and_announced(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    (file_root / "line-a").mkdir()
    (file_root / "line-a" / "frame.jpg").write_bytes(JPEG)

    source.on_message(
        {
            "header": {"correlation_id": "corr-7", "reply_to": "reply/topic"},
            "body": {
                "relativePath": "line-a/frame.jpg",
                "sha256": sha256_bytes(JPEG),
                "bytes": len(JPEG),
            },
        }
    )

    route_id, identity, staged = events.discovered_calls[0]
    assert identity.kind is SourceKind.REFERENCE
    assert identity.relative_path == "line-a/frame.jpg"
    assert identity.sha256 == sha256_bytes(JPEG)
    assert identity.correlation_id == "corr-7"
    assert identity.reply_to == "reply/topic"
    assert staged.read_bytes() == JPEG
    assert staged != file_root / "line-a" / "frame.jpg"


def test_a_reference_escaping_the_file_root_is_refused(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(JPEG)

    source.on_message(
        {
            "body": {
                "relativePath": "../outside/secret.jpg",
                "sha256": sha256_bytes(JPEG),
                "bytes": len(JPEG),
            }
        }
    )

    assert events.discovered_calls == []
    assert events.reasons == ["PATH_ESCAPE"]


def test_a_reference_naming_an_absolute_path_is_refused(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(JPEG)

    source.on_message(
        {
            "body": {
                "relativePath": str(outside),
                "sha256": sha256_bytes(JPEG),
                "bytes": len(JPEG),
            }
        }
    )

    assert events.reasons == ["ABSOLUTE_PATH"]


def test_a_reference_whose_digest_does_not_match_is_refused(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    (file_root / "frame.jpg").write_bytes(JPEG)

    source.on_message(
        {"body": {"relativePath": "frame.jpg", "sha256": "0" * 64, "bytes": len(JPEG)}}
    )

    assert events.discovered_calls == []
    assert events.reasons == ["DIGEST_MISMATCH"]
    assert not (tmp_path / "staging").exists()


def test_a_reference_whose_size_does_not_match_is_refused_before_it_is_hashed(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    (file_root / "frame.jpg").write_bytes(JPEG)

    source.on_message(
        {
            "body": {
                "relativePath": "frame.jpg",
                "sha256": sha256_bytes(JPEG),
                "bytes": len(JPEG) + 10,
            }
        }
    )

    assert events.reasons == ["SIZE_MISMATCH"]


def test_a_reference_to_a_file_that_is_not_there_is_refused(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)

    source.on_message(
        {"body": {"relativePath": "absent.jpg", "sha256": "a" * 64, "bytes": 10}}
    )

    assert events.reasons == ["MISSING"]


@pytest.mark.parametrize(
    "body",
    [
        {"relativePath": "frame.jpg", "sha256": "short", "bytes": 10},
        {"relativePath": "frame.jpg", "sha256": "a" * 64},
        {"relativePath": "frame.jpg", "sha256": "a" * 64, "bytes": "ten"},
        {"relativePath": "frame.jpg", "sha256": "a" * 64, "bytes": True},
    ],
)
def test_a_reference_missing_its_declarations_is_refused(tmp_path, events, body):
    source, file_root = reference_source(tmp_path, events)

    source.on_message({"body": body})

    assert events.reasons == ["MALFORMED_BODY"]


def test_a_reference_route_with_no_file_root_refuses_references(tmp_path, events):
    source = TriggerSource(trigger_route(inline_staging=tmp_path / "staging"), events)

    source.on_message({"body": {"relativePath": "a.jpg", "sha256": "a" * 64, "bytes": 1}})

    assert events.reasons == ["NO_FILE_ROOT"]


def test_a_route_with_no_staging_directory_is_refused(events):
    with pytest.raises(TriggerError) as caught:
        TriggerSource(trigger_route(), events)
    assert caught.value.code == "INLINE_STAGING_REQUIRED"


def test_a_route_with_no_id_is_refused(tmp_path, events):
    with pytest.raises(TriggerError):
        TriggerSource(SimpleNamespace(id="", source=SimpleNamespace()), events)
    with pytest.raises(TriggerError):
        TriggerSource(SimpleNamespace(source=SimpleNamespace()), events)


def test_the_subscribed_filters_come_from_configuration(tmp_path, events):
    many = source_for(tmp_path, events, subscribe=("a/#", "b/#"))
    one = source_for(tmp_path, events, subscribe="a/#")

    assert many.subscribe == ("a/#", "b/#")
    assert one.subscribe == ("a/#",)


def test_a_reference_that_changes_while_it_is_read_is_refused(tmp_path, events, monkeypatch):
    source, file_root = reference_source(tmp_path, events)
    target = file_root / "frame.jpg"
    target.write_bytes(JPEG)
    import image_processor.sources.trigger as trigger_module

    real = trigger_module.sha256_file

    def hash_then_change(path, chunk=1 << 20):
        digest = real(path, chunk)
        import os

        os.utime(target, ns=(0, 0))
        return digest

    monkeypatch.setattr(trigger_module, "sha256_file", hash_then_change)

    source.on_message(
        {"body": {"relativePath": "frame.jpg", "sha256": sha256_bytes(JPEG),
                  "bytes": len(JPEG)}}
    )

    assert events.reasons == ["CHANGED_DURING_READ"]


def test_a_staging_failure_is_reported_rather_than_raised(tmp_path, events, monkeypatch):
    import image_processor.sources.trigger as trigger_module

    def refuse(*args, **kwargs):
        raise OSError("the staging volume is full")

    monkeypatch.setattr(trigger_module, "stage_bytes", refuse)
    source = source_for(tmp_path, events)

    source.on_message({"body": {"image": JPEG}})

    assert events.reasons == ["STAGING_FAILED"]


def test_a_reference_staging_failure_is_reported_rather_than_raised(tmp_path, events, monkeypatch):
    import image_processor.sources.trigger as trigger_module

    source, file_root = reference_source(tmp_path, events)
    (file_root / "frame.jpg").write_bytes(JPEG)
    monkeypatch.setattr(
        trigger_module,
        "stage_copy",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    source.on_message(
        {"body": {"relativePath": "frame.jpg", "sha256": sha256_bytes(JPEG),
                  "bytes": len(JPEG)}}
    )

    assert events.reasons == ["STAGING_FAILED"]


def test_a_reference_to_something_that_is_not_a_regular_file_is_refused(tmp_path, events):
    source, file_root = reference_source(tmp_path, events)
    (file_root / "a-directory").mkdir()

    source.on_message(
        {"body": {"relativePath": "a-directory", "sha256": "a" * 64, "bytes": 1}}
    )

    assert events.reasons == ["DIRECTORY"]


@pytest.mark.parametrize(
    "message, expected",
    [
        (SimpleNamespace(), (None, None)),
        ({"header": {"correlation_id": "c", "reply_to": "r"}}, ("c", "r")),
        ({"header": {"correlationId": "c", "replyTo": "r"}}, ("c", "r")),
        ({"header": {"correlation_id": 17}}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
)
def test_request_correlation_reads_every_envelope_shape(message, expected):
    assert request_correlation(message) == expected


def test_a_bare_byte_string_message_is_an_inline_image(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message(JPEG)

    assert events.discovered_calls[0][1].sha256 == sha256_bytes(JPEG)


def test_an_envelope_whose_body_is_bytes_is_an_inline_image(tmp_path, events):
    source = source_for(tmp_path, events)

    source.on_message({"body": JPEG})

    assert events.discovered_calls[0][1].sha256 == sha256_bytes(JPEG)


def test_a_reference_that_cannot_be_read_is_refused(tmp_path, events, monkeypatch):
    import image_processor.sources.trigger as trigger_module

    source, file_root = reference_source(tmp_path, events)
    (file_root / "frame.jpg").write_bytes(JPEG)

    def refuse(path, chunk=1 << 20):
        raise PermissionError("another process holds the file")

    monkeypatch.setattr(trigger_module, "sha256_file", refuse)

    source.on_message(
        {"body": {"relativePath": "frame.jpg", "sha256": sha256_bytes(JPEG),
                  "bytes": len(JPEG)}}
    )

    assert events.reasons == ["MISSING"]
