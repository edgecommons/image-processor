"""Shared value types (LLD §2): the identity derivation every package depends on."""

import re

from image_processor.types import (
    TERMINAL_STATES,
    CompletionAction,
    JobState,
    SecretRef,
    SourceKind,
    derive_inference_id,
)

DIGEST_A = "sha256:" + "ab" * 32
DIGEST_B = "sha256:" + "cd" * 32
SOURCE_SHA = "ef" * 32


def derive(**overrides):
    arguments = {
        "route_id": "clearance-cam-01",
        "capture_id": None,
        "source_sha256": SOURCE_SHA,
        "normalized_source_id": "2026/08/22/cap-0001.jpg",
        "model_digest": DIGEST_A,
    }
    arguments.update(overrides)
    return derive_inference_id(**arguments)


def test_the_same_input_under_the_same_model_yields_the_same_id():
    assert derive() == derive()
    assert derive(capture_id="01KZ8Q4M") == derive(capture_id="01KZ8Q4M")


def test_a_new_model_digest_is_a_new_inference():
    assert derive() != derive(model_digest=DIGEST_B)
    assert derive(capture_id="01KZ8Q4M") != derive(capture_id="01KZ8Q4M", model_digest=DIGEST_B)


def test_the_capture_key_and_the_fallback_key_are_distinct():
    """A capture id keys the job on its own; without one the path and digest stand in."""
    assert derive(capture_id="01KZ8Q4M") != derive()
    # With a capture id the path and digest no longer take part.
    assert derive(capture_id="01KZ8Q4M", normalized_source_id="elsewhere.jpg") == derive(
        capture_id="01KZ8Q4M"
    )
    # Without one, they do.
    assert derive(normalized_source_id="elsewhere.jpg") != derive()
    assert derive(source_sha256="00" * 32) != derive()


def test_two_routes_never_share_an_inference_id():
    assert derive(route_id="clearance-cam-02") != derive()


def test_the_id_is_a_bounded_lowercase_token():
    identifier = derive()
    assert re.fullmatch(r"[a-z2-7]{26}", identifier), identifier


def test_the_enums_spell_what_the_wire_carries():
    assert SourceKind.SPOOL.value == "spool"
    assert CompletionAction.RETAIN.value == "retain"
    assert JobState.COMPLETED in TERMINAL_STATES
    assert JobState.PUBLISH_PENDING not in TERMINAL_STATES


def test_a_secret_reference_round_trips_through_its_configuration_form():
    assert SecretRef.parse({"$secret": "a"}).to_config() == {"$secret": "a"}
    assert SecretRef.parse({"$secret": "a", "field": "b"}).to_config() == {
        "$secret": "a",
        "field": "b",
    }
