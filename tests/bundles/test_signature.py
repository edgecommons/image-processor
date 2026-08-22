"""Ed25519 manifest signing and verification (D-IP-10)."""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from image_processor.bundles import (
    BundleError,
    generate_keypair,
    load_private_key,
    load_public_key,
    private_key_pem,
    public_key_pem,
    public_key_raw,
    sign_manifest,
    verify_manifest_signature,
)

MANIFEST = b'{"modelId": "line-clearance-cam-01", "version": "2026.08.20"}\n'


def test_sign_and_verify_round_trip(signing_key) -> None:
    private_pem, public_pem, public_raw = signing_key
    signature = sign_manifest(MANIFEST, private_pem)
    assert len(signature) == 64
    verify_manifest_signature(MANIFEST, signature, public_raw)
    verify_manifest_signature(MANIFEST, signature, public_pem)


def test_verification_fails_when_the_manifest_changed(signing_key) -> None:
    private_pem, _public_pem, public_raw = signing_key
    signature = sign_manifest(MANIFEST, private_pem)
    with pytest.raises(BundleError) as caught:
        verify_manifest_signature(MANIFEST.replace(b"2026.08.20", b"2026.08.21"), signature, public_raw)
    assert caught.value.code == "BAD_SIGNATURE"


def test_verification_fails_against_another_key(signing_key) -> None:
    private_pem, _public_pem, _public_raw = signing_key
    _other_private, _other_pem, other_raw = generate_keypair()
    signature = sign_manifest(MANIFEST, private_pem)
    with pytest.raises(BundleError) as caught:
        verify_manifest_signature(MANIFEST, signature, other_raw)
    assert caught.value.code == "BAD_SIGNATURE"


@pytest.mark.parametrize("encode", [lambda raw: raw, base64.b64encode, lambda raw: raw.hex().encode()])
def test_public_keys_are_accepted_raw_base64_and_hex(signing_key, encode) -> None:
    private_pem, _public_pem, public_raw = signing_key
    signature = sign_manifest(MANIFEST, private_pem)
    verify_manifest_signature(MANIFEST, signature, encode(public_raw))


def test_key_material_is_accepted_as_text(signing_key) -> None:
    private_pem, public_pem, _public_raw = signing_key
    signature = sign_manifest(MANIFEST, private_pem.decode("ascii"))
    verify_manifest_signature(MANIFEST, signature, public_pem.decode("ascii"))


def test_signatures_are_accepted_base64_and_hex(signing_key) -> None:
    private_pem, _public_pem, public_raw = signing_key
    signature = sign_manifest(MANIFEST, private_pem)
    verify_manifest_signature(MANIFEST, base64.b64encode(signature), public_raw)
    verify_manifest_signature(MANIFEST, signature.hex(), public_raw)


def test_a_short_signature_is_a_bad_signature(signing_key) -> None:
    _private_pem, _public_pem, public_raw = signing_key
    with pytest.raises(BundleError) as caught:
        verify_manifest_signature(MANIFEST, b"too short", public_raw)
    assert caught.value.code == "BAD_SIGNATURE"


def test_der_and_raw_private_keys_are_accepted() -> None:
    key = Ed25519PrivateKey.generate()
    raw_private = key.private_bytes_raw()
    signature = sign_manifest(MANIFEST, raw_private)
    verify_manifest_signature(MANIFEST, signature, public_key_raw(key))
    assert load_private_key(key) is key
    loaded = load_public_key(public_key_pem(key))
    assert load_public_key(loaded) is loaded


def test_encrypted_private_key_pems_round_trip() -> None:
    key = Ed25519PrivateKey.generate()
    encrypted = private_key_pem(key, password=b"line-clearance")
    assert b"ENCRYPTED" in encrypted
    signature = sign_manifest(MANIFEST, load_private_key(encrypted, password=b"line-clearance"))
    verify_manifest_signature(MANIFEST, signature, public_key_raw(key))


def test_a_wrong_passphrase_is_an_invalid_key() -> None:
    encrypted = private_key_pem(Ed25519PrivateKey.generate(), password=b"right")
    with pytest.raises(BundleError) as caught:
        load_private_key(encrypted, password=b"wrong")
    assert caught.value.code == "SIGNING_KEY_INVALID"


@pytest.mark.parametrize("material", [b"not a key at all", "", 42])
def test_unreadable_key_material_is_rejected(material) -> None:
    with pytest.raises(BundleError) as caught:
        load_public_key(material)
    assert caught.value.code == "SIGNING_KEY_INVALID"


def test_a_non_ed25519_key_is_rejected() -> None:
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(BundleError) as caught:
        load_public_key(public_key_pem_of_rsa(rsa_key))
    assert caught.value.code == "SIGNING_KEY_INVALID"
    with pytest.raises(BundleError) as caught:
        load_private_key(private_key_pem_of_rsa(rsa_key))
    assert caught.value.code == "SIGNING_KEY_INVALID"


def public_key_pem_of_rsa(key) -> bytes:
    """Serialize an RSA public key, which is not a key this component accepts."""
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def private_key_pem_of_rsa(key) -> bytes:
    """Serialize an RSA private key, which is not a key this component accepts."""
    from cryptography.hazmat.primitives import serialization

    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def test_generate_keypair_produces_matching_halves() -> None:
    private_pem, public_pem, public_raw = generate_keypair()
    assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert public_pem.startswith(b"-----BEGIN PUBLIC KEY-----")
    assert len(public_raw) == 32
    signature = sign_manifest(MANIFEST, private_pem)
    verify_manifest_signature(MANIFEST, signature, public_pem)


def test_der_encoded_keys_are_accepted() -> None:
    from cryptography.hazmat.primitives import serialization

    key = Ed25519PrivateKey.generate()
    der_private = key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    der_public = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signature = sign_manifest(MANIFEST, der_private)
    verify_manifest_signature(MANIFEST, signature, der_public)


def test_a_der_key_of_the_wrong_algorithm_is_rejected() -> None:
    from cryptography.hazmat.primitives import serialization

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    der_public = rsa_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(BundleError) as caught:
        load_public_key(der_public)
    assert caught.value.code == "SIGNING_KEY_INVALID"


def test_text_keys_starting_with_zero_are_not_mistaken_for_der():
    """A base64 or hex key whose first character is ``0`` (ASCII 0x30, the DER SEQUENCE tag)
    is text, not DER. Found by WP1: ~1.7% of fresh keys failed to load."""
    import base64 as _b64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization as _ser

    from image_processor.bundles import signature as sig

    found = None
    for _ in range(5000):
        key = Ed25519PrivateKey.generate()
        raw_pub = key.public_key().public_bytes(
            _ser.Encoding.Raw, _ser.PublicFormat.Raw
        )
        if _b64.b64encode(raw_pub)[:1] == b"0":
            found = (key, raw_pub)
            break
    assert found is not None, "no key with a base64 encoding starting with '0' in 5000 tries"
    key, raw_pub = found
    loaded = sig.load_public_key(_b64.b64encode(raw_pub))
    assert loaded.public_bytes(_ser.Encoding.Raw, _ser.PublicFormat.Raw) == raw_pub
    hex_text = raw_pub.hex().encode("ascii")
    if hex_text[:1] == b"0":
        assert (
            sig.load_public_key(hex_text).public_bytes(
                _ser.Encoding.Raw, _ser.PublicFormat.Raw
            )
            == raw_pub
        )
    # A raw 32-byte key starting with the SEQUENCE tag byte is still raw, not DER.
    raw_starting_with_0x30 = bytes([0x30]) + raw_pub[1:]
    assert (
        sig.load_public_key(raw_starting_with_0x30).public_bytes(
            _ser.Encoding.Raw, _ser.PublicFormat.Raw
        )
        == raw_starting_with_0x30
    )
    # Real DER still loads.
    der = key.public_key().public_bytes(
        _ser.Encoding.DER, _ser.PublicFormat.SubjectPublicKeyInfo
    )
    assert sig.load_public_key(der).public_bytes(_ser.Encoding.Raw, _ser.PublicFormat.Raw) == raw_pub
