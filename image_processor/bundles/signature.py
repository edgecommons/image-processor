"""Ed25519 detached signatures over a bundle manifest (D-IP-10).

A bundle carries ``manifest.sig``: the raw 64-byte Ed25519 signature over the exact bytes of
``manifest.json``. The manifest names the signing key in ``keyId``, and configuration maps that
id to a trusted public key, so verification needs no PKI, network, or registry. Digest
verification is always on; signature verification is what the regulated profile requires
(DESIGN.md section 15).

Key material arrives from configuration through a ``$secret`` reference, so both PEM and raw
32-byte keys are accepted, as text or bytes.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Optional, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .archive import BundleError

logger = logging.getLogger(__name__)

#: Length of a raw Ed25519 key, in bytes.
RAW_KEY_BYTES = 32

#: Length of an Ed25519 signature, in bytes.
SIGNATURE_BYTES = 64

_PEM_PREFIX = b"-----BEGIN"

KeyMaterial = Union[bytes, bytearray, str]


def _as_bytes(material: KeyMaterial, what: str) -> bytes:
    """Return ``material`` as bytes, accepting text from a resolved secret."""
    if isinstance(material, str):
        return material.strip().encode("utf-8")
    if isinstance(material, (bytes, bytearray)):
        return bytes(material)
    raise BundleError(
        "SIGNING_KEY_INVALID", f"{what} must be bytes or text, got {type(material).__name__}"
    )


def _decode_raw(material: bytes, length: int) -> bytes:
    """Decode ``material`` to exactly ``length`` raw bytes, allowing base64 or hex text."""
    if len(material) == length:
        return material
    text = material.strip()
    try:
        decoded = base64.b64decode(text, validate=True)
        if len(decoded) == length:
            return decoded
    except (binascii.Error, ValueError):
        pass
    try:
        decoded = binascii.unhexlify(text)
        if len(decoded) == length:
            return decoded
    except (binascii.Error, ValueError):
        pass
    raise ValueError(f"not {length} raw bytes, base64, or hex")


def _looks_der(data: bytes) -> bool:
    """Return whether ``data`` is a DER encoding rather than base64 or hex text.

    DER starts with the SEQUENCE tag ``0x30`` -- which is also ASCII ``'0'``, so a base64 or hex
    key whose text happens to start with ``0`` must not be mistaken for DER. DER is binary and a
    text encoding is printable, so the tag alone is never enough: the material must also contain
    at least one non-printable byte. A raw 32-byte key is never DER either (an Ed25519
    SubjectPublicKeyInfo is 44 bytes and a PKCS#8 private key 48).
    """
    if data[:1] != b"\x30" or len(data) == RAW_KEY_BYTES:
        return False
    return any((b < 0x20 and b not in b"\t\r\n") or b > 0x7E for b in data)


def load_public_key(material: Union[Ed25519PublicKey, KeyMaterial]) -> Ed25519PublicKey:
    """Load a trusted Ed25519 public key.

    Args:
        material: An already-loaded key, a PEM or DER encoding, or the raw 32 bytes, given as
            bytes or as base64 or hex text.

    Returns:
        The loaded public key.

    Raises:
        BundleError: ``SIGNING_KEY_INVALID`` when the material is not an Ed25519 public key.
    """
    if isinstance(material, Ed25519PublicKey):
        return material
    data = _as_bytes(material, "public key")
    try:
        if data.lstrip().startswith(_PEM_PREFIX):
            key = serialization.load_pem_public_key(data.lstrip())
        elif _looks_der(data):
            key = serialization.load_der_public_key(data)
        else:
            return Ed25519PublicKey.from_public_bytes(_decode_raw(data, RAW_KEY_BYTES))
    except Exception as exc:
        raise BundleError("SIGNING_KEY_INVALID", f"unreadable public key: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise BundleError(
            "SIGNING_KEY_INVALID", f"public key is {type(key).__name__}, expected Ed25519"
        )
    return key


def load_private_key(
    material: Union[Ed25519PrivateKey, KeyMaterial], password: Optional[bytes] = None
) -> Ed25519PrivateKey:
    """Load an Ed25519 private key.

    Args:
        material: An already-loaded key, a PEM or DER encoding, or the raw 32 bytes, given as
            bytes or as base64 or hex text.
        password: Passphrase for an encrypted PEM, or ``None``.

    Returns:
        The loaded private key.

    Raises:
        BundleError: ``SIGNING_KEY_INVALID`` when the material is not an Ed25519 private key.
    """
    if isinstance(material, Ed25519PrivateKey):
        return material
    data = _as_bytes(material, "private key")
    try:
        if data.lstrip().startswith(_PEM_PREFIX):
            key = serialization.load_pem_private_key(data.lstrip(), password=password)
        elif _looks_der(data):
            key = serialization.load_der_private_key(data, password=password)
        else:
            return Ed25519PrivateKey.from_private_bytes(_decode_raw(data, RAW_KEY_BYTES))
    except Exception as exc:
        raise BundleError("SIGNING_KEY_INVALID", f"unreadable private key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise BundleError(
            "SIGNING_KEY_INVALID", f"private key is {type(key).__name__}, expected Ed25519"
        )
    return key


def verify_manifest_signature(
    manifest_bytes: bytes,
    signature: Union[bytes, str],
    public_key_pem_or_raw: Union[Ed25519PublicKey, KeyMaterial],
) -> None:
    """Verify a detached signature over the exact manifest bytes.

    Args:
        manifest_bytes: The bytes of ``manifest.json`` as the bundle carries them, byte for byte.
        signature: The contents of ``manifest.sig``: 64 raw bytes, or base64 or hex text.
        public_key_pem_or_raw: The trusted public key for the manifest ``keyId``.

    Raises:
        BundleError: ``SIGNING_KEY_INVALID`` when the key is unreadable, ``BAD_SIGNATURE`` when
            the signature is malformed or does not verify.
    """
    key = load_public_key(public_key_pem_or_raw)
    raw = _as_bytes(signature, "signature")
    try:
        raw = _decode_raw(raw, SIGNATURE_BYTES)
    except ValueError as exc:
        raise BundleError(
            "BAD_SIGNATURE", f"signature is not {SIGNATURE_BYTES} bytes: {exc}"
        ) from exc
    try:
        key.verify(raw, manifest_bytes)
    except InvalidSignature as exc:
        raise BundleError(
            "BAD_SIGNATURE", "manifest.sig does not verify against the trusted key"
        ) from exc


def sign_manifest(
    manifest_bytes: bytes, private_key: Union[Ed25519PrivateKey, KeyMaterial]
) -> bytes:
    """Sign manifest bytes with an Ed25519 private key.

    Args:
        manifest_bytes: The exact bytes that go into the bundle as ``manifest.json``.
        private_key: A PEM, DER, or raw 32-byte private key, or an already-loaded key.

    Returns:
        The raw 64-byte signature, which the bundle carries as ``manifest.sig``.

    Raises:
        BundleError: ``SIGNING_KEY_INVALID`` when the key is unreadable.
    """
    key = load_private_key(private_key)
    return key.sign(manifest_bytes)


def public_key_pem(key: Union[Ed25519PrivateKey, Ed25519PublicKey]) -> bytes:
    """Return the PEM encoding of a public key.

    Args:
        key: A public key, or the private key whose public half you want.

    Returns:
        The PEM-encoded SubjectPublicKeyInfo.
    """
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def public_key_raw(key: Union[Ed25519PrivateKey, Ed25519PublicKey]) -> bytes:
    """Return the raw 32 bytes of a public key.

    Args:
        key: A public key, or the private key whose public half you want.

    Returns:
        The raw public key bytes.
    """
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def private_key_pem(key: Ed25519PrivateKey, password: Optional[bytes] = None) -> bytes:
    """Return the PKCS8 PEM encoding of a private key.

    Args:
        key: The private key to serialize.
        password: Passphrase to encrypt the PEM with, or ``None`` for an unencrypted PEM.

    Returns:
        The PEM-encoded private key.
    """
    encryption = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )


def generate_keypair(password: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
    """Generate an Ed25519 signing keypair.

    Args:
        password: Passphrase to encrypt the private PEM with, or ``None``.

    Returns:
        A tuple of the private key PEM, the public key PEM, and the raw 32-byte public key. The
        raw form is what a ``trustedKeys[].publicKey`` secret usually holds.
    """
    key = Ed25519PrivateKey.generate()
    return private_key_pem(key, password), public_key_pem(key), public_key_raw(key)
