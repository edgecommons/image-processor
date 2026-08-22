"""Fixtures for the bundle suite: real tarballs, real keys, and a real TLS server.

Nothing here is mocked out of the code under test. Bundles are built with
``tools/make_bundle.py``, signed with a generated Ed25519 key, and served over a self-signed TLS
endpoint, so the tests exercise the same paths the component uses at run time.
"""

from __future__ import annotations

import datetime
import http.server
import io
import ipaddress
import json
import ssl
import tarfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from image_processor.bundles import generate_keypair
from tools.make_bundle import make_bundle

#: The bundle-manifest contract (WP1). The suite validates against the shipped schema, so a
#: bundle these tests accept is one the component accepts.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "model-bundle-manifest.schema.json"

#: A model payload small enough to keep the suite fast and large enough to stream.
MODEL_BYTES = b"onnx-graph-" + bytes(range(256)) * 8

KEY_ID = "pharma-model-publisher-1"


def manifest_document(**overrides: Any) -> Dict[str, Any]:
    """Build a manifest document with every DESIGN.md section 8 field populated.

    Args:
        **overrides: Fields to replace or add.

    Returns:
        The manifest document, without ``files``, which the packer computes.
    """
    document: Dict[str, Any] = {
        "schemaVersion": 1,
        "modelId": "line-clearance-cam-01",
        "version": "2026.08.20",
        "minOnnxRuntime": "1.18.0",
        "providersPermitted": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "providerPolicy": "requireListed",
        "inputs": [{"name": "images", "dtype": "float32", "shape": ["N", 3, 224, 224]}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": ["N", 2]}],
        "dynamicBatch": True,
        "family": "classification",
        "familyParams": {"labels": ["clear", "hold"], "topK": 2, "activation": "softmax"},
        "preprocess": {
            "resize": {"mode": "letterbox", "width": 224, "height": 224},
            "scale": 0.00392156862745098,
            "mean": [0.485],
            "std": [0.229],
            "layout": "NCHW",
            "colorOrder": "RGB",
            "dtype": "float32",
        },
        "decisionRules": {
            "pass": {"path": "$.classes[0].label", "op": "==", "value": "clear"},
            "confidence": "$.classes[0].score",
            "threshold": 0.8,
        },
        "maxResultItems": 10,
        "estimatedDeviceMiB": 512,
        "warmup": [{"input": "warmup/input-01.bin", "expected": "warmup/expected-01.json"}],
        "tolerances": {"absolute": 0.001, "relative": 0.001},
        "compatibilityKeys": {"gpuClass": "sm_86", "tensorrt": "10.0"},
        "provenance": {"publisher": "pharma-mlops", "publishedAt": "2026-08-22T00:00:00Z"},
        "keyId": KEY_ID,
        "transformVersion": "2026.08.20-1",
    }
    document.update(overrides)
    return document


def write_source(root: Path, manifest: Optional[Dict[str, Any]] = None) -> Path:
    """Write a bundle source directory that make_bundle can pack.

    Args:
        root: Directory to create and fill.
        manifest: The manifest document to author, or ``None`` for the default one.

    Returns:
        The source directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.onnx").write_bytes(MODEL_BYTES)
    (root / "labels.json").write_text(json.dumps(["clear", "hold"]), encoding="utf-8")
    (root / "transforms.json").write_text(json.dumps({"resize": [224, 224]}), encoding="utf-8")
    (root / "result.schema.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    (root / "model-card.json").write_text(json.dumps({"license": "Apache-2.0"}), encoding="utf-8")
    warmup = root / "warmup"
    warmup.mkdir(exist_ok=True)
    (warmup / "input-01.bin").write_bytes(bytes(range(64)))
    (warmup / "expected-01.json").write_text(json.dumps({"label": "clear"}), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(manifest if manifest is not None else manifest_document(), indent=2),
        encoding="utf-8",
    )
    return root


class Built:
    """A packed bundle and everything a test needs to stage it.

    Attributes:
        archive: The tarball on disk.
        digest: Its ``sha256:<hex>`` digest.
        source: The directory it was packed from.
        key_id: The signing key id recorded in the manifest.
        trusted: A ``{key_id: public key}`` mapping for the signing key.
        manifest: The manifest document that was authored.
    """

    def __init__(
        self,
        archive: Path,
        digest: str,
        source: Path,
        key_id: Optional[str],
        trusted: Dict[str, bytes],
        manifest: Dict[str, Any],
    ) -> None:
        self.archive = archive
        self.digest = digest
        self.source = source
        self.key_id = key_id
        self.trusted = trusted
        self.manifest = manifest


@pytest.fixture
def schema_path() -> Path:
    """Return the bundle-manifest schema this suite validates against."""
    return SCHEMA_PATH


@pytest.fixture
def signing_key() -> Tuple[bytes, bytes, bytes]:
    """Return a generated Ed25519 keypair as (private PEM, public PEM, raw public key)."""
    return generate_keypair()


@pytest.fixture
def build_bundle(tmp_path: Path, signing_key: Tuple[bytes, bytes, bytes]) -> Callable[..., Built]:
    """Return a factory that packs a signed bundle and reports its digest.

    The factory takes optional manifest overrides, a ``sign`` flag, a ``compress`` flag, and a
    ``name`` so a test can build more than one bundle.
    """
    private_pem, _public_pem, public_raw = signing_key
    counter = {"n": 0}

    def _build(
        sign: bool = True,
        compress: bool = False,
        manifest: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        key: Optional[bytes] = None,
        key_id: Optional[str] = KEY_ID,
    ) -> Built:
        counter["n"] += 1
        label = name or f"bundle-{counter['n']}"
        source = write_source(tmp_path / "src" / label, manifest)
        suffix = ".tar.gz" if compress else ".tar"
        out = tmp_path / "dist" / f"{label}{suffix}"
        document = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        digest = make_bundle(
            src_dir=source,
            out_path=out,
            key=(key if key is not None else private_pem) if sign else None,
            key_id=key_id,
            compress=compress,
            schema_path=SCHEMA_PATH,
        )
        return Built(out, digest, source, key_id, {KEY_ID: public_raw}, document)

    return _build


def write_tar(path: Path, members: Iterable[Tuple[tarfile.TarInfo, Optional[bytes]]], compress: bool = False) -> Path:
    """Write a tar archive from explicit members, including unsafe ones.

    Args:
        path: The archive to write.
        members: ``(TarInfo, payload or None)`` pairs. A payload of ``None`` means the member
            carries no data, as for a directory, symlink, or device node.
        compress: Whether to gzip the archive.

    Returns:
        The archive path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if compress else "w"
    with tarfile.open(path, mode) as tar:
        for info, payload in members:
            if payload is None:
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return path


def file_member(name: str, payload: bytes) -> Tuple[tarfile.TarInfo, bytes]:
    """Build a regular-file member for ``write_tar``."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    return info, payload


def link_member(name: str, target: str, kind: bytes = tarfile.SYMTYPE) -> Tuple[tarfile.TarInfo, None]:
    """Build a symlink or hardlink member for ``write_tar``."""
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info, None


def _self_signed_cert(directory: Path) -> Tuple[Path, Path]:
    """Create a self-signed certificate for localhost and return (cert path, key path)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    cert_path = directory / "server.crt"
    key_path = directory / "server.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


class TlsServer:
    """A throwaway https origin serving one directory.

    Attributes:
        root: The directory served.
        ca_file: The self-signed certificate, which the client context trusts.
        redirects: Path to Location mapping, so a test can exercise the redirect policy.
    """

    def __init__(self, root: Path, certificate: Path, key: Path) -> None:
        self.root = root
        self.ca_file = certificate
        self.redirects: Dict[str, str] = {}
        server_root = str(root)
        redirects = self.redirects

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=server_root, **kwargs)

            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                target = redirects.get(self.path)
                if target:
                    self.send_response(302)
                    self.send_header("Location", target)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                super().do_GET()

            def log_message(self, *args: Any) -> None:
                pass

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(certificate), str(key))
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self, name: str) -> str:
        """Return the https URL of ``name`` under the served directory."""
        return f"https://localhost:{self.port}/{name}"

    def client_context(self) -> ssl.SSLContext:
        """Return a TLS context that trusts this server and nothing else."""
        return ssl.create_default_context(cafile=str(self.ca_file))

    def stop(self) -> None:
        """Shut the server down."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def tls_server(tmp_path: Path):
    """Serve ``tmp_path/www`` over https with a self-signed certificate."""
    root = tmp_path / "www"
    root.mkdir(parents=True, exist_ok=True)
    certificate, key = _self_signed_cert(tmp_path / "tls")
    server = TlsServer(root, certificate, key)
    try:
        yield server
    finally:
        server.stop()
