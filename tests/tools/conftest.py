"""Fixtures for the WP7 tool suite: a real HTTP server and a manifest builder.

Nothing here is mocked out of the code under test. The fetch tool talks to a real
``http.server`` over a real socket, so the streaming download, the digest check, and the ``.part``
rename are exercised the way they run against the pinned corpus. The suite still needs no network:
the server is bound to the loopback interface and serves a temporary directory.

The helpers are fixtures rather than importable functions because this directory shares its name
with the top-level ``tools`` package, so it deliberately is not a package of its own.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """A directory server that does not write a line per request to stderr."""

    def log_message(self, *args) -> None:
        """Swallow the access log.

        Args:
            *args: The format string and its arguments, as the base class passes them.
        """
        return


@pytest.fixture
def served(tmp_path):
    """Serve a temporary directory over HTTP on the loopback interface.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        A ``(root, base_url)`` pair, where a file written under ``root`` is reachable at
        ``base_url`` plus its name.
    """
    root = tmp_path / "served"
    root.mkdir()

    def factory(*args, **kwargs):
        return _QuietHandler(*args, directory=str(root), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def publish(served):
    """Write a file into the served directory and describe it as a manifest part.

    Args:
        served: The running server.

    Returns:
        A callable taking a name and the bytes, returning the ``name``, ``sha256``, and ``bytes``
        an asset entry needs.
    """
    root, _ = served

    def write(name: str, payload: bytes) -> Dict[str, Any]:
        (root / name).write_bytes(payload)
        return {"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}

    return write


@pytest.fixture
def asset_entry(served):
    """Build a single-file asset entry pointing at the served directory.

    Args:
        served: The running server.

    Returns:
        A callable taking an id, a part, and any manifest overrides.
    """
    _, base = served

    def build(asset_id: str, part: Dict[str, Any], **extra) -> Dict[str, Any]:
        entry = {
            "id": asset_id,
            "kind": "model",
            "license": "Apache-2.0",
            "source": "the test server",
            "notes": "",
            "uri": base + part["name"],
            "sha256": part["sha256"],
            "bytes": part["bytes"],
            "name": part["name"],
        }
        entry.update(extra)
        return entry

    return build


@pytest.fixture
def write_manifest(tmp_path):
    """Write an asset manifest into the test's temporary directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A callable taking the asset entries and returning the manifest path.
    """

    def write(assets: Iterable[Dict[str, Any]], name: str = "assets.json") -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps({"schemaVersion": 1, "assets": list(assets)}, indent=2), encoding="utf-8"
        )
        return path

    return write
