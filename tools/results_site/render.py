"""The files the site is made of.

Four of them, and no more: ``index.html``, ``site.css``, ``site.js``, and ``data.js``. The
stylesheet and the script are hand-written and shipped verbatim out of ``assets/``; nothing is
compiled, minified, or fetched. ``data.js`` assigns ``window.RESULTS``, which is what keeps the
site working from ``file://``: a browser refuses to read a JSON file over that scheme, and
refuses nothing about a script tag.

``index.html`` is the one generated page. Its shell comes from ``assets/index.html`` and the
generator substitutes a ``<noscript>`` index of every entry into it, so the page still names what
it holds when scripting is off and so the source carries every entry id.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

#: Where the hand-written files live.
ASSETS = Path(__file__).resolve().parent / "assets"

#: The token in ``assets/index.html`` the entry index replaces.
NOSCRIPT_TOKEN = "<!--ENTRY-INDEX-->"

#: The token in ``assets/index.html`` the build stamp replaces.
GENERATED_TOKEN = "<!--GENERATED-->"

#: The files copied straight through.
STATIC_FILES = ("site.css", "site.js")


def data_js(data: Dict[str, Any]) -> str:
    """Render the document as the one assignment ``data.js`` holds.

    Args:
        data: The site document.

    Returns:
        The file content. The JSON is compact: a few hundred whole result bodies are the bulk of
        the site, and the page pretty-prints the one body it is showing anyway.
    """
    from tools.results_site.model import ASSIGNMENT

    payload = json.dumps(data, separators=(",", ":"), allow_nan=False, sort_keys=False)
    return f"{ASSIGNMENT}{payload};\n"


def _noscript_rows(data: Dict[str, Any]) -> str:
    """Render the entry index the page shows when scripting is off.

    Args:
        data: The site document.

    Returns:
        The table rows, one per entry.
    """
    rows: List[str] = []
    for entry in data.get("entries", []):
        image = entry["image"]
        decision = entry.get("decision") or {}
        links = [f'<a href="{html.escape(image["src"])}">image</a>']
        if image.get("overlay"):
            links.append(f'<a href="{html.escape(image["overlay"])}">overlay</a>')
        rows.append(
            "<tr id={id}><td>{model}</td><td>{run}</td><td>{name}</td>"
            "<td>{outcome}</td><td>{ms:.2f} ms</td><td>{links}</td></tr>".format(
                id=html.escape(entry["id"], quote=True),
                model=html.escape(entry["modelKey"]),
                run=html.escape(entry["runId"]),
                name=html.escape(image["name"]),
                outcome=html.escape(str(decision.get("outcome", "?"))),
                ms=float(entry["timings"]["sessionMs"]),
                links=" ".join(links),
            )
        )
    return "\n".join(rows)


def index_html(data: Dict[str, Any]) -> str:
    """Render the page.

    Args:
        data: The site document.

    Returns:
        The whole document, shell and entry index together.
    """
    shell = (ASSETS / "index.html").read_text(encoding="utf-8")
    generated = html.escape(str(data.get("generated", "")))
    counts = (
        f"{len(data.get('entries', []))} entries, "
        f"{len(data.get('models', []))} model runs, "
        f"{len(data.get('runs', []))} runs, generated {generated}"
    )
    return shell.replace(NOSCRIPT_TOKEN, _noscript_rows(data)).replace(GENERATED_TOKEN, counts)


def write_site(out_dir: Path, data: Dict[str, Any]) -> List[Path]:
    """Write the four files of the site.

    Args:
        out_dir: The site root. Created if missing. Images, thumbnails, and overlays are already
            in it by the time this runs.
        data: The site document.

    Returns:
        The paths written, in the order they were written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in STATIC_FILES:
        target = out_dir / name
        shutil.copyfile(ASSETS / name, target)
        written.append(target)
    index = out_dir / "index.html"
    index.write_text(index_html(data), encoding="utf-8")
    written.append(index)
    payload = out_dir / "data.js"
    payload.write_text(data_js(data), encoding="utf-8")
    written.append(payload)
    return written


def site_bytes(out_dir: Path) -> int:
    """Measure everything under the site root.

    Args:
        out_dir: The site root.

    Returns:
        The total size in bytes.
    """
    return sum(path.stat().st_size for path in Path(out_dir).rglob("*") if path.is_file())