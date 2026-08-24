"""Build the inference results explorer: a local static site of what the models actually did.

The tool runs every model of the chosen suites over its corpus, keeps the whole
``app/inference/result`` body for every image, draws the regions each result reports onto a copy
of the picture, and writes a site that opens straight off disk. Nothing it writes is committed:
``tests/.site/`` is gitignored.

Two suites are available. ``synthetic`` is the tier-1 corpus, whose seven bundles have fixed
weights and computable answers; the tool builds that corpus into a temporary directory first, so
it needs no network and no cache. ``live`` is the tier-2 corpus of seven real exports, and needs
``tests/.cache`` filled by ``tools/fetch_test_assets.py``.

Examples:
    Build both suites on CPU::

        python tools/build_results_site.py --out tests/.site --suites synthetic,live

    Add a CUDA leg to the same site, keeping the CPU one::

        python tools/build_results_site.py --out tests/.site --suites live \
            --provider CUDAExecutionProvider --merge

    Build a small site and serve it::

        python tools/build_results_site.py --out tests/.site --suites synthetic --limit 2 --serve
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_processor.engine.protocol import CPU_PROVIDER, CUDA_PROVIDER  # noqa: E402
from tools.results_site import build as build_support  # noqa: E402
from tools.results_site import corpus as corpus_support  # noqa: E402
from tools.results_site import render  # noqa: E402
from tools.results_site.model import MergeError  # noqa: E402

#: The default site root, which ``.gitignore`` excludes.
DEFAULT_OUT = REPO_ROOT / "tests" / ".site"

#: The port ``--serve`` uses when it is given none.
DEFAULT_PORT = 8000


def parse_suites(value: str) -> List[str]:
    """Read the ``--suites`` list.

    Args:
        value: The comma-separated list.

    Returns:
        The suite names, in the order this tool runs them.

    Raises:
        argparse.ArgumentTypeError: A name is not a suite.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in corpus_support.SUITES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown suite {', '.join(unknown)}; choose from {', '.join(corpus_support.SUITES)}"
        )
    if not names:
        raise argparse.ArgumentTypeError("name at least one suite")
    return [name for name in corpus_support.SUITES if name in names]


def parse_keys(value: str) -> List[str]:
    """Read the ``--models`` list.

    Args:
        value: The comma-separated list.

    Returns:
        The model keys.
    """
    return [name.strip() for name in value.split(",") if name.strip()]


def parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parsed = argparse.ArgumentParser(
        description="Build the ImageProcessor inference results explorer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parsed.add_argument("--out", default=str(DEFAULT_OUT), help="site root (default tests/.site)")
    parsed.add_argument(
        "--provider",
        default=CPU_PROVIDER,
        choices=[CPU_PROVIDER, CUDA_PROVIDER],
        help="execution provider to request; the session is refused if it is not assigned",
    )
    parsed.add_argument("--device-id", type=int, default=0, help="CUDA ordinal")
    parsed.add_argument(
        "--suites",
        type=parse_suites,
        default=list(corpus_support.SUITES),
        help="comma-separated: synthetic, live",
    )
    parsed.add_argument("--models", type=parse_keys, default=None, help="only these model keys")
    parsed.add_argument("--limit", type=int, default=None, help="images per model")
    parsed.add_argument(
        "--merge",
        action="store_true",
        help="fold this run into the site already in --out instead of replacing it",
    )
    parsed.add_argument(
        "--corpus",
        default=None,
        help="an already-built tier-1 corpus; the default builds one into a temporary directory",
    )
    parsed.add_argument(
        "--serve",
        nargs="?",
        type=int,
        const=DEFAULT_PORT,
        default=None,
        metavar="PORT",
        help=f"serve the site after building it (default port {DEFAULT_PORT})",
    )
    parsed.add_argument("--quiet", action="store_true", help="print only the final summary")
    return parsed


def serve(out_dir: Path, port: int, forever: bool = True) -> int:
    """Serve the built site over HTTP.

    The site works from ``file://`` as well; a server is here for the case where a browser is
    configured to treat local files strictly, and for looking at a build from another machine.

    Args:
        out_dir: The site root.
        port: The port to bind on the loopback interface.
        forever: Whether to block. A test binds and returns instead.

    Returns:
        The port that was bound, which is the requested one unless it was ``0``.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        bound = server.server_address[1]
        print(f"serving {out_dir} at http://127.0.0.1:{bound}/  (ctrl-c to stop)")
        if not forever:
            return bound
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
        return bound


def _build_corpus(target: Path) -> Path:
    """Generate the tier-1 corpus the synthetic suite runs over.

    Args:
        target: Where to build it.

    Returns:
        The corpus root.
    """
    from tests.fixtures import build as fixtures

    fixtures.build(target)
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the tool from the command line.

    Args:
        argv: The argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    arguments = parser().parse_args(argv)
    out_dir = Path(arguments.out).resolve()
    report = None if arguments.quiet else (lambda line: print(line, flush=True))

    with tempfile.TemporaryDirectory(prefix="results-site-") as scratch:
        scratch_root = Path(scratch)
        corpus_root = None
        if "synthetic" in arguments.suites:
            corpus_root = (
                Path(arguments.corpus).resolve()
                if arguments.corpus
                else _build_corpus(scratch_root / "corpus")
            )
        try:
            data = build_support.build_site(
                out_dir=out_dir,
                scratch=scratch_root / "bundles",
                suites=arguments.suites,
                provider=arguments.provider,
                device_id=arguments.device_id,
                corpus_root=corpus_root,
                keys=arguments.models,
                limit=arguments.limit,
                merge=arguments.merge,
                report=report,
            )
        except (corpus_support.CorpusError, MergeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    size = render.site_bytes(out_dir)
    print(
        f"{len(data['models'])} model runs, {len(data['entries'])} entries, "
        f"{len(data['runs'])} runs, {size / 2 ** 20:.1f} MiB on disk"
    )
    print(f"open {(out_dir / 'index.html').as_uri()}")
    if arguments.serve is not None:
        serve(out_dir, arguments.serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())