"""The run itself: from a staged bundle to a written site.

One session per model, one pass over its images, and one entry per image. The session is opened
through ``tests/live_models/runner.open_session``, which preloads the NVIDIA libraries the CUDA
provider links against and then refuses a session the runtime did not actually assign the
requested provider -- ONNX Runtime drops a provider it cannot build and carries on, and a site
that accepted that would label CPU numbers as CUDA.

Each image goes through ``runner.infer``, which is the same decode, preprocess, run, postprocess,
and decision-rule sequence the tier-2 suite measures. The answer is then put into the shape the
component publishes: ``outputs/result.build_result_body`` builds the body from a
:class:`~image_processor.types.Job` and an :class:`~image_processor.types.InferenceResult` this
module constructs, and validates it against ``schemas/inference-result.schema.json`` on the way
out. What the site shows is therefore the message, not a rendering of one.

The builder times two things: the graph, which ``runner.infer`` reports, and the wall clock around
the whole per-image sequence. The body's per-stage ``timingsMs`` fields it does not measure --
queue, model load, preprocess, postprocess -- are reported as zero rather than estimated.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import shutil
import socket
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from image_processor.engine.decode import DecodeError, DecodeLimits, decode_image
from image_processor.engine.protocol import CPU_PROVIDER, CUDA_PROVIDER
from image_processor.engine.residency import probe_for
from image_processor.outputs.result import build_result_body
from image_processor.types import (
    InferenceResult,
    Job,
    JobState,
    ModelRef,
    SourceIdentity,
    SourceKind,
    Timings,
    derive_inference_id,
)
from tests.live_models import runner
from tools.results_site import corpus as corpus_support
from tools.results_site import model as model_support
from tools.results_site import overlays, render

#: Where a copied source image, its thumbnail, and its overlay live under the site root.
IMAGE_DIR = "images"
THUMB_DIR = "thumbs"
OVERLAY_DIR = "overlays"


def utc_now() -> str:
    """Return the moment this build ran.

    Returns:
        An ISO 8601 timestamp in UTC, to the second.
    """
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def gpu_name(provider: str, device_id: int) -> Optional[str]:
    """Name the device a CUDA run is on.

    The probe is the component own NVML probe, so the site reports the device the way a result
    body reports it. Without NVML, or on a CPU run, there is no name and the site says so rather
    than guessing one.

    Args:
        provider: The execution provider.
        device_id: The CUDA ordinal.

    Returns:
        The device name, or ``None``.
    """
    if provider != CUDA_PROVIDER:
        return None
    return probe_for(device_id).snapshot(device_id).device_class or None


def as_image(array: np.ndarray) -> Image.Image:
    """Turn a decoded array into a Pillow image.

    The array is what the task family measured its regions against, so drawing on it rather than
    on a second decode of the file keeps the boxes and the pixels in the same coordinate system.

    Args:
        array: The ``HWC`` RGB array :func:`decode_image` produced.

    Returns:
        The image, reduced to eight bits per sample when the source carried more.
    """
    if array.dtype != np.uint8:
        array = (np.asarray(array) >> 8).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _job_for(
    route_id: str, name: str, data: bytes, manifest: Any, digest: str
) -> Job:
    """Build the durable job one entry stands for.

    Args:
        route_id: The route the site names this model run under.
        name: The image name, relative to its corpus.
        data: The encoded image bytes.
        manifest: The staged bundle manifest.
        digest: The staged bundle tarball digest.

    Returns:
        The job.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    source = SourceIdentity(
        kind=SourceKind.SPOOL,
        route_id=route_id,
        relative_path=name,
        bytes=len(data),
        sha256=sha256,
    )
    return Job(
        inference_id=derive_inference_id(route_id, None, sha256, name, digest),
        route_id=route_id,
        source=source,
        model=ModelRef(id=manifest.model_id, version=manifest.version, digest=digest),
        transform_version=manifest.transform_version,
        state=JobState.INFERENCING,
    )


def result_body(
    route_id: str,
    name: str,
    data: bytes,
    bundle: Any,
    normalized: Any,
    decision: Any,
    providers: Sequence[str],
    device: Optional[str],
    device_class: Optional[str],
    session_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    """Build the wire-shaped result body for one image.

    Args:
        route_id: The route the site names this model run under.
        name: The image name, relative to its corpus.
        data: The encoded image bytes.
        bundle: The staged :class:`~image_processor.types.CachedBundle`.
        normalized: The family normalized output.
        decision: The verdict the decision rules reached.
        providers: The session actual provider assignment.
        device: The CUDA ordinal as a string, or ``None``.
        device_class: The device name, or ``None``.
        session_ms: The graph time.
        total_ms: The wall time for the whole per-image sequence.

    Returns:
        The body, already checked against ``schemas/inference-result.schema.json``.
    """
    job = _job_for(route_id, name, data, bundle.manifest, bundle.digest)
    answer = InferenceResult(
        inference_id=job.inference_id,
        status="SUCCEEDED",
        normalized=normalized,
        decision=decision,
        providers=list(providers),
        gpu_device=device,
        gpu_class=device_class,
        timings=Timings(
            queue_ms=0.0,
            model_load_ms=0.0,
            preprocess_ms=0.0,
            inference_ms=session_ms,
            postprocess_ms=0.0,
            total_ms=total_ms,
        ),
        memory_high_water_mib=None,
    )
    return build_result_body(job, answer, bundle.manifest)


def _install_image(
    out_dir: Path,
    run: str,
    suite: str,
    model_key: str,
    name: str,
    source: Path,
    picture: Image.Image,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy one source image into the site and draw its thumbnail and overlay.

    The picture and its thumbnail are the same bytes whatever provider ran the model, so both runs
    of a model share one copy. The overlay is not: it is drawn from the regions that run reported,
    and a CPU and a CUDA session can disagree about a box. It therefore lives under the run id, so
    a merged site never shows one run drawing beside another run numbers.

    Args:
        out_dir: The site root.
        run: The run id this drawing belongs to.
        suite: The suite the model belongs to.
        model_key: The model key.
        name: The image name, relative to its corpus.
        source: The source file to copy.
        picture: The decoded image, as the family measured it.
        body: The wire-shaped result body.

    Returns:
        The entry ``image`` block: the three site-relative paths, the source size, and the name.
    """
    flat = model_support.file_slug(name)
    relative = f"{suite}/{model_support.slug(model_key)}/{flat}"

    target = out_dir / IMAGE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

    thumb = overlays.save_thumbnail(
        overlays.thumbnail(picture), out_dir / THUMB_DIR / relative
    )

    overlay_path = None
    drawn = overlays.draw_overlay(picture, body)
    if drawn is not None:
        overlay_path = (out_dir / OVERLAY_DIR / run / relative).with_suffix(".png")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        drawn.save(overlay_path, format="PNG", optimize=True)

    return {
        "src": f"{IMAGE_DIR}/{relative}",
        "thumb": f"{THUMB_DIR}/{thumb.relative_to(out_dir / THUMB_DIR).as_posix()}",
        "overlay": (
            f"{OVERLAY_DIR}/{overlay_path.relative_to(out_dir / OVERLAY_DIR).as_posix()}"
            if overlay_path
            else None
        ),
        "w": picture.width,
        "h": picture.height,
        "name": name,
    }


def run_model(
    entry_model: corpus_support.CorpusModel,
    out_dir: Path,
    scratch: Path,
    run: str,
    provider: str,
    device_id: int,
    device_class: Optional[str],
    limit: Optional[int] = None,
    report: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Stage one model, run it over its images, and install everything the site shows.

    Args:
        entry_model: The model and its images.
        out_dir: The site root.
        scratch: Scratch space for the tarball, the staging directory, and the cache.
        run: The run id every row is filed under.
        provider: The execution provider to request.
        device_id: The CUDA ordinal, when the provider is CUDA.
        device_class: The device name, or ``None``.
        limit: How many images to run, or ``None`` for all of them.
        report: Called with a progress line per model, or ``None`` for a silent build.

    Returns:
        A mapping with the model row under ``"model"`` and the entries under ``"entries"``.
    """
    bundle = entry_model.stage(scratch)
    session = runner.open_session(bundle, provider, device_id)
    providers = list(session.get_providers())
    device = str(device_id) if provider == CUDA_PROVIDER else None
    route_id = model_support.slug(f"{entry_model.suite}-{entry_model.key}")

    card = model_support.model_card(bundle.manifest, providers, device_class)
    card["preprocessSummary"] = model_support.preprocess_summary(card)

    pairs = list(zip(entry_model.images, entry_model.names))
    if limit is not None:
        pairs = pairs[: max(0, limit)]

    entries: List[Dict[str, Any]] = []
    for path, name in pairs:
        started = time.perf_counter()
        data = runner.read_image(path)
        normalized, decision, session_ms = runner.infer(session, bundle.manifest, data)
        total_ms = (time.perf_counter() - started) * 1000.0
        body = result_body(
            route_id,
            name,
            data,
            bundle,
            normalized,
            decision,
            providers,
            device,
            device_class,
            session_ms,
            total_ms,
        )
        picture = as_image(decode_image(data, DecodeLimits()))
        image = _install_image(
            out_dir, run, entry_model.suite, entry_model.key, name, path, picture, body
        )
        entries.append(
            {
                "id": model_support.entry_id(run, entry_model.key, name),
                "runId": run,
                "modelKey": entry_model.key,
                "image": image,
                "timings": {
                    "sessionMs": round(session_ms, 4),
                    "totalMs": round(total_ms, 4),
                },
                "decision": body["decision"],
                "resultBody": body,
                "summary": model_support.summarize(body),
            }
        )

    if report:
        report(
            f"{entry_model.suite}/{entry_model.key}: {len(entries)} images on "
            f"{'+'.join(name.replace('ExecutionProvider', '') for name in providers)}"
        )
    return {
        "model": {
            "key": entry_model.key,
            "runId": run,
            "suite": entry_model.suite,
            "family": entry_model.family,
            "corpus": entry_model.corpus,
            "card": card,
        },
        "entries": entries,
    }


def refusal_rows(corpus_root: Path, run: str) -> List[Dict[str, Any]]:
    """Put the tier-1 bad-input set through the component decoder.

    The fixtures are the hostile half of the tier-1 corpus: truncated, empty, mislabelled, and
    header-forged files, each with the decode code it is required to raise. Running them costs
    nothing and states on the site what the pipeline refuses, next to what it accepts.

    Args:
        corpus_root: The directory ``tests/fixtures/build.py`` wrote.
        run: The run id the rows are filed under.

    Returns:
        One row per fixture.
    """
    rows: List[Dict[str, Any]] = []
    limits = DecodeLimits()
    for record in corpus_support.bad_inputs(corpus_root):
        data = (Path(corpus_root) / record["path"]).read_bytes()
        observed = None
        with warnings.catch_warnings():
            # The forged-header fixtures exist to be refused, and Pillow warns about the header
            # before the decoder gets to refuse it. The refusal is the result; the warning is not.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            try:
                decode_image(data, limits)
            except DecodeError as exc:
                observed = exc.code
        rows.append(
            {
                "runId": run,
                "name": record["path"],
                "bytes": int(record["bytes"]),
                "expected": record.get("code"),
                "observed": observed,
                "refused": observed is not None,
            }
        )
    return rows


def collect_models(
    suites: Sequence[str],
    provider: str,
    corpus_root: Optional[Path],
    keys: Optional[Sequence[str]] = None,
) -> List[corpus_support.CorpusModel]:
    """Assemble every model the requested suites hold.

    Args:
        suites: The suites to include.
        provider: The execution provider the run uses.
        corpus_root: Where the tier-1 corpus was built, required by the synthetic suite.
        keys: Only these model keys, or ``None`` for all of them.

    Returns:
        The models, synthetic first.

    Raises:
        CorpusError: A requested suite cannot be assembled, or the key filter matched nothing.
    """
    models: List[corpus_support.CorpusModel] = []
    if "synthetic" in suites:
        models.extend(corpus_support.synthetic_models(corpus_root, provider))
    if "live" in suites:
        models.extend(corpus_support.live_models(provider))
    if keys:
        wanted = set(keys)
        models = [model for model in models if model.key in wanted]
        missing = wanted - {model.key for model in models}
        if missing:
            raise corpus_support.CorpusError(
                f"no model named {', '.join(sorted(missing))} in suites {', '.join(suites)}"
            )
    if not models:
        raise corpus_support.CorpusError("the selection holds no models")
    return models


def build_site(
    out_dir: Path,
    scratch: Path,
    suites: Sequence[str] = corpus_support.SUITES,
    provider: str = CPU_PROVIDER,
    device_id: int = 0,
    corpus_root: Optional[Path] = None,
    keys: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    merge: bool = False,
    report: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run every model of every requested suite and write the site.

    Args:
        out_dir: The site root. Created if missing.
        scratch: Scratch space for staged bundles.
        suites: Which suites to run.
        provider: The execution provider to request.
        device_id: The CUDA ordinal, when the provider is CUDA.
        corpus_root: Where the tier-1 corpus was built. Required by the synthetic suite.
        keys: Only these model keys, or ``None`` for all of them.
        limit: How many images per model, or ``None`` for all of them.
        merge: Fold this run into the site already in ``out_dir`` rather than replacing it.
        report: Called with one progress line per model, or ``None`` for a silent build.

    Returns:
        The site document that was written.

    Raises:
        MergeError: ``merge`` was asked for and the existing ``data.js`` cannot be read.
    """
    import onnxruntime as ort

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch)

    device_class = gpu_name(provider, device_id)
    run = model_support.run_id(provider, device_class)
    generated = utc_now()

    models = collect_models(suites, provider, corpus_root, keys)
    rows: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    for index, entry_model in enumerate(models):
        produced = run_model(
            entry_model,
            out_dir,
            scratch / f"{index:02d}-{model_support.slug(entry_model.key)}",
            run,
            provider,
            device_id,
            device_class,
            limit=limit,
            report=report,
        )
        rows.append(produced["model"])
        entries.extend(produced["entries"])

    refusals: List[Dict[str, Any]] = []
    if "synthetic" in suites and corpus_root is not None:
        refusals = refusal_rows(Path(corpus_root), run)

    fresh = model_support.document(
        generated=generated,
        runs=[
            {
                "runId": run,
                "provider": provider,
                "gpu": device_class,
                "host": socket.gethostname(),
                "onnxruntime": ort.__version__,
                "generated": generated,
            }
        ],
        models=rows,
        entries=entries,
        refusals=refusals,
    )

    data = fresh
    existing_path = out_dir / "data.js"
    if merge and existing_path.is_file():
        existing = model_support.parse_data_js(existing_path.read_text(encoding="utf-8"))
        data = model_support.merge(existing, fresh)

    render.write_site(out_dir, data)
    return data