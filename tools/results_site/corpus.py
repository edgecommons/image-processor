"""Which models the site runs, and over which images.

Two suites feed the explorer, and they answer different questions.

``synthetic`` runs the seven bundles ``tests/fixtures/build.py`` generates. Their weights are
fixed and their answers are computable, so the site shows a picture whose correct output is known
by construction next to the output the pipeline produced.

``live`` runs the seven real exports of the tier-2 corpus over exactly the images the committed
goldens are made of. The image selection is not re-stated here: the three selection functions are
imported out of ``tests/live_models/conftest.py`` and called directly, so the site cannot drift
from the goldens by picking a different twenty pictures.

Both suites stage every bundle the way the component installs one: packed by
``tools/make_bundle.py`` and promoted through ``image_processor.bundles.stage_bundle``, with the
tarball digest, the per-file digests, the manifest schema, and the family own manifest check on
the way in. The synthetic bundles are unsigned, because the corpus builder writes no ``keyId``;
the tier-2 bundles are signed with a fresh key per run, as the tier-2 suite signs them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from image_processor.bundles import BundleCache, stage_bundle
from image_processor.engine.families import family_for
from image_processor.engine.protocol import CPU_PROVIDER
from image_processor.types import CachedBundle
from tools.make_bundle import make_bundle

#: Repository root, so the defaults work from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The bundle-manifest schema every staged bundle is checked against.
SCHEMA_PATH = REPO_ROOT / "schemas" / "model-bundle-manifest.schema.json"

#: The suite names ``--suites`` accepts, in the order the site lists them.
SUITES = ("synthetic", "live")


@dataclass(frozen=True)
class CorpusModel:
    """One model of one suite, with the images it runs over.

    Attributes:
        key: The model key. It names the model in the site and, for a tier-2 model, its golden.
        suite: ``"synthetic"`` or ``"live"``.
        family: The task family the manifest declares.
        corpus: What the images are, in one phrase: the dataset for a live model, the fixture set
            for a synthetic one.
        images: The image files to run, in the order the site lists them.
        names: How each image is named, parallel to ``images``. A live name is the path relative
            to ``tests/.cache``, which is what the goldens record; a synthetic name is the path
            relative to the generated corpus.
        stage: Stages the bundle and returns it. Called once per model per run, with a scratch
            directory.
    """

    key: str
    suite: str
    family: str
    corpus: str
    images: Sequence[Path]
    names: Sequence[str]
    stage: Callable[[Path], CachedBundle]


class CorpusError(Exception):
    """A suite cannot be assembled: a missing asset, or a corpus that was never built."""


def fixture_function(fixture):
    """Return the plain function a pytest fixture wraps.

    ``tests/live_models/conftest.py`` states the tier-2 image selection once, as three
    zero-argument session fixtures. pytest wraps each one in a definition object and keeps the
    original function on ``__wrapped__``; calling that is what lets this module use the selection
    itself rather than a second copy of it that can drift.

    Args:
        fixture: The fixture as the conftest module exposes it.

    Returns:
        The underlying function.
    """
    return getattr(fixture, "__wrapped__", fixture)


def select_images(fixture) -> List[Path]:
    """Run one tier-2 image selection.

    Args:
        fixture: The conftest fixture that selects the images.

    Returns:
        The selected image paths.

    Raises:
        CorpusError: The assets the selection reads are not in ``tests/.cache``. The selection
            calls ``pytest.skip`` for that, which is an instruction to a test run and not an
            answer a build tool can act on, so it becomes a failure naming the fetch command.
    """
    import pytest

    try:
        return list(fixture_function(fixture)())
    except pytest.skip.Exception as exc:
        raise CorpusError(str(exc)) from exc


def stage_directory(
    source: Path,
    workdir: Path,
    cache_root: Path,
    provider: str,
    private_key: Optional[bytes] = None,
    key_id: Optional[str] = None,
    trusted_keys: Optional[Dict[str, bytes]] = None,
) -> CachedBundle:
    """Pack, optionally sign, and stage one bundle directory.

    Args:
        source: The bundle source directory: the graph, the labels, the transforms, the manifest.
        workdir: Scratch space for the tarball and the staging directory.
        cache_root: The content-addressed cache to promote into.
        provider: The execution provider the run uses, checked against ``providersPermitted``.
        private_key: The Ed25519 private key PEM to sign ``manifest.json`` with, or ``None`` for
            an unsigned bundle.
        key_id: The signing key id, when the bundle is signed.
        trusted_keys: The public keys staging trusts, when the bundle is signed.

    Returns:
        The staged, verified bundle.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    archive = workdir / f"{source.name}.tar"
    digest = make_bundle(
        src_dir=source,
        out_path=archive,
        key=private_key,
        key_id=key_id,
        compress=False,
        schema_path=SCHEMA_PATH,
    )
    document = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    providers = [provider] if provider == CPU_PROVIDER else [provider, CPU_PROVIDER]
    return stage_bundle(
        uri=str(archive),
        digest=digest,
        staging_root=workdir / "staging",
        cache=BundleCache(cache_root, schema_path=SCHEMA_PATH),
        signing_required=private_key is not None,
        trusted_keys=trusted_keys or {},
        schema_path=SCHEMA_PATH,
        model_id=document["modelId"],
        version=document["version"],
        available_providers=providers,
        validators=[lambda manifest: family_for(manifest).validate_manifest(manifest)],
    )


def synthetic_models(corpus_root: Path, provider: str) -> List[CorpusModel]:
    """Describe the synthetic suite over an already-built tier-1 corpus.

    Every bundle of ``expected.json`` becomes one model, and its images are the ones the oracle
    states an expected answer for, in the order the oracle lists them.

    Args:
        corpus_root: The directory ``tests/fixtures/build.py`` wrote.
        provider: The execution provider the run uses.

    Returns:
        The models, ordered by key.

    Raises:
        CorpusError: The corpus has no ``expected.json``.
    """
    oracle_path = Path(corpus_root) / "expected.json"
    if not oracle_path.is_file():
        raise CorpusError(f"{oracle_path} is missing; the tier-1 corpus was not built")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))

    models: List[CorpusModel] = []
    for key, record in sorted(oracle["bundles"].items()):
        names = [case["image"] for case in record["cases"]]
        source = Path(corpus_root) / record["path"]
        models.append(
            CorpusModel(
                key=key,
                suite="synthetic",
                family=record["family"],
                corpus="tier-1 fixtures",
                images=[Path(corpus_root) / name for name in names],
                names=names,
                stage=lambda scratch, source=source: stage_directory(
                    source, scratch, scratch / "cache", provider
                ),
            )
        )
    return models


def bad_inputs(corpus_root: Path) -> List[Dict[str, object]]:
    """Read the tier-1 bad-input set and what each fixture is required to do.

    Args:
        corpus_root: The directory ``tests/fixtures/build.py`` wrote.

    Returns:
        One record per fixture, carrying its path, size, and the decode code it must raise, or
        ``None`` for a fixture that is supposed to decode.
    """
    oracle = json.loads((Path(corpus_root) / "expected.json").read_text(encoding="utf-8"))
    return list(oracle.get("badInputs", []))


def live_models(provider: str) -> List[CorpusModel]:
    """Describe the tier-2 suite over the fetched corpus.

    The models, their manifests, and the image selection all come from the tier-2 suite itself, so
    the site runs what the goldens were recorded from.

    Args:
        provider: The execution provider the run uses.

    Returns:
        The models, ordered by key.

    Raises:
        CorpusError: The tier-2 asset cache is missing an asset the suite needs.
    """
    import pytest

    from tests.live_models import CACHE_ROOT, asset_path, relative_name, require_asset
    from tests.live_models import bundles as bundle_support
    from tests.live_models import conftest as live_conftest
    from tests.live_models import labels as label_sets

    if not CACHE_ROOT.is_dir():
        raise CorpusError(
            f"{CACHE_ROOT} is missing; run python tools/fetch_test_assets.py to fetch the corpus"
        )

    def asset(asset_id: str, *parts: str) -> Path:
        try:
            return require_asset(asset_id, *parts)
        except pytest.skip.Exception as exc:
            raise CorpusError(str(exc)) from exc

    imagenet = label_sets.imagenet_1000(asset("labels-imagenet-synset", "synset.txt"))

    build_record_path = asset_path(live_conftest.PATCHCORE_ASSET, "build.json")
    build_record = (
        json.loads(build_record_path.read_text(encoding="utf-8"))
        if build_record_path.is_file()
        else None
    )
    described = bundle_support.live_models(imagenet, build_record)

    datasets = {
        "imagenette": ("Imagenette val", live_conftest.imagenette_images),
        "coco": ("COCO val2017 slice", live_conftest.coco_images),
        "visa": ("VisA capsules", live_conftest.visa_capsule_images),
    }
    selected: Dict[str, List[Path]] = {}
    private_key, public_key = bundle_support.keypair()

    models: List[CorpusModel] = []
    for key, described_model in sorted(described.items()):
        label, fixture = datasets[described_model.dataset]
        if described_model.dataset not in selected:
            selected[described_model.dataset] = select_images(fixture)
        images = selected[described_model.dataset]
        graph = asset(described_model.asset_id, described_model.filename)

        def stage(scratch: Path, described_model=described_model, graph=graph) -> CachedBundle:
            source = bundle_support.write_source(
                described_model, graph, scratch / "src" / described_model.key
            )
            return stage_directory(
                source,
                scratch,
                scratch / "cache",
                provider,
                private_key=private_key,
                key_id=bundle_support.KEY_ID,
                trusted_keys={bundle_support.KEY_ID: public_key},
            )

        models.append(
            CorpusModel(
                key=key,
                suite="live",
                family=described_model.family,
                corpus=label,
                images=images,
                names=[relative_name(path) for path in images],
                stage=stage,
            )
        )
    if not models:
        raise CorpusError("the tier-2 corpus described no models")
    return models
