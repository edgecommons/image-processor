"""The results explorer (WP11): geometry, the data model, the page, and the tool end to end.

Everything here is tier 1: no network, no tier-2 cache, no GPU. The drawing tests assert pixels
rather than that a file appeared, because the one claim an overlay makes is that the rectangle
sits exactly where the result body said it did; the merge tests assert that a second run lands
beside the first rather than on top of it; and the end-to-end test runs the real command line over
the real synthetic corpus, so the wiring between the corpus, the session, the result body, the
images, and the written site is exercised in one piece.

The tier-2 model description is covered too, against a fabricated cache: describing the corpus
reads file names and a synset, never a model, so a fake tree proves the selection without
downloading two gigabytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageColor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures import build as fixtures  # noqa: E402
from tools import build_results_site as cli  # noqa: E402
from tools.results_site import build as build_support  # noqa: E402
from tools.results_site import corpus as corpus_support  # noqa: E402
from tools.results_site import model as model_support  # noqa: E402
from tools.results_site import overlays, render  # noqa: E402

BLACK = (0, 0, 0)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """Build the tier-1 corpus once for the whole module.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        The corpus root.
    """
    root = tmp_path_factory.mktemp("wp11-corpus")
    fixtures.build(root)
    return root


def detection_body(box, label="bolt", family="detection"):
    """Build the smallest result body an overlay can be drawn from.

    Args:
        box: The normalized region.
        label: The class label.
        family: The task family to claim.

    Returns:
        The body.
    """
    return {
        "outputs": {
            "family": family,
            "detections": [{"label": label, "index": 0, "score": 0.9, "box": list(box)}],
        }
    }


# --- geometry ---------------------------------------------------------------------------------


def test_denormalize_maps_a_region_onto_inclusive_pixel_corners():
    """A region becomes the pixel span it covers, both corners inside the canvas."""
    assert overlays.denormalize([0.1, 0.2, 0.3, 0.4], 200, 100) == (20, 20, 79, 59)


def test_denormalize_keeps_a_full_frame_box_inside_the_canvas():
    """A box covering the whole picture ends on the last column and row, not one past them."""
    assert overlays.denormalize([0.0, 0.0, 1.0, 1.0], 200, 100) == (0, 0, 199, 99)


def test_denormalize_keeps_a_box_against_the_right_and_bottom_edges_inside():
    """A box that ends exactly at the edge is still drawable."""
    assert overlays.denormalize([0.5, 0.5, 0.5, 0.5], 200, 100) == (100, 50, 199, 99)


def test_denormalize_gives_a_degenerate_box_one_pixel():
    """A zero-area region still covers a pixel rather than inverting."""
    assert overlays.denormalize([1.0, 1.0, 0.0, 0.0], 64, 64) == (63, 63, 63, 63)
    assert overlays.denormalize([0.25, 0.25, 0.0, 0.0], 64, 64) == (16, 16, 16, 16)


def test_denormalize_clamps_a_region_that_runs_past_the_frame():
    """A region outside the unit square is clipped to the canvas rather than drawn off it."""
    assert overlays.denormalize([-0.5, -0.5, 3.0, 3.0], 40, 20) == (0, 0, 39, 19)


def test_denormalize_refuses_a_region_that_is_not_four_numbers():
    """A malformed region is a failure, not a guess."""
    with pytest.raises(ValueError, match="four numbers"):
        overlays.denormalize([0.1, 0.2, 0.3], 10, 10)


def test_denormalize_refuses_an_empty_canvas():
    """There is no pixel to draw on."""
    with pytest.raises(ValueError, match="at least one pixel"):
        overlays.denormalize([0.0, 0.0, 1.0, 1.0], 0, 10)


def test_canvas_size_leaves_a_large_source_alone():
    """A picture already big enough is drawn at its own size."""
    assert overlays.canvas_size(1024, 768) == (1024, 768)


def test_canvas_size_scales_a_small_source_by_a_whole_number():
    """A 32-pixel fixture is enlarged by an integer factor, which keeps its pixels square."""
    assert overlays.canvas_size(32, 32, minimum=640) == (640, 640)
    assert overlays.canvas_size(128, 64, minimum=640) == (640, 320)


def test_canvas_size_tolerates_an_empty_source():
    """A zero-sized source comes back unchanged rather than dividing by zero."""
    assert overlays.canvas_size(0, 0, minimum=640) == (0, 0)


def test_label_colour_is_stable_and_comes_from_the_palette():
    """A label keeps its colour across runs, because the index comes from a digest."""
    assert overlays.label_color("person") == overlays.label_color("person")
    assert overlays.label_color("person") in overlays.PALETTE
    assert overlays.label_color("person") != overlays.label_color("bicycle")


# --- drawing ----------------------------------------------------------------------------------


def test_overlay_draws_the_rectangle_the_record_asked_for():
    """The outline lands on the pixels the record box denormalizes to, and not beside them."""
    source = Image.new("RGB", (800, 400), BLACK)
    box = [0.1, 0.2, 0.3, 0.4]
    drawn = overlays.draw_overlay(source, detection_body(box))

    assert drawn.size == (800, 400)
    x0, y0, x1, y1 = overlays.denormalize(box, 800, 400)
    colour = ImageColor.getrgb(overlays.label_color("bolt"))
    pixels = drawn.load()
    middle = (y0 + y1) // 2
    assert pixels[x0, middle] == colour
    assert pixels[x1, middle] == colour
    assert pixels[(x0 + x1) // 2, y1] == colour
    assert pixels[x0 - 1, middle] == BLACK
    assert pixels[x1 + 1, middle] == BLACK
    assert pixels[(x0 + x1) // 2, y1 + 1] == BLACK


def test_overlay_draws_a_box_against_the_frame_edges():
    """A detection filling the frame is drawn on the last row and column, not past them."""
    source = Image.new("RGB", (700, 700), BLACK)
    drawn = overlays.draw_overlay(source, detection_body([0.0, 0.0, 1.0, 1.0]))
    colour = ImageColor.getrgb(overlays.label_color("bolt"))
    pixels = drawn.load()
    assert pixels[0, 350] == colour
    assert pixels[699, 350] == colour
    assert pixels[350, 699] == colour


def test_overlay_is_deterministic():
    """Two draws of one record are the same bytes, which is what makes a rebuild a no-op."""
    source = Image.new("RGB", (700, 400), (20, 30, 40))
    body = detection_body([0.2, 0.1, 0.4, 0.5], label="washer")
    assert overlays.draw_overlay(source, body).tobytes() == overlays.draw_overlay(source, body).tobytes()


def test_overlay_enlarges_a_small_source_before_drawing():
    """A 32-pixel fixture is drawn on a canvas a caption fits on."""
    source = Image.new("RGB", (32, 32), BLACK)
    drawn = overlays.draw_overlay(source, detection_body([0.25, 0.25, 0.5, 0.5]))
    assert drawn.size == (640, 640)
    colour = ImageColor.getrgb(overlays.label_color("bolt"))
    x0, y0, x1, y1 = overlays.denormalize([0.25, 0.25, 0.5, 0.5], 640, 640)
    assert drawn.load()[x0, (y0 + y1) // 2] == colour


def test_classification_has_no_overlay():
    """A ranking has nothing to point at, so the detail view shows the table instead."""
    source = Image.new("RGB", (64, 64), BLACK)
    body = {
        "outputs": {"family": "classification", "classes": [{"label": "red", "index": 0, "score": 1.0}]}
    }
    assert overlays.draw_overlay(source, body) is None


def test_segmentation_draws_every_class_that_claims_a_region():
    """Each class with a bounding box is outlined; background and null boxes are left out."""
    outputs = {
        "family": "segmentation",
        "segments": {
            "background": {"pixels": 700, "fraction": 0.7, "bbox": [0.0, 0.0, 1.0, 1.0]},
            "part": {"pixels": 200, "fraction": 0.2, "bbox": [0.1, 0.1, 0.4, 0.4]},
            "defect": {"pixels": 0, "fraction": 0.0, "bbox": None},
        },
    }
    regions = overlays.segment_regions(outputs)
    assert [region[0] for region in regions] == [[0.1, 0.1, 0.4, 0.4]]
    assert "part 200 px (20.0%)" in regions[0][2]

    drawn = overlays.draw_overlay(Image.new("RGB", (700, 700), BLACK), {"outputs": outputs})
    x0, y0, x1, y1 = overlays.denormalize([0.1, 0.1, 0.4, 0.4], 700, 700)
    assert drawn.load()[x0, (y0 + y1) // 2] == ImageColor.getrgb(overlays.label_color("part"))


def test_segmentation_falls_back_to_background_when_it_is_all_there_is():
    """A frame the model called background entirely still gets an outline rather than nothing."""
    outputs = {
        "family": "segmentation",
        "segments": {"background": {"pixels": 1000, "fraction": 1.0, "bbox": [0.0, 0.0, 1.0, 1.0]}},
    }
    regions = overlays.segment_regions(outputs)
    assert len(regions) == 1
    assert regions[0][0] == [0.0, 0.0, 1.0, 1.0]


def test_anomaly_draws_its_summary_region_in_red():
    """A map-reducing anomaly model points at the region that crossed the threshold."""
    outputs = {
        "family": "anomaly",
        "anomaly": {
            "score": 0.9,
            "threshold": 0.5,
            "anomalous": True,
            "direction": "higherIsAnomalous",
            "summary": {"bbox": [0.25, 0.25, 0.5, 0.5]},
        },
    }
    drawn = overlays.draw_overlay(Image.new("RGB", (700, 700), BLACK), {"outputs": outputs})
    x0, y0, x1, y1 = overlays.denormalize([0.25, 0.25, 0.5, 0.5], 700, 700)
    assert drawn.load()[x0, (y0 + y1) // 2] == ImageColor.getrgb(overlays.ANOMALY_COLOR)


def test_a_scalar_anomaly_gets_a_banner_instead_of_a_region():
    """A model that reports one number has no region, so the reading is stated across the top."""
    outputs = {
        "family": "anomaly",
        "anomaly": {
            "score": 0.9,
            "threshold": 0.5,
            "anomalous": True,
            "direction": "higherIsAnomalous",
        },
    }
    drawn = overlays.draw_overlay(Image.new("RGB", (600, 600), BLACK), {"outputs": outputs})
    assert drawn.load()[4, 2] == ImageColor.getrgb(overlays.ANOMALY_COLOR)
    assert drawn.load()[300, 300] == BLACK


def test_a_scalar_anomaly_within_threshold_gets_the_calm_banner():
    """An image the model cleared says so in the palette colour, not in the alarm one."""
    outputs = {
        "family": "anomaly",
        "anomaly": {
            "score": 0.1,
            "threshold": 0.5,
            "anomalous": False,
            "direction": "higherIsAnomalous",
        },
    }
    drawn = overlays.draw_overlay(Image.new("RGB", (600, 600), BLACK), {"outputs": outputs})
    assert drawn.load()[4, 2] == ImageColor.getrgb(overlays.PALETTE[1])


def test_a_caption_against_the_top_edge_stays_on_the_canvas():
    """A box at the top of the frame carries its label inside the picture rather than above it."""
    source = Image.new("RGB", (700, 400), BLACK)
    drawn = overlays.draw_overlay(source, detection_body([0.0, 0.0, 0.5, 0.5]))
    assert drawn.load()[2, 2] == ImageColor.getrgb(overlays.label_color("bolt"))


# --- thumbnails -------------------------------------------------------------------------------


def test_thumbnail_reduces_a_large_image_to_the_bound():
    """A photograph comes down to 320 pixels on its longest side, keeping its aspect ratio."""
    thumb = overlays.thumbnail(Image.new("RGB", (1000, 500), BLACK))
    assert max(thumb.size) == overlays.THUMB_PX
    assert thumb.size == (320, 160)


def test_thumbnail_leaves_a_small_image_alone():
    """A 32-pixel fixture is not blown up into a blurred square."""
    assert overlays.thumbnail(Image.new("RGB", (32, 32), BLACK)).size == (32, 32)


def test_a_small_thumbnail_is_written_as_png_and_a_large_one_as_jpeg(tmp_path):
    """Lossless where it is free, lossy where a few hundred files would otherwise be the site."""
    small = overlays.save_thumbnail(overlays.thumbnail(Image.new("RGB", (32, 32))), tmp_path / "a")
    large = overlays.save_thumbnail(
        overlays.thumbnail(Image.new("RGB", (1000, 800))), tmp_path / "b"
    )
    assert small.suffix == ".png"
    assert large.suffix == ".jpg"
    assert Image.open(large).size == (320, 256)

# --- the data model ---------------------------------------------------------------------------


def test_run_ids_name_the_provider_and_the_device():
    """A run is one provider on one device, and its id says which."""
    assert model_support.run_id("CPUExecutionProvider", None) == "cpu"
    assert (
        model_support.run_id("CUDAExecutionProvider", "NVIDIA GeForce RTX 5080")
        == "cuda-nvidia-geforce-rtx-5080"
    )


def test_entry_ids_are_stable():
    """The same picture under the same model in the same run keeps its link across rebuilds."""
    first = model_support.entry_id("cpu", "yolox-s", "coco/000000000139.jpg")
    assert first == model_support.entry_id("cpu", "yolox-s", "coco/000000000139.jpg")
    assert first != model_support.entry_id("cuda", "yolox-s", "coco/000000000139.jpg")
    assert first.startswith("e") and len(first) == 13


def test_file_slugs_keep_two_images_of_the_same_name_apart():
    """VisA calls one good and one bad capsule 000.JPG; one must not overwrite the other."""
    good = model_support.file_slug("capsules/Normal/000.JPG")
    bad = model_support.file_slug("capsules/Anomaly/000.JPG")
    assert good != bad
    assert "/" not in good and "\\" not in good


def test_summaries_follow_the_task_family():
    """Each family reduces to the line the gallery reads, and to nothing from another family."""
    classification = model_support.summarize(
        {
            "outputs": {
                "family": "classification",
                "classes": [{"label": "tench", "index": 0, "score": 0.8}],
            }
        }
    )
    assert classification == {
        "family": "classification",
        "label": "tench",
        "score": 0.8,
        "classes": 1,
    }

    detection = model_support.summarize(
        {
            "outputs": {
                "family": "detection",
                "detections": [
                    {"label": "person", "index": 0, "score": 0.9, "box": [0, 0, 1, 1]},
                    {"label": "person", "index": 0, "score": 0.4, "box": [0, 0, 1, 1]},
                ],
            }
        }
    )
    assert detection == {"family": "detection", "count": 2, "labels": ["person"], "topScore": 0.9}

    segmentation = model_support.summarize(
        {
            "outputs": {
                "family": "segmentation",
                "segments": {
                    "background": {"pixels": 9, "fraction": 0.9},
                    "person": {"pixels": 1, "fraction": 0.1},
                },
            }
        }
    )
    assert segmentation["classes"] == 2
    assert segmentation["top"][0] == {"label": "background", "fraction": 0.9}

    anomaly = model_support.summarize(
        {
            "outputs": {
                "family": "anomaly",
                "anomaly": {
                    "score": 0.7,
                    "threshold": 0.5,
                    "anomalous": True,
                    "direction": "higherIsAnomalous",
                },
            }
        }
    )
    assert anomaly["anomalous"] is True and anomaly["score"] == 0.7


def test_an_empty_classification_summarizes_without_a_top_class():
    """A model that reported nothing says so rather than indexing an empty list."""
    summary = model_support.summarize({"outputs": {"family": "classification", "classes": []}})
    assert summary == {"family": "classification", "label": None, "score": None, "classes": 0}


def test_an_empty_detection_summarizes_to_no_labels():
    """An empty frame is a real answer and carries no top score."""
    summary = model_support.summarize({"outputs": {"family": "detection", "detections": []}})
    assert summary == {"family": "detection", "count": 0, "labels": [], "topScore": None}


def fake_run(run_id="cpu", provider="CPUExecutionProvider", gpu=None, keys=("alpha",), images=2):
    """Fabricate one build rows, without running anything.

    Args:
        run_id: The run id every row is filed under.
        provider: The provider the run claims.
        gpu: The device name, or ``None``.
        keys: The model keys to fabricate.
        images: How many entries per model.

    Returns:
        The document one build would have produced.
    """
    models = []
    entries = []
    for key in keys:
        models.append(
            {
                "key": key,
                "runId": run_id,
                "suite": "synthetic",
                "family": "detection",
                "corpus": "tier-1 fixtures",
                "card": {"modelId": key, "version": "1.0.0", "providers": [provider], "gpu": gpu},
            }
        )
        for index in range(images):
            name = f"images/{key}-{index}.png"
            body = detection_body([0.1, 0.1, 0.2, 0.2])
            entries.append(
                {
                    "id": model_support.entry_id(run_id, key, name),
                    "runId": run_id,
                    "modelKey": key,
                    "image": {
                        "src": f"images/synthetic/{key}/{index}.png",
                        "thumb": f"thumbs/synthetic/{key}/{index}.png",
                        "overlay": f"overlays/synthetic/{key}/{index}.png",
                        "w": 64,
                        "h": 64,
                        "name": name,
                    },
                    "timings": {"sessionMs": 1.0 + index, "totalMs": 5.0 + index},
                    "decision": {
                        "outcome": "CLEAR",
                        "pass": True,
                        "confidence": 0.9,
                        "threshold": 0.5,
                    },
                    "resultBody": body,
                    "summary": model_support.summarize(body),
                }
            )
    return model_support.document(
        generated="2026-08-23T00:00:00Z",
        runs=[
            {
                "runId": run_id,
                "provider": provider,
                "gpu": gpu,
                "host": "test",
                "onnxruntime": "1.0.0",
                "generated": "2026-08-23T00:00:00Z",
            }
        ],
        models=models,
        entries=entries,
        refusals=[
            {
                "runId": run_id,
                "name": "bad/zero-byte.jpg",
                "bytes": 0,
                "expected": "EMPTY_IMAGE",
                "observed": "EMPTY_IMAGE",
                "refused": True,
            }
        ],
    )


def test_a_document_carries_its_runs_models_and_entries():
    """The assembled document is the shape the page reads."""
    data = fake_run(keys=("alpha", "beta"), images=3)
    assert data["schemaVersion"] == model_support.SITE_SCHEMA_VERSION
    assert len(data["runs"]) == 1
    assert len(data["models"]) == 2
    assert len(data["entries"]) == 6
    assert {entry["runId"] for entry in data["entries"]} == {"cpu"}


def test_a_model_card_reports_the_assignment_and_the_preprocessing(corpus):
    """The card states the session actual providers and states the transforms in one line."""
    from tests.fixtures.build import load_bundle_manifest

    manifest = load_bundle_manifest(corpus / "bundles" / "synthetic-classification-1.0.0")
    card = model_support.model_card(manifest, ["CUDAExecutionProvider", "CPUExecutionProvider"], "RTX 5080")
    assert card["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert card["gpu"] == "RTX 5080"
    assert card["inputShape"] == ["1", "3", "64", "64"]
    assert card["signed"] is False
    assert model_support.preprocess_summary(card) == "RGB stretch 64x64 NCHW float32"


def test_data_js_round_trips():
    """What the tool writes is what a merge reads back."""
    data = fake_run()
    text = render.data_js(data)
    assert text.startswith(model_support.ASSIGNMENT)
    assert model_support.parse_data_js(text) == data


def test_parsing_refuses_a_file_that_is_not_a_data_js():
    """A stray file in the output directory is a failure, not a silent overwrite."""
    with pytest.raises(model_support.MergeError, match="does not assign"):
        model_support.parse_data_js("console.log(1);")


def test_parsing_refuses_a_broken_payload():
    """A truncated data.js is reported rather than half read."""
    with pytest.raises(model_support.MergeError, match="not JSON"):
        model_support.parse_data_js(model_support.ASSIGNMENT + '{"runs":')


def test_parsing_refuses_another_data_model_version():
    """A site an older tool built is not merged into blindly."""
    text = model_support.ASSIGNMENT + json.dumps({"schemaVersion": 99, "runs": []}) + ";"
    with pytest.raises(model_support.MergeError, match="data model 99"):
        model_support.parse_data_js(text)


# --- merging ----------------------------------------------------------------------------------


def test_merging_a_second_run_keeps_both():
    """A CUDA build lands beside the CPU one, and neither loses a row."""
    cpu = fake_run(run_id="cpu")
    cuda = fake_run(run_id="cuda-rtx", provider="CUDAExecutionProvider", gpu="RTX 5080")
    merged = model_support.merge(cpu, cuda)

    assert [run["runId"] for run in merged["runs"]] == ["cpu", "cuda-rtx"]
    assert len(merged["models"]) == 2
    assert len(merged["entries"]) == 4
    assert {entry["runId"] for entry in merged["entries"]} == {"cpu", "cuda-rtx"}
    assert len(merged["refusals"]) == 2


def test_merging_the_same_run_replaces_it():
    """Rebuilding one leg refreshes its numbers instead of doubling them."""
    merged = model_support.merge(fake_run(run_id="cpu", images=2), fake_run(run_id="cpu", images=3))
    assert len(merged["runs"]) == 1
    assert len(merged["models"]) == 1
    assert len(merged["entries"]) == 3
    assert len(merged["refusals"]) == 1


def test_merging_leaves_the_other_run_untouched():
    """Replacing the CPU leg changes nothing about the CUDA one."""
    cuda = fake_run(run_id="cuda-rtx", provider="CUDAExecutionProvider", gpu="RTX 5080")
    site = model_support.merge(fake_run(run_id="cpu"), cuda)
    rebuilt = model_support.merge(site, fake_run(run_id="cpu", images=5))

    kept = [entry for entry in rebuilt["entries"] if entry["runId"] == "cuda-rtx"]
    assert kept == list(cuda["entries"])
    assert len([entry for entry in rebuilt["entries"] if entry["runId"] == "cpu"]) == 5


# --- the page ---------------------------------------------------------------------------------


def test_the_page_names_every_entry_it_holds():
    """The generated index carries every entry id, so the source says what the site holds."""
    data = fake_run(keys=("alpha", "beta"), images=2)
    page = render.index_html(data)
    for entry in data["entries"]:
        assert entry["id"] in page
        assert entry["image"]["src"] in page
    assert "4 entries" in page


def test_the_page_escapes_what_it_writes():
    """A name carrying markup is written as text, not as markup."""
    data = fake_run()
    data["entries"][0]["image"]["name"] = "<script>alert(1)</script>"
    page = render.index_html(data)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_writing_the_site_produces_four_files(tmp_path):
    """The site is index.html, site.css, site.js, and data.js, and nothing else is generated."""
    written = render.write_site(tmp_path, fake_run())
    assert sorted(path.name for path in written) == ["data.js", "index.html", "site.css", "site.js"]
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
    assert render.site_bytes(tmp_path) == sum(path.stat().st_size for path in written)


def test_the_page_fetches_nothing_and_links_nothing_external(tmp_path):
    """The site opens from file://, so it may not reach for a stylesheet, a font, or a JSON file."""
    render.write_site(tmp_path, fake_run())
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    script = (tmp_path / "site.js").read_text(encoding="utf-8")
    style = (tmp_path / "site.css").read_text(encoding="utf-8")
    for text in (page, script, style):
        assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in text
    for forbidden in ("fetch(", "XMLHttpRequest", "@import"):
        assert forbidden not in script and forbidden not in style

# --- corpus description -----------------------------------------------------------------------


def test_the_synthetic_suite_describes_every_generated_bundle(corpus):
    """Each bundle of the oracle becomes a model, with the images the oracle answers for."""
    models = corpus_support.synthetic_models(corpus, "CPUExecutionProvider")
    assert len(models) == 7
    assert {model.family for model in models} == {
        "classification",
        "detection",
        "segmentation",
        "anomaly",
    }
    for model in models:
        assert model.suite == "synthetic"
        assert len(model.images) == len(model.names) > 0
        assert all(path.is_file() for path in model.images)


def test_the_synthetic_suite_refuses_a_corpus_that_was_never_built(tmp_path):
    """A missing oracle names the problem rather than producing an empty site."""
    with pytest.raises(corpus_support.CorpusError, match="was not built"):
        corpus_support.synthetic_models(tmp_path, "CPUExecutionProvider")


def test_the_bad_input_set_comes_out_of_the_oracle(corpus):
    """Every hostile fixture is described, with the decode code it has to raise."""
    records = corpus_support.bad_inputs(corpus)
    assert len(records) == 12
    assert any(record["code"] == "EMPTY_IMAGE" for record in records)


def test_the_decoder_refuses_every_fixture_that_has_to_be_refused(corpus):
    """The refusals section states what the pipeline turns away, and agrees with the oracle."""
    rows = build_support.refusal_rows(corpus, "cpu")
    assert len(rows) == 12
    assert all(row["observed"] == row["expected"] for row in rows)
    assert any(row["refused"] for row in rows)


@pytest.fixture
def fake_tier2_cache(tmp_path, monkeypatch):
    """Fabricate a tier-2 cache with the file names the corpus description reads.

    Describing the tier-2 corpus never opens a model: it reads a synset, lists image directories,
    and checks that each graph file exists. A tree of empty files therefore proves the selection
    without two gigabytes of downloads.

    Args:
        tmp_path: pytest per-test temporary directory.
        monkeypatch: pytest attribute patcher, used to point the suite at the fake cache.

    Returns:
        The fabricated cache root.
    """
    from tests import live_models as live_package

    root = tmp_path / "cache"
    synset = root / "labels-imagenet-synset"
    synset.mkdir(parents=True)
    (synset / "synset.txt").write_text(
        "".join(f"n{index:08d} class-{index}, alias\n" for index in range(1000)), encoding="utf-8"
    )

    val = root / "dataset-imagenette2-160" / "extracted" / "imagenette2-160" / "val"
    for index in range(10):
        directory = val / f"n{index:08d}"
        directory.mkdir(parents=True)
        for image in range(4):
            (directory / f"image-{image}.JPEG").write_bytes(b"")

    coco = root / "dataset-coco-val2017-slice"
    coco.mkdir(parents=True)
    for index in range(30):
        (coco / f"{index:012d}.jpg").write_bytes(b"")

    visa = root / "dataset-visa" / "extracted" / "capsules" / "Data" / "Images"
    for split in ("Normal", "Anomaly"):
        (visa / split).mkdir(parents=True)
        for index in range(15):
            (visa / split / f"{index:03d}.JPG").write_bytes(b"")

    graphs = {
        "model-mobilenetv2-12": "mobilenetv2-12.onnx",
        "model-resnet50-v1-12": "resnet50-v1-12.onnx",
        "model-yolox-nano": "yolox_nano.onnx",
        "model-yolox-s": "yolox_s.onnx",
        "model-ssd-mobilenetv1-12": "ssd_mobilenet_v1_12.onnx",
        "model-fcn-resnet50-12": "fcn-resnet50-12.onnx",
        "model-patchcore-visa-capsules": "model.onnx",
    }
    for asset_id, filename in graphs.items():
        directory = root / asset_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(b"")
    (root / "model-patchcore-visa-capsules" / "build.json").write_text(
        json.dumps({"imageSize": 256, "threshold": 41.98}), encoding="utf-8"
    )

    monkeypatch.setattr(live_package, "CACHE_ROOT", root)
    return root


def test_the_live_suite_describes_all_seven_models_over_twenty_images_each(fake_tier2_cache):
    """The tier-2 description is the goldens own: seven models, and the same twenty pictures."""
    models = corpus_support.live_models("CPUExecutionProvider")
    assert len(models) == 7
    assert {model.key for model in models} == {
        "mobilenetv2-12",
        "resnet50-v1-12",
        "yolox-nano",
        "yolox-s",
        "ssd-mobilenetv1-12",
        "fcn-resnet50-12",
        "patchcore-visa-capsules",
    }
    for model in models:
        assert model.suite == "live"
        assert len(model.images) == 20
        assert len(model.names) == 20
        assert not any(name.startswith("/") for name in model.names)


def test_the_live_suite_selects_what_the_tier_two_fixtures_select(fake_tier2_cache):
    """The selection is imported, not restated, so the site cannot drift from the goldens."""
    from tests.live_models import conftest as live_conftest

    models = {model.key: model for model in corpus_support.live_models("CPUExecutionProvider")}
    expected = corpus_support.select_images(live_conftest.coco_images)
    assert list(models["yolox-s"].images) == expected
    assert list(models["yolox-nano"].images) == expected
    assert list(models["fcn-resnet50-12"].images) == expected


def test_the_live_suite_drops_the_anomaly_model_when_it_was_never_built(fake_tier2_cache):
    """A machine without the PatchCore build gets the six fetched models, not a failure."""
    (fake_tier2_cache / "model-patchcore-visa-capsules" / "build.json").unlink()
    models = corpus_support.live_models("CPUExecutionProvider")
    assert len(models) == 6
    assert "patchcore-visa-capsules" not in {model.key for model in models}


def test_the_live_suite_names_the_fetch_command_when_an_asset_is_missing(fake_tier2_cache):
    """A partial cache is reported with the command that fills it."""
    (fake_tier2_cache / "model-yolox-s" / "yolox_s.onnx").unlink()
    with pytest.raises(corpus_support.CorpusError, match="fetch_test_assets"):
        corpus_support.live_models("CPUExecutionProvider")


def test_the_live_suite_refuses_a_cache_that_is_not_there(tmp_path, monkeypatch):
    """No cache at all is reported before anything is staged."""
    from tests import live_models as live_package

    monkeypatch.setattr(live_package, "CACHE_ROOT", tmp_path / "absent")
    with pytest.raises(corpus_support.CorpusError, match="fetch_test_assets"):
        corpus_support.live_models("CPUExecutionProvider")


def test_selecting_models_by_key_refuses_a_name_that_is_not_there(corpus):
    """A typo in --models is a failure, not a quietly smaller site."""
    with pytest.raises(corpus_support.CorpusError, match="no model named nope"):
        build_support.collect_models(["synthetic"], "CPUExecutionProvider", corpus, keys=["nope"])


def test_selecting_models_by_key_keeps_only_those(corpus):
    """--models narrows the run to the named bundles."""
    models = build_support.collect_models(
        ["synthetic"], "CPUExecutionProvider", corpus, keys=["synthetic-classification-1.0.0"]
    )
    assert [model.key for model in models] == ["synthetic-classification-1.0.0"]


def test_an_empty_selection_is_refused(corpus):
    """A run with no suites at all is a failure rather than an empty site."""
    with pytest.raises(corpus_support.CorpusError, match="holds no models"):
        build_support.collect_models([], "CPUExecutionProvider", corpus)


# --- the device -------------------------------------------------------------------------------


def test_a_cpu_run_names_no_device():
    """There is no GPU to name, and the site says so rather than guessing."""
    assert build_support.gpu_name("CPUExecutionProvider", 0) is None


class _Probe:
    """A device probe that answers with one fixed name."""

    def __init__(self, name, expect=None):
        """Initialize the probe.

        Args:
            name: The device name to report.
            expect: The device ordinal the caller must ask for, or ``None``.
        """
        self.name = name
        self.expect = expect

    def snapshot(self, device_id):
        """Return the fixed reading.

        Args:
            device_id: The device ordinal.

        Returns:
            An object carrying ``device_class``.
        """
        if self.expect is not None:
            assert device_id == self.expect
        return type("Reading", (), {"device_class": self.name})()


def test_a_cuda_run_names_the_device_nvml_reports(monkeypatch):
    """The device name comes from the component own probe, as a result body does."""
    monkeypatch.setattr(
        build_support, "probe_for", lambda device_id: _Probe("NVIDIA GeForce RTX 5080", 3)
    )
    assert build_support.gpu_name("CUDAExecutionProvider", 3) == "NVIDIA GeForce RTX 5080"


def test_a_cuda_run_without_nvml_names_no_device(monkeypatch):
    """An unreadable device leaves the field empty instead of inventing a name."""
    monkeypatch.setattr(build_support, "probe_for", lambda device_id: _Probe(""))
    assert build_support.gpu_name("CUDAExecutionProvider", 0) is None


def test_a_decoded_array_becomes_an_image():
    """Drawing happens on what the family measured, including a high-bit-depth source."""
    import numpy as np

    eight = build_support.as_image(np.zeros((4, 6, 3), dtype=np.uint8))
    assert eight.size == (6, 4) and eight.mode == "RGB"
    sixteen = build_support.as_image(np.full((2, 2, 3), 0xFF00, dtype=np.uint16))
    assert sixteen.getpixel((0, 0)) == (255, 255, 255)


def test_the_build_stamp_is_utc():
    """The site says when it was built, in one timezone."""
    assert build_support.utc_now().endswith("Z")

# --- the command line -------------------------------------------------------------------------


def test_the_suite_list_is_checked():
    """A misspelt suite is refused with the names that exist."""
    assert cli.parse_suites("live,synthetic") == ["synthetic", "live"]
    with pytest.raises(Exception, match="unknown suite"):
        cli.parse_suites("tier9")
    with pytest.raises(Exception, match="at least one suite"):
        cli.parse_suites(" , ")


def test_the_model_list_is_split():
    """--models takes a comma-separated list and tolerates spacing."""
    assert cli.parse_keys("a, b ,c") == ["a", "b", "c"]


def test_the_tool_builds_a_site_from_the_synthetic_suite(tmp_path, capsys):
    """The whole command line, over the real corpus, on CPU: one image per model."""
    out = tmp_path / "site"
    assert cli.main(["--out", str(out), "--suites", "synthetic", "--limit", "1", "--quiet"]) == 0

    data = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    assert len(data["runs"]) == 1
    assert data["runs"][0]["runId"] == "cpu"
    assert data["runs"][0]["provider"] == "CPUExecutionProvider"
    assert len(data["models"]) == 7
    assert len(data["entries"]) == 7
    assert len(data["refusals"]) == 12

    page = (out / "index.html").read_text(encoding="utf-8")
    for entry in data["entries"]:
        assert entry["id"] in page
        assert (out / entry["image"]["src"]).is_file()
        assert (out / entry["image"]["thumb"]).is_file()
        if entry["image"]["overlay"]:
            assert (out / entry["image"]["overlay"]).is_file()
        body = entry["resultBody"]
        assert body["schemaVersion"] == "1.0"
        assert body["status"] == "SUCCEEDED"
        assert body["model"]["providers"] == ["CPUExecutionProvider"]
        assert body["model"]["digest"].startswith("sha256:")
        assert body["source"]["relativePath"] == entry["image"]["name"]
        assert entry["decision"]["outcome"] in {"CLEAR", "HOLD", "FAIL"}
        assert entry["timings"]["sessionMs"] > 0
        assert entry["timings"]["totalMs"] >= entry["timings"]["sessionMs"]

    assert {model["family"] for model in data["models"]} == {
        "classification",
        "detection",
        "segmentation",
        "anomaly",
    }
    assert "7 model runs, 7 entries" in capsys.readouterr().out


def test_a_classification_entry_has_no_overlay_and_a_detection_entry_has_one(tmp_path, corpus):
    """The site draws where there is something to draw, and states the ranking where there is not."""
    out = tmp_path / "site"
    assert (
        cli.main(
            [
                "--out",
                str(out),
                "--suites",
                "synthetic",
                "--corpus",
                str(corpus),
                "--models",
                "synthetic-classification-1.0.0,synthetic-detection-grid-1.0.0",
                "--limit",
                "1",
                "--quiet",
            ]
        )
        == 0
    )
    data = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    overlay = {entry["modelKey"]: entry["image"]["overlay"] for entry in data["entries"]}
    assert overlay["synthetic-classification-1.0.0"] is None
    drawn = overlay["synthetic-detection-grid-1.0.0"]
    assert drawn.endswith(".png")
    # The picture is the same bytes on any provider and is shared; the drawing is that run own,
    # so a merged site keeps one run boxes out of another run entry.
    assert drawn.startswith("overlays/cpu/")
    assert (out / drawn).is_file()
    for entry in data["entries"]:
        assert entry["image"]["src"].startswith("images/synthetic/")
        assert entry["image"]["thumb"].startswith("thumbs/synthetic/")


def test_a_second_run_merges_into_the_site(tmp_path, corpus):
    """--merge folds a leg in beside the one already there; without it the site is replaced."""
    out = tmp_path / "site"
    common = [
        "--out",
        str(out),
        "--suites",
        "synthetic",
        "--corpus",
        str(corpus),
        "--models",
        "synthetic-classification-1.0.0",
        "--limit",
        "1",
        "--quiet",
    ]
    assert cli.main(common) == 0

    data = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    data["runs"].append(dict(data["runs"][0], runId="cuda-rtx", provider="CUDAExecutionProvider"))
    data["models"].append(dict(data["models"][0], runId="cuda-rtx"))
    data["entries"].append(dict(data["entries"][0], runId="cuda-rtx", id="ecuda00000001"))
    (out / "data.js").write_text(render.data_js(data), encoding="utf-8")

    assert cli.main(common + ["--merge"]) == 0
    merged = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    assert sorted(run["runId"] for run in merged["runs"]) == ["cpu", "cuda-rtx"]
    assert len([entry for entry in merged["entries"] if entry["runId"] == "cpu"]) == 1
    assert len([entry for entry in merged["entries"] if entry["runId"] == "cuda-rtx"]) == 1

    assert cli.main(common) == 0
    replaced = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    assert [run["runId"] for run in replaced["runs"]] == ["cpu"]


def test_merging_into_an_empty_directory_just_builds(tmp_path, corpus):
    """--merge on a site that does not exist yet is the first build, not a failure."""
    out = tmp_path / "site"
    assert (
        cli.main(
            [
                "--out",
                str(out),
                "--suites",
                "synthetic",
                "--corpus",
                str(corpus),
                "--models",
                "synthetic-classification-1.0.0",
                "--limit",
                "1",
                "--merge",
                "--quiet",
            ]
        )
        == 0
    )
    data = model_support.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    assert len(data["runs"]) == 1


def test_the_tool_reports_progress_unless_it_is_told_not_to(tmp_path, corpus, capsys):
    """A build says what it is running, so a long corpus is not silent."""
    assert (
        cli.main(
            [
                "--out",
                str(tmp_path / "site"),
                "--suites",
                "synthetic",
                "--corpus",
                str(corpus),
                "--models",
                "synthetic-classification-1.0.0",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert "synthetic/synthetic-classification-1.0.0: 1 images on CPU" in capsys.readouterr().out


def test_the_tool_reports_a_corpus_it_cannot_assemble(tmp_path, capsys):
    """A missing tier-1 corpus exits non-zero with the reason on stderr."""
    code = cli.main(
        [
            "--out",
            str(tmp_path / "site"),
            "--suites",
            "synthetic",
            "--corpus",
            str(tmp_path / "nope"),
        ]
    )
    assert code == 2
    assert "was not built" in capsys.readouterr().err


def test_the_tool_reports_a_data_js_it_cannot_merge_into(tmp_path, corpus, capsys):
    """A site written by another tool is refused rather than overwritten."""
    out = tmp_path / "site"
    out.mkdir()
    (out / "data.js").write_text("window.SOMETHING = 1;\n", encoding="utf-8")
    code = cli.main(
        [
            "--out",
            str(out),
            "--suites",
            "synthetic",
            "--corpus",
            str(corpus),
            "--models",
            "synthetic-classification-1.0.0",
            "--limit",
            "1",
            "--merge",
            "--quiet",
        ]
    )
    assert code == 2
    assert "does not assign" in capsys.readouterr().err


def test_the_server_binds_the_site(tmp_path, capsys):
    """--serve puts the built site on the loopback interface and says where."""
    render.write_site(tmp_path, fake_run())
    port = cli.serve(tmp_path, 0, forever=False)
    assert port > 0
    assert f"http://127.0.0.1:{port}/" in capsys.readouterr().out