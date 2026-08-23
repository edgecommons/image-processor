"""MobileNetV2 and ResNet-50 against Imagenette (DESIGN.md section 16.1, tier 2).

Imagenette names its directories by WordNet id, so the expected ImageNet class of every image is
known without an annotation file. That gives the suite two independent checks: the run agrees with
the committed golden, and the golden itself agrees with the dataset's own labels.
"""

from __future__ import annotations

import pytest

from tests.live_models import relative_name, require_asset, require_live_models
from tests.live_models import labels as label_sets
from tests.live_models import verify

require_live_models()

#: The two ImageNet classifiers of the corpus.
CLASSIFIERS = ["mobilenetv2-12", "resnet50-v1-12"]


@pytest.fixture(scope="module")
def wordnet_to_label():
    """Map every WordNet id to the class name the bundles label it with.

    Returns:
        A mapping of WordNet id to label.
    """
    synset = require_asset("labels-imagenet-synset", "synset.txt")
    return dict(zip(label_sets.wordnet_ids(synset), label_sets.imagenet_1000(synset)))


@pytest.mark.parametrize("key", CLASSIFIERS)
def test_classifier_matches_its_golden(key, staged, imagenette_images, update_goldens_mode):
    """The staged classifier reproduces its committed golden on the Imagenette slice."""
    model, records = verify.run_corpus(staged, key, imagenette_images, relative_name)
    assert len(records) == len(imagenette_images)
    assert all(len(entry["classes"]) == 5 for entry in records)
    verify.check(model, records, update_goldens_mode)


@pytest.mark.parametrize("key", CLASSIFIERS)
def test_classifier_agrees_with_imagenette_directories(
    key, staged, imagenette_images, wordnet_to_label
):
    """The Imagenette class the image came from is what the model reports.

    ImageNet-1k holds classes a human would not separate either, so top-1 is scored loosely and
    top-5 tightly: `cassette player` and `tape player` are the same object, and a model that puts
    the dataset's own class in its top five has decoded, resized, normalized, and ranked correctly.
    A chain with a broken colour order, normalization, or crop misses on both counts.
    """
    model, records = verify.run_corpus(staged, key, imagenette_images, relative_name)
    expected = [wordnet_to_label[path.parent.name] for path in imagenette_images]
    top_one = [entry["classes"][0]["label"] for entry in records]
    exact = sum(1 for got, want in zip(top_one, expected) if got == want)
    within_five = sum(
        1
        for entry, want in zip(records, expected)
        if want in {item["label"] for item in entry["classes"]}
    )
    assert exact >= len(records) * 0.5, (
        f"{model.key} matched the Imagenette class on {exact} of {len(records)} images; "
        f"got {top_one}, expected {expected}"
    )
    assert within_five >= len(records) * 0.75, (
        f"{model.key} carried the Imagenette class in its top five on {within_five} of "
        f"{len(records)} images"
    )
