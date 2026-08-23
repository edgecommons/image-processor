"""FCN-ResNet50 against the COCO slice (DESIGN.md section 16.1, tier 2).

DESIGN.md names DeepLabV3-MobileNetV3 for this row. The ONNX Model Zoo publishes no DeepLabV3 of
any backbone and no other prebuilt permissively licensed export exists, so the corpus uses the
zoo's FCN-ResNet50 instead: the same Pascal VOC 21-class head, the same normalization, and the
same argmax reduction the segmentation family performs.

The model also carries an auxiliary classifier, so the bundle declares only the ``out`` tensor and
names it in ``familyParams.outputName``. That is the case a real export produces and a synthetic
one does not.
"""

from __future__ import annotations

from tests.live_models import relative_name, require_live_models
from tests.live_models import labels as label_sets
from tests.live_models import verify

require_live_models()


def test_segmentation_matches_its_golden(staged, coco_images, update_goldens_mode):
    """The staged FCN reproduces its committed golden on the COCO slice."""
    model, records = verify.run_corpus(staged, "fcn-resnet50-12", coco_images, relative_name)
    assert len(records) == len(coco_images)
    verify.check(model, records, update_goldens_mode)


def test_every_voc_class_is_reported(staged, coco_images):
    """Argmax mode reports all 21 classes, including the ones claiming no pixels.

    A rule such as "no defect pixels" has to evaluate on a clean image rather than fail to resolve
    its path, which is only true if a class with zero pixels still gets an entry.
    """
    _, records = verify.run_corpus(staged, "fcn-resnet50-12", coco_images, relative_name)
    for entry in records:
        assert sorted(entry["segments"]) == sorted(label_sets.VOC_21), entry["image"]
        total = sum(item["fraction"] for item in entry["segments"].values())
        assert abs(total - 1.0) < 1e-3, f"{entry['image']}: fractions sum to {total}"


def test_background_dominates_a_typical_scene(staged, coco_images):
    """Background claims most of the class map on most images.

    A colour order, normalization, or layout mistake collapses the argmax onto one foreground
    class, which shows up as background losing the frame.
    """
    _, records = verify.run_corpus(staged, "fcn-resnet50-12", coco_images, relative_name)
    dominant = sum(1 for entry in records if entry["segments"]["background"]["fraction"] >= 0.4)
    assert dominant >= len(records) * 0.7, (
        f"background held at least 40 percent of the map on {dominant} of {len(records)} images"
    )
