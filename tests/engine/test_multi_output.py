"""A manifest declares every graph output; the family reads ``familyParams.outputName``.

FCN-ResNet50 exports ``out`` and ``aux`` (WP7 finding); a manifest that declares both must be
accepted when the family is told which one to read, and refused when it is not.
"""

import dataclasses

import pytest

from image_processor.engine.families import FAMILIES, FamilyError, single_output
from image_processor.types import Family, TensorSpec

from tests.engine.conftest import make_manifest, spec


@pytest.fixture
def segmentation_manifest():
    return make_manifest(
        family=Family.CLASSIFICATION,
        outputs=[spec("logits", (1, 3))],
        family_params={"labels": ["a", "b", "c"], "outputName": "logits"},
    )


def _with_outputs(manifest, outputs, **params):
    merged = dict(manifest.family_params or {})
    merged.update(params)
    return dataclasses.replace(manifest, outputs=list(outputs), family_params=merged)


def test_single_output_accepts_the_named_output_among_several(segmentation_manifest):
    main = segmentation_manifest.outputs[0]
    aux = TensorSpec(name="aux", dtype=main.dtype, shape=main.shape)
    m = _with_outputs(segmentation_manifest, [main, aux], outputName=main.name)
    assert single_output(m, "classification") == main
    FAMILIES[Family.CLASSIFICATION].validate_manifest(m)


def test_single_output_refuses_several_outputs_without_a_name(segmentation_manifest):
    main = segmentation_manifest.outputs[0]
    aux = TensorSpec(name="aux", dtype=main.dtype, shape=main.shape)
    params = dict(segmentation_manifest.family_params or {})
    params.pop("outputName", None)
    m = dataclasses.replace(segmentation_manifest, outputs=[main, aux], family_params=params)
    with pytest.raises(FamilyError) as excinfo:
        single_output(m, "classification")
    assert excinfo.value.code == "UNSUPPORTED_OUTPUT_COUNT"


def test_single_output_refuses_an_undeclared_name(segmentation_manifest):
    m = _with_outputs(segmentation_manifest, segmentation_manifest.outputs, outputName="nope")
    with pytest.raises(FamilyError) as excinfo:
        single_output(m, "classification")
    assert excinfo.value.code == "MISSING_OUTPUT"


def test_single_output_refuses_no_outputs(segmentation_manifest):
    m = dataclasses.replace(segmentation_manifest, outputs=[])
    with pytest.raises(FamilyError) as excinfo:
        single_output(m, "classification")
    assert excinfo.value.code == "UNSUPPORTED_OUTPUT_COUNT"
