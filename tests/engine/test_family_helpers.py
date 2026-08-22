"""The small shared helpers every family leans on (LLD §6)."""

from __future__ import annotations

import numpy as np
import pytest

from image_processor.engine.families import label_for, pick_output, static_dim
from image_processor.engine.families.classification import ClassificationFamily
from image_processor.engine.families.segmentation import _class_axis
from image_processor.types import Family
from tests.engine.conftest import make_manifest, spec


def test_a_class_index_outside_the_label_set_gets_a_positional_name():
    assert label_for(["a", "b"], 1) == "b"
    assert label_for(["a", "b"], 7) == "class-7"
    assert label_for([], 0) == "class-0"


def test_an_axis_outside_the_declared_shape_reads_as_dynamic():
    declared = spec("logits", (1, 4))
    assert static_dim(declared, -1) == 4
    assert static_dim(declared, 5) is None
    assert static_dim(declared, -5) is None
    assert static_dim(spec("logits", (1, "C")), -1) is None


def test_several_outputs_without_a_name_fall_back_to_the_first_declared():
    m = make_manifest(outputs=[spec("head", (1, 3)), spec("aux", (1, 3))])
    chosen = pick_output({"aux": np.ones((1, 3)), "head": np.zeros((1, 3))}, m)
    assert chosen.tolist() == [[0.0, 0.0, 0.0]]


def test_a_single_output_needs_no_name_at_all():
    m = make_manifest(outputs=[])
    assert pick_output({"only": np.ones((1, 2))}, m).tolist() == [[1.0, 1.0]]


def test_a_two_dimensional_segmentation_output_declares_one_channel():
    m = make_manifest(family=Family.SEGMENTATION, outputs=[spec("mask", (4, 4))])
    assert _class_axis(m, "NCHW") == 1


def test_the_classification_family_reports_its_own_family():
    assert ClassificationFamily().family is Family.CLASSIFICATION
