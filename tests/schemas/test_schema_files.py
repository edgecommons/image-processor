"""The three shipped contracts are valid JSON Schema 2020-12 and stay strict."""

import json

import pytest
from jsonschema import Draft202012Validator

SCHEMA_FILES = (
    "config.schema.json",
    "schemas/inference-result.schema.json",
    "schemas/model-bundle-manifest.schema.json",
)

#: Object schemas that deliberately leave `additionalProperties` open. `familyParams` is the
#: dispatch point the per-family conditions constrain, so a blanket rule there would forbid
#: every family's own parameters.
OPEN_OBJECTS = {"schemas/model-bundle-manifest.schema.json": {"/properties/familyParams"}}


def _walk(node, pointer=""):
    """Yield every subschema in a schema document with its JSON pointer."""
    if isinstance(node, dict):
        yield pointer, node
        for key, value in node.items():
            yield from _walk(value, f"{pointer}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{pointer}/{index}")


def _is_object_schema(node):
    declared = node.get("type")
    if isinstance(declared, list):
        return "object" in declared
    return declared == "object"


@pytest.mark.parametrize("relative", SCHEMA_FILES)
def test_the_schema_is_valid_2020_12(component_root, relative):
    schema = json.loads((component_root / relative).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].startswith("https://docs.edgecommons.mbreissi.com/schemas/ImageProcessor/")
    assert schema["title"]
    assert schema["description"]


@pytest.mark.parametrize("relative", SCHEMA_FILES)
def test_every_internal_ref_resolves(component_root, relative):
    schema = json.loads((component_root / relative).read_text(encoding="utf-8"))
    for pointer, node in _walk(schema):
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            continue
        target = schema
        for part in ref[2:].split("/"):
            assert part in target, f"{relative}{pointer}: $ref {ref} does not resolve"
            target = target[part]


@pytest.mark.parametrize("relative", SCHEMA_FILES)
def test_every_object_schema_bounds_its_properties(component_root, relative):
    schema = json.loads((component_root / relative).read_text(encoding="utf-8"))
    exempt = OPEN_OBJECTS.get(relative, set())
    for pointer, node in _walk(schema):
        if not _is_object_schema(node) or pointer in exempt:
            continue
        assert "additionalProperties" in node, f"{relative}{pointer} accepts unknown keys"


def test_the_config_schema_rejects_every_unknown_key(component_root, config_schema):
    """`config.schema.json` is strict throughout, so a typo fails at deploy time."""
    for pointer, node in _walk(config_schema):
        if not _is_object_schema(node):
            continue
        assert node["additionalProperties"] is False, f"{pointer} is not strict"


def test_the_config_schema_describes_one_instance_as_a_route(config_schema):
    assert "route" in config_schema["$defs"]
    assert config_schema["$defs"]["route"]["required"] == ["id", "source", "modelRef"]


@pytest.mark.parametrize("relative", SCHEMA_FILES)
def test_every_declared_property_is_documented(component_root, relative):
    """A contract that does not say what a field means is not a contract."""
    schema = json.loads((component_root / relative).read_text(encoding="utf-8"))
    for pointer, node in _walk(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        if {"if", "then", "else"} & set(pointer.split("/")):
            # Conditional applicators restate a property to select a branch; the declaration
            # they narrow carries the description.
            continue
        for name, child in properties.items():
            described = any(
                key in child for key in ("description", "$ref", "title", "const")
            )
            assert described, f"{relative}{pointer}/properties/{name} has no description"
