"""The configuration reference documents a real configuration, not an approximation."""

import json
import re

from jsonschema import Draft202012Validator

from image_processor.config import parse_component_config


def documented_example(component_root) -> dict:
    """The complete example from `docs/reference/configuration.md`."""
    page = (component_root / "docs" / "reference" / "configuration.md").read_text(encoding="utf-8")
    match = re.search(r"## Complete example.*?```jsonc\n(.*?)\n```", page, re.S)
    assert match is not None, "the configuration reference has no complete example"
    return json.loads(match.group(1))


def test_the_documented_example_validates(component_root, config_schema):
    example = documented_example(component_root)
    validator = Draft202012Validator(config_schema)
    assert list(validator.iter_errors(example["component"]["global"])) == []
    routes = Draft202012Validator(
        {
            "$schema": config_schema["$schema"],
            "$id": config_schema["$id"] + "#route",
            "$ref": "#/$defs/route",
            "$defs": config_schema["$defs"],
        }
    )
    for instance in example["component"]["instances"]:
        assert list(routes.iter_errors(instance)) == [], instance["id"]


def test_the_documented_example_parses(component_root):
    example = documented_example(component_root)["component"]
    config = parse_component_config(example["global"], example["instances"])
    assert [route.id for route in config.routes] == ["clearance-cam-01", "adhoc-inspect"]
    for route in config.routes:
        assert config.model_entry(route.model_ref) is not None


def test_every_documented_rejection_code_is_one_the_validator_raises(component_root):
    """A code in the reference that no rule produces would send an operator hunting."""
    page = (component_root / "docs" / "reference" / "configuration.md").read_text(encoding="utf-8")
    section = page.split("## Validation", 1)[1].split("## Complete example", 1)[0]
    documented = set(re.findall(r"`([A-Z][A-Z0-9_]{2,})`", section))

    sources = "".join(
        (component_root / "image_processor" / "config" / name).read_text(encoding="utf-8")
        for name in ("models.py", "validate.py")
    )
    raised = set(re.findall(r'ConfigError\(\s*"([A-Z][A-Z0-9_]*)"', sources))
    raised.add("SCHEMA_INVALID")

    assert documented - raised == set(), "documented codes no rule raises"
