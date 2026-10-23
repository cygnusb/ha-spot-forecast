"""Translation files stay in lockstep with each other and with the config flow's forms.

Runs without Home Assistant: the form fields are read from config_flow.py with `ast`
and resolved against const.py, which is stdlib-only.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "custom_components" / "spot_forecast"
TRANSLATIONS = PACKAGE / "translations"
REFERENCE_LANGUAGE = "en"
LANGUAGES = sorted(p.stem for p in TRANSLATIONS.glob("*.json"))


def _load(language: str) -> dict:
    return json.loads((TRANSLATIONS / f"{language}.json").read_text(encoding="utf-8"))


def _keys(node: dict, prefix: str = "") -> set[str]:
    found = set()
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        found.add(path)
        if isinstance(value, dict):
            found |= _keys(value, path)
    return found


def _string_constants(path: Path) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` assignments."""
    return {
        node.targets[0].id: node.value.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _flow_fields() -> dict[tuple[str, str], set[str]]:
    """(section, step id) -> keys of the ``vol.Required``/``vol.Optional`` markers.

    The section is "config" for the ConfigFlow class and "options" for the OptionsFlow
    class; the step id comes from the ``async_step_<id>`` method that builds the form.
    """
    const = _string_constants(PACKAGE / "const.py")
    fields: dict[tuple[str, str], set[str]] = {}
    for cls in ast.parse((PACKAGE / "config_flow.py").read_text()).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        bases = {getattr(b, "attr", getattr(b, "id", None)) for b in cls.bases}
        if "OptionsFlow" in bases:
            section = "options"
        elif "ConfigFlow" in bases:
            section = "config"
        else:
            continue
        for method in cls.body:
            if not (
                isinstance(method, ast.AsyncFunctionDef) and method.name.startswith("async_step_")
            ):
                continue
            keys = {
                const[call.args[0].id] if isinstance(call.args[0], ast.Name) else call.args[0].value
                for call in ast.walk(method)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("Required", "Optional")
                and call.args
            }
            if keys:
                fields[(section, method.name.removeprefix("async_step_"))] = keys
    return fields


def test_reference_and_other_languages_present():
    assert REFERENCE_LANGUAGE in LANGUAGES
    assert "de" in LANGUAGES


@pytest.mark.parametrize("language", LANGUAGES)
def test_translation_files_have_same_keys(language):
    reference = _keys(_load(REFERENCE_LANGUAGE))
    keys = _keys(_load(language))
    assert not reference - keys, f"{language}.json is missing {sorted(reference - keys)}"
    assert not keys - reference, f"{language}.json has unknown keys {sorted(keys - reference)}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_config_field_has_label(language):
    fields = _flow_fields()
    assert fields == {
        ("config", "user"): {"zone"},
        ("options", "init"): {"peak_threshold", "backfill_days"},
    }
    doc = _load(language)
    for (section, step), expected in fields.items():
        body = doc[section]["step"][step]
        assert set(body["data"]) == expected, f"{language} {section}.{step} labels"
        assert set(body["data_description"]) == expected, f"{language} {section}.{step} help"
        assert all(text.strip() for text in body["data"].values())
        assert all(text.strip() for text in body["data_description"].values())
