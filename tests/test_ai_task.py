"""Tests for ``ai_task.py``'s ``_structure_to_json_schema`` (#34).

Regression coverage for: a ``structure`` field with a selector using
``multiple: true`` was flattened to a scalar JSON-schema type instead of
an array, because (a) the primary ``voluptuous_openapi.convert()`` call
never passed HA's selector-aware ``custom_serializer`` and so always
raised on HA selector instances, and (b) the manual fallback it fell
through to never inspected ``multiple`` either. Also covers the
``_sanitize()`` safety net applied to both paths' output (same class of
sentinel-leak issue fixed for tool schemas in #36).
"""
from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import voluptuous as vol
from homeassistant.helpers import selector

from custom_components.mistral_conversation.ai_task import MistralAITaskEntity


def _fake_converters(convert: Any) -> Any:
    """Replace the schema converter for one test.

    ``_schema_to_openapi`` imports ``probatio.to_openapi`` (HA 2026.9+) or
    falls back to ``voluptuous_openapi.convert``; both names point at
    *convert* here, so the test controls the result on every HA version.
    """
    return patch.dict(
        sys.modules,
        {
            "probatio": SimpleNamespace(to_openapi=convert),
            "voluptuous_openapi": SimpleNamespace(convert=convert),
        },
    )


def _force_convert_to_raise() -> Any:
    """Make the primary conversion path fall through.

    Mirrors real HA behaviour: without a selector-aware custom_serializer,
    convert() can't hash HA's selector instances and raises TypeError —
    which is exactly the bug in #34 (no custom_serializer was passed).
    """
    return _fake_converters(MagicMock(side_effect=TypeError("unhashable type")))


class StructureToJsonSchemaFallbackTests(unittest.TestCase):
    """The manual fallback path must respect `multiple: true` (#34)."""

    def setUp(self) -> None:
        patcher = _force_convert_to_raise()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_multiple_text_selector_becomes_array_of_strings(self) -> None:
        structure = vol.Schema({
            vol.Required("lines", description="three announcements"): selector.TextSelector(
                {"multiple": True}
            ),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(
            result["properties"]["lines"],
            {"type": "array", "items": {"type": "string"}, "description": "three announcements"},
        )
        self.assertEqual(result["required"], ["lines"])

    def test_non_multiple_text_selector_stays_scalar(self) -> None:
        """Documents the previously-correct single-value case still works."""
        structure = vol.Schema({
            vol.Optional("summary"): selector.TextSelector({}),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(result["properties"]["summary"], {"type": "string"})
        self.assertNotIn("required", result)

    def test_multiple_select_selector_preserves_enum_inside_array(self) -> None:
        structure = vol.Schema({
            vol.Required("colors"): selector.SelectSelector(
                {"multiple": True, "options": ["red", "green", "blue"]}
            ),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(
            result["properties"]["colors"],
            {
                "type": "array",
                "items": {"type": "string", "enum": ["red", "green", "blue"]},
            },
        )

    def test_output_is_json_serializable(self) -> None:
        structure = vol.Schema({
            vol.Required("lines"): selector.TextSelector({"multiple": True}),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        json.dumps(result)  # must not raise


class StructureToJsonSchemaSanitizeTests(unittest.TestCase):
    """A non-dict/sentinel convert() result must never reach Mistral raw.

    Regression coverage for the probatio/voluptuous_openapi sentinel
    mismatch reported after #36 shipped: HA 2026.9+'s custom_serializer can
    return a sentinel voluptuous_openapi.convert() doesn't recognize as its
    own, and because the serializer runs on the top-level schema node too,
    the bare sentinel can replace the *entire* schema instead of one field.
    `_schema_to_openapi()` catches a non-dict result and substitutes an
    empty object schema; unlike tool schemas, ai_task then has a real
    fallback available and uses it.
    """

    def test_sentinel_replacing_whole_schema_falls_through_to_manual_walk(self) -> None:
        structure = vol.Schema({
            vol.Required("summary"): selector.TextSelector({}),
        })
        with _fake_converters(MagicMock(return_value="UNSUPPORTED")):
            result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(
            result,
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        )
        json.dumps(result)

    def test_sentinel_with_nothing_to_fall_back_on_yields_empty_object(self) -> None:
        """No fields for the manual walk to recover either — still a valid,
        if empty, JSON schema object, never the bare sentinel/string."""
        with _fake_converters(MagicMock(return_value="UNSUPPORTED")):
            result = MistralAITaskEntity._structure_to_json_schema(vol.Schema({}))
        self.assertEqual(result, {"type": "object", "properties": {}})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
