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
# ruff: noqa: I001 - import order below is intentional: `_ha_stubs` must run
# before the `mistral_conversation` import so Home Assistant is stubbed first.
from __future__ import annotations

import json
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation.ai_task import MistralAITaskEntity


# ---------------------------------------------------------------------------
# Minimal real stand-ins for voluptuous / HA selectors.
#
# The generic stubs in _ha_stubs.py make `voluptuous` and
# `homeassistant.helpers.selector` plain MagicMock *modules*, so attributes
# like `vol.Required` are MagicMock instances, not classes — `isinstance()`
# against them raises TypeError. The manual fallback branch of
# `_structure_to_json_schema` needs real, narrow types to exercise its
# `isinstance` checks, so we install minimal real classes for the duration
# of this module only.
# ---------------------------------------------------------------------------

class _Required:
    def __init__(self, schema: str, description: str | None = None) -> None:
        self.schema = schema
        self.description = description


class _Optional:
    def __init__(self, schema: str, description: str | None = None) -> None:
        self.schema = schema
        self.description = description


class _TextSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}


class _NumberSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}


class _BooleanSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}


class _SelectSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}


sys.modules["voluptuous"].Required = _Required
sys.modules["voluptuous"].Optional = _Optional
sys.modules["homeassistant.helpers.selector"].TextSelector = _TextSelector
sys.modules["homeassistant.helpers.selector"].NumberSelector = _NumberSelector
sys.modules["homeassistant.helpers.selector"].BooleanSelector = _BooleanSelector
sys.modules["homeassistant.helpers.selector"].SelectSelector = _SelectSelector


class _FakeStructure:
    """Stand-in for a voluptuous ``Schema`` — only ``.schema`` is read."""

    def __init__(self, schema: dict[Any, Any]) -> None:
        self.schema = schema


def _force_convert_to_raise() -> None:
    """Make the primary voluptuous_openapi.convert() path fall through.

    Mirrors real HA behaviour: without a selector-aware custom_serializer,
    convert() can't hash HA's selector instances and raises TypeError —
    which is exactly the bug in #34 (no custom_serializer was passed).
    """
    sys.modules["voluptuous_openapi"].convert = MagicMock(
        side_effect=TypeError("unhashable type")
    )


class StructureToJsonSchemaFallbackTests(unittest.TestCase):
    """The manual fallback path must respect `multiple: true` (#34)."""

    def setUp(self) -> None:
        _force_convert_to_raise()

    def test_multiple_text_selector_becomes_array_of_strings(self) -> None:
        structure = _FakeStructure({
            _Required("lines", "three announcements"): _TextSelector(
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
        structure = _FakeStructure({
            _Optional("summary"): _TextSelector({}),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(result["properties"]["summary"], {"type": "string"})
        self.assertNotIn("required", result)

    def test_multiple_number_selector_becomes_array_of_numbers(self) -> None:
        structure = _FakeStructure({
            _Required("scores"): _NumberSelector({"multiple": True}),
        })
        result = MistralAITaskEntity._structure_to_json_schema(structure)
        self.assertEqual(
            result["properties"]["scores"],
            {"type": "array", "items": {"type": "number"}},
        )

    def test_multiple_select_selector_preserves_enum_inside_array(self) -> None:
        structure = _FakeStructure({
            _Required("colors"): _SelectSelector(
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
        structure = _FakeStructure({
            _Required("lines"): _TextSelector({"multiple": True}),
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
        sys.modules["voluptuous_openapi"].convert = MagicMock(return_value="UNSUPPORTED")
        structure = _FakeStructure({
            _Required("summary"): _TextSelector({}),
        })
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
        sys.modules["voluptuous_openapi"].convert = MagicMock(return_value="UNSUPPORTED")
        result = MistralAITaskEntity._structure_to_json_schema(_FakeStructure({}))
        self.assertEqual(result, {"type": "object", "properties": {}})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
