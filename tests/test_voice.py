"""Tests for voice discovery and resolution helpers.

Covers ``async_discover_mistral_voices``, ``_resolve_voice``,
``async_get_supported_voices``, and ``MistralOptionsFlow._build_voice_options``.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from homeassistant.components.tts import Voice
from homeassistant.helpers.selector import SelectOptionDict

import mistral_conversation.const as const
from mistral_conversation.__init__ import async_discover_mistral_voices
from mistral_conversation.tts import MistralTTSEntity
from mistral_conversation.config_flow import MistralOptionsFlow


# ---------------------------------------------------------------------------
# Discover helper
# ---------------------------------------------------------------------------

class DiscoverVoicesTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_response_returns_empty_list(self) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"items": []})
        session = MagicMock()
        session.get = MagicMock(return_value=resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(return_value=ctx)

        voices = await async_discover_mistral_voices(session, {})
        self.assertEqual(voices, [])

    async def test_filters_invalid_items(self) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={
                "items": [
                    {"id": "abc-123", "name": "Alice", "slug": None},
                    "not-a-dict",
                    {"id": "", "name": "Empty ID"},
                    {"name": "No ID"},
                    {"id": "abc-123", "name": "Duplicate", "slug": "dup"},
                ]
            }
        )
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get = MagicMock(return_value=ctx)

        voices = await async_discover_mistral_voices(session, {})
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0]["id"], "abc-123")
        self.assertEqual(voices[0]["name"], "Alice")
        self.assertIsNone(voices[0]["slug"])

    async def test_non_200_returns_empty(self) -> None:
        resp = MagicMock()
        resp.status = 404
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get = MagicMock(return_value=ctx)

        voices = await async_discover_mistral_voices(session, {})
        self.assertEqual(voices, [])

    async def test_client_error_returns_empty(self) -> None:
        session = MagicMock()
        session.get = MagicMock(side_effect=OSError("no network"))

        voices = await async_discover_mistral_voices(session, {})
        self.assertEqual(voices, [])


# ---------------------------------------------------------------------------
# TTS entity helpers
# ---------------------------------------------------------------------------

class _FakeEntry:
    """Minimal stand-in for ConfigEntry."""

    def __init__(self, options: dict | None = None) -> None:
        self.entry_id = "test-entry"
        self.options = options or {}


class ResolveVoiceTests(unittest.TestCase):
    """``_resolve_voice`` must respect its documented priority."""

    def setUp(self) -> None:
        self.entity = MistralTTSEntity.__new__(MistralTTSEntity)
        self.entity.hass = MagicMock()
        # wire _runtime through hass.data
        self.runtime = MagicMock(discovered_voices=[])
        self.entity.hass.data = {"mistral_conversation": {"test-entry": self.runtime}}

    def test_options_voice_takes_highest_priority(self) -> None:
        entry = _FakeEntry(options={
            const.CONF_TTS_VOICE: "en_paul_angry",
            const.CONF_TTS_VOICE_OVERRIDE: "custom-uuid",
        })
        self.entity._entry = entry
        result = self.entity._resolve_voice({"voice": "per_request_voice"})
        self.assertEqual(result, "per_request_voice")

    def test_override_takes_priority_over_dropdown(self) -> None:
        entry = _FakeEntry(options={
            const.CONF_TTS_VOICE: "en_paul_angry",
            const.CONF_TTS_VOICE_OVERRIDE: "custom-uuid",
        })
        self.entity._entry = entry
        result = self.entity._resolve_voice({})
        self.assertEqual(result, "custom-uuid")

    def test_dropdown_fallback_when_no_override(self) -> None:
        entry = _FakeEntry(options={
            const.CONF_TTS_VOICE: "en_paul_happy",
            const.CONF_TTS_VOICE_OVERRIDE: "",
        })
        self.entity._entry = entry
        result = self.entity._resolve_voice({})
        self.assertEqual(result, "en_paul_happy")

    def test_builtin_default_when_nothing_set(self) -> None:
        entry = _FakeEntry(options={})
        self.entity._entry = entry
        result = self.entity._resolve_voice({})
        self.assertEqual(result, const.DEFAULT_TTS_VOICE)


class GetSupportedVoicesTests(unittest.TestCase):
    """``async_get_supported_voices`` must list presets + discovered."""

    def setUp(self) -> None:
        self.entity = MistralTTSEntity.__new__(MistralTTSEntity)
        self.entity.hass = MagicMock()

    def test_presets_always_present(self) -> None:
        runtime = MagicMock(discovered_voices=[])
        self.entity.hass.data = {"mistral_conversation": {"test-entry": runtime}}
        self.entity._entry = _FakeEntry()
        voices = self.entity.async_get_supported_voices("en")
        ids = [v.voice_id for v in voices]
        self.assertIn("en_paul_neutral", ids)
        self.assertIn("fr_marie_neutral", ids)
        # Discovered list empty
        self.assertEqual(len(voices), len(const.TTS_VOICES))

    def test_discovered_voices_appended_with_labels(self) -> None:
        runtime = MagicMock(discovered_voices=[
            {"id": "custom-a", "name": "Alice", "slug": "alice-v1"},
            {"id": "custom-b", "name": "Bob", "slug": None},
        ])
        self.entity.hass.data = {"mistral_conversation": {"test-entry": runtime}}
        self.entity._entry = _FakeEntry()
        voices = self.entity.async_get_supported_voices("en")
        ids = [v.voice_id for v in voices]
        names = {v.voice_id: v.name for v in voices}
        self.assertIn("custom-a", ids)
        self.assertIn("custom-b", ids)
        self.assertIn("Custom: Alice (alice-v1)", names["custom-a"])
        self.assertIn("Custom: Bob (no slug)", names["custom-b"])

    def test_duplicate_ids_ignored(self) -> None:
        runtime = MagicMock(discovered_voices=[
            {"id": "en_paul_neutral", "name": "Same", "slug": "dup"},
        ])
        self.entity.hass.data = {"mistral_conversation": {"test-entry": runtime}}
        self.entity._entry = _FakeEntry()
        voices = self.entity.async_get_supported_voices("en")
        ids = [v.voice_id for v in voices]
        self.assertEqual(ids.count("en_paul_neutral"), 1)


# ---------------------------------------------------------------------------
# Options flow helpers
# ---------------------------------------------------------------------------

class BuildVoiceOptionsTests(unittest.TestCase):
    """``_build_voice_options`` must integrate discovered voices."""

    def test_presets_first_then_discovered(self) -> None:
        flow = MistralOptionsFlow.__new__(MistralOptionsFlow)
        flow.config_entry = _FakeEntry()
        flow.hass = MagicMock()
        runtime = MagicMock(discovered_voices=[
            {"id": "v1", "name": "Alice", "slug": "alice-v1"},
        ])
        flow.hass.data = {"mistral_conversation": {"test-entry": runtime}}

        options = flow._build_voice_options()
        values = [opt.value for opt in options]
        # Presets come first
        self.assertEqual(values[0], const.TTS_VOICES[0])
        # Discovered appended
        self.assertIn("v1", values)
        # Labels
        alice_opt = [o for o in options if o.value == "v1"][0]
        self.assertEqual(alice_opt.label, "Custom: Alice (alice-v1)")

    def test_no_slug_labelled_explicitly(self) -> None:
        flow = MistralOptionsFlow.__new__(MistralOptionsFlow)
        flow.config_entry = _FakeEntry()
        flow.hass = MagicMock()
        runtime = MagicMock(discovered_voices=[
            {"id": "v2", "name": "Bob", "slug": None},
        ])
        flow.hass.data = {"mistral_conversation": {"test-entry": runtime}}

        options = flow._build_voice_options()
        bob_opt = [o for o in options if o.value == "v2"][0]
        self.assertEqual(bob_opt.label, "Custom: Bob (no slug)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
