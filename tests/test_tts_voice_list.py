"""Tests for the account voice list and the refresh button (MA-06, MA-13).

No real Home Assistant and no network: the HTTP session is a fake that serves
canned pages of ``GET /v1/audio/voices``.
"""
# ruff: noqa: I001 - `_ha_stubs` must run before the `mistral_conversation` import.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from homeassistant.exceptions import HomeAssistantError
from mistral_conversation import button as button_module
from mistral_conversation import tts as tts_module
from mistral_conversation.const import DOMAIN, TTS_LANGUAGES

COMPONENT = Path(__file__).resolve().parent.parent / "custom_components" / "mistral_conversation"


class _Response:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def json(self):
        return self._payload

    async def text(self) -> str:
        return "error body"


class _Session:
    """Serves queued outcomes, one per GET: a _Response or an exception."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _page(items: list[dict], total: int | None = None) -> _Response:
    return _Response(200, {"items": items, "total": len(items) if total is None else total})


PRESET = {"id": "id-paul", "name": "en_paul_neutral"}
CUSTOM = {"id": "id-mine", "name": "My Voice"}


def _make_entity(session) -> tts_module.MistralTTSEntity:
    runtime = SimpleNamespace(session=session, headers={}, tts_entity=None)
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", options={})
    return tts_module.MistralTTSEntity(hass, entry)


def _ids(entity) -> list[str]:
    return [v.voice_id for v in entity.async_get_supported_voices("en")]


class VoiceListTests(unittest.IsolatedAsyncioTestCase):
    async def test_picker_is_empty_before_the_first_fetch(self) -> None:
        self.assertEqual(_ids(_make_entity(_Session())), [])

    async def test_custom_and_preset_voices_appear_with_readable_labels(self) -> None:
        entity = _make_entity(_Session(_page([PRESET, CUSTOM])))
        self.assertTrue(await entity.async_refresh_voices())
        voices = entity.async_get_supported_voices("en")
        self.assertEqual(
            [(v.voice_id, v.name) for v in voices],
            [("id-mine", "My Voice"), ("id-paul", "Paul – Neutral (English)")],
        )

    async def test_all_pages_are_fetched(self) -> None:
        session = _Session(_page([PRESET], total=2), _page([CUSTOM], total=2))
        entity = _make_entity(session)
        self.assertTrue(await entity.async_refresh_voices())
        self.assertEqual(session.calls, 2)
        self.assertCountEqual(_ids(entity), ["id-paul", "id-mine"])

    async def test_no_static_fallback_when_first_fetch_fails(self) -> None:
        entity = _make_entity(_Session(_Response(500)))
        self.assertFalse(await entity.async_refresh_voices())
        self.assertEqual(_ids(entity), [])

    async def test_failed_refresh_keeps_last_good_list(self) -> None:
        failures = [
            lambda: _Response(500),
            lambda: aiohttp.ClientError("down"),
            lambda: TimeoutError(),
            lambda: _page([]),
        ]
        for make_failure in failures:
            with self.subTest(failure=repr(make_failure())):
                entity = _make_entity(_Session(_page([PRESET]), make_failure()))
                self.assertTrue(await entity.async_refresh_voices())
                self.assertFalse(await entity.async_refresh_voices())
                self.assertEqual(_ids(entity), ["id-paul"])

    async def test_successful_refresh_picks_up_a_new_voice(self) -> None:
        entity = _make_entity(_Session(_page([PRESET]), _page([PRESET, CUSTOM])))
        await entity.async_refresh_voices()
        self.assertEqual(_ids(entity), ["id-paul"])
        await entity.async_refresh_voices()
        self.assertCountEqual(_ids(entity), ["id-paul", "id-mine"])

    def test_supported_languages_are_the_documented_nine(self) -> None:
        self.assertEqual(_make_entity(_Session()).supported_languages, TTS_LANGUAGES)


class RefreshButtonTests(unittest.IsolatedAsyncioTestCase):
    def _button(self, tts_entity):
        runtime = SimpleNamespace(tts_entity=tts_entity)
        hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
        entry = SimpleNamespace(entry_id="entry1")
        return button_module.MistralRefreshVoicesButton(hass, entry)

    async def test_press_refreshes_the_tts_voices(self) -> None:
        entity = _make_entity(_Session(_page([CUSTOM])))
        await self._button(entity).async_press()
        self.assertEqual(_ids(entity), ["id-mine"])

    async def test_press_raises_a_translated_error_when_refresh_fails(self) -> None:
        button = self._button(_make_entity(_Session(_Response(500))))
        with self.assertRaises(HomeAssistantError) as ctx:
            await button.async_press()
        self.assertEqual(ctx.exception.translation_key, "voices_refresh_failed")

    async def test_press_raises_when_tts_entity_is_not_loaded(self) -> None:
        with self.assertRaises(HomeAssistantError):
            await self._button(None).async_press()

    def test_button_is_a_config_entity_with_a_translation_key(self) -> None:
        button = self._button(None)
        self.assertEqual(button._attr_translation_key, "refresh_voices")
        self.assertIsNotNone(button._attr_entity_category)


class StringsTests(unittest.TestCase):
    def _files(self) -> list[Path]:
        files = [COMPONENT / "strings.json"]
        files += sorted((COMPONENT / "translations").glob("*.json"))
        self.assertGreaterEqual(len(files), 4)  # strings + en, nl, fr
        return files

    def test_button_name_and_error_exist_in_every_language(self) -> None:
        for path in self._files():
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data["entity"]["button"]["refresh_voices"]["name"]
            message = data["exceptions"]["voices_refresh_failed"]["message"]
            self.assertTrue(name and message, path.name)

    def test_stale_stt_language_strings_are_gone(self) -> None:
        for path in self._files():
            self.assertNotIn("stt_language", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
