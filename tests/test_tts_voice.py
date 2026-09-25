"""Tests for the TTS voice fallback (#75).

The integration options dialog no longer has a ``tts_voice`` field. The voice
comes from the assistant settings or a ``tts.speak`` call; without one the
built-in ``DEFAULT_TTS_VOICE`` is used. A ``tts_voice`` option left over on an
existing config entry is ignored.

No real Home Assistant and no network: the HTTP session is a fake that records
the payload of the speech request.
"""
# ruff: noqa: I001 - import order below is intentional: `_ha_stubs` must run
# before the `mistral_conversation` import so Home Assistant is stubbed first.
from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation import const
from mistral_conversation import tts as tts_module
from mistral_conversation.const import DEFAULT_TTS_VOICE, DOMAIN

COMPONENT = Path(__file__).resolve().parent.parent / "custom_components" / "mistral_conversation"
LEFTOVER_VOICE = "fr_marie_neutral"  # what an old config entry may still hold


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def json(self):
        return {"audio_data": base64.b64encode(b"mp3-bytes").decode()}


class _FakeSession:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def request(self, method, url, **kwargs):
        """``aiohttp.ClientSession.request``: dispatch on the HTTP verb."""
        return getattr(self, method.lower())(url, **kwargs)

    def post(self, url, *, json, **kwargs):
        self.payloads.append(json)
        return _FakeResponse()


def _make_entity(options=None, session=None) -> tts_module.MistralTTSEntity:
    runtime = SimpleNamespace(session=session, headers={})
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", options=options or {})
    return tts_module.MistralTTSEntity(hass, entry)


class DefaultOptionsTests(unittest.TestCase):
    def test_default_options_use_built_in_voice(self) -> None:
        self.assertEqual(_make_entity().default_options, {"voice": DEFAULT_TTS_VOICE})

    def test_leftover_tts_voice_option_is_ignored(self) -> None:
        entity = _make_entity({"tts_voice": LEFTOVER_VOICE})
        self.assertEqual(entity.default_options, {"voice": DEFAULT_TTS_VOICE})


class BatchVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def _voice_sent(self, options: dict, entry_options=None) -> str:
        session = _FakeSession()
        entity = _make_entity(entry_options, session)
        await entity.async_get_tts_audio("hello", "en", options)
        return session.payloads[0]["voice_id"]

    async def test_no_voice_falls_back_to_default(self) -> None:
        self.assertEqual(await self._voice_sent({}), DEFAULT_TTS_VOICE)

    async def test_empty_voice_falls_back_to_default(self) -> None:
        self.assertEqual(await self._voice_sent({"voice": ""}), DEFAULT_TTS_VOICE)

    async def test_explicit_voice_wins(self) -> None:
        self.assertEqual(await self._voice_sent({"voice": "some-uuid"}), "some-uuid")

    async def test_leftover_option_still_speaks_with_default(self) -> None:
        voice = await self._voice_sent({}, {"tts_voice": LEFTOVER_VOICE})
        self.assertEqual(voice, DEFAULT_TTS_VOICE)


class StreamVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def _voice_streamed(self, options: dict, entry_options=None) -> str:
        entity = _make_entity(entry_options)
        seen: list[str] = []

        def fake_pipelined(message_gen, voice):
            seen.append(voice)

        request = SimpleNamespace(options=options, message_gen=None, message="hello")
        # The HA stub's TTSAudioResponse takes no arguments; swap in a plain one.
        with (
            patch.object(entity, "_pipelined_stream", fake_pipelined),
            patch.object(tts_module, "TTSAudioResponse", SimpleNamespace),
        ):
            await entity.async_stream_tts_audio(request)
        return seen[0]

    async def test_no_voice_falls_back_to_default(self) -> None:
        self.assertEqual(await self._voice_streamed({}), DEFAULT_TTS_VOICE)

    async def test_explicit_voice_wins(self) -> None:
        self.assertEqual(await self._voice_streamed({"voice": "some-uuid"}), "some-uuid")

    async def test_leftover_option_still_speaks_with_default(self) -> None:
        voice = await self._voice_streamed({}, {"tts_voice": LEFTOVER_VOICE})
        self.assertEqual(voice, DEFAULT_TTS_VOICE)


class OptionRemovedTests(unittest.TestCase):
    def test_option_key_is_gone(self) -> None:
        self.assertFalse(hasattr(const, "CONF_TTS_VOICE"))

    def test_config_flow_has_no_voice_field(self) -> None:
        source = (COMPONENT / "config_flow.py").read_text(encoding="utf-8")
        self.assertNotIn("tts_voice", source.lower())

    def test_no_label_or_description_left_in_any_language(self) -> None:
        files = [COMPONENT / "strings.json"]
        files += sorted((COMPONENT / "translations").glob("*.json"))
        self.assertGreaterEqual(len(files), 4)  # strings + en, nl, fr
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            step = data["options"]["step"]["init"]
            self.assertNotIn("tts_voice", step["data"], path.name)
            self.assertNotIn("tts_voice", step["data_description"], path.name)


if __name__ == "__main__":
    unittest.main()
