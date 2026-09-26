"""Tests for the follow-ups to the release-1 code review.

Covers: no API key in the setup error, a fresh STT upload per 429 retry, the
voice fetch running in the background at startup, and the web-search
conversation cleanup going through the shared request helper.

No real Home Assistant and no network.
"""
from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from homeassistant.exceptions import ConfigEntryNotReady

import custom_components.mistral_conversation as init_module
from custom_components.mistral_conversation import api
from custom_components.mistral_conversation import conversation as conv_module
from custom_components.mistral_conversation import stt as stt_module
from custom_components.mistral_conversation import tts as tts_module

from .helpers import attach_runtime, with_hass

API_KEY = "sk-secret-key"


class _LeakyError(aiohttp.ClientError):
    """Mimics ClientResponseError: repr() includes the request headers."""

    def __str__(self) -> str:
        return "503, message='Service Unavailable', url='https://api.mistral.ai/v1/models'"

    def __repr__(self) -> str:
        return f"ClientResponseError(headers={{'Authorization': 'Bearer {API_KEY}'}})"


class _Response:
    def __init__(self, status: int, payload=None) -> None:
        self.status = status
        self.headers: dict = {}
        self._payload = payload if payload is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def json(self):
        return self._payload

    async def text(self) -> str:
        return ""


class SetupErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_ready_message_does_not_contain_the_api_key(self) -> None:
        class _Session:
            def get(self, *args, **kwargs):
                raise _LeakyError

        entry = SimpleNamespace(data={"api_key": API_KEY}, entry_id="entry1")
        with (
            patch.object(init_module, "async_get_clientsession", lambda hass: _Session()),
            self.assertRaises(ConfigEntryNotReady) as ctx,
        ):
            await init_module.async_setup_entry(SimpleNamespace(data={}), entry)
        self.assertNotIn(API_KEY, str(ctx.exception))
        self.assertIn("503", str(ctx.exception))


class SttRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_attempt_sends_a_new_form(self) -> None:
        sent: list = []
        outcomes = [_Response(429), _Response(200, {"text": "hallo"})]

        def request(method, url, **kwargs):
            sent.append(kwargs["data"])
            return outcomes.pop(0)

        async def audio():
            yield b"\x00\x00" * 160

        entry = SimpleNamespace(entry_id="entry1")
        attach_runtime(entry, SimpleNamespace(request=request))
        entity = with_hass(stt_module.MistralSTTEntity(entry), SimpleNamespace(data={}))
        metadata = SimpleNamespace(language="nl", sample_rate=16000, channel=1, bit_rate=16)
        with (
            patch.object(stt_module.aiohttp, "FormData", MagicMock(side_effect=lambda: MagicMock())),
            patch.object(api, "_sleep", AsyncMock()),
            patch.object(stt_module, "SpeechResult", lambda text, state: (text, state)),
            patch.object(stt_module, "SpeechResultState", SimpleNamespace(SUCCESS="ok", ERROR="error")),
        ):
            result = await entity.async_process_audio_stream(metadata, audio())

        self.assertEqual(result, ("hallo", "ok"))
        self.assertEqual(len(sent), 2)
        self.assertIsNot(sent[0], sent[1])

    async def test_non_text_transcription_is_an_error_result(self) -> None:
        async def audio():
            yield b"\x00\x00" * 160

        entry = SimpleNamespace(entry_id="entry1")
        attach_runtime(
            entry, SimpleNamespace(request=lambda *a, **k: _Response(200, {"text": None}))
        )
        entity = with_hass(stt_module.MistralSTTEntity(entry), SimpleNamespace(data={}))
        metadata = SimpleNamespace(language="nl", sample_rate=16000, channel=1, bit_rate=16)
        with (
            patch.object(stt_module, "SpeechResult", lambda text, state: (text, state)),
            patch.object(stt_module, "SpeechResultState", SimpleNamespace(SUCCESS="ok", ERROR="error")),
        ):
            result = await entity.async_process_audio_stream(metadata, audio())
        self.assertEqual(result, ("", "error"))


class VoiceFetchAtStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_runs_as_a_background_task(self) -> None:
        hass = SimpleNamespace(data={})
        entry = SimpleNamespace(
            entry_id="entry1", options={}, async_create_background_task=MagicMock()
        )
        runtime = attach_runtime(entry)
        entity = with_hass(tts_module.MistralTTSEntity(entry), hass)
        refresh = AsyncMock()
        with (
            patch.object(
                tts_module.TextToSpeechEntity, "async_added_to_hass", AsyncMock(), create=True
            ),
            patch.object(entity, "async_refresh_voices", refresh),
        ):
            await entity.async_added_to_hass()

        entry.async_create_background_task.assert_called_once()
        self.assertIs(runtime.tts_entity, entity)
        # Handed to HA to run later, not awaited during setup.
        refresh.assert_not_awaited()
        entry.async_create_background_task.call_args.args[1].close()


class DeleteConversationTests(unittest.IsolatedAsyncioTestCase):
    def _entity(self, request):
        hass = SimpleNamespace(data={})
        entry = SimpleNamespace(entry_id="entry1", options={}, async_start_reauth=MagicMock())
        attach_runtime(entry, SimpleNamespace(request=request))
        return with_hass(conv_module.MistralConversationEntity(entry), hass)

    async def test_delete_uses_the_shared_helper(self) -> None:
        calls: list = []

        def request(method, url, **kwargs):
            calls.append((method, url))
            return _Response(204)

        await self._entity(request)._delete_mistral_conversation("m-1")
        self.assertEqual(calls, [("DELETE", "https://api.mistral.ai/v1/conversations/m-1")])

    async def test_delete_failures_are_only_debug_logged(self) -> None:
        def request(method, url, **kwargs):
            raise TimeoutError

        with self.assertLogs(level=logging.DEBUG) as logs:
            await self._entity(request)._delete_mistral_conversation("m-1")
        self.assertFalse([line for line in logs.output if not line.startswith("DEBUG")])


if __name__ == "__main__":
    unittest.main()
