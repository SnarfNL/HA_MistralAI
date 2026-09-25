"""Tests for structured AI Task output (MA-09) and timeout handling in setup,
the config flow and STT (MA-03).

No real Home Assistant and no network: sessions are fakes that raise.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError

import custom_components.mistral_conversation as init_module
from custom_components.mistral_conversation import ai_task as ai_task_module
from custom_components.mistral_conversation import config_flow as config_flow_module
from custom_components.mistral_conversation import stt as stt_module

from .helpers import attach_runtime


class _Invalid(Exception):
    """Stand-in for voluptuous.Invalid."""


def _structure(data):
    """Stand-in for a voluptuous schema: needs a dict with an int 'count'."""
    if not isinstance(data, dict) or not isinstance(data.get("count"), int):
        raise _Invalid("expected {'count': int}")
    return {"count": data["count"], "coerced": True}


class StructuredOutputTests(unittest.TestCase):
    def test_valid_json_is_returned_unchanged(self) -> None:
        data = ai_task_module._parse_structured(_structure, '{"count": 3}')
        self.assertEqual(data, {"count": 3})

    def test_invalid_json_raises_json_parse_error(self) -> None:
        with self.assertRaises(HomeAssistantError) as ctx:
            ai_task_module._parse_structured(_structure, "The count is three.")
        self.assertEqual(ctx.exception.translation_key, "json_parse_error")

    def test_wrong_shape_raises_structure_mismatch(self) -> None:
        with self.assertRaises(HomeAssistantError) as ctx:
            ai_task_module._parse_structured(_structure, '{"total": 3}')
        self.assertEqual(ctx.exception.translation_key, "structure_mismatch")


class _RaisingSession:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def get(self, *args, **kwargs):
        raise self.error

    post = get

    def request(self, method, *args, **kwargs):
        raise self.error


FAILURES = (TimeoutError(), aiohttp.ClientError("down"))


class SetupTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_at_setup_means_not_ready(self) -> None:
        entry = SimpleNamespace(data={"api_key": "k"}, entry_id="entry1")
        for failure in FAILURES:
            with (
                self.subTest(failure=repr(failure)),
                patch.object(
                    init_module, "async_get_clientsession", lambda hass, f=failure: _RaisingSession(f)
                ),
                self.assertRaises(ConfigEntryNotReady),
            ):
                    await init_module.async_setup_entry(SimpleNamespace(data={}), entry)


class ConfigFlowTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_is_cannot_connect_not_unknown(self) -> None:
        flow = config_flow_module.MistralConversationConfigFlow()
        flow.hass = None
        for failure in FAILURES:
            with (
                self.subTest(failure=repr(failure)),
                patch.object(
                    config_flow_module, "async_get_clientsession", lambda hass, f=failure: _RaisingSession(f)
                ),
            ):
                self.assertEqual(await flow._test_api_key("k"), "cannot_connect")


class SttTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_gives_an_error_result_not_a_traceback(self) -> None:
        async def audio():
            yield b"\x00\x00" * 160

        metadata = SimpleNamespace(language="nl", sample_rate=16000, channel=1, bit_rate=16)
        for failure in FAILURES:
            with self.subTest(failure=repr(failure)):
                entry = SimpleNamespace(entry_id="entry1")
                attach_runtime(entry, _RaisingSession(failure))
                entity = stt_module.MistralSTTEntity(SimpleNamespace(data={}), entry)
                with (
                    patch.object(stt_module, "SpeechResult", lambda text, state: (text, state)),
                    patch.object(stt_module, "SpeechResultState", SimpleNamespace(ERROR="error")),
                ):
                    result = await entity.async_process_audio_stream(metadata, audio())
                self.assertEqual(result, ("", "error"))


if __name__ == "__main__":
    unittest.main()
