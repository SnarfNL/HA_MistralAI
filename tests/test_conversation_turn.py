"""Tests for a conversation turn: spoken errors (MA-07) and the direct
web-search answer landing in the chat log (MA-02).

No real Home Assistant and no network: the chat log is a small fake that
records the deltas it receives, and the Mistral calls are patched.
"""
# ruff: noqa: I001 - `_ha_stubs` must run before the `mistral_conversation` import.
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation import conversation as conv_module
from mistral_conversation._api import mistral_error
from mistral_conversation.const import (
    CONF_MODEL,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_MODE,
    CONF_WEB_SEARCH_TRIGGER,
    DOMAIN,
    WEB_SEARCH_MODE_ALWAYS,
)


class _ChatLog:
    conversation_id = "conv-1"
    llm_api = None
    unresponded_tool_results = False

    def __init__(self) -> None:
        self.deltas: list[dict] = []

    async def async_provide_llm_data(self, *args) -> None:
        return None

    async def async_add_delta_content_stream(self, agent_id, stream):
        async for delta in stream:
            self.deltas.append(delta)
            yield delta


def _user_input(text: str = "search the weather") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        language="nl",
        agent_id="conversation.mistral",
        extra_system_prompt=None,
        as_llm_context=lambda domain: None,
    )


def _entity(options: dict) -> conv_module.MistralConversationEntity:
    runtime = SimpleNamespace(session=None, headers={})
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", options=options)
    return conv_module.MistralConversationEntity(hass, entry)


WEB_SEARCH_ALWAYS = {
    CONF_MODEL: "mistral-small-latest",
    CONF_WEB_SEARCH: True,
    CONF_WEB_SEARCH_MODE: WEB_SEARCH_MODE_ALWAYS,
}


class DirectWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, options: dict, text: str):
        entity = _entity(options)
        chat_log = _ChatLog()
        result_from_log = MagicMock(return_value="result")
        with (
            patch.object(entity, "_conversations_chat", AsyncMock(return_value="Sunny.")),
            patch.object(conv_module.conversation, "async_get_result_from_chat_log", result_from_log),
        ):
            result = await entity._async_converse(_user_input(text), chat_log)
        return result, chat_log, result_from_log

    async def test_answer_is_added_to_the_chat_log_role_first(self) -> None:
        result, chat_log, result_from_log = await self._run(WEB_SEARCH_ALWAYS, "weather tomorrow")
        self.assertEqual(chat_log.deltas, [{"role": "assistant"}, {"content": "Sunny."}])
        self.assertEqual(result, "result")
        result_from_log.assert_called_once()

    async def test_trigger_phrase_route_also_uses_the_chat_log(self) -> None:
        options = {**WEB_SEARCH_ALWAYS, CONF_WEB_SEARCH_TRIGGER: "search"}
        _, chat_log, _ = await self._run(options, "search the weather")
        self.assertEqual(chat_log.deltas, [{"role": "assistant"}, {"content": "Sunny."}])


class SpokenErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_is_returned_as_a_spoken_error(self) -> None:
        entity = _entity({})
        response = MagicMock()
        with (
            patch.object(entity, "_async_converse", AsyncMock(side_effect=mistral_error("rate_limited"))),
            patch.object(conv_module, "async_spoken_error", AsyncMock(return_value="Druk.")) as spoken,
            patch.object(conv_module.intent, "IntentResponse", MagicMock(return_value=response)),
            patch.object(conv_module, "ConversationResult", SimpleNamespace),
        ):
            result = await entity._async_handle_message(_user_input(), _ChatLog())

        self.assertIs(result.response, response)
        self.assertEqual(result.conversation_id, "conv-1")
        self.assertEqual(spoken.await_args.args[2], "nl")  # the pipeline language
        self.assertEqual(response.async_set_error.call_args.args[1], "Druk.")

    async def test_success_is_returned_unchanged(self) -> None:
        entity = _entity({})
        with patch.object(entity, "_async_converse", AsyncMock(return_value="ok")):
            self.assertEqual(await entity._async_handle_message(_user_input(), _ChatLog()), "ok")


if __name__ == "__main__":
    unittest.main()
