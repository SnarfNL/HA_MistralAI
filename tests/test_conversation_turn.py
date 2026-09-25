"""Tests for a conversation turn: spoken errors (MA-07) and the direct
web-search answer landing in the chat log (MA-02).

No real Home Assistant and no network: the chat log is a small fake that
records the deltas it receives, and the Mistral calls are patched.
"""
from __future__ import annotations

import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.exceptions import HomeAssistantError

from custom_components.mistral_conversation import conversation as conv_module
from custom_components.mistral_conversation.api import mistral_error
from custom_components.mistral_conversation.const import (
    CONF_MODEL,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_MODE,
    CONF_WEB_SEARCH_TRIGGER,
    WEB_SEARCH_MODE_ALWAYS,
)

from .helpers import attach_runtime, with_hass


class _ChatLog:
    conversation_id = "conv-1"
    llm_api = None
    unresponded_tool_results = False

    def __init__(self, content: list | None = None, stream_error: BaseException | None = None) -> None:
        self.content: list = list(content or [])
        self.deltas: list[dict] = []
        self.stream_error = stream_error

    async def async_provide_llm_data(self, *args) -> None:
        return None

    async def async_add_delta_content_stream(self, agent_id, stream):
        async for delta in stream:
            self.deltas.append(delta)
            if self.stream_error is not None:
                # Stands in for an HA tool failing while the reply streams.
                raise self.stream_error
            yield delta

    def async_add_assistant_content_without_tools(self, content) -> None:
        self.content.append(content)


def _user_input(text: str = "search the weather") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        language="nl",
        agent_id="conversation.mistral",
        extra_system_prompt=None,
        as_llm_context=lambda domain: None,
    )


def _entity(options: dict, session=None) -> conv_module.MistralConversationEntity:
    hass = SimpleNamespace(data={})
    entry = SimpleNamespace(entry_id="entry1", options=options, async_start_reauth=MagicMock())
    attach_runtime(entry, session)
    return with_hass(conv_module.MistralConversationEntity(entry), hass)


WEB_SEARCH_ALWAYS = {
    CONF_MODEL: "mistral-small-latest",
    CONF_WEB_SEARCH: True,
    CONF_WEB_SEARCH_MODE: WEB_SEARCH_MODE_ALWAYS,
}
WEB_SEARCH_BY_MODEL = {CONF_MODEL: "mistral-small-latest", CONF_WEB_SEARCH: True}


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


class NoDoubleRetryTests(unittest.IsolatedAsyncioTestCase):
    """A bad key or rate limit is not retried again via chat completions."""

    async def test_direct_route_does_not_fall_back_on_unrecoverable_errors(self) -> None:
        for key in ("rate_limited", "invalid_auth"):
            with self.subTest(key=key):
                entity = _entity(WEB_SEARCH_ALWAYS)
                stream = AsyncMock()
                with (
                    patch.object(entity, "_conversations_chat", AsyncMock(side_effect=mistral_error(key))),
                    patch.object(entity, "_stream_and_collect", stream),
                    self.assertRaises(HomeAssistantError) as ctx,
                ):
                    await entity._async_converse(_user_input("weather"), _ChatLog())
                self.assertEqual(ctx.exception.translation_key, key)
                stream.assert_not_awaited()

    async def test_direct_route_still_falls_back_on_other_errors(self) -> None:
        entity = _entity(WEB_SEARCH_ALWAYS)
        stream = AsyncMock()
        with (
            patch.object(entity, "_conversations_chat", AsyncMock(side_effect=mistral_error("cannot_connect"))),
            patch.object(entity, "_stream_and_collect", stream),
            patch.object(conv_module.conversation, "async_get_result_from_chat_log", MagicMock()),
        ):
            await entity._async_converse(_user_input("weather"), _ChatLog())
        stream.assert_awaited_once()

    async def test_model_requested_search_does_not_retry_on_unrecoverable_errors(self) -> None:
        entity = _entity(WEB_SEARCH_BY_MODEL)

        async def model_asks_for_search(payload, chat_log, user_input, *, intercept_tool, intercepted):
            intercepted.append("weather amsterdam")

        stream = AsyncMock(side_effect=model_asks_for_search)
        with (
            patch.object(entity, "_conversations_chat", AsyncMock(side_effect=mistral_error("rate_limited"))),
            patch.object(entity, "_stream_and_collect", stream),
            self.assertRaises(HomeAssistantError) as ctx,
        ):
            await entity._async_converse(_user_input("what's the weather?"), _ChatLog())
        self.assertEqual(ctx.exception.translation_key, "rate_limited")
        self.assertEqual(stream.await_count, 1)


class ConversationsParsingTests(unittest.IsolatedAsyncioTestCase):
    """Odd Conversations API answers must not raise non-HA exceptions."""

    async def _reply(self, data) -> str:
        entity = _entity(WEB_SEARCH_ALWAYS)

        class _Resp:
            status = 200
            headers = MappingProxyType({})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def json(self):
                return data

        runtime = attach_runtime(
            entity._entry, SimpleNamespace(request=lambda *a, **k: _Resp())
        )
        runtime.web_search_convs = MagicMock(
            pop_expired=MagicMock(return_value=[]), get=MagicMock(return_value=None)
        )
        return await entity._conversations_chat(
            model="mistral-small-latest", user_text="q", language="nl", conv_id=None
        )

    async def test_non_dict_outputs_are_ignored(self) -> None:
        reply = await self._reply({"outputs": ["junk", {"type": "message.output", "content": "Hi."}]})
        self.assertEqual(reply, "Hi.")

    async def test_non_list_outputs_and_message_give_empty_reply(self) -> None:
        self.assertEqual(await self._reply({"outputs": "junk", "message": 3}), "")

    async def test_non_object_json_is_a_translated_error(self) -> None:
        with self.assertRaises(HomeAssistantError) as ctx:
            await self._reply(["not", "an", "object"])
        self.assertEqual(ctx.exception.translation_key, "invalid_response")


class SpokenErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _spoken_turn(self, error: BaseException, chat_log: _ChatLog):
        entity = _entity({})
        response = MagicMock()

        async def converse(user_input, log):
            # The turn got as far as a tool call and its result before failing.
            log.content.extend(["assistant tool_calls", "tool result"])
            raise error

        with (
            patch.object(entity, "_async_converse", converse),
            patch.object(conv_module, "async_spoken_error", AsyncMock(return_value="Druk.")) as spoken,
            patch.object(conv_module.intent, "IntentResponse", MagicMock(return_value=response)),
            patch.object(conv_module, "ConversationResult", SimpleNamespace),
            patch.object(conv_module.conversation, "AssistantContent", SimpleNamespace),
        ):
            result = await entity._async_handle_message(_user_input(), chat_log)
        return result, response, spoken

    async def test_failure_is_returned_as_a_spoken_error(self) -> None:
        result, response, spoken = await self._spoken_turn(mistral_error("rate_limited"), _ChatLog())
        self.assertIs(result.response, response)
        self.assertEqual(result.conversation_id, "conv-1")
        self.assertEqual(spoken.await_args.args[2], "nl")  # the pipeline language
        self.assertEqual(response.async_set_error.call_args.args[1], "Druk.")

    async def test_partial_turn_is_replaced_by_the_error_reply(self) -> None:
        """The saved log must end in an assistant reply, not a tool result."""
        chat_log = _ChatLog(content=["system", "earlier user", "earlier reply", "user"])
        await self._spoken_turn(mistral_error("rate_limited"), chat_log)
        self.assertEqual(chat_log.content[:4], ["system", "earlier user", "earlier reply", "user"])
        self.assertEqual(len(chat_log.content), 5)
        last = chat_log.content[-1]
        self.assertEqual((last.agent_id, last.content), ("conversation.mistral", "Druk."))

    async def test_unexpected_exception_is_spoken_too(self) -> None:
        _, response, spoken = await self._spoken_turn(ValueError("bad json"), _ChatLog())
        self.assertIsInstance(spoken.await_args.args[1], ValueError)
        self.assertEqual(response.async_set_error.call_args.args[1], "Druk.")

    async def test_success_is_returned_unchanged(self) -> None:
        entity = _entity({})
        with patch.object(entity, "_async_converse", AsyncMock(return_value="ok")):
            self.assertEqual(await entity._async_handle_message(_user_input(), _ChatLog()), "ok")


class ToolErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_timeout_is_not_reported_as_mistral_unreachable(self) -> None:
        """A TimeoutError from an HA tool (raised by chat_log) passes through."""

        class _Resp:
            status = 200
            headers = MappingProxyType({})
            content = SimpleNamespace(iter_any=None)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

        entity = _entity({}, session=SimpleNamespace(request=lambda *a, **k: _Resp()))
        with self.assertRaises(TimeoutError):
            await entity._stream_and_collect(
                {"model": "m"}, _ChatLog(stream_error=TimeoutError()), _user_input()
            )


if __name__ == "__main__":
    unittest.main()
