"""Tests for web-search routing.

Covers the two pure pieces of the routing decision:

* ``_resolve_trigger`` — optional trigger-phrase matching and query stripping.
* ``_filter_intercepted_tool`` — removing the synthetic ``web_search`` tool call
  from the delta stream before HA's ``chat_log`` sees it.

Plus ``_web_search_tool_def`` (shape of the advertised tool) and the
``_convert_chat_log_to_messages`` guard that drops content-less assistant turns
left behind by an intercepted call.
"""
# ruff: noqa: I001 - import order below is intentional: `_ha_stubs` must run
# before the `mistral_conversation` import so Home Assistant is stubbed first.
from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Any

from homeassistant.components import conversation as ha_conversation

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation.const import WEB_SEARCH_TOOL_NAME
from mistral_conversation.conversation import (
    _convert_chat_log_to_messages,
    _filter_intercepted_tool,
    _resolve_trigger,
    _web_search_tool_def,
)


class ResolveTriggerTests(unittest.TestCase):
    """``_resolve_trigger`` matches phrases and strips them from the query."""

    def test_empty_trigger_never_matches(self) -> None:
        self.assertEqual(_resolve_trigger("zoek op het weer", ""), (False, "zoek op het weer"))

    def test_whitespace_only_trigger_never_matches(self) -> None:
        self.assertEqual(_resolve_trigger("anything", "  ,  , "), (False, "anything"))

    def test_simple_match_strips_phrase(self) -> None:
        self.assertEqual(_resolve_trigger("zoek op het weer", "zoek op"), (True, "het weer"))

    def test_non_match_returns_text_unchanged(self) -> None:
        self.assertEqual(
            _resolve_trigger("zet de lamp aan", "zoek op, google"),
            (False, "zet de lamp aan"),
        )

    def test_case_insensitive(self) -> None:
        self.assertEqual(_resolve_trigger("ZOEK OP het weer", "zoek op"), (True, "het weer"))

    def test_leading_whitespace_ignored(self) -> None:
        self.assertEqual(_resolve_trigger("   zoek op iets", "zoek op"), (True, "iets"))

    def test_colon_after_phrase_stripped(self) -> None:
        self.assertEqual(_resolve_trigger("google: het weer", "google"), (True, "het weer"))

    def test_multiple_phrases_any_matches(self) -> None:
        trigger = "zoek op, zoek online, google"
        self.assertEqual(_resolve_trigger("google het weer", trigger), (True, "het weer"))
        self.assertEqual(_resolve_trigger("zoek online iets", trigger), (True, "iets"))

    def test_longest_match_wins(self) -> None:
        """A specific phrase must beat a prefix of itself."""
        self.assertEqual(
            _resolve_trigger("zoek online het weer", "zoek, zoek online"),
            (True, "het weer"),
        )

    def test_phrase_only_utterance_yields_empty_query(self) -> None:
        """Caller must treat this as 'no usable query' rather than searching ''."""
        self.assertEqual(_resolve_trigger("google", "google"), (True, ""))

    def test_phrase_must_be_at_start(self) -> None:
        self.assertEqual(
            _resolve_trigger("kun je google gebruiken", "google"),
            (False, "kun je google gebruiken"),
        )

    def test_phrases_are_trimmed(self) -> None:
        self.assertEqual(_resolve_trigger("google iets", "  google  "), (True, "iets"))


class WebSearchToolDefTests(unittest.TestCase):
    """The advertised tool must be a plain function tool Mistral accepts."""

    def test_is_function_type(self) -> None:
        self.assertEqual(_web_search_tool_def()["type"], "function")

    def test_name_matches_constant(self) -> None:
        self.assertEqual(
            _web_search_tool_def()["function"]["name"], WEB_SEARCH_TOOL_NAME
        )

    def test_requires_query_parameter(self) -> None:
        params = _web_search_tool_def()["function"]["parameters"]
        self.assertIn("query", params["properties"])
        self.assertEqual(params["required"], ["query"])

    def test_description_warns_off_device_control(self) -> None:
        """The description is the only steering signal the model gets."""
        desc = _web_search_tool_def()["function"]["description"].lower()
        self.assertIn("home", desc)


class _FakeToolInput:
    """Stand-in for llm.ToolInput with the attributes the filter reads."""

    def __init__(self, tool_name: str, tool_args: Any) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args


class FilterInterceptedToolTests(unittest.IsolatedAsyncioTestCase):
    """``_filter_intercepted_tool`` hides our synthetic tool from chat_log."""

    @staticmethod
    async def _gen(items: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        for item in items:
            yield item

    async def _run(
        self, items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        collected: list[str] = []
        out = [
            d
            async for d in _filter_intercepted_tool(
                self._gen(items), WEB_SEARCH_TOOL_NAME, collected
            )
        ]
        return out, collected

    async def test_non_tool_deltas_pass_through(self) -> None:
        items = [{"role": "assistant"}, {"content": "hallo"}]
        out, collected = await self._run(items)
        self.assertEqual(out, items)
        self.assertEqual(collected, [])

    async def test_web_search_call_removed_and_query_collected(self) -> None:
        items = [
            {"role": "assistant"},
            {"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "tour 2026"})]},
        ]
        out, collected = await self._run(items)
        # The delta is suppressed entirely — chat_log rejects empty tool_calls.
        self.assertEqual(out, [{"role": "assistant"}])
        self.assertEqual(collected, ["tour 2026"])

    async def test_other_tool_calls_are_preserved(self) -> None:
        keep = _FakeToolInput("HassTurnOn", {"name": "tafellamp"})
        out, collected = await self._run([{"tool_calls": [keep]}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool_calls"], [keep])
        self.assertEqual(collected, [])

    async def test_mixed_delta_keeps_ha_tool_drops_web_search(self) -> None:
        keep = _FakeToolInput("HassTurnOn", {"name": "tafellamp"})
        drop = _FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "weer"})
        out, collected = await self._run([{"tool_calls": [keep, drop]}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool_calls"], [keep])
        self.assertEqual(collected, ["weer"])

    async def test_query_is_stripped(self) -> None:
        items = [{"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "  weer  "})]}]
        _, collected = await self._run(items)
        self.assertEqual(collected, ["weer"])

    async def test_missing_query_is_dropped_not_collected(self) -> None:
        items = [{"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {})]}]
        out, collected = await self._run(items)
        self.assertEqual(out, [])
        self.assertEqual(collected, [])

    async def test_blank_query_is_dropped_not_collected(self) -> None:
        items = [{"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "   "})]}]
        out, collected = await self._run(items)
        self.assertEqual(out, [])
        self.assertEqual(collected, [])

    async def test_non_dict_args_do_not_raise(self) -> None:
        items = [{"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, "not-a-dict")]}]
        out, collected = await self._run(items)
        self.assertEqual(out, [])
        self.assertEqual(collected, [])

    async def test_multiple_searches_all_collected(self) -> None:
        items = [
            {"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "a"})]},
            {"tool_calls": [_FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "b"})]},
        ]
        out, collected = await self._run(items)
        self.assertEqual(out, [])
        self.assertEqual(collected, ["a", "b"])

    async def test_delta_metadata_preserved_when_filtering(self) -> None:
        keep = _FakeToolInput("HassTurnOn", {})
        drop = _FakeToolInput(WEB_SEARCH_TOOL_NAME, {"query": "q"})
        out, _ = await self._run([{"role": "assistant", "tool_calls": [keep, drop]}])
        self.assertEqual(out[0]["role"], "assistant")


class _Content:
    """Base for chat_log content stubs (the real classes are stub types)."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ChatLog:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


def _assistant(content: str | None, tool_calls: Any = None) -> Any:
    obj = _Content(content=content, tool_calls=tool_calls)
    obj.__class__ = type(
        "AssistantContentStub", (ha_conversation.AssistantContent,), {}
    )
    obj.content = content
    obj.tool_calls = tool_calls
    return obj


def _user(content: str) -> Any:
    obj = ha_conversation.UserContent()
    obj.content = content
    return obj


class ConvertChatLogEmptyAssistantTests(unittest.TestCase):
    """Content-less assistant turns must not reach the Mistral payload.

    An intercepted web_search call leaves an assistant turn with no text and no
    surviving tool calls. Sending it yields HTTP 400 "Assistant message must
    have either content or tool_calls, but not none." (code 3240).
    """

    def test_empty_assistant_turn_is_skipped(self) -> None:
        chat_log = _ChatLog([_user("wie won de tour"), _assistant(None, None)])
        messages = _convert_chat_log_to_messages(chat_log)
        self.assertEqual(messages, [{"role": "user", "content": "wie won de tour"}])

    def test_empty_string_assistant_turn_is_skipped(self) -> None:
        chat_log = _ChatLog([_user("hoi"), _assistant("", None)])
        messages = _convert_chat_log_to_messages(chat_log)
        self.assertEqual(len(messages), 1)

    def test_assistant_turn_with_text_is_kept(self) -> None:
        chat_log = _ChatLog([_user("hoi"), _assistant("hallo", None)])
        messages = _convert_chat_log_to_messages(chat_log)
        self.assertEqual(messages[-1], {"role": "assistant", "content": "hallo"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
