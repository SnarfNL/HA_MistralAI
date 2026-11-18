"""Tool results in the messages sent to Mistral (HA 2026.10 ``ToolResult``).

HA 2026.10 replaced ``ToolResultContent.tool_result`` (plain JSON) with
``ToolResultContent.result`` (an ``llm.ToolResult`` with ``data`` and
``error``). The old attribute became a deprecated shim that logs a warning on
every access and is removed in HA 2027.11; older HA versions only have
``tool_result``. Both shapes must produce a valid ``tool`` message.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from homeassistant.components import conversation
from homeassistant.helpers import llm

from custom_components.mistral_conversation.conversation import (
    _convert_chat_log_to_messages,
    _to_mistral_id,
    _tool_result_to_json,
)

HAS_TOOL_RESULT = hasattr(llm, "ToolResult")


class _NewStyleResult:
    """HA 2026.10+ shape; reading the deprecated ``tool_result`` fails the test."""

    def __init__(self, data: Any, error: bool = False) -> None:
        self.tool_call_id = "call-1"
        self.tool_name = "HassTurnOn"
        self.result = SimpleNamespace(data=data, error=error)

    @property
    def tool_result(self) -> Any:
        raise AssertionError("deprecated ToolResultContent.tool_result was read")


def _legacy_result(tool_result: Any) -> SimpleNamespace:
    """HA < 2026.10 shape: plain JSON on ``tool_result``, no ``result``."""
    return SimpleNamespace(
        tool_call_id="call-1", tool_name="HassTurnOn", tool_result=tool_result
    )


def _real_tool_result_content(data: dict[str, Any], error: bool = False) -> Any:
    """A real ToolResultContent for the installed HA version."""
    common = {"agent_id": "conversation.mistral", "tool_call_id": "call-1",
              "tool_name": "HassTurnOn"}
    if HAS_TOOL_RESULT:
        return conversation.ToolResultContent(
            **common, result=llm.ToolResult(data=data, error=error)
        )
    return conversation.ToolResultContent(**common, tool_result=data)


class ToolResultToJsonTests(unittest.TestCase):
    """``_tool_result_to_json`` handles both HA shapes."""

    def test_new_style_success(self) -> None:
        res = _NewStyleResult({"success": True})
        self.assertEqual(
            json.loads(_tool_result_to_json(res)),  # type: ignore[arg-type]
            {"data": {"success": True}, "error": False},
        )

    def test_new_style_error_flag_is_passed_on(self) -> None:
        res = _NewStyleResult({"error": "HomeAssistantError", "error_text": "boom"}, True)
        self.assertEqual(
            json.loads(_tool_result_to_json(res)),  # type: ignore[arg-type]
            {
                "data": {"error": "HomeAssistantError", "error_text": "boom"},
                "error": True,
            },
        )

    def test_new_style_non_json_data_is_sanitized(self) -> None:
        class Custom:
            def __repr__(self) -> str:
                return "<Custom>"

        res = _NewStyleResult({1: Custom()})
        self.assertEqual(
            json.loads(_tool_result_to_json(res)),  # type: ignore[arg-type]
            {"data": {"1": "<Custom>"}, "error": False},
        )

    def test_legacy_dict_is_sent_unchanged(self) -> None:
        res = _legacy_result({"speech": {"plain": {"speech": "Done"}}})
        self.assertEqual(
            json.loads(_tool_result_to_json(res)),  # type: ignore[arg-type]
            {"speech": {"plain": {"speech": "Done"}}},
        )

    def test_legacy_non_dict_values(self) -> None:
        for value in ("ok", 3, None, [1, {2: "b"}]):
            res = _legacy_result(value)
            expected = [1, {"2": "b"}] if isinstance(value, list) else value
            self.assertEqual(
                json.loads(_tool_result_to_json(res)),  # type: ignore[arg-type]
                expected,
            )


class ConvertChatLogToolResultTests(unittest.TestCase):
    """A full chat log with a tool call and its result, on the installed HA."""

    def _messages(self, result_content: Any) -> list[dict[str, Any]]:
        chat_log = SimpleNamespace(
            content=[
                conversation.UserContent(content="turn on the lamp"),
                conversation.AssistantContent(
                    agent_id="conversation.mistral",
                    tool_calls=[
                        llm.ToolInput(
                            id="call-1",
                            tool_name="HassTurnOn",
                            tool_args={"name": "lamp"},
                        )
                    ],
                ),
                result_content,
            ]
        )
        return _convert_chat_log_to_messages(chat_log)  # type: ignore[arg-type]

    def test_tool_message_is_built_from_real_content(self) -> None:
        messages = self._messages(_real_tool_result_content({"success": True}))
        tool_msg = messages[-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["name"], "HassTurnOn")
        self.assertEqual(tool_msg["tool_call_id"], _to_mistral_id("call-1"))
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], _to_mistral_id("call-1"))
        expected: Any = (
            {"data": {"success": True}, "error": False}
            if HAS_TOOL_RESULT
            else {"success": True}
        )
        self.assertEqual(json.loads(tool_msg["content"]), expected)

    @unittest.skipUnless(HAS_TOOL_RESULT, "llm.ToolResult exists from HA 2026.10")
    def test_no_deprecation_warning_on_ha_2026_10(self) -> None:
        """The deprecated ``tool_result`` shim must not be touched."""
        content = _real_tool_result_content({"error": "Invalid"}, error=True)
        with patch("homeassistant.helpers.frame.report_usage") as report_usage:
            messages = self._messages(content)
        report_usage.assert_not_called()
        self.assertEqual(
            json.loads(messages[-1]["content"]),
            {"data": {"error": "Invalid"}, "error": True},
        )
