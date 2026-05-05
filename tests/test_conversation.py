"""Tests for ``conversation.py`` helpers.

Covered: ``_sanitize`` (recursive JSON-safe coercion), ``_to_mistral_id``
(stable 9-char hex ID), and ``_async_stream_delta`` (SSE parser for
chat-completions streaming responses).
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, AsyncIterator

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation.conversation import (  # noqa: E402
    _async_stream_delta,
    _sanitize,
    _to_mistral_id,
)


class SanitizeTests(unittest.TestCase):
    """``_sanitize`` recursively makes objects JSON-serializable."""

    def test_primitives_pass_through(self) -> None:
        for value in ("hi", 1, 1.0, True, False, None):
            self.assertEqual(_sanitize(value), value)

    def test_dict_int_keys_become_str(self) -> None:
        self.assertEqual(_sanitize({1: "a", 2: "b"}), {"1": "a", "2": "b"})

    def test_nested_structures(self) -> None:
        self.assertEqual(
            _sanitize({"a": [1, {"b": 2}]}),
            {"a": [1, {"b": 2}]},
        )

    def test_non_json_object_repr_fallback(self) -> None:
        class Custom:
            def __repr__(self) -> str:
                return "<Custom>"

        self.assertEqual(_sanitize(Custom()), "<Custom>")

    def test_tuple_falls_back_to_repr(self) -> None:
        """Tuples aren't recognized as containers; they hit the repr branch.

        Documenting current behaviour. If a future change adds tuple support,
        this test will fail loudly so the contract change is conscious.
        """
        result = _sanitize((1, 2))
        self.assertEqual(result, repr((1, 2)))

    def test_mixed_dict_keys_and_values(self) -> None:
        self.assertEqual(
            _sanitize({1: [2, 3], None: "x"}),
            {"1": [2, 3], "None": "x"},
        )

    def test_output_is_json_serializable(self) -> None:
        data = {1: [None, True, 3.14, {"nested": "ok"}]}
        json.dumps(_sanitize(data))  # must not raise


class ToMistralIdTests(unittest.TestCase):
    """``_to_mistral_id`` returns a stable 9-char alphanumeric ID."""

    def test_length_is_nine(self) -> None:
        self.assertEqual(len(_to_mistral_id("anything")), 9)

    def test_only_hex_chars(self) -> None:
        result = _to_mistral_id("hello world")
        self.assertTrue(all(c in "0123456789abcdef" for c in result))

    def test_deterministic(self) -> None:
        self.assertEqual(_to_mistral_id("foo"), _to_mistral_id("foo"))

    def test_different_inputs_differ(self) -> None:
        self.assertNotEqual(_to_mistral_id("foo"), _to_mistral_id("bar"))

    def test_handles_unicode(self) -> None:
        self.assertEqual(len(_to_mistral_id("café")), 9)

    def test_known_md5_prefix_for_empty_string(self) -> None:
        # md5("") = d41d8cd98f00b204e9800998ecf8427e
        self.assertEqual(_to_mistral_id(""), "d41d8cd98")


# ---------------------------------------------------------------------------
# Helpers for SSE tests
# ---------------------------------------------------------------------------


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeContent(chunks)


def _data_frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _content_delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def _tool_delta(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    args: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    tc: dict[str, Any] = {"index": index}
    if call_id is not None:
        tc["id"] = call_id
    fn: dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        tc["function"] = fn
    delta: dict[str, Any] = {"tool_calls": [tc]}
    return {"choices": [{"delta": delta, "finish_reason": finish_reason}]}


class AsyncStreamDeltaTests(unittest.IsolatedAsyncioTestCase):
    """``_async_stream_delta`` parses Mistral chat completions SSE stream."""

    async def _collect(self, resp: Any) -> list[dict[str, Any]]:
        return [item async for item in _async_stream_delta(resp)]

    async def test_role_assistant_yielded_first(self) -> None:
        """Every stream MUST start with {"role": "assistant"} — without this
        HA's pipeline drops all content and tts_start_streaming never fires.
        """
        out = await self._collect(_FakeResponse([b"data: [DONE]\n\n"]))
        self.assertEqual(out, [{"role": "assistant"}])

    async def test_plain_text_content(self) -> None:
        chunks = [
            _data_frame(_content_delta("Hello")),
            _data_frame(_content_delta(" world")),
            b"data: [DONE]\n\n",
        ]
        out = await self._collect(_FakeResponse(chunks))
        self.assertEqual(
            out,
            [{"role": "assistant"}, {"content": "Hello"}, {"content": " world"}],
        )

    async def test_done_terminates(self) -> None:
        chunks = [
            _data_frame(_content_delta("X")),
            b"data: [DONE]\n\n",
            _data_frame(_content_delta("ignored")),
        ]
        out = await self._collect(_FakeResponse(chunks))
        self.assertEqual(out, [{"role": "assistant"}, {"content": "X"}])

    async def test_malformed_json_skipped(self) -> None:
        chunks = [
            b"data: not-valid-json\n\n",
            _data_frame(_content_delta("ok")),
            b"data: [DONE]\n\n",
        ]
        out = await self._collect(_FakeResponse(chunks))
        self.assertEqual(out, [{"role": "assistant"}, {"content": "ok"}])

    async def test_frame_split_across_wire_chunks(self) -> None:
        full = (
            _data_frame(_content_delta("Streaming"))
            + _data_frame(_content_delta(" works"))
            + b"data: [DONE]\n\n"
        )
        wire = [full[:5], full[5:20], full[20:50], full[50:]]
        out = await self._collect(_FakeResponse(wire))
        self.assertEqual(
            out,
            [{"role": "assistant"}, {"content": "Streaming"}, {"content": " works"}],
        )

    async def test_tool_call_accumulation_and_flush(self) -> None:
        """Tool-call fragments span multiple deltas; flushed on finish_reason."""
        chunks = [
            _data_frame(_tool_delta(index=0, call_id="c1", name="get_weather")),
            _data_frame(_tool_delta(index=0, args='{"location":')),
            _data_frame(_tool_delta(index=0, args='"Paris"}')),
            _data_frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            b"data: [DONE]\n\n",
        ]
        out = await self._collect(_FakeResponse(chunks))
        # Expect role marker, then exactly one yielded dict with one ToolInput
        self.assertEqual(out[0], {"role": "assistant"})
        self.assertEqual(len(out), 2)
        self.assertIn("tool_calls", out[1])
        tool_calls = out[1]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        tc = tool_calls[0]
        self.assertEqual(tc.id, "c1")
        self.assertEqual(tc.tool_name, "get_weather")
        self.assertEqual(tc.tool_args, {"location": "Paris"})

    async def test_tool_call_with_invalid_json_args_falls_back_to_empty(self) -> None:
        """Malformed tool arguments shouldn't crash; empty dict is the fallback."""
        chunks = [
            _data_frame(_tool_delta(index=0, call_id="c1", name="x", args="{not json")),
            _data_frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            b"data: [DONE]\n\n",
        ]
        out = await self._collect(_FakeResponse(chunks))
        self.assertEqual(out[0], {"role": "assistant"})
        self.assertEqual(out[1]["tool_calls"][0].tool_args, {})

    async def test_mixed_text_then_tool_call(self) -> None:
        """A text response followed by a tool call yields both, in order."""
        chunks = [
            _data_frame(_content_delta("Sure, ")),
            _data_frame(_content_delta("calling: ")),
            _data_frame(_tool_delta(index=0, call_id="c2", name="lights_off", args="{}")),
            _data_frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            b"data: [DONE]\n\n",
        ]
        out = await self._collect(_FakeResponse(chunks))
        self.assertEqual(out[0], {"role": "assistant"})
        self.assertEqual(out[1], {"content": "Sure, "})
        self.assertEqual(out[2], {"content": "calling: "})
        self.assertEqual(len(out), 4)
        self.assertEqual(out[3]["tool_calls"][0].tool_name, "lights_off")
        self.assertEqual(out[3]["tool_calls"][0].tool_args, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
