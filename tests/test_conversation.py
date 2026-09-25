"""Tests for ``conversation.py`` helpers.

Covered: ``_sanitize`` (recursive JSON-safe coercion), ``_to_mistral_id``
(stable 9-char hex ID), and ``_async_stream_delta`` (SSE parser for
chat-completions streaming responses).
"""
from __future__ import annotations

import json
import unittest
from typing import Any, AsyncIterator
from unittest.mock import patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation.conversation import (  # noqa: E402
    _async_stream_delta,
    _format_tool,
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


class _FakeTool:
    """Stand-in for ``llm.Tool`` — only the attributes ``_format_tool`` reads."""

    def __init__(self, name: str, description: str, parameters: Any) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


class FormatToolTests(unittest.TestCase):
    """``_format_tool`` must always return a JSON-serializable schema.

    Regression coverage for #36: on HA 2026.9, ``voluptuous_openapi.convert()``
    left an HA-internal sentinel object (``_Unsupported``) inside the
    ``parameters`` schema for a selector type it couldn't represent. That
    object reached aiohttp's JSON encoder unresolved and crashed every
    Mistral request that offered a tool with such a parameter — a single
    unresponsive/unrecognized selector in the exposed HA tools took the
    whole conversation agent down.
    """

    def test_output_is_always_json_serializable(self) -> None:
        tool = _FakeTool("any_tool", "desc", parameters={})
        result = _format_tool(tool)
        json.dumps(result)  # must not raise, regardless of what convert() returns

    def test_unconvertible_schema_value_is_sanitized_not_left_raw(self) -> None:
        """Simulates convert() returning something that isn't a JSON schema.

        ``voluptuous_openapi.convert`` is mocked out in this test environment
        (see ``_ha_stubs``), so a bare call returns a ``MagicMock`` rather
        than a real dict — standing in for any non-dict value a schema
        conversion could produce, including HA 2026.9's probatio/
        voluptuous_openapi sentinel mismatch, which replaces the *entire*
        schema with a bare string (see ``_schema_to_openapi``'s docstring).
        The fix must never ship that value as `parameters` — Mistral
        requires a JSON Schema object there, and a stray string produces a
        422 with no exception anywhere in the HA log. It must fall back to
        a valid, if empty, object schema instead.
        """
        tool = _FakeTool("broken_tool", "desc", parameters={})
        result = _format_tool(tool)
        params = result["function"]["parameters"]
        self.assertEqual(params, {"type": "object", "properties": {}})
        json.dumps(result)

    def test_sentinel_replacing_whole_schema_falls_back_to_empty_object(self) -> None:
        """Reproduces the exact failure mode reported after #36 shipped.

        On HA 2026.9+, HA's custom_serializer returns probatio.UNSUPPORTED
        for a non-selector node, voluptuous_openapi.convert() doesn't
        recognize that as its own sentinel, and — because the serializer
        runs on the top-level schema node too — returns the bare sentinel
        as the WHOLE schema for nearly every tool, not just one field
        within it. ``_sanitize()`` alone (the #36 fix) turned that into
        the *valid JSON but wrong shape* string ``"UNSUPPORTED"``, which
        Mistral rejected with a 422 on every tool call. The fix must
        recognize a non-dict result and substitute an empty object schema.
        """
        tool = _FakeTool("basic-utilities__calculate", "Calculator", parameters={})

        with patch("voluptuous_openapi.convert", return_value="UNSUPPORTED"):
            result = _format_tool(tool)

        self.assertEqual(
            result["function"]["parameters"], {"type": "object", "properties": {}}
        )
        json.dumps(result)


    def test_probatio_is_preferred_over_voluptuous_openapi_when_available(self) -> None:
        """On HA 2026.9+, probatio.to_openapi() should be used, not
        voluptuous_openapi.convert() — it's the library that actually
        understands HA's own UNSUPPORTED sentinel correctly.
        """
        import types

        fake_probatio = types.ModuleType("probatio")
        fake_probatio.to_openapi = lambda schema, custom_serializer=None: {
            "type": "object",
            "properties": {"via": {"type": "string", "const": "probatio"}},
        }
        with patch.dict("sys.modules", {"probatio": fake_probatio}):
            tool = _FakeTool("any_tool", "desc", parameters={})
            result = _format_tool(tool)

        self.assertEqual(
            result["function"]["parameters"]["properties"]["via"]["const"], "probatio"
        )
        json.dumps(result)


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
