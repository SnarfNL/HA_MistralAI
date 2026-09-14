"""Conversation platform for Mistral AI."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

import aiohttp
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    AGENT_CAPABLE_MODELS,
    CONF_CONTINUE_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_MODE,
    CONF_WEB_SEARCH_TRIGGER,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEB_SEARCH,
    DEFAULT_WEB_SEARCH_MODE,
    DEFAULT_WEB_SEARCH_TRIGGER,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    MISTRAL_API_BASE,
    WEB_SEARCH_MODE_ALWAYS,
    WEB_SEARCH_TOOL_NAME,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI conversation entity."""
    async_add_entities([MistralConversationEntity(hass, config_entry)])


# ---------------------------------------------------------------------------
# Payload sanitizer
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively make obj fully JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def _schema_to_openapi(
    schema: Any,
    custom_serializer: Any = None,
    *,
    log_context: str = "schema",
) -> dict[str, Any]:
    """Convert a voluptuous/HA-selector schema to an OpenAPI/JSON-Schema dict.

    HA core 2026.9+ serializes schemas with ``probatio`` instead of
    ``voluptuous_openapi``. HA's own custom_serializer (``llm.selector_serializer``,
    what ``chat_log.llm_api.custom_serializer`` hands us) now returns
    ``probatio.UNSUPPORTED`` for any node it doesn't recognize as a
    selector. ``voluptuous_openapi.convert()`` checks a serializer's return
    value against its *own* ``UNSUPPORTED`` sentinel first — a different
    object — so it takes ``probatio.UNSUPPORTED`` as a valid serialized
    value instead of falling back to its normal conversion. Because the
    custom_serializer runs on every node including the top-level schema
    itself, this doesn't corrupt one odd field: it replaces the *entire*
    schema with the bare sentinel for every tool that isn't itself a single
    selector (i.e. nearly all of them) — see #36's follow-up report.
    ``probatio.to_openapi()`` checks against probatio's own sentinel
    correctly and is tried first; ``voluptuous_openapi.convert()`` remains
    the fallback for HA versions that never shipped probatio.

    The result is always sanitized — a leftover non-JSON object anywhere
    inside it degrades to a harmless string instead of reaching aiohttp's
    JSON encoder unresolved (the original #36 crash) — and is always a
    dict: if serialization still produces something else (a bare sentinel,
    an unexpected type), ``empty_schema`` is returned instead of shipping a
    schema Mistral will silently reject with a 422 that never surfaces in
    the HA log as an exception.
    """
    empty_schema: dict[str, Any] = {"type": "object", "properties": {}}

    try:
        try:
            from probatio import to_openapi as convert  # noqa: PLC0415
        except ImportError:
            from voluptuous_openapi import convert  # noqa: PLC0415
        result = convert(schema, custom_serializer=custom_serializer)
    except Exception:  # pylint: disable=broad-except
        _LOGGER.debug("Could not serialize %s, using empty schema", log_context)
        return empty_schema

    result = _sanitize(result)
    if not isinstance(result, dict):
        _LOGGER.warning(
            "%s serialized to a non-dict (%r) instead of a JSON schema object "
            "— using empty schema",
            log_context,
            result,
        )
        return empty_schema
    return result


def _format_tool(tool: llm.Tool, custom_serializer: Any = None) -> dict[str, Any]:
    """Convert an HA LLM tool to Mistral function-calling format."""
    parameters = _schema_to_openapi(
        tool.parameters,
        custom_serializer,
        log_context=f"tool parameters for '{tool.name}'",
    )
    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(tool.description or ""),
            "parameters": parameters,
        },
    }


def _to_mistral_id(ha_id: str) -> str:
    """Convert an HA tool_call ID to a Mistral-compatible 9-char alphanumeric ID."""
    import hashlib
    return hashlib.md5(ha_id.encode()).hexdigest()[:9]


def _web_search_tool_def() -> dict[str, Any]:
    """Return the synthetic ``web_search`` function tool offered to the model.

    Mistral's built-in ``{"type": "web_search"}`` tool is rejected by
    /v1/chat/completions, so web search is advertised as a normal function and
    serviced by us against the Agents API. The description is what steers the
    model, so it states both when to use it and when not to — without it the
    model reaches for search on questions it can answer locally.
    """
    return {
        "type": "function",
        "function": {
            "name": WEB_SEARCH_TOOL_NAME,
            "description": (
                "Search the public web for information you do not already know: "
                "current events, news, sports results, prices, opening hours, or "
                "facts that change over time. Do NOT use this to control or query "
                "devices in this home, and do not use it for general knowledge you "
                "can already answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Make it self-contained — resolve any "
                            "pronouns or context from the conversation first."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }


def _resolve_trigger(text: str, trigger: str) -> tuple[bool, str]:
    """Match ``text`` against comma-separated trigger phrases.

    Returns ``(matched, query)``. When matched, ``query`` is ``text`` with the
    trigger phrase removed (a leading ``:`` or whitespace after it is stripped
    too). If several phrases match, the LONGEST wins so that "zoek online"
    beats a bare "zoek". Matching is case-insensitive and ignores surrounding
    whitespace. An empty ``trigger`` never matches.

    A stripped query can come back empty (utterance was only the phrase, e.g.
    just "google"); callers must treat that as "no usable query".
    """
    phrases = [p.strip().lower() for p in (trigger or "").split(",") if p.strip()]
    if not phrases:
        return False, text

    stripped = text.strip()
    lowered = stripped.lower()
    matches = [p for p in phrases if lowered.startswith(p)]
    if not matches:
        return False, text

    matched = max(matches, key=len)
    return True, stripped[len(matched):].lstrip(" :").strip()


def _convert_chat_log_to_messages(
    chat_log: conversation.ChatLog,
) -> list[dict[str, Any]]:
    """Convert HA ChatLog content into Mistral chat completions messages."""
    messages: list[dict[str, Any]] = []

    tool_results: dict[str, conversation.ToolResultContent] = {
        c.tool_call_id: c
        for c in chat_log.content
        if isinstance(c, conversation.ToolResultContent)
    }

    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": str(content.content)})

        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": str(content.content)})

        elif isinstance(content, conversation.AssistantContent):
            # Skip turns that carry neither text nor tool calls. Mistral rejects
            # those with "Assistant message must have either content or
            # tool_calls, but not none." (HTTP 400, code 3240). A turn like this
            # occurs when the only tool call in it was intercepted by us (see
            # the web_search interception in _async_handle_message).
            if not content.content and not content.tool_calls:
                continue

            if content.tool_calls:
                all_have_results = all(tc.id in tool_results for tc in content.tool_calls)
                if not all_have_results:
                    if content.content:
                        messages.append({"role": "assistant", "content": str(content.content)})
                    continue

                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": _to_mistral_id(str(tc.id)),
                            "type": "function",
                            "function": {
                                "name": str(tc.tool_name),
                                "arguments": json.dumps(
                                    _sanitize(tc.tool_args) if isinstance(tc.tool_args, dict)
                                    else tc.tool_args
                                ),
                            },
                        }
                        for tc in content.tool_calls
                    ],
                }
                messages.append(msg)

                for tc in content.tool_calls:
                    if res := tool_results.get(tc.id):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _to_mistral_id(str(res.tool_call_id)),
                            "name": str(res.tool_name),
                            "content": json.dumps(
                                _sanitize(res.tool_result)
                                if isinstance(res.tool_result, (dict, list))
                                else res.tool_result
                            ),
                        })
            else:
                messages.append({"role": "assistant", "content": str(content.content or "")})

    return messages


# ---------------------------------------------------------------------------
# SSE stream parser
#
# HA 2026.4 chat_log.async_add_delta_content_stream does:
#   if "role" not in delta: ...
# which means it ALWAYS expects dicts, never plain str/ToolInput.
#
# The first delta of every assistant turn MUST be {"role": "assistant"}.
# HA's pipeline (assist_pipeline/pipeline.py chat_log_delta_listener) gates
# all forwarding to tts_input_stream on chat_log_role == "assistant". Without
# the role yield, content deltas are silently dropped by the pipeline,
# tts_start_streaming never fires, and TTS streaming is dead. The tool-call
# loop in _async_handle_message calls _stream_and_collect once per round, so
# a fresh role yield per call correctly signals each new assistant message.
#
# Each subsequent delta carries exactly one of:
#   {"content": "text string"}      — text delta
#   {"tool_calls": [ToolInput(...)]} — completed tool call
# Never combined in one dict — that previously caused
# "can only concatenate str (not list)" inside chat_log.
# ---------------------------------------------------------------------------

async def _async_stream_delta(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[dict[str, Any], None]:
    """Parse SSE stream from Mistral and yield delta dicts for HA's chat_log.

    Yields, in order:
      {"role": "assistant"}           — start of a new assistant message
      {"content": str}                — text content delta (zero or more)
      {"tool_calls": [llm.ToolInput]} — one completed tool call per yield
                                        (zero or more, after content)

    HA 2026.4 does `if "role" not in delta` on each item, so plain
    str/ToolInput objects will raise TypeError — dicts are required.
    """
    buffer = b""
    current_tool_calls: dict[int, dict] = {}

    # Required first delta — see module-level note above.
    yield {"role": "assistant"}

    async def _flush() -> AsyncGenerator[dict[str, Any], None]:
        """Yield each buffered tool call as its own dict and clear the buffer."""
        for tc in current_tool_calls.values():
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=tc["id"],
                        tool_name=tc["name"],
                        tool_args=args,
                    )
                ]
            }
        current_tool_calls.clear()

    async for raw_chunk in resp.content.iter_any():
        buffer += raw_chunk
        while b"\n\n" in buffer:
            frame, buffer = buffer.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    async for item in _flush():
                        yield item
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                # Yield text delta — always a separate dict, never mixed with tool_calls
                if delta.get("content"):
                    yield {"content": str(delta["content"])}

                # Accumulate streaming tool call fragments
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        else:
                            if tc_delta.get("id"):
                                current_tool_calls[idx]["id"] = tc_delta["id"]
                            if tc_delta.get("function", {}).get("name"):
                                current_tool_calls[idx]["name"] = tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            current_tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]

                # Flush each completed tool call as its own dict
                if choice.get("finish_reason") in ("tool_calls", "stop") and current_tool_calls:
                    async for item in _flush():
                        yield item


async def _filter_intercepted_tool(
    stream: AsyncGenerator[dict[str, Any], None],
    tool_name: str,
    collected: list[str],
) -> AsyncGenerator[dict[str, Any], None]:
    """Strip calls to ``tool_name`` from a delta stream, collecting their queries.

    The named tool is one we service ourselves (web search via the Agents API),
    so HA must never see it — it isn't in HA's LLM API and executing it would
    raise. Each intercepted call's ``query`` argument is appended to
    ``collected``; a call with no usable query is dropped silently.

    Deltas that end up with an empty ``tool_calls`` list are suppressed entirely,
    since chat_log treats an empty list as a malformed delta.
    """
    async for delta in stream:
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            yield delta
            continue

        kept = []
        for tc in tool_calls:
            if getattr(tc, "tool_name", None) != tool_name:
                kept.append(tc)
                continue
            args = getattr(tc, "tool_args", None) or {}
            query = args.get("query") if isinstance(args, dict) else None
            if isinstance(query, str) and query.strip():
                collected.append(query.strip())
            else:
                _LOGGER.debug("Ignoring %s call without a usable query", tool_name)

        if kept:
            yield {**delta, "tool_calls": kept}


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class MistralConversationEntity(ConversationEntity):
    """Mistral AI conversation agent entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def _runtime(self):
        return self.hass.data[DOMAIN][self._entry.entry_id]

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    @property
    def device_info(self) -> DeviceInfo:
        model = self._entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_conversation")},
            name="Mistral AI Conversation",
            manufacturer="Mistral AI",
            model=model,
            entry_type=DeviceEntryType.SERVICE,
            configuration_url="https://console.mistral.ai",
        )

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Process a conversation turn using HA's ChatLog and LLM API."""
        opts = self._entry.options
        continue_conversation_enabled = opts.get(
            CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION
        )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                opts.get(CONF_LLM_HASS_API),
                opts.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        model = opts.get(CONF_MODEL, DEFAULT_MODEL)
        max_tokens = int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = max(0.0, min(1.0, float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))))
        web_search = opts.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)
        web_search_mode = opts.get(CONF_WEB_SEARCH_MODE, DEFAULT_WEB_SEARCH_MODE)
        web_search_trigger = opts.get(
            CONF_WEB_SEARCH_TRIGGER, DEFAULT_WEB_SEARCH_TRIGGER
        )

        # Web search needs an agent-capable model — it is only reachable through
        # the Agents/Conversations API.
        web_search_available = web_search and any(
            model.startswith(m) for m in AGENT_CAPABLE_MODELS
        )

        # Trigger phrases, when configured, are leading: a match goes straight to
        # the Agents API and a non-match skips web search for this turn. An empty
        # trigger list (the default) leaves the decision to web_search_mode.
        trigger_matched, trigger_query = _resolve_trigger(
            user_input.text, web_search_trigger if web_search_available else ""
        )
        has_trigger = bool((web_search_trigger or "").strip()) and web_search_available

        # Offer web search as a tool only when the model is the one deciding.
        offer_web_search_tool = (
            web_search_available
            and not has_trigger
            and web_search_mode != WEB_SEARCH_MODE_ALWAYS
        )
        if offer_web_search_tool:
            tools = (tools or []) + [_web_search_tool_def()]

        # --- Direct web search path: Agents/Conversations API -------------
        # Used when a trigger phrase matched, or in legacy "always" mode. This
        # path carries no HA tools, so device control cannot happen on it.
        use_direct_web_search = web_search_available and (
            trigger_matched
            or (not has_trigger and web_search_mode == WEB_SEARCH_MODE_ALWAYS)
        )
        if use_direct_web_search:
            # A trigger-only utterance ("google") leaves nothing to search for;
            # fall through to the normal path rather than querying for "".
            search_text = trigger_query if trigger_matched else user_input.text
            if not search_text.strip():
                _LOGGER.debug(
                    "Web search trigger matched but query was empty; "
                    "falling through to chat completions"
                )
            else:
                system_content = chat_log.content[0] if chat_log.content else None
                system_prompt = (
                    system_content.content if hasattr(system_content, "content") else ""
                )
                try:
                    ws_reply = await self._conversations_chat(
                        model=model,
                        system_prompt=system_prompt,
                        user_text=search_text,
                        conv_id=chat_log.conversation_id,
                    )
                except HomeAssistantError as err:
                    _LOGGER.debug(
                        "Web search failed, falling back to chat completions: %s", err
                    )
                    ws_reply = None

                if ws_reply:
                    should_continue = continue_conversation_enabled and "?" in ws_reply
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_speech(ws_reply)
                    return ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                        continue_conversation=should_continue,
                    )

        # --- Standard path: chat completions with tool-call loop ---------
        # Synthetic messages holding web-search results. They live only in the
        # outgoing payload, never in chat_log, so conversation history stays a
        # clean user/assistant transcript.
        injected: list[dict[str, Any]] = []
        # Queries the model asked us to search for, captured from the stream.
        captured_searches: list[str] = []

        for _iteration in range(MAX_TOOL_ITERATIONS):
            raw_payload: dict[str, Any] = {
                "model": model,
                "messages": _convert_chat_log_to_messages(chat_log) + injected,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            if tools:
                raw_payload["tools"] = tools
                raw_payload["tool_choice"] = "auto"
            # Sanitize AFTER tools are attached. Tool parameter schemas come
            # from voluptuous_openapi.convert() via HA's LLM API and can
            # contain HA-internal sentinel objects (e.g. a schema-conversion
            # placeholder for a selector type _sanitize has never seen) for
            # constructs voluptuous_openapi doesn't know how to represent.
            # Sanitizing before attaching tools let such objects reach
            # aiohttp's JSON encoder untouched, crashing every turn that
            # exposed that tool (#36). Sanitizing the whole payload in one
            # pass, after tools are in it, is the same defense already
            # applied to messages/tool results elsewhere in this file.
            payload: dict[str, Any] = _sanitize(raw_payload)

            captured_searches.clear()
            try:
                await self._stream_and_collect(
                    payload,
                    chat_log,
                    user_input,
                    intercept_tool=(
                        WEB_SEARCH_TOOL_NAME if offer_web_search_tool else None
                    ),
                    intercepted=captured_searches,
                )
            except HomeAssistantError:
                raise
            except Exception as err:
                _LOGGER.exception("Unexpected error in Mistral conversation")
                raise HomeAssistantError(
                    f"Unexpected error talking to Mistral: {err}"
                ) from err

            # The model asked for a web search: service it against the Agents
            # API and hand the result back on the next round. Checked before
            # unresponded_tool_results, because an intercepted call leaves no
            # pending tool result behind.
            if captured_searches:
                query = captured_searches[0]
                _LOGGER.debug("Model requested web search: %s", query)
                try:
                    results = await self._conversations_chat(
                        model=model,
                        system_prompt=(
                            "Answer the search query factually and concisely. "
                            "Cite nothing; plain prose only."
                        ),
                        user_text=query,
                        conv_id=chat_log.conversation_id,
                    )
                except HomeAssistantError as err:
                    _LOGGER.debug("Web search lookup failed: %s", err)
                    results = ""

                injected.append({
                    "role": "user",
                    "content": (
                        f'Web search results for "{query}":\n'
                        f"{results or 'No results were returned.'}\n\n"
                        "Using these results, answer my original question directly. "
                        "Do not mention that you performed a search."
                    ),
                })
                # Withdraw the tool so a turn can trigger at most one search and
                # cannot loop on it.
                tools = [
                    t
                    for t in (tools or [])
                    if t.get("function", {}).get("name") != WEB_SEARCH_TOOL_NAME
                ] or None
                offer_web_search_tool = False
                continue

            if not chat_log.unresponded_tool_results:
                break

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)

        if continue_conversation_enabled:
            reply_text = result.response.speech.get("plain", {}).get("speech", "")
            if "?" in reply_text:
                return ConversationResult(
                    response=result.response,
                    conversation_id=result.conversation_id,
                    continue_conversation=True,
                )

        return result

    # ------------------------------------------------------------------
    # Agents / Conversations API for web search
    # ------------------------------------------------------------------
    async def _ensure_web_search_agent(self, model: str, system_prompt: str) -> str:
        """Create (or reuse) a Mistral Agent with web_search enabled."""
        runtime = self._runtime
        if runtime.web_search_agent_id:
            return runtime.web_search_agent_id

        payload = _sanitize({
            "model": model,
            "name": "HA Mistral Web Search",
            "description": "Home Assistant conversation agent with web search",
            "instructions": system_prompt,
            "tools": [{"type": "web_search"}],
            "completion_args": {"temperature": 0.3, "top_p": 0.95},
        })
        async with self._runtime.session.post(
            f"{MISTRAL_API_BASE}/agents",
            headers=self._runtime.headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise HomeAssistantError(
                    f"Failed to create Mistral web-search agent: {resp.status} {body}"
                )
            data = await resp.json()
            agent_id = data["id"]
            self._runtime.web_search_agent_id = agent_id
            _LOGGER.debug("Created Mistral web-search agent: %s", agent_id)
            return agent_id

    async def _conversations_chat(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        conv_id: str,
    ) -> str:
        """Use the Mistral Conversations API (beta) with web search."""
        runtime = self._runtime
        agent_id = await self._ensure_web_search_agent(model, system_prompt)

        mistral_conv_id = getattr(runtime, "_ws_convs", {}).get(conv_id)
        if mistral_conv_id:
            url = f"{MISTRAL_API_BASE}/conversations/{mistral_conv_id}"
            payload: dict[str, Any] = {"inputs": user_text}
        else:
            url = f"{MISTRAL_API_BASE}/conversations"
            payload = {"agent_id": agent_id, "inputs": user_text}

        async with runtime.session.post(
            url,
            headers=runtime.headers,
            json=_sanitize(payload),
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise HomeAssistantError(
                    f"Mistral Conversations API error {resp.status}: {body}"
                )
            data = await resp.json()

        new_conv_id = data.get("conversation_id") or data.get("id")
        if new_conv_id:
            if not hasattr(runtime, "_ws_convs"):
                runtime._ws_convs = {}
            runtime._ws_convs[conv_id] = new_conv_id

        parts: list[str] = []
        for output in data.get("outputs", []):
            if output.get("type") == "tool.execution":
                continue
            content = output.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        parts.append(chunk.get("text", ""))
        return "".join(parts).strip() or data.get("message", "")

    # ------------------------------------------------------------------
    # Streaming HTTP + chat_log delta integration
    # ------------------------------------------------------------------
    async def _stream_and_collect(
        self,
        payload: dict[str, Any],
        chat_log: conversation.ChatLog,
        user_input: ConversationInput,
        intercept_tool: str | None = None,
        intercepted: list[str] | None = None,
    ) -> None:
        """POST to Mistral, stream deltas into chat_log.

        When ``intercept_tool`` is set, calls to that tool are removed from the
        delta stream before chat_log sees them and their ``query`` argument is
        appended to ``intercepted``. HA would otherwise try to execute a tool
        that isn't in its LLM API and raise.
        """
        runtime = self._runtime
        try:
            async with runtime.session.post(
                f"{MISTRAL_API_BASE}/chat/completions",
                headers=runtime.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status == 401:
                    raise HomeAssistantError("Invalid Mistral AI API key")
                if resp.status == 429:
                    raise HomeAssistantError("Mistral AI rate limit exceeded")
                if resp.status >= 400:
                    body = await resp.text()
                    _LOGGER.error(
                        "Mistral API HTTP %s — model=%s body=%s",
                        resp.status, payload.get("model"), body,
                    )
                    raise HomeAssistantError(
                        f"Mistral API error {resp.status}: {body}"
                    )

                stream = _async_stream_delta(resp)
                if intercept_tool is not None:
                    stream = _filter_intercepted_tool(
                        stream, intercept_tool, intercepted if intercepted is not None else []
                    )

                async for _content in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    stream,
                ):
                    pass

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral AI request failed: %s", err)
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err
