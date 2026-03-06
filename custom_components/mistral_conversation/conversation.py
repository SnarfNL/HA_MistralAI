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
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEB_SEARCH,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    MISTRAL_API_BASE,
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

#: JSON-safe scalar types that need no further processing
_JSON_SCALARS = (str, int, float, bool, type(None))


def _sanitize(obj: Any) -> Any:
    """Recursively make obj fully JSON-serializable.

    - Dict keys   → cast to str  (prevents OPT_NON_STR_KEYS)
    - Dict values → recurse
    - Lists       → recurse
    - Scalars     → pass through
    - Anything else (Python types, callables, vol validators, …)
                  → repr string  (prevents "Type is not JSON serializable")
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, _JSON_SCALARS):
        return obj
    # Anything non-serializable (function, type, voluptuous validator, …)
    return repr(obj)




def _format_tool(tool: llm.Tool, custom_serializer: Any = None) -> dict[str, Any]:
    """Convert an HA LLM tool to Mistral function-calling format.

    Uses voluptuous_openapi.convert() to produce proper JSON Schema,
    matching the pattern used by the official OpenAI and Anthropic
    integrations.
    """
    try:
        from voluptuous_openapi import convert

        parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    except Exception:  # pylint: disable=broad-except
        _LOGGER.debug(
            "Could not serialize tool parameters for '%s', using empty schema",
            tool.name,
        )
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(tool.description or ""),
            "parameters": parameters,
        },
    }


def _to_mistral_id(ha_id: str) -> str:
    """Convert an HA tool_call ID to a Mistral-compatible 9-char alphanumeric ID.

    Mistral requires tool_call IDs to be exactly 9 characters, a-z A-Z 0-9.
    HA's chat_log uses 26-char ULIDs. We hash deterministically so the same
    HA ID always maps to the same Mistral ID.
    """
    import hashlib
    return hashlib.md5(ha_id.encode()).hexdigest()[:9]


def _convert_chat_log_to_messages(
    chat_log: conversation.ChatLog,
) -> list[dict[str, Any]]:
    """Convert HA ChatLog content into Mistral chat completions messages.

    Mistral requires:
    - Each assistant message with tool_calls must be followed by exactly
      one tool result per tool_call, in order.
    - An assistant message with tool_calls should not have text content.
    """
    messages: list[dict[str, Any]] = []

    # Collect tool results indexed by tool_call_id for pairing
    tool_results: dict[str, conversation.ToolResultContent] = {}
    for content in chat_log.content:
        if isinstance(content, conversation.ToolResultContent):
            tool_results[content.tool_call_id] = content

    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": str(content.content)})

        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": str(content.content)})

        elif isinstance(content, conversation.AssistantContent):
            if content.tool_calls:
                # Assistant message with tool calls — no text content
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

                # Immediately append matching tool results in order
                for tc in content.tool_calls:
                    result = tool_results.get(tc.id)
                    if result:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _to_mistral_id(str(result.tool_call_id)),
                            "name": str(result.tool_name),
                            "content": json.dumps(
                                _sanitize(result.tool_result)
                                if isinstance(result.tool_result, (dict, list))
                                else result.tool_result
                            ),
                        })
            else:
                # Regular assistant message (no tool calls)
                messages.append({
                    "role": "assistant",
                    "content": str(content.content or ""),
                })

        # ToolResultContent is handled above paired with tool_calls
        # Skip standalone tool results to avoid duplicates

    return messages


async def _async_stream_delta(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[dict[str, Any]]:
    """Parse SSE stream from Mistral and yield delta dicts for chat_log."""
    buffer = b""
    current_tool_calls: dict[int, dict] = {}

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
                    if current_tool_calls:
                        for tc in current_tool_calls.values():
                            yield {
                                "tool_calls": [
                                    llm.ToolInput(
                                        id=tc["id"],
                                        tool_name=tc["name"],
                                        tool_args=json.loads(tc["arguments"] or "{}"),
                                    )
                                ]
                            }
                        current_tool_calls.clear()
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if delta.get("content"):
                    yield {"content": delta["content"]}

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

                if choice.get("finish_reason") in ("tool_calls", "stop") and current_tool_calls:
                    for tc in current_tool_calls.values():
                        yield {
                            "tool_calls": [
                                llm.ToolInput(
                                    id=tc["id"],
                                    tool_name=tc["name"],
                                    tool_args=json.loads(tc["arguments"] or "{}"),
                                )
                            ]
                        }
                    current_tool_calls.clear()


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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
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

        # Let HA build system prompt and expose tools
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

        # --- Web search path: use Agents/Conversations API ---------------
        if web_search and any(model.startswith(m) for m in AGENT_CAPABLE_MODELS):
            system_content = chat_log.content[0] if chat_log.content else None
            system_prompt = (
                system_content.content if hasattr(system_content, "content") else ""
            )
            try:
                ws_reply = await self._conversations_chat(
                    model=model,
                    system_prompt=system_prompt,
                    user_text=user_input.text,
                    conv_id=chat_log.conversation_id,
                )
            except HomeAssistantError as err:
                _LOGGER.debug(
                    "Web search failed, falling back to chat completions: %s", err
                )
                ws_reply = None

            if ws_reply:
                should_continue = (
                    continue_conversation_enabled and "?" in ws_reply
                )
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(ws_reply)
                return ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                    continue_conversation=should_continue,
                )

        # --- Standard path: chat completions with tool-call loop ---------
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_chat_log_to_messages(chat_log)

            _LOGGER.warning(
                "Mistral request iteration=%d, messages=%s",
                _iteration,
                json.dumps([{"role": m["role"], "has_tool_calls": "tool_calls" in m, "tool_call_id": m.get("tool_call_id")} for m in messages], default=str),
            )

            # Build payload — _sanitize ensures ALL keys are strings
            payload: dict[str, Any] = _sanitize({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            })
            if tools:
                payload["tools"] = tools  # already sanitized by _format_tool
                payload["tool_choice"] = "auto"

            try:
                await self._stream_and_collect(payload, chat_log, user_input)
            except HomeAssistantError:
                raise
            except Exception as err:
                _LOGGER.exception("Unexpected error in Mistral conversation")
                raise HomeAssistantError(
                    f"Unexpected error talking to Mistral: {err}"
                ) from err

            if not chat_log.unresponded_tool_results:
                break

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)

        # Apply continue_conversation flag if the reply ends with a question
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
        async with runtime.session.post(
            f"{MISTRAL_API_BASE}/agents",
            headers=runtime.headers,
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
            runtime.web_search_agent_id = agent_id
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
    ) -> None:
        """POST to Mistral, stream deltas into chat_log."""
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

                async for _content in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    _async_stream_delta(resp),
                ):
                    pass

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral AI request failed: %s", err)
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err
