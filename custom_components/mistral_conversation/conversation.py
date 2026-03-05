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
from homeassistant.const import CONF_LLM_HASS_API, EVENT_STATE_CHANGED, MATCH_ALL
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    AGENT_CAPABLE_MODELS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_WEB_SEARCH,
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
# Helpers: convert between HA chat_log and Mistral API formats
# ---------------------------------------------------------------------------

def _format_tool(tool: llm.Tool) -> dict[str, Any]:
    """Convert an HA LLM tool to Mistral function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters.schema if hasattr(tool.parameters, "schema") else {},
        },
    }


def _convert_chat_log_to_messages(
    chat_log: conversation.ChatLog,
) -> list[dict[str, Any]]:
    """Convert HA ChatLog content into Mistral chat completions messages."""
    messages: list[dict[str, Any]] = []
    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": content.content})
        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": content.content})
        elif isinstance(content, conversation.AssistantContent):
            msg: dict[str, Any] = {"role": "assistant"}
            if content.content:
                msg["content"] = content.content
            if content.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_args),
                        },
                    }
                    for tc in content.tool_calls
                ]
            if "content" not in msg and "tool_calls" not in msg:
                msg["content"] = ""
            messages.append(msg)
        elif isinstance(content, conversation.ToolResultContent):
            messages.append({
                "role": "tool",
                "tool_call_id": content.tool_call_id,
                "name": content.tool_name,
                "content": json.dumps(content.tool_result),
            })
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
                    # Flush any pending tool calls
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

                # Text content
                if delta.get("content"):
                    yield {"content": delta["content"]}

                # Tool calls (streamed incrementally)
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

                # If finish_reason is tool_calls or stop, flush tool calls
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

class MistralConversationEntity(ConversationEntity, conversation.AbstractConversationAgent):
    """Mistral AI conversation agent entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        # Set CONTROL feature if an LLM API is configured
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def _runtime(self):
        """Return the shared MistralRuntimeData for this entry."""
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
    # Main entry point — uses HA's native chat_log
    # ------------------------------------------------------------------
    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Process a conversation turn using HA's ChatLog and LLM API."""
        opts = self._entry.options

        # Let HA build the system prompt, expose tools, etc.
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                opts.get(CONF_LLM_HASS_API),
                opts.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Build Mistral tools from HA's LLM API
        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [_format_tool(tool) for tool in chat_log.llm_api.tools]

        model = opts.get(CONF_MODEL, DEFAULT_MODEL)
        max_tokens = int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = max(0.0, min(1.0, float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))))
        web_search = opts.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)

        # --- Web search path: use Agents/Conversations API ---------------
        if web_search and any(model.startswith(m) for m in AGENT_CAPABLE_MODELS):
            _LOGGER.warning(
                "Web search enabled — using Conversations API (model=%s)", model
            )
            system_content = chat_log.content[0] if chat_log.content else None
            system_prompt = system_content.content if hasattr(system_content, "content") else ""
            try:
                ws_reply = await self._conversations_chat(
                    model=model,
                    system_prompt=system_prompt,
                    user_text=user_input.text,
                    conv_id=chat_log.conversation_id,
                )
            except HomeAssistantError as err:
                _LOGGER.warning("Web search failed, falling back to chat completions: %s", err)
                ws_reply = None

            if ws_reply:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(ws_reply)
                return ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )
        else:
            _LOGGER.debug(
                "Web search skipped (web_search=%s, model=%s, capable=%s)",
                web_search, model, AGENT_CAPABLE_MODELS,
            )

        # --- Standard path: chat completions with tool calling -----------
        # Tool-call loop: model may request tools, we execute and re-send
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_chat_log_to_messages(chat_log)

            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            try:
                result = await self._stream_and_collect(
                    payload, chat_log, user_input
                )
                if isinstance(result, ConversationResult):
                    return result
            except HomeAssistantError:
                raise
            except Exception as err:
                _LOGGER.exception("Unexpected error in Mistral conversation")
                raise HomeAssistantError(
                    f"Unexpected error talking to Mistral: {err}"
                ) from err

            # If no tool results are pending, we're done
            if not chat_log.unresponded_tool_results:
                break

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    # ------------------------------------------------------------------
    # Legacy API (< 2024.6) — redirect to new path
    # ------------------------------------------------------------------
    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Legacy entry point — create a minimal chat_log and delegate."""
        # On older HA versions, _async_handle_message won't be called.
        # Fall back to a simple non-tool-calling path.
        return await self._legacy_process(user_input)

    async def _legacy_process(self, user_input: ConversationInput) -> ConversationResult:
        """Simple non-chat_log fallback for older HA versions."""
        opts = self._entry.options
        runtime = self._runtime
        model = opts.get(CONF_MODEL, DEFAULT_MODEL)
        max_tokens = int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = max(0.0, min(1.0, float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))))

        messages = [
            {"role": "system", "content": opts.get(CONF_PROMPT, "You are a helpful assistant.")},
            {"role": "user", "content": user_input.text},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with runtime.session.post(
                f"{MISTRAL_API_BASE}/chat/completions",
                headers=runtime.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise HomeAssistantError(f"Mistral API error {resp.status}: {body}")
                data = await resp.json()
                reply = data["choices"][0]["message"]["content"].strip()
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(reply)
        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id or "legacy",
        )

    # ------------------------------------------------------------------
    # Agents / Conversations API for web search
    # ------------------------------------------------------------------
    async def _ensure_web_search_agent(self, model: str, system_prompt: str) -> str:
        """Create (or reuse) a Mistral Agent with web_search enabled."""
        runtime = self._runtime
        if runtime.web_search_agent_id:
            return runtime.web_search_agent_id

        payload = {
            "model": model,
            "name": "HA Mistral Web Search",
            "description": "Home Assistant conversation agent with web search",
            "instructions": system_prompt,
            "tools": [{"type": "web_search"}],
            "completion_args": {"temperature": 0.3, "top_p": 0.95},
        }
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

        # Check if we have an existing Mistral conversation_id
        mistral_conv_key = f"_ws_conv_{conv_id}"
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
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise HomeAssistantError(
                    f"Mistral Conversations API error {resp.status}: {body}"
                )
            data = await resp.json()

        # Store conversation_id for follow-ups
        new_conv_id = data.get("conversation_id") or data.get("id")
        if new_conv_id:
            if not hasattr(runtime, "_ws_convs"):
                runtime._ws_convs = {}
            runtime._ws_convs[conv_id] = new_conv_id

        # Extract text from response
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
    ) -> ConversationResult | None:
        """POST to Mistral, stream deltas into chat_log, handle errors."""
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
                    raise HomeAssistantError(f"Mistral API error {resp.status}: {body}")

                # Stream deltas into chat_log — HA handles tool execution
                async for _content in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    _async_stream_delta(resp),
                ):
                    pass  # chat_log accumulates content internally

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral AI request failed: %s", err)
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err

        return None

    @staticmethod
    def _new_id() -> str:
        from homeassistant.util import ulid
        return ulid.ulid_now()
