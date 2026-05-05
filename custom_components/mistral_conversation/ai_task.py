"""AI Task platform for Mistral AI."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import aiohttp
from homeassistant.components import conversation
from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

if TYPE_CHECKING:
    from pathlib import Path

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    MISTRAL_API_BASE,
)
from .conversation import (
    _async_stream_delta,
    _convert_chat_log_to_messages,
    _sanitize,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI task entity."""
    async_add_entities([MistralAITaskEntity(hass, config_entry)])


class MistralAITaskEntity(AITaskEntity):
    """Mistral AI task entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = (
        AITaskEntityFeature.GENERATE_DATA | AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_ai_task"

    @property
    def _runtime(self):
        return self.hass.data[DOMAIN][self._entry.entry_id]

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

    async def _read_attachment(self, path: Path, mime_type: str) -> dict[str, Any]:
        data = await self.hass.async_add_executor_job(path.read_bytes)
        b64 = base64.b64encode(data).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }

    async def _build_image_parts(self, task: GenDataTask) -> list[dict[str, Any]]:
        image_attachments = [
            a for a in (task.attachments or [])
            if a.mime_type.startswith("image/")
        ]

        async def _read_one(attachment) -> dict[str, Any] | None:
            try:
                return await self._read_attachment(attachment.path, attachment.mime_type)
            except OSError as err:
                _LOGGER.warning("Failed to read attachment %s: %s", attachment.path, err)
                return None

        results = await asyncio.gather(*[_read_one(a) for a in image_attachments])
        return [r for r in results if r is not None]

    @staticmethod
    def _build_response_format(task: GenDataTask) -> dict[str, Any] | None:
        if task.structure is None:
            return None
        json_schema = MistralAITaskEntity._structure_to_json_schema(task.structure)
        if json_schema is None:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": task.name.replace(" ", "_"),
                "schema": json_schema,
                "strict": False,
            },
        }

    @staticmethod
    def _structure_to_json_schema(structure) -> dict[str, Any] | None:
        try:
            from voluptuous_openapi import convert  # noqa: PLC0415
            return convert(structure)
        except ImportError:
            return None
        except (ValueError, TypeError):
            pass  # HA selector types are unhashable — fall through to manual conversion

        try:
            import voluptuous as vol  # noqa: PLC0415
            from homeassistant.helpers import selector as sel  # noqa: PLC0415

            properties: dict[str, Any] = {}
            required: list[str] = []

            for key, validator in structure.schema.items():
                name = key.schema if isinstance(key, (vol.Required, vol.Optional)) else str(key)
                if isinstance(key, vol.Required):
                    required.append(name)

                if isinstance(validator, sel.NumberSelector):
                    prop: dict[str, Any] = {"type": "number"}
                elif isinstance(validator, sel.BooleanSelector):
                    prop = {"type": "boolean"}
                elif isinstance(validator, sel.SelectSelector):
                    options = [
                        o if isinstance(o, str) else o.get("value", "")
                        for o in (validator.config.get("options") or [])
                    ]
                    prop = {"type": "string", "enum": options} if options else {"type": "string"}
                else:
                    prop = {"type": "string"}

                if hasattr(key, "description") and key.description:
                    prop["description"] = key.description
                properties[name] = prop

            result: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                result["required"] = required
            return result
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.warning("Could not build JSON schema from HA selectors: %s", err)
            return None

    @staticmethod
    def _inject_image_parts(
        messages: list[dict[str, Any]],
        image_parts: list[dict[str, Any]],
    ) -> None:
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = [{"type": "text", "text": msg["content"]}, *image_parts]
                return

    async def _async_generate_data(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> GenDataTaskResult:
        """Generate data from instructions using Mistral."""
        opts = self._entry.options
        model = opts.get(CONF_MODEL, DEFAULT_MODEL)
        max_tokens = int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = max(0.0, min(1.0, float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))))

        image_parts = await self._build_image_parts(task)
        messages = _convert_chat_log_to_messages(chat_log)
        if image_parts:
            self._inject_image_parts(messages, image_parts)

        payload: dict[str, Any] = _sanitize({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        })
        response_format = self._build_response_format(task)
        if response_format:
            payload["response_format"] = response_format

        try:
            async with self._runtime.session.post(
                f"{MISTRAL_API_BASE}/chat/completions",
                headers=self._runtime.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status == HTTPStatus.UNAUTHORIZED:
                    raise HomeAssistantError("Invalid Mistral AI API key")
                if resp.status == HTTPStatus.TOO_MANY_REQUESTS:
                    raise HomeAssistantError("Mistral AI rate limit exceeded")
                if resp.status >= HTTPStatus.BAD_REQUEST:
                    body = await resp.text()
                    raise HomeAssistantError(f"Mistral API error {resp.status}: {body}")

                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _async_stream_delta(resp),
                ):
                    pass
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err

        response_text = ""
        for c in reversed(chat_log.content):
            if isinstance(c, conversation.AssistantContent):
                response_text = c.content or ""
                break

        if task.structure is not None:
            try:
                return GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=json.loads(response_text),
                )
            except json.JSONDecodeError:
                _LOGGER.warning(
                    "Failed to parse AI task response as JSON: %s", response_text[:200]
                )

        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=response_text,
        )
