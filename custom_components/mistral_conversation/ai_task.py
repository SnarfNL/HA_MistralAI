"""AI Task platform for Mistral AI."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

if TYPE_CHECKING:
    from pathlib import Path

from .api import mistral_error, translate_stream
from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from .conversation import (
    _async_stream_delta,
    _convert_chat_log_to_messages,
    _sanitize,
    _schema_to_openapi,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI task entity."""
    async_add_entities([MistralAITaskEntity(hass, config_entry)])


def _parse_structured(structure: Any, response_text: str) -> Any:
    """Parse and validate a structured AI Task response, or raise.

    An automation that asked for a ``structure`` always gets data of that
    shape or a clear error, never a raw string or silently wrong data (#34).
    Mistral's own schema enforcement is not relied on ("strict": False is
    sent, see ``_build_response_format``), so the result is checked here.
    """
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as err:
        _LOGGER.warning(
            "Failed to parse AI task response as JSON: %s", response_text[:200]
        )
        raise mistral_error("json_parse_error") from err
    try:
        # Validate only; return the model's data unchanged, as before.
        structure(parsed)
    except Exception as err:
        _LOGGER.warning(
            "AI task response does not match requested structure: %s (response: %s)",
            err,
            response_text[:200],
        )
        raise mistral_error("structure_mismatch") from err
    return parsed


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
        return self._entry.runtime_data

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
        """Convert an HA voluptuous ``structure`` schema to a Mistral JSON schema.

        Uses ``_schema_to_openapi()`` — the same probatio/voluptuous_openapi
        conversion ``_format_tool()`` in conversation.py uses for Assist
        tools — so selector features like ``multiple: true`` come through
        as a JSON array instead of being silently flattened to a scalar
        (#34), and so a probatio/voluptuous_openapi sentinel mismatch on
        HA 2026.9+ can't replace the whole schema with a bare string
        without HA_MistralAI noticing (the tool-schema version of that bug
        was reported and fixed after #36 shipped).

        ``_schema_to_openapi()`` only ever returns ``{"type": "object",
        "properties": {}}`` when it couldn't build a real schema — either
        because HA's selector instances aren't hashable for a plain
        ``convert()``/``to_openapi()`` call, or because of the sentinel
        mismatch above. In that case we fall through to a manual per-key
        walk here, which knows enough about the common HA selector types
        (including ``multiple``) to still produce a useful schema instead
        of an empty one — needs no external schema library at all, just
        ``voluptuous`` and HA's own ``selector`` module, both always
        present.
        """
        from homeassistant.helpers import llm

        custom_serializer = getattr(llm, "selector_serializer", None) or getattr(
            llm, "_selector_serializer", None
        )
        result = _schema_to_openapi(
            structure, custom_serializer, log_context="ai_task structure"
        )
        if result != {"type": "object", "properties": {}}:
            return result

        try:
            import voluptuous as vol
            from homeassistant.helpers import selector as sel

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

                # A `multiple: true` selector (of any of the above types)
                # means the field is a list of that type, e.g.
                # TextSelector(multiple=True) -> array of strings, not a
                # single string (#34) — regardless of which scalar `prop`
                # was picked above.
                if getattr(validator, "config", None) and validator.config.get("multiple"):
                    prop = {"type": "array", "items": prop}

                if hasattr(key, "description") and key.description:
                    prop["description"] = key.description
                properties[name] = prop

            result: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                result["required"] = required
            return _sanitize(result)
        except Exception as err:  # noqa: BLE001 - HA selector shapes vary; log and fall back rather than crash
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

        async with self._runtime.client.chat_completions(
            payload, source="ai_task"
        ) as resp:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id,
                translate_stream(_async_stream_delta(resp)),
            ):
                pass

        response_text = ""
        for c in reversed(chat_log.content):
            if isinstance(c, conversation.AssistantContent):
                response_text = c.content or ""
                break

        if task.structure is not None:
            return GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=_parse_structured(task.structure, response_text),
            )

        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=response_text,
        )
