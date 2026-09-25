"""The Mistral AI Conversation integration."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ._web_search import WebSearchConversations
from .api import MistralClient
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS = ["ai_task", "button", "conversation", "stt", "tts"]


@dataclass
class MistralRuntimeData:
    """Shared runtime data for a config entry."""

    client: MistralClient
    # The last errors from calls to Mistral, for the diagnostics file.
    errors: deque[dict[str, Any]]
    # HA conversation -> Mistral conversation, for follow-ups on the direct
    # web-search route. Bounded and expiring.
    web_search_convs: WebSearchConversations = field(
        default_factory=WebSearchConversations
    )
    # The TTS entity, registered while it is loaded so the refresh-voices
    # button can reach it.
    tts_entity: Any | None = field(default=None)
    # Last successful GET /v1/models result; None until fetched.
    models: list[dict[str, Any]] | None = field(default=None)


type MistralConfigEntry = ConfigEntry[MistralRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Set up Mistral AI Conversation from a config entry."""
    api_key = entry.data[CONF_API_KEY]
    session = async_get_clientsession(hass)

    error, detail = await MistralClient.validate_key(session, api_key)
    if error == "invalid_auth":
        raise ConfigEntryAuthFailed("Invalid Mistral AI API key")
    if error:
        raise ConfigEntryNotReady(f"Cannot connect to Mistral AI: {detail}")

    errors: deque[dict[str, Any]] = deque(maxlen=10)
    entry.runtime_data = MistralRuntimeData(
        client=MistralClient(hass, entry, session, api_key, errors), errors=errors
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload a config entry.

    HA drops ``entry.runtime_data`` only after the platforms have unloaded, so
    entities can finish their teardown (closing streams, cancelling pipelined
    TTS tasks) with the client still available.
    """
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Reload entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
