"""The Mistral AI Conversation integration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN, MISTRAL_API_BASE

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS = ["conversation", "stt", "tts"]


@dataclass
class MistralRuntimeData:
    """Shared runtime data for a config entry."""

    session: aiohttp.ClientSession
    headers: dict[str, str]
    # Cached Mistral Agent ID for web-search conversations
    web_search_agent_id: str | None = field(default=None)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Mistral AI Conversation from a config entry."""
    api_key = entry.data[CONF_API_KEY]
    session = async_get_clientsession(hass)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with session.get(
            f"{MISTRAL_API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 401:
                raise ConfigEntryAuthFailed("Invalid Mistral AI API key")
            resp.raise_for_status()
    except aiohttp.ClientError as err:
        raise ConfigEntryNotReady(f"Cannot connect to Mistral AI: {err}") from err

    runtime = MistralRuntimeData(session=session, headers=headers)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry.

    Order matters: platforms must be unloaded *before* the runtime data is
    cleared, so entities can complete their teardown (closing streams,
    cancelling pipelined TTS tasks, etc.) using ``self._runtime`` while it
    still exists. Clearing first races with in-flight ``async_stream_tts_audio``
    iterations that resolve ``self._runtime`` lazily on each chunk.
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
