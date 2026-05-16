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

PLATFORMS = ["ai_task", "conversation", "stt", "tts"]


@dataclass
class MistralRuntimeData:
    """Shared runtime data for a config entry."""

    session: aiohttp.ClientSession
    headers: dict[str, str]
    # Cached Mistral Agent ID for web-search conversations
    web_search_agent_id: str | None = field(default=None)
    # Custom voices discovered from /v1/audio/voices at setup time
    discovered_voices: list[dict[str, str | None]] = field(default_factory=list)


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

    # Discover custom voices from Mistral so they appear in the options
    # flow and the Voice Assistants dialog. Failures are non-fatal.
    try:
        runtime.discovered_voices = await async_discover_mistral_voices(
            session, headers
        )
    except Exception:  # pylint: disable=broad-except
        _LOGGER.debug("Voice discovery failed during setup, continuing without it")
        runtime.discovered_voices = []

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry.

    Order matters: platforms must be unloaded *before* the runtime data is
    cleared, so entities can complete their teardown (closing streams,
    cancelling pipelined TTS tasks, etc.) using ``self._runtime`` lazily on each chunk.
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


# ---------------------------------------------------------------------------
# Voice discovery helper
# ---------------------------------------------------------------------------

async def async_discover_mistral_voices(
    session: aiohttp.ClientSession,
    headers: dict[str, str],
) -> list[dict[str, str | None]]:
    """Fetch custom voices from Mistral's /v1/audio/voices endpoint.

    Returns a list of voice dicts with keys ``id``, ``name``, ``slug``.
    Preset voices (the hard-coded ``TTS_VOICES``) are **not** included here —
    they are managed separately in ``const.py``.

    Errors are swallowed and logged: if the voices endpoint is unavailable
    or returns an unexpected shape, we fall back to an empty list so the
    integration keeps working with preset voices only.
    """
    try:
        async with session.get(
            f"{MISTRAL_API_BASE}/audio/voices",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                _LOGGER.debug(
                    "Mistral voice discovery returned HTTP %s — skipping", resp.status
                )
                return []
            data = await resp.json()
    except aiohttp.ClientError as err:
        _LOGGER.debug("Mistral voice discovery request failed: %s", err)
        return []
    except Exception:  # pylint: disable=broad-except
        _LOGGER.exception("Unexpected error during Mistral voice discovery")
        return []

    items = data.get("items", []) if isinstance(data, dict) else []
    voices: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        voice_id = item.get("id")
        if not voice_id or not isinstance(voice_id, str):
            continue
        if voice_id in seen:
            continue
        seen.add(voice_id)
        voices.append(
            {
                "id": voice_id,
                "name": item.get("name") or voice_id,
                "slug": item.get("slug"),
            }
        )
    _LOGGER.debug("Discovered %d custom Mistral voice(s)", len(voices))
    return voices
