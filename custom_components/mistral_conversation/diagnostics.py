"""Diagnostics download for a Mistral config entry (MA-14)."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from . import MistralConfigEntry
from ._models import model_available
from .const import CONF_MODEL, DEFAULT_MODEL, DOMAIN

TO_REDACT = {CONF_API_KEY}
REDACTED = "**REDACTED**"


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: MistralConfigEntry
) -> dict[str, Any]:
    """Settings, versions and the last errors — never the API key or bodies."""
    runtime = entry.runtime_data
    integration = await async_get_integration(hass, DOMAIN)
    model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    api_key = str(entry.data.get(CONF_API_KEY) or "")

    def scrub(value: Any) -> Any:
        # Records hold only a translation key or class name, never the key;
        # this is a second safety net.
        if api_key and isinstance(value, str):
            return value.replace(api_key, REDACTED)
        return value

    tts_entity = runtime.tts_entity
    return {
        "entry": {
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": dict(entry.options),
        },
        "versions": {
            "home_assistant": HA_VERSION,
            "integration": str(integration.version),
        },
        "model": {
            "configured": model,
            "available": (
                None
                if runtime.models is None
                else model_available(model, runtime.models)
            ),
        },
        "voices_loaded": (
            len(tts_entity.async_get_supported_voices("en")) if tts_entity else 0
        ),
        "recent_errors": [
            {k: scrub(v) for k, v in record.items()} for record in runtime.errors
        ],
    }
