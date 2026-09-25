"""Button platform: refresh the voice list of the Mistral TTS entity."""
from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .tts import tts_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the refresh-voices button."""
    async_add_entities([MistralRefreshVoicesButton(hass, config_entry)])


class MistralRefreshVoicesButton(ButtonEntity):
    """Re-fetch the account's voices, e.g. after creating one in Mistral Studio."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "refresh_voices"
    _attr_icon = "mdi:refresh"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_refresh_voices"

    @property
    def device_info(self) -> DeviceInfo:
        return tts_device_info(self._entry)

    async def async_press(self) -> None:
        """Refresh the voice list; a failed refresh keeps the old list."""
        runtime = self._entry.runtime_data
        tts_entity = runtime.tts_entity
        if tts_entity is None or not await tts_entity.async_refresh_voices():
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="voices_refresh_failed",
            )
