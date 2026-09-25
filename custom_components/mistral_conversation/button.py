"""Button platform: refresh the voice list of the Mistral TTS entity."""
from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import MistralConfigEntry
from .const import DOMAIN
from .entity import TTS_DEVICE, MistralEntity

# Cloud service: nothing polls, calls may run in parallel.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the refresh-voices button."""
    async_add_entities([MistralRefreshVoicesButton(config_entry)])


class MistralRefreshVoicesButton(MistralEntity, ButtonEntity):
    """Re-fetch the account's voices, e.g. after creating one in Mistral Studio."""

    _device = TTS_DEVICE
    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "refresh_voices"
    _attr_icon = "mdi:refresh"

    def __init__(self, entry: MistralConfigEntry) -> None:
        super().__init__(entry, "refresh_voices")

    async def async_press(self) -> None:
        """Refresh the voice list; a failed refresh keeps the old list."""
        runtime = self._runtime
        tts_entity = runtime.tts_entity
        if tts_entity is None or not await tts_entity.async_refresh_voices():
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="voices_refresh_failed",
            )
