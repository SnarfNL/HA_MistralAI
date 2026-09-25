"""Base class for all Mistral entities: config entry, client and device info."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity

from .const import DOMAIN, STT_MODEL, TTS_MODEL

if TYPE_CHECKING:
    from . import MistralConfigEntry, MistralRuntimeData
    from .api import MistralClient


@dataclass(frozen=True)
class DeviceSpec:
    """One of the three service devices this integration creates."""

    suffix: str
    name: str
    model: str
    configuration_url: str


# Identifiers and names must never change: HA would create new devices.
CONVERSATION_DEVICE = DeviceSpec(
    "conversation", "Mistral AI Conversation", "", "https://console.mistral.ai"
)
STT_DEVICE = DeviceSpec(
    "stt",
    "Mistral AI STT",
    STT_MODEL,
    "https://docs.mistral.ai/capabilities/audio_transcription",
)
TTS_DEVICE = DeviceSpec(
    "tts",
    "Mistral AI TTS",
    TTS_MODEL,
    "https://docs.mistral.ai/capabilities/audio_generation",
)


class MistralEntity(Entity):
    """Shared plumbing: ``_entry``, ``_runtime``, ``_client`` and ``device_info``."""

    _attr_has_entity_name = True
    _device: DeviceSpec

    def __init__(self, entry: MistralConfigEntry, unique_suffix: str) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"

    @property
    def _runtime(self) -> MistralRuntimeData:
        return self._entry.runtime_data

    @property
    def _client(self) -> MistralClient:
        return self._entry.runtime_data.client

    def _device_model(self) -> str:
        """Model shown on the device page; the conversation device overrides it."""
        return self._device.model

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_{self._device.suffix}")},
            name=self._device.name,
            manufacturer="Mistral AI",
            model=self._device_model(),
            entry_type=DeviceEntryType.SERVICE,
            configuration_url=self._device.configuration_url,
        )
