"""Text-to-Speech platform for Mistral AI."""
from __future__ import annotations

import base64
import logging
from typing import Any

import aiohttp
from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_TTS_VOICE,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    MISTRAL_API_BASE,
    TTS_MODEL,
    TTS_VOICES,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI TTS entity."""
    async_add_entities([MistralTTSEntity(hass, config_entry)])


class MistralTTSEntity(TextToSpeechEntity):
    """Mistral AI text-to-speech entity.

    Voice selection priority (highest to lowest):
      1. Voice Assistants dialog (Settings → Voice Assistants → Text-to-speech
         voice). HA passes this selection via options["voice"] in each call.
      2. Integration default (Settings → Devices & Services → Configure →
         Text-to-speech voice). Used as fallback when no voice is chosen in
         the Voice Assistants dialog or when TTS is called from an automation
         without an explicit voice option.
    """

    _attr_has_entity_name = True
    _attr_name = "Mistral AI TTS"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_tts"

    @property
    def _runtime(self):
        return self.hass.data[DOMAIN][self._entry.entry_id]

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_tts")},
            name="Mistral AI TTS",
            manufacturer="Mistral AI",
            model=TTS_MODEL,
            entry_type=DeviceEntryType.SERVICE,
            configuration_url="https://docs.mistral.ai/capabilities/audio_generation",
        )

    @property
    def default_language(self) -> str:
        """Return default language — Mistral TTS is language-agnostic."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Languages exposed to HA; Mistral TTS handles all of these natively."""
        return ["en", "nl", "fr", "de", "es", "it", "pt", "pl", "ru", "ja", "zh"]

    @property
    def supported_options(self) -> list[str]:
        return ["voice"]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return the integration-configured default voice as fallback."""
        voice = self._entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)
        return {"voice": voice}

    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return all available Mistral TTS voices for the Voice Assistants dialog."""
        return [
            Voice(voice_id=v, name=v.replace("_", " ").title())
            for v in TTS_VOICES
        ]

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Synthesise speech via the Mistral audio/speech endpoint.

        Voice priority: options["voice"] (from Voice Assistants dialog) wins
        over the integration default (CONF_TTS_VOICE).
        """
        voice = options.get("voice") or self._entry.options.get(
            CONF_TTS_VOICE, DEFAULT_TTS_VOICE
        )

        payload = {
            "model": TTS_MODEL,
            "input": message,
            "voice_id": voice,
            "response_format": "mp3",
        }

        runtime = self._runtime
        try:
            async with runtime.session.post(
                f"{MISTRAL_API_BASE}/audio/speech",
                headers=runtime.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 401:
                    raise HomeAssistantError("Invalid Mistral AI API key")
                if resp.status == 429:
                    raise HomeAssistantError("Mistral AI rate limit exceeded")
                if resp.status >= 400:
                    body = await resp.text()
                    _LOGGER.error(
                        "Mistral TTS HTTP %s — voice=%s body=%s",
                        resp.status, voice, body,
                    )
                    raise HomeAssistantError(
                        f"Mistral TTS error {resp.status}: {body}"
                    )
                # Mistral returns JSON with base64-encoded MP3 in audio_data
                data = await resp.json()
                audio_b64 = data.get("audio_data", "")
                if not audio_b64:
                    raise HomeAssistantError("Mistral TTS returned empty audio_data")
                audio_bytes = base64.b64decode(audio_b64)

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral TTS request failed: %s", err)
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err

        _LOGGER.debug(
            "Mistral TTS: synthesised %d bytes (voice=%s)", len(audio_bytes), voice
        )
        return "mp3", audio_bytes
