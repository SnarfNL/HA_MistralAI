"""Text-to-Speech platform for Mistral AI."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
from homeassistant.components.tts import (
    ATTR_AUDIO_OUTPUT,
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
    """Mistral AI text-to-speech entity."""

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
        """Mistral TTS supports all languages the model knows; expose as wildcard."""
        return ["en", "nl", "fr", "de", "es", "it", "pt", "pl", "ru", "ja", "zh"]

    @property
    def supported_options(self) -> list[str]:
        return ["voice"]

    @property
    def default_options(self) -> dict[str, Any]:
        voice = self._entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)
        return {"voice": voice}

    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return all available Mistral TTS voices."""
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
        """Synthesise speech via the Mistral audio/speech endpoint."""
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
                # API returns JSON with base64-encoded audio in audio_data field
                data = await resp.json()
                audio_b64 = data.get("audio_data", "")
                if not audio_b64:
                    raise HomeAssistantError("Mistral TTS returned empty audio_data")
                import base64
                audio_bytes = base64.b64decode(audio_b64)

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral TTS request failed: %s", err)
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err

        _LOGGER.debug(
            "Mistral TTS: synthesised %d bytes (voice=%s)", len(audio_bytes), voice
        )
        # Mistral returns MP3 by default
        return "mp3", audio_bytes
