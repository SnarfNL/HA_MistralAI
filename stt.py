"""Speech-to-Text platform for Mistral AI using Voxtral."""
from __future__ import annotations

import io
import logging
import wave
from typing import AsyncIterable

import aiohttp
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    MISTRAL_API_BASE,
    STT_MODEL,
)

_LOGGER = logging.getLogger(__name__)

# BCP-47 code → display name (exposed via supported_languages so HA can
# show a language picker in the Voice Assistants dialog)
LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("af", "Afrikaans"),
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kk", "Kazakh"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Burmese"),
    ("nb", "Norwegian Bokmål"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("th", "Thai"),
    ("tl", "Filipino"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Voxtral STT entity."""
    async_add_entities([MistralSTTEntity(hass, config_entry)])


class MistralSTTEntity(SpeechToTextEntity):
    """Mistral AI / Voxtral speech-to-text — separate device from conversation entity.

    Language selection is handled entirely by the HA Voice Assistants dialog
    (Settings → Voice Assistants → Speech-to-text language). HA passes the
    selected language via SpeechMetadata.language; no duplicate setting is
    needed in the integration options.

    If no language is selected in the voice assistant (metadata.language is
    empty), Voxtral uses automatic language detection.
    """

    _attr_has_entity_name = True
    _attr_name = "Mistral AI STT (Voxtral)"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_stt"

    @property
    def device_info(self) -> DeviceInfo:
        """Separate device from the conversation entity."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_stt")},
            name="Mistral AI STT",
            manufacturer="Mistral AI",
            model=STT_MODEL,
            entry_type=DeviceEntryType.SERVICE,
            configuration_url="https://docs.mistral.ai/capabilities/audio_transcription",
        )

    @property
    def supported_languages(self) -> list[str]:
        """Expose supported languages so HA can show the language picker."""
        return [code for code, _ in LANGUAGE_OPTIONS]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> SpeechResult:
        """Collect raw PCM from HA pipeline, wrap in WAV, transcribe via Voxtral.

        Language comes from metadata.language, which HA populates from the
        voice assistant pipeline language setting. If empty, Voxtral will
        automatically detect the spoken language.
        """
        pcm_data = b""
        async for chunk in stream:
            pcm_data += chunk

        if not pcm_data:
            _LOGGER.warning("STT: received empty audio stream")
            return SpeechResult("", SpeechResultState.ERROR)

        # Language is set in the Voice Assistants dialog; empty = auto-detect
        lang_code = (metadata.language or "").strip()

        _LOGGER.debug(
            "STT: %d bytes PCM — rate=%s channels=%s bits=%s lang=%s",
            len(pcm_data),
            metadata.sample_rate,
            metadata.channel,
            metadata.bit_rate,
            lang_code or "auto",
        )

        # HA always delivers raw PCM frames — always wrap in a proper WAV container
        wav_bytes = _pcm_to_wav(
            pcm_data,
            sample_rate=int(metadata.sample_rate),
            channels=int(metadata.channel),
            sample_width=int(metadata.bit_rate) // 8,
        )

        runtime = self.hass.data[DOMAIN][self._entry.entry_id]
        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_bytes,
                filename="audio.wav",
                content_type="application/octet-stream",
            )
            form.add_field("model", STT_MODEL)
            if lang_code:
                form.add_field("language", lang_code)

            # Use only the Authorization header for multipart (no Content-Type override)
            auth_header = {"Authorization": runtime.headers["Authorization"]}
            async with runtime.session.post(
                f"{MISTRAL_API_BASE}/audio/transcriptions",
                headers=auth_header,
                data=form,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    _LOGGER.error("Mistral STT HTTP %s: %s", resp.status, body)
                    return SpeechResult("", SpeechResultState.ERROR)
                result = await resp.json()

        except aiohttp.ClientError as err:
            _LOGGER.error("Mistral STT request failed: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)

        text = result.get("text", "").strip()
        if not text:
            _LOGGER.warning("Voxtral returned empty transcription")
            return SpeechResult("", SpeechResultState.ERROR)

        _LOGGER.debug("Voxtral transcription: %s", text)
        return SpeechResult(text, SpeechResultState.SUCCESS)


def _pcm_to_wav(
    pcm_data: bytes, sample_rate: int, channels: int, sample_width: int
) -> bytes:
    """Wrap raw PCM bytes in a RIFF/WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
