"""Text-to-Speech platform for Mistral AI.

Two operating modes selectable via integration options (CONF_TTS_MODE):

* ``stream``: uses the streaming /v1/audio/speech endpoint with
  ``response_format=wav`` and ``stream=true``. Mistral returns Server-Sent
  Events containing base64-encoded WAV chunks while synthesis is still in
  progress. For multi-sentence LLM responses, sentences are extracted from
  the incoming token stream and dispatched to Mistral with bounded
  concurrency. Audio is reassembled in strict sentence order with the WAV
  header from sentence 0 followed by raw PCM samples from sentences 1..N.
  The pipeline keeps up to TTS_MAX_INFLIGHT_SENTENCES Mistral requests in
  flight simultaneously while preserving playback order.

* ``batch``: single-shot path that POSTs the full message and waits for the
  whole mp3 (returned as base64 in a JSON body) before yielding any audio.
  ``async_get_tts_audio`` always uses this path regardless of the setting,
  so direct ``tts.speak`` service calls keep working. When CONF_TTS_MODE is
  ``batch``, ``async_stream_tts_audio`` also delegates here via the base
  class default implementation.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator

import aiohttp
from homeassistant.components.tts import (
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from ._streaming import (
    has_speakable_content,
    iter_sse_audio_chunks,
    pop_complete_sentences,
)
from .const import (
    CONF_TTS_MODE,
    CONF_TTS_VOICE,
    DEFAULT_TTS_MODE,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    MISTRAL_API_BASE,
    TTS_INTER_SENTENCE_SILENCE_BYTES,
    TTS_MAX_INFLIGHT_SENTENCES,
    TTS_MIN_SENTENCE_CHARS,
    TTS_MODE_BATCH,
    TTS_MODEL,
    TTS_VOICES,
    TTS_WAV_HEADER_SIZE,
)

_LOGGER = logging.getLogger(__name__)

# Pre-computed once at import. ``bytes(N)`` materialises N zero bytes — valid
# PCM silence at any sample rate / channel count. Yielded between sentences
# in the pipelined streaming engine to produce natural inter-sentence pauses.
_INTER_SENTENCE_SILENCE: bytes = bytes(TTS_INTER_SENTENCE_SILENCE_BYTES)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI TTS entity."""
    async_add_entities([MistralTTSEntity(hass, config_entry)])


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


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
        return [Voice(voice_id=v, name=v.replace("_", " ").title()) for v in TTS_VOICES]

    # ------------------------------------------------------------------
    # Batch path
    # Always used by direct tts.speak service calls. Also used by
    # async_stream_tts_audio's base-class fallback when CONF_TTS_MODE
    # is 'batch'.
    # ------------------------------------------------------------------
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
        voice = options.get("voice") or self._entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)

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
                        resp.status,
                        voice,
                        body,
                    )
                    raise HomeAssistantError(f"Mistral TTS error {resp.status}: {body}")
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
            "Mistral TTS (batch): synthesised %d bytes (voice=%s)",
            len(audio_bytes),
            voice,
        )
        return "mp3", audio_bytes

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        """Stream WAV audio while the LLM is still generating its reply.

        For CONF_TTS_MODE == 'batch' we defer to the inherited default which
        collects the full message and calls ``async_get_tts_audio``.
        """
        # Diagnostic: when is HA actually invoking us? Compare this timestamp
        # against the pipeline's "tts_start_streaming: true" event time to see
        # where time is being lost in the LLM-to-TTS pipeline.
        _LOGGER.debug("async_stream_tts_audio called at monotonic=%.3fs", time.monotonic())

        mode = self._entry.options.get(CONF_TTS_MODE, DEFAULT_TTS_MODE)
        if mode == TTS_MODE_BATCH:
            return await super().async_stream_tts_audio(request)

        voice = request.options.get("voice") or self._entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)

        return TTSAudioResponse(
            extension="wav",
            data_gen=self._pipelined_stream(request.message_gen, voice),
        )

    # ------------------------------------------------------------------
    # Pipelined per-sentence streaming engine
    # ------------------------------------------------------------------
    async def _pipelined_stream(
        self,
        message_gen: AsyncGenerator[str, None],
        voice: str,
    ) -> AsyncGenerator[bytes, None]:
        """Aggressive sentence-pipelined TTS generator.

        Architecture::

            message_gen ─► producer ─► outer_q (FIFO of inner queues)
                              │            │
                              │            ▼
                              ▼          consumer (this generator)
                        per-sentence
                        fetcher tasks
                        (≤ MAX_INFLIGHT
                        concurrent)

        * Producer reads tokens, segments to complete sentences, spawns one
          fetcher task per sentence, and pushes that sentence's inner audio
          queue onto the outer queue immediately (preserving order).
        * Each fetcher acquires a semaphore (bounding outbound concurrency),
          POSTs to Mistral, parses SSE, decodes base64 audio, optionally
          strips the WAV header for sentences after the first, and pushes
          chunks into its (unbounded) inner queue.
        * Consumer (this method) drains inner queues in strict order. The
          output is a single contiguous WAV stream: header from sentence 0,
          PCM samples concatenated from sentences 0..N.
        """
        sem = asyncio.Semaphore(TTS_MAX_INFLIGHT_SENTENCES)
        outer_q: asyncio.Queue[asyncio.Queue[Any] | None] = asyncio.Queue()
        fetch_tasks: list[asyncio.Task] = []
        end_marker = object()

        async def fetch_sentence(idx: int, text: str, drop_header: bool) -> asyncio.Queue:
            # Unbounded queue: avoids a cancellation-time deadlock where a
            # worker stuck on ``await inner.put(...)`` (queue at maxsize) can't
            # reach its finally block, leaving the consumer's ``gather`` to
            # wait forever. Memory ceiling is per-sentence audio size ×
            # in-flight count (a few MB worst-case), which is fine on any HA
            # host. Backpressure on the wire still applies because Mistral
            # streams chunks at their generation rate.
            inner: asyncio.Queue = asyncio.Queue()

            async def worker() -> None:
                try:
                    async with sem:
                        _LOGGER.debug(
                            "TTS sentence %d START (drop_header=%s, %d chars)",
                            idx,
                            drop_header,
                            len(text),
                        )
                        await self._stream_one_sentence_into(text, voice, drop_header, inner, idx=idx)
                        _LOGGER.debug("TTS sentence %d DONE", idx)
                except asyncio.CancelledError:
                    raise
                except Exception as err:  # pylint: disable=broad-except
                    _LOGGER.warning("TTS sentence %d failed: %s", idx, err)
                    await inner.put(err)
                finally:
                    await inner.put(end_marker)

            fetch_tasks.append(asyncio.create_task(worker()))
            return inner

        async def producer() -> None:
            try:
                token_buffer = ""
                next_idx = 0
                async for token in message_gen:
                    if not token:
                        continue
                    token_buffer += token
                    sentences, token_buffer = pop_complete_sentences(token_buffer, TTS_MIN_SENTENCE_CHARS)
                    for sentence in sentences:
                        inner = await fetch_sentence(next_idx, sentence, drop_header=next_idx > 0)
                        await outer_q.put(inner)
                        next_idx += 1
                # Flush any trailing text without a terminator. Skip if it
                # has no speakable content (emoji- or punctuation-only) —
                # Mistral would reject it with HTTP 400. The segmenter
                # applies the same check via has_speakable_content() but
                # skips the min-length gate that the loop above uses, since
                # this is the last chance to emit whatever's left.
                trailing = token_buffer.strip()
                if trailing and has_speakable_content(trailing):
                    inner = await fetch_sentence(next_idx, trailing, drop_header=next_idx > 0)
                    await outer_q.put(inner)
            finally:
                await outer_q.put(None)

        producer_task = asyncio.create_task(producer())

        try:
            sentence_idx = 0
            while True:
                inner = await outer_q.get()
                if inner is None:
                    return
                # Inject a brief silence before every sentence after the
                # first to give an audible pause at sentence boundaries.
                # Without this, the back-to-back per-sentence Mistral calls
                # concatenate with no gap — each call's audio ends at the
                # last phoneme. Silence is plain zero PCM appended to the
                # data subchunk (size = 0xFFFFFFFF, no length to update).
                if sentence_idx > 0:
                    yield _INTER_SENTENCE_SILENCE
                sentence_idx += 1
                while True:
                    item = await inner.get()
                    if item is end_marker:
                        break
                    if isinstance(item, BaseException):
                        # Skip this sentence rather than aborting the whole
                        # stream — a single sentence's failure (e.g. Mistral
                        # 4xx, network blip) shouldn't truncate the rest of
                        # the response. The worker has already logged the
                        # specifics at WARNING level.
                        _LOGGER.warning("Skipping sentence due to error: %s", item)
                        break
                    yield item
        finally:
            # Cleanup. The CancelledError raised by ``await producer_task``
            # below originates from *our* cancel of the producer, not from
            # this generator being cancelled — swallowing it does not lose
            # any caller-visible signal because Python's try/finally rule is
            # that if the finally block doesn't raise, the original exception
            # (be it CancelledError from a real cancel or GeneratorExit from
            # aclose) keeps propagating.
            if not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.debug("TTS producer ended with error: %s", err)
            for task in fetch_tasks:
                if not task.done():
                    task.cancel()
            if fetch_tasks:
                await asyncio.gather(*fetch_tasks, return_exceptions=True)

    async def _stream_one_sentence_into(
        self,
        text: str,
        voice: str,
        drop_header: bool,
        out_queue: asyncio.Queue,
        idx: int | None = None,
    ) -> None:
        """POST one sentence and pump SSE-decoded audio bytes into *out_queue*.

        When *drop_header* is True, swallow the first TTS_WAV_HEADER_SIZE
        decoded bytes regardless of how many SSE frames they span. This
        produces a continuous PCM tail that concatenates cleanly behind
        sentence 0's RIFF/WAVE header.

        *idx* is purely for log correlation with the surrounding START/DONE
        lines emitted by the worker; logic does not depend on it.
        """
        runtime = self._runtime
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice_id": voice,
            "response_format": "wav",
            "stream": True,
        }
        bytes_skipped = 0
        request_start = time.monotonic()
        first_chunk_logged = False
        try:
            async with runtime.session.post(
                f"{MISTRAL_API_BASE}/audio/speech",
                headers=runtime.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 401:
                    raise HomeAssistantError("Invalid Mistral AI API key")
                if resp.status == 429:
                    raise HomeAssistantError("Mistral AI rate limit exceeded")
                if resp.status >= 400:
                    body = await resp.text()
                    _LOGGER.error(
                        "Mistral TTS HTTP %s — voice=%s body=%s",
                        resp.status,
                        voice,
                        body,
                    )
                    raise HomeAssistantError(f"Mistral TTS error {resp.status}: {body}")
                async for audio in iter_sse_audio_chunks(resp):
                    if not first_chunk_logged:
                        # Time from POST to first decoded audio bytes — the
                        # actual TTFA contributed by Mistral. Logged before
                        # the optional header-strip so it reflects what the
                        # network delivered, not what we forwarded onward.
                        _LOGGER.debug(
                            "TTS sentence %s first audio chunk after %.3fs" " (%d decoded bytes)",
                            idx if idx is not None else "?",
                            time.monotonic() - request_start,
                            len(audio),
                        )
                        first_chunk_logged = True
                    if drop_header and bytes_skipped < TTS_WAV_HEADER_SIZE:
                        skip = min(TTS_WAV_HEADER_SIZE - bytes_skipped, len(audio))
                        audio = audio[skip:]
                        bytes_skipped += skip
                        if not audio:
                            continue
                    await out_queue.put(audio)
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot reach Mistral AI: {err}") from err
