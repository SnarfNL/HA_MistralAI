"""Tests for the optional TTS volume boost on the batch path (#89).

Mistral's speech endpoint has no volume parameter, so with a boost above 0 dB
the integration asks for wav, scales the samples and returns wav. With 0 dB
(the default) nothing changes: mp3 is requested and returned as it came.

No real Home Assistant and no network: the HTTP session is a fake that records
the payload and answers with a small WAV file.
"""
from __future__ import annotations

import base64
import io
import struct
import unittest
import wave
from types import SimpleNamespace

from custom_components.mistral_conversation import tts as tts_module
from custom_components.mistral_conversation._streaming import apply_gain

from .helpers import attach_runtime, with_hass

LOGGER = tts_module._LOGGER.name
SAMPLES = (0, 100, -100, 1000, -1000, 20000)


def _wav(samples: tuple[int, ...], *, width: int = 2, channels: int = 1) -> bytes:
    fmt = {1: "<{}B", 2: "<{}h"}[width]
    pcm = struct.pack(fmt.format(len(samples)), *samples)
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(24000)
        writer.writeframes(pcm)
    return out.getvalue()


def _pcm_of(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        return reader.readframes(reader.getnframes())


class _FakeResponse:
    status = 200

    def __init__(self, audio: bytes) -> None:
        self._audio = audio

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def json(self):
        return {"audio_data": base64.b64encode(self._audio).decode()}


class _FakeSession:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio
        self.payloads: list[dict] = []

    def request(self, method, url, *, json, **kwargs):
        self.payloads.append(json)
        return _FakeResponse(self.audio)


def _make_entity(options: dict, session: _FakeSession) -> tts_module.MistralTTSEntity:
    entry = SimpleNamespace(entry_id="entry1", options=options)
    attach_runtime(entry, session)
    executor_jobs: list = []

    async def run_in_executor(func, *args):
        executor_jobs.append(func)
        return func(*args)

    hass = SimpleNamespace(data={}, async_add_executor_job=run_in_executor)
    entity = with_hass(tts_module.MistralTTSEntity(entry), hass)
    entity.executor_jobs = executor_jobs
    return entity


class BatchGainTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_gain_asks_for_mp3_and_returns_it_untouched(self) -> None:
        session = _FakeSession(b"mp3-bytes")
        entity = _make_entity({}, session)
        result = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(result, ("mp3", b"mp3-bytes"))
        self.assertEqual(session.payloads[0]["response_format"], "mp3")
        self.assertEqual(entity.executor_jobs, [])

    async def test_zero_gain_is_the_same_as_no_gain(self) -> None:
        session = _FakeSession(b"mp3-bytes")
        entity = _make_entity({"tts_gain_db": 0}, session)
        result = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(result, ("mp3", b"mp3-bytes"))
        self.assertEqual(session.payloads[0]["response_format"], "mp3")

    async def test_gain_asks_for_wav_and_returns_scaled_wav(self) -> None:
        session = _FakeSession(_wav(SAMPLES))
        entity = _make_entity({"tts_gain_db": 6.0}, session)
        extension, audio = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(session.payloads[0]["response_format"], "wav")
        self.assertEqual(extension, "wav")
        self.assertEqual(
            _pcm_of(audio), apply_gain(_pcm_of(_wav(SAMPLES)), 6.0)
        )
        scaled = struct.unpack(f"<{len(SAMPLES)}h", _pcm_of(audio))
        self.assertAlmostEqual(scaled[1], 200, delta=2)  # +6 dB doubles it
        self.assertEqual(scaled[0], 0)
        with wave.open(io.BytesIO(audio), "rb") as reader:
            self.assertEqual(
                (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()),
                (1, 2, 24000),
            )

    async def test_the_scaling_runs_in_the_executor(self) -> None:
        entity = _make_entity({"tts_gain_db": 6}, _FakeSession(_wav(SAMPLES)))
        await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(entity.executor_jobs, [tts_module._boost_wav])

    async def test_stereo_is_kept_stereo(self) -> None:
        wav_bytes = _wav((100, -100, 200, -200), channels=2)
        entity = _make_entity({"tts_gain_db": 6}, _FakeSession(wav_bytes))
        _, audio = await entity.async_get_tts_audio("hello", "en", {})
        with wave.open(io.BytesIO(audio), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 2)
            self.assertEqual(reader.getnframes(), 2)

    async def test_gain_is_clamped_to_the_option_range(self) -> None:
        wav_bytes = _wav(SAMPLES)
        for configured, effective in ((99, 12.0), (-5, 0.0), ("junk", 0.0), (None, 0.0)):
            with self.subTest(configured=configured):
                entity = _make_entity({"tts_gain_db": configured}, _FakeSession(wav_bytes))
                self.assertEqual(entity._gain_db(), effective)

    async def test_non_16_bit_wav_is_returned_unchanged_with_one_warning(self) -> None:
        wav_bytes = _wav((10, 20, 30), width=1)
        entity = _make_entity({"tts_gain_db": 6}, _FakeSession(wav_bytes))
        with self.assertLogs(LOGGER, "WARNING") as logs:
            result = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(result, ("wav", wav_bytes))
        self.assertIn("volume boost is not applied", logs.output[0])
        # The second time is silent: the warning is logged once per entity.
        with self.assertNoLogs(LOGGER, "WARNING"):
            result = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(result, ("wav", wav_bytes))

    async def test_audio_that_is_not_a_wav_is_returned_unchanged(self) -> None:
        entity = _make_entity({"tts_gain_db": 6}, _FakeSession(b"not a wav file"))
        with self.assertLogs(LOGGER, "WARNING"):
            result = await entity.async_get_tts_audio("hello", "en", {})
        self.assertEqual(result, ("wav", b"not a wav file"))


if __name__ == "__main__":
    unittest.main()
