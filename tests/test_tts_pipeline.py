"""Tests for the TTS streaming pipeline in ``tts.py`` (MA-01).

Regression coverage for: when sentence 0 failed, the stream started with raw
PCM and no RIFF/WAVE header. Now every sentence worker hands over its own
header as a separate ``("header", bytes)`` item and the consumer yields the
header of the first sentence that really delivers audio, exactly once.

No real Home Assistant and no network: the per-sentence HTTP call is replaced
by a fake that writes a real 44-byte WAV header plus recognisable PCM bytes.
"""
# ruff: noqa: I001 - import order below is intentional: `_ha_stubs` must run
# before the `mistral_conversation` import so Home Assistant is stubbed first.
from __future__ import annotations

import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from homeassistant.exceptions import HomeAssistantError
from mistral_conversation import tts as tts_module
from mistral_conversation.const import (
    DOMAIN,
    TTS_INTER_SENTENCE_SILENCE_BYTES,
    TTS_WAV_HEADER_SIZE,
)


def _wav_header(sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
        b"data",
        0xFFFFFFFF,
    )


HEADER = _wav_header()
PCM = [b"\x11" * 100, b"\x22" * 100, b"\x33" * 100]
SILENCE = bytes(TTS_INTER_SENTENCE_SILENCE_BYTES)
TEXTS = ["First sentence here.", "Second sentence here.", "Third sentence here."]


async def _message_gen(*sentences: str):
    for sentence in sentences:
        yield sentence + " "


def _make_entity(runtime=None) -> tts_module.MistralTTSEntity:
    runtime = runtime or SimpleNamespace(session=None, headers={})
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", options={})
    return tts_module.MistralTTSEntity(hass, entry)


def _fake_sentence(behaviour: dict[str, tuple]):
    """Build a stand-in for ``_stream_one_sentence_into``.

    *behaviour* maps sentence text to ``("ok", pcm)``, ``("header_only",)``
    or ``("fail",)``.
    """

    async def fake(text, voice, out_queue, idx=None):
        kind = behaviour[text]
        if kind[0] == "fail":
            raise HomeAssistantError("boom")
        await out_queue.put(("header", HEADER))
        if kind[0] == "ok":
            await out_queue.put(kind[1])

    return fake


async def _collect(entity, *sentences: str) -> bytes:
    async def run() -> bytes:
        chunks = [c async for c in entity._pipelined_stream(_message_gen(*sentences), "v")]
        return b"".join(chunks)

    return await asyncio.wait_for(run(), timeout=5)


class PipelinedStreamTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, behaviour: dict[str, tuple]) -> bytes:
        entity = _make_entity()
        entity._stream_one_sentence_into = _fake_sentence(behaviour)
        return await _collect(entity, *TEXTS)

    async def test_normal_three_sentences(self) -> None:
        out = await self._run({t: ("ok", p) for t, p in zip(TEXTS, PCM)})
        self.assertEqual(out, HEADER + PCM[0] + SILENCE + PCM[1] + SILENCE + PCM[2])

    async def test_first_sentence_fails(self) -> None:
        out = await self._run(
            {TEXTS[0]: ("fail",), TEXTS[1]: ("ok", PCM[1]), TEXTS[2]: ("ok", PCM[2])}
        )
        self.assertTrue(out.startswith(b"RIFF"))
        self.assertEqual(out, HEADER + PCM[1] + SILENCE + PCM[2])

    async def test_middle_sentence_fails_no_double_silence(self) -> None:
        out = await self._run(
            {TEXTS[0]: ("ok", PCM[0]), TEXTS[1]: ("fail",), TEXTS[2]: ("ok", PCM[2])}
        )
        self.assertEqual(out, HEADER + PCM[0] + SILENCE + PCM[2])

    async def test_last_sentence_fails_no_trailing_silence(self) -> None:
        out = await self._run(
            {TEXTS[0]: ("ok", PCM[0]), TEXTS[1]: ("ok", PCM[1]), TEXTS[2]: ("fail",)}
        )
        self.assertEqual(out, HEADER + PCM[0] + SILENCE + PCM[1])

    async def test_sentence_with_header_but_no_audio_is_skipped(self) -> None:
        out = await self._run(
            {TEXTS[0]: ("header_only",), TEXTS[1]: ("ok", PCM[1]), TEXTS[2]: ("ok", PCM[2])}
        )
        self.assertEqual(out, HEADER + PCM[1] + SILENCE + PCM[2])

    async def test_all_sentences_fail_raises(self) -> None:
        with self.assertRaises(HomeAssistantError) as ctx:
            await self._run({t: ("fail",) for t in TEXTS})
        self.assertIn("no audio", str(ctx.exception))

    async def test_silence_length_follows_header(self) -> None:
        header_16k = _wav_header(sample_rate=16000)
        entity = _make_entity()

        async def fake(text, voice, out_queue, idx=None):
            await out_queue.put(("header", header_16k))
            await out_queue.put(PCM[TEXTS.index(text)])

        entity._stream_one_sentence_into = fake
        out = await _collect(entity, TEXTS[0], TEXTS[1])
        # 300 ms at 16 kHz, mono, 16-bit = 4800 samples * 2 bytes.
        self.assertEqual(out, header_16k + PCM[0] + bytes(9600) + PCM[1])


class SilenceForHeaderTests(unittest.TestCase):
    def test_default_format(self) -> None:
        self.assertEqual(tts_module._silence_for_header(HEADER), SILENCE)

    def test_stereo_44k(self) -> None:
        header = _wav_header(sample_rate=44100, channels=2, bits=16)
        self.assertEqual(len(tts_module._silence_for_header(header)), 13230 * 4)

    def test_unparsable_header_falls_back_to_constant(self) -> None:
        self.assertEqual(tts_module._silence_for_header(b""), SILENCE)
        self.assertEqual(tts_module._silence_for_header(bytes(44)), SILENCE)


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class HeaderSplitAcrossChunksTests(unittest.IsolatedAsyncioTestCase):
    async def test_header_spread_over_several_chunks(self) -> None:
        self.assertEqual(len(HEADER), TTS_WAV_HEADER_SIZE)
        chunks = [HEADER[:10], HEADER[10:30], HEADER[30:] + b"\x01\x02", b"\x03"]

        async def fake_iter(resp):
            for chunk in chunks:
                yield chunk

        session = SimpleNamespace(post=lambda *a, **k: _FakeResponse())
        entity = _make_entity(SimpleNamespace(session=session, headers={}))
        queue: asyncio.Queue = asyncio.Queue()
        with patch.object(tts_module, "iter_sse_audio_chunks", fake_iter):
            await entity._stream_one_sentence_into("Some text here.", "v", queue)

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        self.assertEqual(items, [("header", HEADER), b"\x01\x02", b"\x03"])


if __name__ == "__main__":
    unittest.main()
