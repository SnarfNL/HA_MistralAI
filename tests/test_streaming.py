"""Unit tests for ``custom_components.mistral_conversation._streaming``.

Run from the repo root::

    python -m unittest tests.test_streaming -v

The tested module (``_streaming.py``) is intentionally stdlib-only so these
tests run without Home Assistant installed. We import it directly via
``importlib.util`` to avoid triggering the package ``__init__.py`` which
itself imports ``homeassistant``.
"""
from __future__ import annotations

import base64
import importlib.util
import itertools
import json
import random
import struct
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# Module loading (skip the package __init__ which depends on Home Assistant)
# ---------------------------------------------------------------------------


def _load_streaming_module() -> ModuleType:
    """Load _streaming.py without importing its package."""
    repo_root = Path(__file__).resolve().parent.parent
    path = (
        repo_root
        / "custom_components"
        / "mistral_conversation"
        / "_streaming.py"
    )
    spec = importlib.util.spec_from_file_location("mistral_streaming_uut", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load streaming module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_STREAMING = _load_streaming_module()
has_speakable_content = _STREAMING.has_speakable_content
pop_complete_sentences = _STREAMING.pop_complete_sentences
iter_sse_audio_chunks = _STREAMING.iter_sse_audio_chunks
PcmGain = _STREAMING.PcmGain
apply_gain = _STREAMING.apply_gain
wav_is_pcm16 = _STREAMING.wav_is_pcm16


# Matches TTS_MIN_SENTENCE_CHARS in const.py. Pinned here so changes there
# don't silently shift these tests' baselines.
MIN_CHARS = 12


# ---------------------------------------------------------------------------
# Sentence segmenter
# ---------------------------------------------------------------------------


class SentenceSegmenterTests(unittest.TestCase):
    """``pop_complete_sentences``: extract complete sentences from a buffer."""

    def test_two_complete_sentences(self) -> None:
        sentences, rest = pop_complete_sentences(
            "It is twelve thirty PM. So we are good.", MIN_CHARS
        )
        self.assertEqual(
            sentences, ["It is twelve thirty PM.", "So we are good."]
        )
        self.assertEqual(rest, "")

    def test_no_terminator_keeps_whole_buffer(self) -> None:
        sentences, rest = pop_complete_sentences(
            "Hello world without terminator", MIN_CHARS
        )
        self.assertEqual(sentences, [])
        self.assertEqual(rest, "Hello world without terminator")

    def test_short_sentence_below_min_is_held(self) -> None:
        """``OK.`` is below MIN_CHARS, kept in remainder for the producer's
        trailing flush rather than emitted on its own."""
        sentences, rest = pop_complete_sentences("OK.", MIN_CHARS)
        self.assertEqual(sentences, [])
        self.assertEqual(rest, "OK.")

    def test_short_then_long_merges(self) -> None:
        """A rejected short fragment is folded into the next long sentence."""
        sentences, rest = pop_complete_sentences(
            "OK. Now I am turning off the lights.", MIN_CHARS
        )
        self.assertEqual(
            sentences, ["OK. Now I am turning off the lights."]
        )
        self.assertEqual(rest, "")

    def test_short_trailing_held_in_remainder(self) -> None:
        sentences, rest = pop_complete_sentences(
            "It is twelve thirty PM. The end.", MIN_CHARS
        )
        self.assertEqual(sentences, ["It is twelve thirty PM."])
        self.assertEqual(rest, "The end.")

    def test_abbreviation_does_not_split(self) -> None:
        """``Mr.`` is recognized; ``Mr.`` mid-sentence does not trigger a split."""
        sentences, rest = pop_complete_sentences(
            "Hello Mr. Smith, please come in.", MIN_CHARS
        )
        self.assertEqual(sentences, ["Hello Mr. Smith, please come in."])
        self.assertEqual(rest, "")

    def test_mixed_terminators(self) -> None:
        sentences, rest = pop_complete_sentences(
            "What time is it?! Right now please. Thanks a lot!", MIN_CHARS
        )
        self.assertEqual(
            sentences,
            ["What time is it?!", "Right now please.", "Thanks a lot!"],
        )
        self.assertEqual(rest, "")

    def test_progressive_token_buffering(self) -> None:
        """Tokens arriving piecemeal accumulate into sentences without dupes."""
        pieces = [
            "It's ", "twelve ", "thirty ", "PM. ",
            "Anything ", "else ", "you ", "need", "?",
        ]
        accumulated: list[str] = []
        buf = ""
        for piece in pieces:
            buf += piece
            sents, buf = pop_complete_sentences(buf, MIN_CHARS)
            accumulated.extend(sents)
        # Producer in tts.py also flushes any non-empty remainder at end of stream.
        if buf.strip():
            accumulated.append(buf.strip())
        self.assertEqual(
            accumulated,
            ["It's twelve thirty PM.", "Anything else you need?"],
        )

    def test_remainder_preserves_incomplete_followup(self) -> None:
        sentences, rest = pop_complete_sentences(
            "First long sentence done. Continuing", MIN_CHARS
        )
        self.assertEqual(sentences, ["First long sentence done."])
        self.assertEqual(rest, "Continuing")

    def test_emoji_only_long_candidate_is_rejected(self) -> None:
        """An emoji-only fragment ≥ MIN_CHARS still passes the length gate
        but must be rejected by the speakable-content check."""
        sentences, rest = pop_complete_sentences(
            "🌿✨💫🌹🌷🌸💐🌼🌻🌺.", MIN_CHARS
        )
        self.assertEqual(sentences, [])
        self.assertEqual(rest, "🌿✨💫🌹🌷🌸💐🌼🌻🌺.")

    def test_punctuation_only_candidate_is_rejected(self) -> None:
        sentences, _rest = pop_complete_sentences("...???!!!....", MIN_CHARS)
        self.assertEqual(sentences, [])

    def test_emoji_decoration_at_end_of_long_sentence_kept(self) -> None:
        """Emojis sprinkled in a real sentence are fine — the sentence has
        speakable content overall."""
        sentences, rest = pop_complete_sentences(
            "Listen closely on a still night you might hear it too 🌿✨.",
            MIN_CHARS,
        )
        self.assertEqual(
            sentences,
            ["Listen closely on a still night you might hear it too 🌿✨."],
        )
        self.assertEqual(rest, "")


class HasSpeakableContentTests(unittest.TestCase):
    """``has_speakable_content`` checks for at least one letter or digit."""

    def test_letters_only(self) -> None:
        self.assertTrue(has_speakable_content("hello"))

    def test_digits_only(self) -> None:
        self.assertTrue(has_speakable_content("12345"))

    def test_mixed_letters_and_punctuation(self) -> None:
        self.assertTrue(has_speakable_content("Hello, world!"))

    def test_unicode_letters(self) -> None:
        # "café" has 'c', 'a', 'f', 'é' — all letters
        self.assertTrue(has_speakable_content("café"))

    def test_japanese_letters(self) -> None:
        self.assertTrue(has_speakable_content("こんにちは"))

    def test_empty_string(self) -> None:
        self.assertFalse(has_speakable_content(""))

    def test_whitespace_only(self) -> None:
        self.assertFalse(has_speakable_content("   \n\t  "))

    def test_punctuation_only(self) -> None:
        self.assertFalse(has_speakable_content("...!?,;:"))

    def test_emoji_only(self) -> None:
        self.assertFalse(has_speakable_content("🌿✨"))
        self.assertFalse(has_speakable_content("🌅☕"))
        self.assertFalse(has_speakable_content("😊"))

    def test_emoji_with_punctuation(self) -> None:
        self.assertFalse(has_speakable_content("🌿✨!?."))

    def test_single_letter_makes_speakable(self) -> None:
        self.assertTrue(has_speakable_content("a 🌿✨"))


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------


class _FakeContent:
    """Stand-in for ``aiohttp.StreamReader``; only ``iter_any()`` is needed."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    """Stand-in for ``aiohttp.ClientResponse``."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeContent(chunks)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _frame(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


class SseAudioChunkTests(unittest.IsolatedAsyncioTestCase):
    """``iter_sse_audio_chunks``: parse SSE frames, decode base64 audio.

    Uses ``IsolatedAsyncioTestCase`` so each test runs in its own event loop.
    """

    async def _collect(self, resp: Any) -> list[bytes]:
        return [chunk async for chunk in iter_sse_audio_chunks(resp)]

    async def test_single_delta_then_done(self) -> None:
        audio = b"\x01\x02\x03"
        chunks = [
            _frame("speech.audio.delta", {"audio_data": _b64(audio)}),
            _frame("speech.audio.done", {"usage": {"x": 1}}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [audio])

    async def test_multiple_deltas(self) -> None:
        a, b = b"AAAA", b"BBBB"
        chunks = [
            _frame("speech.audio.delta", {"audio_data": _b64(a)}),
            _frame("speech.audio.delta", {"audio_data": _b64(b)}),
            _frame("speech.audio.done", {}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [a, b])

    async def test_frame_split_across_wire_chunks(self) -> None:
        """A single SSE frame's bytes arriving over multiple TCP reads must
        still parse correctly — the parser buffers until ``\\n\\n``."""
        audio = b"hello-audio"
        full = (
            _frame("speech.audio.delta", {"audio_data": _b64(audio)})
            + _frame("speech.audio.done", {})
        )
        # Split the wire stream at arbitrary byte boundaries.
        wire_chunks = [full[:5], full[5:11], full[11:30], full[30:]]
        result = await self._collect(_FakeResponse(wire_chunks))
        self.assertEqual(result, [audio])

    async def test_unknown_events_are_ignored(self) -> None:
        chunks = [
            _frame("ping", {"keep": "alive"}),
            _frame("speech.audio.delta", {"audio_data": _b64(b"\xff")}),
            _frame("unknown.event", {"foo": "bar"}),
            _frame("speech.audio.done", {}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [b"\xff"])

    async def test_done_terminates_stream(self) -> None:
        """Frames after ``speech.audio.done`` must not be emitted."""
        chunks = [
            _frame("speech.audio.delta", {"audio_data": _b64(b"x")}),
            _frame("speech.audio.done", {}),
            _frame("speech.audio.delta", {"audio_data": _b64(b"y")}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [b"x"])

    async def test_empty_audio_data_skipped(self) -> None:
        chunks = [
            _frame("speech.audio.delta", {"audio_data": ""}),
            _frame("speech.audio.delta", {"audio_data": _b64(b"x")}),
            _frame("speech.audio.done", {}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [b"x"])

    async def test_eof_without_done_event(self) -> None:
        """Streams terminating without ``speech.audio.done`` still emit data."""
        chunks = [
            _frame("speech.audio.delta", {"audio_data": _b64(b"abc")}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [b"abc"])

    async def test_malformed_json_skipped(self) -> None:
        """A frame whose ``data:`` line isn't valid JSON must not crash."""
        chunks = [
            b"event: speech.audio.delta\ndata: {not json\n\n",
            _frame("speech.audio.delta", {"audio_data": _b64(b"ok")}),
            _frame("speech.audio.done", {}),
        ]
        result = await self._collect(_FakeResponse(chunks))
        self.assertEqual(result, [b"ok"])


# ---------------------------------------------------------------------------
# Volume boost (#89)
# ---------------------------------------------------------------------------


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _samples(pcm: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)


class PcmGainTests(unittest.TestCase):
    def test_zero_db_is_identity(self) -> None:
        # Including loud samples: nothing is limited when there is no boost.
        pcm = _pcm(0, 1, -1, 1000, 24575, -24575, 32767, -32768)
        self.assertEqual(apply_gain(pcm, 0), pcm)
        gain = PcmGain(0)
        self.assertEqual(gain.process(pcm), pcm)
        self.assertEqual(gain.flush(), b"")

    def test_six_db_roughly_doubles_a_small_sample(self) -> None:
        out = _samples(apply_gain(_pcm(100, -100, 1000, -1000), 6))
        self.assertAlmostEqual(out[0], 200, delta=2)
        self.assertEqual(out[1], -out[0])
        self.assertAlmostEqual(out[2], 1995, delta=3)
        self.assertEqual(out[3], -out[2])

    def test_output_never_leaves_the_int16_range(self) -> None:
        extremes = _pcm(32767, -32768, 30000, -30000, 20000, -20000)
        for gain_db in (1, 6, 12):
            with self.subTest(gain_db=gain_db):
                out = _samples(apply_gain(extremes, gain_db))
                self.assertTrue(all(-32767 <= v <= 32767 for v in out), out)

    def test_loud_samples_are_compressed_not_clipped(self) -> None:
        out = _samples(apply_gain(_pcm(20000, 24000, 28000, 32767), 3))
        self.assertEqual(sorted(out), list(out))  # still ordered
        self.assertEqual(len(set(out)), 4)  # no two collapsed onto full scale

    def test_gain_is_monotonic_over_the_whole_range(self) -> None:
        ramp = _pcm(*range(0, 32768, 7))
        out = _samples(apply_gain(ramp, 12))
        self.assertTrue(all(b >= a for a, b in itertools.pairwise(out)))

    def test_chunking_does_not_change_the_output(self) -> None:
        rng = random.Random(1)
        data = bytes(rng.randrange(256) for _ in range(10_001))  # odd length
        expected = apply_gain(data, 6)
        for size in (1, 3, 4096):
            with self.subTest(chunk_size=size):
                gain = PcmGain(6)
                out = b"".join(
                    gain.process(data[i : i + size]) for i in range(0, len(data), size)
                )
                self.assertEqual(out + gain.flush(), expected)

    def test_odd_trailing_byte_is_carried_then_flushed_unscaled(self) -> None:
        gain = PcmGain(6)
        first = gain.process(_pcm(100) + b"")
        self.assertEqual(_samples(first), (_samples(apply_gain(_pcm(100), 6))[0],))
        self.assertEqual(gain.process(b""), b"")
        self.assertEqual(gain.flush(), b"")
        self.assertEqual(gain.flush(), b"")

    def test_sample_split_across_chunks_is_scaled_whole(self) -> None:
        pcm = _pcm(1000)
        gain = PcmGain(6)
        self.assertEqual(gain.process(pcm[:1]), b"")
        self.assertEqual(gain.process(pcm[1:]), apply_gain(pcm, 6))


class WavIsPcm16Tests(unittest.TestCase):
    @staticmethod
    def _header(tag: int = 1, bits: int = 16) -> bytes:
        header = bytearray(44)
        struct.pack_into("<H", header, 20, tag)
        struct.pack_into("<H", header, 34, bits)
        return bytes(header)

    def test_pcm16_is_accepted(self) -> None:
        self.assertTrue(wav_is_pcm16(self._header()))

    def test_other_bit_depths_and_formats_are_skipped(self) -> None:
        self.assertFalse(wav_is_pcm16(self._header(bits=24)))
        self.assertFalse(wav_is_pcm16(self._header(bits=8)))
        self.assertFalse(wav_is_pcm16(self._header(tag=3, bits=32)))  # float

    def test_short_header_is_skipped(self) -> None:
        self.assertFalse(wav_is_pcm16(b""))
        self.assertFalse(wav_is_pcm16(self._header()[:30]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
