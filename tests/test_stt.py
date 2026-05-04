"""Tests for ``stt.py``: ``_pcm_to_wav`` helper and ``LANGUAGE_OPTIONS`` data.

Loads ``stt.py`` via the integration package, with HA stubs installed first.
"""
from __future__ import annotations

import struct
import unittest

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation.stt import LANGUAGE_OPTIONS, _pcm_to_wav  # noqa: E402


class PcmToWavTests(unittest.TestCase):
    """``_pcm_to_wav`` wraps raw PCM bytes in a RIFF/WAVE container."""

    def test_riff_wave_magic(self) -> None:
        wav = _pcm_to_wav(
            b"\x00\x00" * 100, sample_rate=16000, channels=1, sample_width=2
        )
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")

    def test_fmt_subchunk_metadata(self) -> None:
        wav = _pcm_to_wav(
            b"\x01\x00" * 50, sample_rate=24000, channels=1, sample_width=2
        )
        self.assertEqual(wav[12:16], b"fmt ")
        audio_format, channels, sample_rate, byte_rate, block_align, bits = (
            struct.unpack("<HHIIHH", wav[20:36])
        )
        self.assertEqual(audio_format, 1, "must be PCM (audio_format=1)")
        self.assertEqual(channels, 1)
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(byte_rate, 24000 * 1 * 2)
        self.assertEqual(block_align, 1 * 2)
        self.assertEqual(bits, 16)

    def test_data_subchunk_payload_round_trips(self) -> None:
        pcm = bytes(range(256))
        wav = _pcm_to_wav(pcm, sample_rate=8000, channels=1, sample_width=1)
        data_idx = wav.find(b"data")
        self.assertGreaterEqual(data_idx, 0)
        size = struct.unpack("<I", wav[data_idx + 4 : data_idx + 8])[0]
        self.assertEqual(size, len(pcm))
        self.assertEqual(wav[data_idx + 8 : data_idx + 8 + size], pcm)

    def test_empty_pcm_input(self) -> None:
        wav = _pcm_to_wav(
            b"", sample_rate=16000, channels=1, sample_width=2
        )
        self.assertEqual(wav[:4], b"RIFF")
        data_idx = wav.find(b"data")
        size = struct.unpack("<I", wav[data_idx + 4 : data_idx + 8])[0]
        self.assertEqual(size, 0)

    def test_stereo_two_byte_samples(self) -> None:
        wav = _pcm_to_wav(
            b"\x01\x02\x03\x04" * 10, sample_rate=44100, channels=2, sample_width=2
        )
        _, channels, sample_rate, byte_rate, block_align, _ = struct.unpack(
            "<HHIIHH", wav[20:36]
        )
        self.assertEqual(channels, 2)
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(block_align, 2 * 2)
        self.assertEqual(byte_rate, 44100 * 2 * 2)

    def test_mistral_voxtral_format(self) -> None:
        """The exact format Voxtral STT consumes per stt.py supported_*."""
        wav = _pcm_to_wav(
            b"\x00" * 320, sample_rate=16000, channels=1, sample_width=2
        )
        _, channels, sample_rate, _, _, bits = struct.unpack(
            "<HHIIHH", wav[20:36]
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)
        self.assertEqual(bits, 16)


class LanguageOptionsTests(unittest.TestCase):
    """``LANGUAGE_OPTIONS`` is exposed via supported_languages and the config flow."""

    def test_non_empty(self) -> None:
        self.assertGreater(len(LANGUAGE_OPTIONS), 0)

    def test_entries_are_code_label_tuples(self) -> None:
        for entry in LANGUAGE_OPTIONS:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
            code, label = entry
            self.assertIsInstance(code, str)
            self.assertIsInstance(label, str)
            self.assertGreater(len(code), 0)
            self.assertGreater(len(label), 0)

    def test_codes_unique(self) -> None:
        codes = [c for c, _ in LANGUAGE_OPTIONS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_codes_are_lowercase_short_iso_like(self) -> None:
        for code, _ in LANGUAGE_OPTIONS:
            self.assertEqual(code, code.lower())
            self.assertGreaterEqual(len(code), 2)
            self.assertLessEqual(len(code), 5)

    def test_includes_voxtral_supported_languages(self) -> None:
        """Mistral's Voxtral STT supports these per their docs; integration
        should expose at least these to HA's language picker."""
        codes = {c for c, _ in LANGUAGE_OPTIONS}
        for required in ("en", "fr", "de", "es", "it", "nl", "ja", "zh", "ar"):
            self.assertIn(required, codes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
