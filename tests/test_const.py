"""Tests for static data and defaults in ``const.py``.

``const.py`` is pure stdlib so no HA stubs are needed; we load it directly
via ``importlib.util`` to bypass the package ``__init__``.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType


def _load_const() -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent
    path = (
        repo_root / "custom_components" / "mistral_conversation" / "const.py"
    )
    spec = importlib.util.spec_from_file_location("mistral_const_uut", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load const module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_const()


class DefaultsConsistencyTests(unittest.TestCase):
    """Defaults must reference values that exist in their respective lists."""

    def test_default_model_in_chat_models(self) -> None:
        self.assertIn(C.DEFAULT_MODEL, C.CHAT_MODELS)

    def test_default_tts_voice_in_tts_voices(self) -> None:
        self.assertIn(C.DEFAULT_TTS_VOICE, C.TTS_VOICES)

    def test_default_tts_mode_in_tts_modes(self) -> None:
        self.assertIn(C.DEFAULT_TTS_MODE, C.TTS_MODES)

    def test_tts_modes_contains_both_values(self) -> None:
        self.assertSetEqual(set(C.TTS_MODES), {"stream", "batch"})

    def test_default_temperature_in_mistral_range(self) -> None:
        self.assertGreaterEqual(C.DEFAULT_TEMPERATURE, 0.0)
        self.assertLessEqual(C.DEFAULT_TEMPERATURE, 1.0)

    def test_default_max_tokens_positive(self) -> None:
        self.assertGreater(C.DEFAULT_MAX_TOKENS, 0)


class TtsConfigurationTests(unittest.TestCase):
    """Voice list and streaming tunables must be sensibly populated."""

    def test_tts_voices_non_empty(self) -> None:
        self.assertGreater(len(C.TTS_VOICES), 0)

    def test_tts_voices_unique(self) -> None:
        self.assertEqual(len(C.TTS_VOICES), len(set(C.TTS_VOICES)))

    def test_voice_naming_convention(self) -> None:
        """Voice IDs follow ``<lang>_<name>_<emotion>`` (3+ tokens, 2+ underscores)."""
        for voice in C.TTS_VOICES:
            self.assertGreaterEqual(
                voice.count("_"), 2,
                f"voice {voice!r} doesn't follow lang_name_emotion convention",
            )

    def test_max_inflight_positive(self) -> None:
        self.assertGreater(C.TTS_MAX_INFLIGHT_SENTENCES, 0)

    def test_min_sentence_chars_positive(self) -> None:
        self.assertGreater(C.TTS_MIN_SENTENCE_CHARS, 0)

    def test_wav_header_size_pinned_to_44(self) -> None:
        """Locks the empirically verified value. If Mistral changes the WAV
        envelope and this fires, the streaming path needs revisiting."""
        self.assertEqual(C.TTS_WAV_HEADER_SIZE, 44)

    def test_inter_sentence_silence_in_natural_pause_range(self) -> None:
        """Silence between sentences should be 100–500 ms — outside that
        window the speech sounds either rushed or disjointed. Mistral's
        streaming WAV is 24 kHz × 16-bit × mono, so 1 s = 48000 bytes."""
        bytes_per_sec = 24000 * 2  # 24 kHz, int16 mono
        ms = (C.TTS_INTER_SENTENCE_SILENCE_BYTES * 1000) // bytes_per_sec
        self.assertGreaterEqual(ms, 100)
        self.assertLessEqual(ms, 500)

    def test_inter_sentence_silence_is_sample_aligned(self) -> None:
        """Byte count must be even so the zeros pack cleanly into int16
        samples (otherwise we'd misalign subsequent audio)."""
        self.assertEqual(C.TTS_INTER_SENTENCE_SILENCE_BYTES % 2, 0)


class AgentCapableModelsTests(unittest.TestCase):
    def test_includes_medium_and_large(self) -> None:
        self.assertIn("mistral-medium-latest", C.AGENT_CAPABLE_MODELS)
        self.assertIn("mistral-large-latest", C.AGENT_CAPABLE_MODELS)


class ApiConstantsTests(unittest.TestCase):
    def test_api_base_uses_https(self) -> None:
        self.assertTrue(C.MISTRAL_API_BASE.startswith("https://"))

    def test_api_base_no_trailing_slash(self) -> None:
        # Endpoints are appended as f"{BASE}/audio/speech", so a trailing
        # slash would produce a double-slash URL.
        self.assertFalse(C.MISTRAL_API_BASE.endswith("/"))

    def test_max_tool_iterations_positive(self) -> None:
        self.assertGreater(C.MAX_TOOL_ITERATIONS, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
