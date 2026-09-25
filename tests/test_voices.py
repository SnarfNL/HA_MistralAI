"""Tests for ``_voices.py``: readable labels and ordering for the voice picker."""
from __future__ import annotations

import unittest

from custom_components.mistral_conversation._voices import build_voice_list, voice_label


class VoiceLabelTests(unittest.TestCase):
    def test_preset_slug_becomes_readable(self) -> None:
        self.assertEqual(voice_label("en_paul_angry"), "Paul – Angry (English)")

    def test_gb_is_british_english(self) -> None:
        self.assertEqual(voice_label("gb_jane_sarcasm"), "Jane – Sarcasm (British English)")

    def test_unknown_language_code_stays_raw(self) -> None:
        self.assertEqual(voice_label("xx_kim_calm"), "Kim – Calm (xx)")

    def test_custom_voice_name_is_untouched(self) -> None:
        for name in ("My Voice", "Mama", "en_only_two_extra", "Paul - Angry", "EN_Paul_Angry"):
            self.assertEqual(voice_label(name), name)


class BuildVoiceListTests(unittest.TestCase):
    def test_voice_id_is_never_changed(self) -> None:
        voices = build_voice_list([{"id": "uuid-1", "name": "en_paul_angry"}])
        self.assertEqual(voices, [("uuid-1", "Paul – Angry (English)")])

    def test_custom_voices_come_first_alphabetically(self) -> None:
        voices = build_voice_list(
            [
                {"id": "1", "name": "en_paul_neutral"},
                {"id": "2", "name": "zed"},
                {"id": "3", "name": "Anna"},
            ]
        )
        self.assertEqual([v[1] for v in voices][:2], ["Anna", "zed"])
        self.assertEqual(voices[2][0], "1")

    def test_presets_sorted_by_speaker_then_neutral_first(self) -> None:
        voices = build_voice_list(
            [
                {"id": "a", "name": "gb_jane_sad"},
                {"id": "b", "name": "en_paul_angry"},
                {"id": "c", "name": "en_paul_neutral"},
                {"id": "d", "name": "gb_jane_neutral"},
            ]
        )
        self.assertEqual([v[0] for v in voices], ["d", "a", "c", "b"])

    def test_item_without_id_is_skipped_and_missing_name_uses_id(self) -> None:
        voices = build_voice_list([{"name": "no id"}, {"id": "only-id"}])
        self.assertEqual(voices, [("only-id", "only-id")])

    def test_empty_input(self) -> None:
        self.assertEqual(build_voice_list([]), [])


if __name__ == "__main__":
    unittest.main()
