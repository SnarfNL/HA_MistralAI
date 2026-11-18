"""Tests for ``_voices.py``: readable labels and ordering for the voice picker."""
from __future__ import annotations

import json
import unittest

from custom_components.mistral_conversation._voices import build_voice_list, voice_label

from .conftest import load_fixture


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
    def test_voice_fixture_uses_the_live_response_fields(self) -> None:
        page = json.loads(load_fixture("voices.json"))
        self.assertEqual(build_voice_list(page["data"]), [
            ("0b6f5c1e-0000-4000-8000-000000000002", "Lotte (Dutch)"),
            ("0b6f5c1e-0000-4000-8000-000000000001", "Paul – Neutral (English)"),
        ])

    def test_voice_id_is_never_changed(self) -> None:
        voices = build_voice_list([
            {"id": "uuid-1", "name": "Paul - Angry", "type": "preset",
             "slug": "en_paul_angry", "languages": ["en_us"]}
        ])
        self.assertEqual(voices, [("uuid-1", "Paul – Angry (English)")])

    def test_custom_voices_come_first_alphabetically(self) -> None:
        voices = build_voice_list(
            [
                {"id": "1", "name": "Paul - Neutral", "type": "preset",
                 "slug": "en_paul_neutral"},
                {"id": "2", "name": "zed", "type": "custom"},
                {"id": "3", "name": "Anna", "type": "custom"},
            ]
        )
        self.assertEqual([v[1] for v in voices][:2], ["Anna", "zed"])
        self.assertEqual(voices[2][0], "1")

    def test_presets_sorted_by_speaker_then_neutral_first(self) -> None:
        voices = build_voice_list(
            [
                {"id": "a", "name": "Jane - Sad", "type": "preset",
                 "slug": "gb_jane_sad"},
                {"id": "b", "name": "Paul - Angry", "type": "preset",
                 "slug": "en_paul_angry"},
                {"id": "c", "name": "Paul - Neutral", "type": "preset",
                 "slug": "en_paul_neutral"},
                {"id": "d", "name": "Jane - Neutral", "type": "preset",
                 "slug": "gb_jane_neutral"},
            ]
        )
        self.assertEqual([v[0] for v in voices], ["d", "a", "c", "b"])

    def test_custom_voice_with_a_preset_like_name_is_untouched(self) -> None:
        voices = build_voice_list([
            {"id": "custom", "name": "en_paul_angry", "type": "custom",
             "slug": "en_paul_angry", "languages": ["nl"]},
            {"id": "preset", "name": "Anna", "type": "preset",
             "slug": "voice_1", "languages": ["en_GB"]},
        ])
        self.assertEqual(voices, [
            ("custom", "en_paul_angry"), ("preset", "Anna (English)")
        ])

    def test_voxtral_3_labels_use_normalised_language_metadata(self) -> None:
        voices = build_voice_list([
            {"id": "lotte", "name": "Lotte", "type": "preset",
             "slug": "voice_3145", "languages": ["nl"]},
            {"id": "marie", "name": "Marie", "type": "preset",
             "slug": "voice_592", "languages": ["fr_fr", "fr", "en-US"]},
            {"id": "sheila", "name": "Sheila", "type": "preset",
             "slug": "voice_478", "languages": ["en_us", "en_GB"]},
        ])
        self.assertEqual(voices, [
            ("lotte", "Lotte (Dutch)"),
            ("marie", "Marie (French, English)"),
            ("sheila", "Sheila (English)"),
        ])

    def test_missing_languages_do_not_add_a_suffix(self) -> None:
        for languages in (None, []):
            with self.subTest(languages=languages):
                self.assertEqual(build_voice_list([
                    {"id": "lotte", "name": "Lotte", "type": "preset",
                     "slug": "voice_3145", "languages": languages},
                ]), [("lotte", "Lotte")])
        self.assertEqual(build_voice_list([
            {"id": "lotte", "name": "Lotte", "type": "preset"},
        ]), [("lotte", "Lotte")])

    def test_unknown_language_code_stays_readable(self) -> None:
        self.assertEqual(build_voice_list([
            {"id": "voice", "name": "Voice", "type": "preset",
             "languages": ["xx_YY"]},
        ]), [("voice", "Voice (xx)")])

    def test_item_without_id_is_skipped_and_missing_name_uses_id(self) -> None:
        voices = build_voice_list([{"name": "no id"}, {"id": "only-id"}])
        self.assertEqual(voices, [("only-id", "only-id")])

    def test_empty_input(self) -> None:
        self.assertEqual(build_voice_list([]), [])


if __name__ == "__main__":
    unittest.main()
