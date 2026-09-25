"""Labels and ordering for the voice picker (stdlib only, no Home Assistant).

Mistral's preset voices are named ``<language>_<speaker>_<emotion>``, for
example ``en_paul_angry``. Shown as-is that is a long, unfriendly list, so
presets get a readable label (``Paul – Angry (English)``). Custom voices, made
in Mistral Studio, have free-form names and are left untouched.

Only the label changes; the ``voice_id`` sent to Mistral is never touched.
"""
from __future__ import annotations

import re

# Language part of a preset slug -> readable name. Unknown codes stay as-is.
_LANGUAGE_NAMES = {
    "en": "English",
    "gb": "British English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "de": "German",
    "hi": "Hindi",
    "ar": "Arabic",
}

_PRESET_SLUG = re.compile(r"^([a-z]{2})_([a-z]+)_([a-z]+)$")


def _parse_preset(name: str) -> tuple[str, str, str] | None:
    """Return (language, speaker, emotion) for a preset slug, else None."""
    match = _PRESET_SLUG.match(name)
    return match.groups() if match else None


def voice_label(name: str) -> str:
    """Return the picker label for a voice name."""
    parsed = _parse_preset(name)
    if parsed is None:
        return name
    language, speaker, emotion = parsed
    language_name = _LANGUAGE_NAMES.get(language, language)
    return f"{speaker.title()} – {emotion.title()} ({language_name})"


def _sort_key(name: str) -> tuple:
    parsed = _parse_preset(name)
    if parsed is None:
        # Custom voices first, alphabetically.
        return (0, name.casefold(), "", 0, "")
    language, speaker, emotion = parsed
    # Presets after that: by speaker, then language, "neutral" first.
    return (1, speaker, language, emotion != "neutral", emotion)


def build_voice_list(items: list[dict]) -> list[tuple[str, str]]:
    """Turn the account's voice items into sorted ``(voice_id, label)`` pairs.

    Items without an ``id`` are skipped. A voice without a ``name`` is shown
    under its id.
    """
    voices = [
        (item["id"], item.get("name") or item["id"])
        for item in items
        if item.get("id")
    ]
    voices.sort(key=lambda voice: _sort_key(voice[1]))
    return [(voice_id, voice_label(name)) for voice_id, name in voices]
