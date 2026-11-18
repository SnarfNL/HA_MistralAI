"""Labels and ordering for the voice picker (stdlib only, no Home Assistant).

Older preset slugs use ``<language>_<speaker>_<emotion>``, for example
``en_paul_angry``. These get a readable label (``Paul – Angry (English)``).
Voxtral 3 presets use their name and language metadata (``Lotte (Dutch)``).
Custom voice names are left untouched.

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
    "ca": "Catalan",
    "hu": "Hungarian",
    "ja": "Japanese",
    "ko": "Korean",
    "no": "Norwegian",
    "pl": "Polish",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_PRESET_SLUG = re.compile(r"^([a-z]{2})_([a-z]+)_([a-z]+)$")


def _parse_preset(name: str) -> tuple[str, str, str] | None:
    """Return (language, speaker, emotion) for a preset slug, else None."""
    match = _PRESET_SLUG.match(name)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def voice_label(name: str) -> str:
    """Return the picker label for a voice name."""
    parsed = _parse_preset(name)
    if parsed is None:
        return name
    language, speaker, emotion = parsed
    language_name = _LANGUAGE_NAMES.get(language, language)
    return f"{speaker.title()} – {emotion.title()} ({language_name})"


def _primary_language(code: str) -> str:
    """Normalise ISO and regional codes such as en_GB, en_us and en-US."""
    return code.strip().lower().replace("_", "-").split("-", 1)[0]


def _item_label(item: dict) -> str:
    """Use the slug for older presets and language metadata for new presets."""
    name = item.get("name") or item["id"]
    if item.get("type") != "preset":
        return name
    slug = item.get("slug") or ""
    if _parse_preset(slug) is not None:
        return voice_label(slug)
    codes = dict.fromkeys(
        _primary_language(code) for code in item.get("languages") or [] if code
    )
    languages = ", ".join(_LANGUAGE_NAMES.get(code, code) for code in codes)
    return f"{name} ({languages})" if languages else name


def _sort_key(item: dict) -> tuple:
    name = item.get("name") or item["id"]
    if item.get("type") != "preset":
        # Custom voices first, alphabetically.
        return (0, name.casefold(), "", 0, "")
    parsed = _parse_preset(item.get("slug") or "")
    if parsed is None:
        return (1, name.casefold(), "", 0, "")
    language, speaker, emotion = parsed
    # Presets after that: by speaker, then language, "neutral" first.
    return (1, speaker, language, emotion != "neutral", emotion)


def build_voice_list(
    items: list[dict], language: str | None = None
) -> list[tuple[str, str]]:
    """Turn the account's voice items into sorted ``(voice_id, label)`` pairs.

    Items without an ``id`` are skipped. A voice without a ``name`` is shown
    under its id. Language matches come first without filtering out other
    voices. Within each language group, custom voices precede presets.
    """
    voices = [item for item in items if item.get("id")]
    primary = _primary_language(language or "")
    voices.sort(
        key=lambda item: (
            primary not in {
                _primary_language(code) for code in item.get("languages") or []
            },
            _sort_key(item),
        )
    )
    return [(item["id"], _item_label(item)) for item in voices]
