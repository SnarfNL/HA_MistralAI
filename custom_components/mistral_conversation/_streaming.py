"""Pure-python streaming helpers for the TTS platform.

Kept dependency-free (stdlib only) so the sentence segmenter and SSE parser
can be unit-tested without bringing in Home Assistant. The TTS entity in
``tts.py`` consumes these.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any, AsyncGenerator, Protocol


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_LAST_WORD_BEFORE_TERMINATOR = re.compile(r"(\S+?)[.!?]+$")
_ABBREV: frozenset[str] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "sr", "jr", "st", "vs", "etc",
        "e.g", "i.e", "fig", "no", "vol", "pp",
    }
)


def has_speakable_content(text: str) -> bool:
    """Return True if *text* contains at least one letter or digit.

    Mistral's /v1/audio/speech rejects inputs that have nothing to vocalise
    (emoji-only, punctuation-only, whitespace-only) with HTTP 400 and the
    message ``Speech input is empty after sanitization``. We mirror that
    policy here so we never even attempt those sentences.
    """
    return any(c.isalpha() or c.isdigit() for c in text)


def pop_complete_sentences(
    buf: str, min_sentence_chars: int
) -> tuple[list[str], str]:
    """Extract complete sentences from *buf*; return ``(sentences, remainder)``.

    A sentence is ``<text><.!?>+<whitespace-or-EOB>`` plus three safeguards:

    1. The candidate must be at least ``min_sentence_chars`` after stripping
       (avoids firing TTS on stray ``OK.`` fragments).
    2. The candidate must contain at least one letter or digit — see
       :func:`has_speakable_content`. Emoji- or punctuation-only fragments
       slip past the length check (e.g. "🌿✨💫🌹🌷🌸💐🌼🌻🌺." is 12
       characters) but Mistral rejects them with HTTP 400.
    3. The token immediately preceding the terminator must not be a known
       abbreviation. False negatives just mean we keep buffering longer.

    Sentences are stripped. The remainder retains content but with leading
    whitespace stripped.
    """
    sentences: list[str] = []
    pos = 0
    while True:
        match = _SENTENCE_END.search(buf, pos)
        if not match:
            break
        end = match.end()
        candidate = buf[:end].strip()
        if len(candidate) < min_sentence_chars:
            pos = end
            continue
        if not has_speakable_content(candidate):
            pos = end
            continue
        last_word_match = _LAST_WORD_BEFORE_TERMINATOR.search(candidate)
        last_word = last_word_match.group(1).lower() if last_word_match else ""
        if last_word in _ABBREV:
            pos = end
            continue
        sentences.append(candidate)
        buf = buf[end:].lstrip()
        pos = 0
    return sentences, buf


# ---------------------------------------------------------------------------
# Server-Sent Events parser for /v1/audio/speech with stream=true
#
# Empirically observed frame shape:
#   event: speech.audio.delta
#   data: {"type":"speech.audio.delta","audio_data":"<base64>"}
#
#   event: speech.audio.done
#   data: {"type":"speech.audio.done","usage":{...}}
# ---------------------------------------------------------------------------


class _AsyncByteStream(Protocol):
    """Minimal protocol matching aiohttp.StreamReader.iter_any()."""

    def iter_any(self) -> AsyncGenerator[bytes, None]: ...


class _AsyncResponse(Protocol):
    """Minimal protocol matching aiohttp.ClientResponse for our needs."""

    content: _AsyncByteStream


async def iter_sse_audio_chunks(
    resp: Any,
) -> AsyncGenerator[bytes, None]:
    """Yield decoded audio bytes from a Mistral TTS SSE response.

    *resp* must expose ``.content.iter_any()`` returning an async iterator of
    bytes (matching ``aiohttp.ClientResponse``). Stops on
    ``speech.audio.done`` or stream EOF; frames without ``audio_data`` are
    skipped.
    """
    buffer = b""
    async for raw in resp.content.iter_any():
        buffer += raw
        while b"\n\n" in buffer:
            frame, buffer = buffer.split(b"\n\n", 1)
            event_name = ""
            data_str = ""
            for line in frame.split(b"\n"):
                decoded = line.decode("utf-8", errors="replace")
                if decoded.startswith("event: "):
                    event_name = decoded[7:]
                elif decoded.startswith("data: "):
                    data_str = decoded[6:]
            if event_name == "speech.audio.done":
                return
            if not data_str:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            audio_b64 = data.get("audio_data")
            if audio_b64:
                yield base64.b64decode(audio_b64)
