"""Re-record tests/fixtures from the real Mistral API.

Usage (never in CI):  MISTRAL_API_KEY=... python scripts/record_fixtures.py
About 7 requests. IDs are replaced by fixed values so fixtures stay stable,
and the API key is never written to a file.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://api.mistral.ai/v1"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
KEY = os.environ.get("MISTRAL_API_KEY", "")


def call(method: str, path: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


def scrub(text: str) -> str:
    text = re.sub(r'"id":\s*"cmpl-[^"]+"', '"id":"cmpl-fixture"', text)
    text = re.sub(r'"(conversation_id)":\s*"[^"]+"', r'"\1":"conv_fixture"', text)
    return text.replace(KEY, "<redacted>")


def save(name: str, data: bytes) -> None:
    (OUT / name).write_text(scrub(data.decode("utf-8")), encoding="utf-8", newline="\n")
    print("wrote", name)


def main() -> int:
    if not KEY:
        print("Set MISTRAL_API_KEY first.")
        return 1
    models = json.loads(call("GET", "/models"))
    models["data"] = [
        {k: m.get(k) for k in ("id", "aliases", "capabilities")} for m in models["data"]
    ]
    save("models.json", json.dumps(models, indent=1).encode())

    voices = json.loads(call("GET", "/audio/voices?limit=5&offset=0"))
    voices["items"] = [{"id": v["id"], "name": v["name"]} for v in voices["items"]]
    voices["total"] = len(voices["items"])
    save("voices.json", json.dumps(voices, indent=1).encode())

    chat = {
        "model": "ministral-8b-latest",
        "stream": True,
        "max_tokens": 30,
        "messages": [{"role": "user", "content": "Zeg in 1 korte zin hoe warm het is."}],
    }
    save("chat_text.sse", call("POST", "/chat/completions", chat))

    tool = {
        "name": "HassTurnOn",
        "description": "Turn on a device",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
    }
    chat_tool = {
        **chat,
        "tools": [{"type": "function", "function": tool}],
        "tool_choice": "any",
        "messages": [{"role": "user", "content": "Doe de keukenlamp aan."}],
    }
    save("chat_tool_call.sse", call("POST", "/chat/completions", chat_tool))

    conv = {
        "model": "mistral-small-latest",
        "inputs": "Weer morgen in Amsterdam?",
        "tools": [{"type": "web_search"}],
        "store": False,
    }
    save("conversation_web_search.json", call("POST", "/conversations", conv))

    speech = {"model": "voxtral-mini-tts-2603", "input": "Test.", "voice_id": "en_paul_neutral"}
    save(
        "tts_stream.sse",
        call("POST", "/audio/speech", {**speech, "response_format": "wav", "stream": True}),
    )
    save("tts_batch.json", call("POST", "/audio/speech", {**speech, "response_format": "mp3"}))
    print("transcription.json is not recorded (needs a multipart upload); keep the hand-built one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
