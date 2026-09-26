# Release 2 bundle (MA-11, MA-10, MA-14, MA-30) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the tests onto real Home Assistant, restructure the integration around one Mistral client and one entity base class, add diagnostics / a retired-model repair / a reconfigure flow, and make web search follow the selected model.

**Architecture:** Tests switch from `tests/_ha_stubs.py` to `pytest-homeassistant-custom-component` (real HA, API mocked with `aioclient_mock` and fixture files). `api.py` holds `MistralClient` (today's `mistral_request` logic as a method) and is reached through `entry.runtime_data`; `entity.py` holds `MistralEntity`. MA-14 adds `diagnostics.py`, `_models.py` + `repairs.py`, and a reconfigure step. MA-30 adds `supports_web_search()` and splits the options flow in two steps.

**Tech Stack:** Python 3.13/3.14, Home Assistant 2025.10–2026.9, aiohttp, voluptuous, pytest + pytest-homeassistant-custom-component, pytest-cov, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-09-26-release-2-ha-native-design.md`

## Global Constraints

- Minimum HA stays **2025.10.0** (`hacs.json`); code must work on 2025.10.0 and 2026.9.x.
- Test matrix: Python **3.13** + `pytest-homeassistant-custom-component==0.13.285` (HA 2025.10.0); Python **3.14** + `pytest-homeassistant-custom-component==0.13.366` (HA 2026.9.3).
- Coverage gates (latest leg): total `--cov-fail-under=90`; `config_flow.py` 100 %.
- `mypy` plain mode (not `--strict`) must pass.
- No entries added to `requirements` in `manifest.json`; do not bump `version`.
- Device identifiers (`{entry_id}_conversation`, `{entry_id}_stt`, `{entry_id}_tts`), device names and entity unique IDs (`_conversation`, `_ai_task`, `_stt`, `_tts`, `_refresh_voices`) must not change.
- All UI text in `strings.json` **and** `translations/en.json`, `nl.json`, `fr.json`, keys in sync.
- Never format an aiohttp exception with `%r`/`!r`; use `describe_error()`. Never put response bodies in user-visible errors.
- English in code, comments, commits, PR. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Existing test expectations are kept; only how HA and the API are set up may change.

## Review Focus

1. **Entry reload during a repair fix or migration** — writing options reloads the entry; a person expects exactly one reload, no loop, and the repair issue gone afterwards. Test: Task 8 `test_fix_flow_applies_replacement_and_clears_issue`, Task 10 `test_migration_does_not_loop`.
2. **Models list fetch fails or returns odd JSON** (free-tier 429, no `data` key) — a person expects no repair issue, no error spam, integration keeps working. Test: Task 8 `test_model_check_failure_changes_nothing` and `test_model_check_ignores_malformed_list`.
3. **Options saved with an unsupported model while mode/trigger were set** — a person expects the trigger phrases to still be there when switching back to a supported model. Test: Task 10 `test_unsupported_model_keeps_mode_and_trigger`.
4. **Diagnostics file leaking the key** — the key must not appear anywhere in the output, including inside recorded errors. Test: Task 7 `test_diagnostics_never_contains_api_key`.
5. **Reconfigure with the same key or while Mistral is down** — a person expects a clear form error, the old key kept. Test: Task 9 `test_reconfigure_cannot_connect_keeps_old_key`.

---

## File map

| File | Status | Responsibility |
|------|--------|----------------|
| `custom_components/__init__.py` | create | Makes `custom_components` a regular package for pytest imports |
| `requirements_test.txt`, `requirements_test_min.txt` | create | Test dependencies per matrix leg |
| `pyproject.toml` | modify | pytest, coverage and mypy config |
| `.github/workflows/tests.yml` | modify | Matrix, mypy, coverage gates |
| `tests/conftest.py` | create | HA fixtures: entry, setup, fixture loader |
| `tests/fixtures/*` | create | Mistral responses (SSE and JSON) |
| `scripts/record_fixtures.py` | create | Re-record fixtures with a real key |
| `tests/_ha_stubs.py` | delete | Replaced by real HA |
| `custom_components/mistral_conversation/api.py` | create (from `_api.py`) | `MistralClient`, error helpers |
| `custom_components/mistral_conversation/entity.py` | create | `MistralEntity`, device descriptions |
| `custom_components/mistral_conversation/_models.py` | create | Model availability + replacement rule + periodic check |
| `custom_components/mistral_conversation/repairs.py` | create | Fix flow for `model_retired` |
| `custom_components/mistral_conversation/diagnostics.py` | create | Diagnostics download |
| `__init__.py`, `conversation.py`, `ai_task.py`, `stt.py`, `tts.py`, `button.py`, `config_flow.py`, `const.py` | modify | Use runtime_data/client/base class; MA-14/MA-30 |
| `strings.json`, `translations/{en,nl,fr}.json` | modify | New steps, issue, reconfigure |
| `CLAUDE.md`, `CONTEXT.md`, `CHANGELOG.md`, `docs/backlog-and-review/backlog.md` | modify | Docs |

---

### Task 1: Real-HA test toolchain and a first setup test

**Files:**
- Create: `custom_components/__init__.py`, `requirements_test.txt`, `requirements_test_min.txt`, `tests/conftest.py`, `tests/fixtures/models.json`, `tests/fixtures/voices.json`, `tests/test_init.py`
- Modify: `pyproject.toml`, `.github/workflows/tests.yml`

**Interfaces:**
- Produces: fixtures `mock_config_entry` (MockConfigEntry, not yet set up), `setup_integration` (entry set up, returns entry), helper `load_fixture(name) -> str`, constants `API_KEY = "sk-test-key"`, `BASE = "https://api.mistral.ai/v1"` in `tests/conftest.py`.

- [ ] **Step 1: Check the test package on this machine (decision point)**

```bash
python -m venv .venv-ha
.venv-ha/Scripts/python -m pip install "pytest-homeassistant-custom-component==0.13.366" pytest-cov mypy ruff
.venv-ha/Scripts/python -c "import pytest_homeassistant_custom_component, homeassistant.const as c; print(c.__version__)"
```
Checked on 2026-09-26 on the maintainer's Windows 11 machine: the install works (HA 2026.9.3), but running pytest fails with `ModuleNotFoundError: No module named 'fcntl'` (then `resource`) — HA's test runner is Linux/macOS only, and stand-in modules are not a viable fix. WSL and Docker are not installed. **Use the test environment chosen by the maintainer at plan approval** (WSL, CI-only, or a remote Linux agent); every `.venv-ha/Scripts/python -m ...` command in this plan runs there as `python -m ...`. Add `.venv-ha/` to `.gitignore`.

- [ ] **Step 2: Add the files**

`custom_components/__init__.py`:
```python
"""Custom integrations (package marker for tests)."""
```

`requirements_test.txt`:
```
pytest-homeassistant-custom-component==0.13.366
pytest-cov
mypy
ruff
```

`requirements_test_min.txt`:
```
pytest-homeassistant-custom-component==0.13.285
pytest-cov
ruff
```

Append to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"

[tool.coverage.run]
source = ["custom_components.mistral_conversation"]

[tool.mypy]
python_version = "3.13"
ignore_missing_imports = true
check_untyped_defs = true
files = ["custom_components/mistral_conversation"]
```

`tests/fixtures/models.json` (shape of `GET /v1/models`, trimmed):
```json
{"object": "list", "data": [
  {"id": "ministral-8b-2410", "aliases": ["ministral-8b-latest"], "capabilities": {"completion_chat": true, "function_calling": true}},
  {"id": "ministral-3b-2410", "aliases": ["ministral-3b-latest"], "capabilities": {"completion_chat": true, "function_calling": true}},
  {"id": "ministral-14b-2512", "aliases": ["ministral-14b-latest"], "capabilities": {"completion_chat": true, "function_calling": true}},
  {"id": "mistral-small-2506", "aliases": ["mistral-small-latest"], "capabilities": {"completion_chat": true, "function_calling": true}},
  {"id": "mistral-medium-2508", "aliases": ["mistral-medium-latest"], "capabilities": {"completion_chat": true, "function_calling": true}},
  {"id": "mistral-large-2512", "aliases": ["mistral-large-latest"], "capabilities": {"completion_chat": true, "function_calling": true}}
]}
```

`tests/fixtures/voices.json` (shape of `GET /v1/audio/voices`):
```json
{"items": [
  {"id": "0b6f5c1e-0000-4000-8000-000000000001", "name": "en_paul_neutral"},
  {"id": "0b6f5c1e-0000-4000-8000-000000000002", "name": "fr_marie_happy"}
], "total": 2}
```

`tests/conftest.py`:
```python
"""Shared fixtures: a Mistral config entry and a set-up integration on real HA."""
from __future__ import annotations

from pathlib import Path

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.mistral_conversation.const import DOMAIN

API_KEY = "sk-test-key"
BASE = "https://api.mistral.ai/v1"
FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    """Return the text of tests/fixtures/<name>."""
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let HA load custom_components/mistral_conversation in every test."""


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A Mistral entry with default options, added to HA but not set up."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mistral AI Conversation",
        data={CONF_API_KEY: API_KEY},
        options={},
        unique_id=DOMAIN,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def setup_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> MockConfigEntry:
    """Set the entry up with /models and /audio/voices mocked."""
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
```

- [ ] **Step 3: Write the first test**

`tests/test_init.py`:
```python
"""Setup and unload through real Home Assistant."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from .conftest import BASE


async def test_setup_and_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("conversation.mistral_ai_conversation") is not None
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_setup_invalid_key_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{BASE}/models", status=401)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress()
    assert any(f["context"]["source"] == "reauth" for f in flows)


async def test_setup_timeout_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{BASE}/models", exc=TimeoutError())
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY
```

- [ ] **Step 4: Run it**

Run: `.venv-ha/Scripts/python -m pytest tests/test_init.py -v -p no:cacheprovider`
(The old test files still import `_ha_stubs`; running only `test_init.py` avoids them for now. `_ha_stubs` replaces `homeassistant` in `sys.modules`, so never run old and new tests in one process — Task 2 removes this conflict.)
Expected: 3 passed. If the entity id differs, read it from `hass.states.async_entity_ids("conversation")` and use that.

- [ ] **Step 5: CI matrix**

Replace the `tests` job in `.github/workflows/tests.yml`:
```yaml
jobs:
  tests:
    name: Tests (HA ${{ matrix.ha }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - ha: "2025.10 (minimum)"
            python: "3.13"
            requirements: requirements_test_min.txt
            latest: false
          - ha: "2026.9 (latest)"
            python: "3.14"
            requirements: requirements_test.txt
            latest: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install dependencies
        run: python -m pip install -r ${{ matrix.requirements }}
      - name: Ruff (lint)
        run: ruff check .
      - name: Pytest
        run: pytest tests
```
(mypy and the coverage gates are added in Task 4 and Task 6, once they can pass.)

- [ ] **Step 6: Commit**

```bash
git add custom_components/__init__.py requirements_test*.txt pyproject.toml .gitignore .github/workflows/tests.yml tests/conftest.py tests/fixtures tests/test_init.py
git commit -m "MA-11: run tests on real Home Assistant (toolchain, fixtures, first setup tests)"
```

---

### Task 2: Port the existing tests and remove the stubs

**Files:**
- Modify: all 15 `tests/test_*.py` files that import `_ha_stubs`
- Delete: `tests/_ha_stubs.py`

**Interfaces:**
- Consumes: `tests/conftest.py` fixtures (`hass`, `mock_config_entry`, `setup_integration`, `aioclient_mock`) from Task 1.
- Produces: the same test names and assertions, now on real HA. Later tasks change imports (`_api` → `api`) in these files.

**Porting rules (apply to every file):**
1. Remove `from . import _ha_stubs` and the `# ruff: noqa: I001` line that exists only for it.
2. Change `mistral_conversation` imports to `custom_components.mistral_conversation` (e.g. `from mistral_conversation import _api` → `from custom_components.mistral_conversation import _api`; `patch("mistral_conversation.tts....")` → `patch("custom_components.mistral_conversation.tts....")`).
3. Keep every `assert` / `self.assert*` and its expected value. A test may change how it builds its inputs (real `hass` fixture instead of `MagicMock()`, a `MockConfigEntry` instead of a `SimpleNamespace` entry), never what it checks.
4. Where a test used `hass = MagicMock()` and the code under test only reads `hass.data`, keep the MagicMock (the unit is still tested in isolation). Where the code calls real HA helpers (translations, chat_log, config entries), switch that test to a pytest function using the `hass` fixture.
5. `unittest.IsolatedAsyncioTestCase` classes may stay when they pass. If HA's plugin rejects them (lingering-task or event-loop errors), convert that class to plain `async def test_...` functions with the same names and bodies.
6. A test that only checked a stub's behaviour (not the integration) is deleted; list it in the commit message body with the reason.

Worked example (from `tests/test_api.py`), before:
```python
# ruff: noqa: I001 - `_ha_stubs` must run before the `mistral_conversation` import.
from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs
import aiohttp
from homeassistant.exceptions import HomeAssistantError
from mistral_conversation import _api
```
after:
```python
import aiohttp
from homeassistant.exceptions import HomeAssistantError

from custom_components.mistral_conversation import _api
```
For `async_spoken_error` tests that stubbed `async_get_translations`, replace the stub with the real function on the `hass` fixture:
```python
async def test_spoken_error_dutch(hass: HomeAssistant) -> None:
    err = _api.mistral_error("rate_limited")
    assert await _api.async_spoken_error(hass, err, "nl") == (
        "Mistral is tijdelijk overbelast. Probeer het zo nog eens."
    )
```

- [ ] **Step 1: Port the pure-helper files** — `test_streaming.py`, `test_voices.py`, `test_const.py`, `test_web_search_conversations.py`, `test_api.py`. Run: `.venv-ha/Scripts/python -m pytest tests/test_streaming.py tests/test_voices.py tests/test_const.py tests/test_web_search_conversations.py tests/test_api.py -v`. Expected: all pass (same count as before: 31 + 9 + 17 + 12 + 31 = 100, minus any deleted stub-only tests listed in the commit).
- [ ] **Step 2: Commit** — `git commit -m "MA-11: port helper tests to real HA"`
- [ ] **Step 3: Port the entity files** — `test_conversation.py`, `test_conversation_turn.py`, `test_web_search_routing.py`, `test_ai_task.py`, `test_stt.py`, `test_tts_pipeline.py`, `test_tts_voice.py`, `test_tts_voice_list.py`, `test_release1_errors.py`, `test_review_fixes.py`. Entities are built as `Entity(hass, entry)` today; keep that signature for now (Task 5 changes it).
- [ ] **Step 4: Delete `tests/_ha_stubs.py`** and run the whole suite: `.venv-ha/Scripts/python -m pytest -v`. Expected: all pass; total = 235 + 3 (Task 1) − deleted stub-only tests.
- [ ] **Step 5: Update CLAUDE.md "Commands" and the `tests/` line in "Layout"**:
```
tests/             pytest on real HA (pytest-homeassistant-custom-component); API mocked with aioclient_mock + tests/fixtures
```
```bash
pip install -r requirements_test.txt
ruff check .
pytest                      # add --cov for the coverage gates
mypy
```
- [ ] **Step 6: Commit** — `git commit -m "MA-11: port entity tests to real HA and remove the HA stubs"`

---

### Task 3: Fixtures, recording script and platform tests through HA

**Files:**
- Create: `tests/fixtures/chat_text.sse`, `chat_tool_call.sse`, `conversation_web_search.json`, `tts_stream.sse`, `tts_batch.json`, `transcription.json`, `scripts/record_fixtures.py`, `tests/test_platforms.py`

**Interfaces:**
- Consumes: `setup_integration`, `load_fixture`, `BASE` (Task 1).
- Produces: fixture files used by Tasks 5–10 tests.

- [ ] **Step 1: Hand-built fixtures**

`tests/fixtures/chat_text.sse` (each event ends with a blank line; file ends with a newline):
```
data: {"id":"cmpl-fixture","object":"chat.completion.chunk","model":"ministral-8b-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"cmpl-fixture","object":"chat.completion.chunk","model":"ministral-8b-latest","choices":[{"index":0,"delta":{"content":"Het is 21 graden"},"finish_reason":null}]}

data: {"id":"cmpl-fixture","object":"chat.completion.chunk","model":"ministral-8b-latest","choices":[{"index":0,"delta":{"content":" in de woonkamer."},"finish_reason":"stop"}]}

data: [DONE]

```

`tests/fixtures/chat_tool_call.sse`:
```
data: {"id":"cmpl-fixture","object":"chat.completion.chunk","model":"ministral-8b-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"cmpl-fixture","object":"chat.completion.chunk","model":"ministral-8b-latest","choices":[{"index":0,"delta":{"tool_calls":[{"id":"abc123xyz","index":0,"function":{"name":"HassTurnOn","arguments":"{\"name\": \"keukenlamp\"}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

```

`tests/fixtures/conversation_web_search.json`:
```json
{"conversation_id": "conv_fixture", "outputs": [
  {"type": "tool.execution", "name": "web_search"},
  {"type": "message.output", "role": "assistant", "content": [
    {"type": "text", "text": "Morgen wordt het 18 graden en zonnig in Amsterdam."}
  ]}
]}
```

`tests/fixtures/tts_stream.sse` — generate with this snippet (a 44-byte WAV header + 480 bytes of silence, split in two chunks) and commit the output:
```python
import base64, struct
header = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF)
pcm = bytes(480)
chunks = [header + pcm[:240], pcm[240:]]
lines = [f'event: speech.audio.delta\ndata: {{"type":"speech.audio.delta","audio_data":"{base64.b64encode(c).decode()}"}}\n\n' for c in chunks]
lines.append('event: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n')
open("tests/fixtures/tts_stream.sse", "w", newline="\n").write("".join(lines))
```

`tests/fixtures/tts_batch.json`:
```json
{"audio_data": "SUQzBAAAAAAAAA=="}
```

`tests/fixtures/transcription.json`:
```json
{"model": "voxtral-mini-latest", "text": "doe de keukenlamp aan", "language": "nl"}
```

- [ ] **Step 2: Recording script** `scripts/record_fixtures.py`:
```python
"""Re-record tests/fixtures from the real Mistral API.

Usage (never in CI):  MISTRAL_API_KEY=... python scripts/record_fixtures.py
About 8 requests. IDs are replaced by fixed values so fixtures stay stable.
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
    chat = {"model": "ministral-8b-latest", "stream": True, "max_tokens": 30,
            "messages": [{"role": "user", "content": "Zeg in 1 korte zin hoe warm het is."}]}
    save("chat_text.sse", call("POST", "/chat/completions", chat))
    tool = {"name": "HassTurnOn", "description": "Turn on a device",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}
    chat_tool = {**chat, "tools": [{"type": "function", "function": tool}], "tool_choice": "any",
                 "messages": [{"role": "user", "content": "Doe de keukenlamp aan."}]}
    save("chat_tool_call.sse", call("POST", "/chat/completions", chat_tool))
    conv = {"model": "mistral-small-latest", "inputs": "Weer morgen in Amsterdam?",
            "tools": [{"type": "web_search"}], "store": False}
    save("conversation_web_search.json", call("POST", "/conversations", conv))
    speech = {"model": "voxtral-mini-tts-2603", "input": "Test.", "voice_id": "en_paul_neutral"}
    save("tts_stream.sse", call("POST", "/audio/speech", {**speech, "response_format": "wav", "stream": True}))
    save("tts_batch.json", call("POST", "/audio/speech", {**speech, "response_format": "mp3"}))
    print("transcription.json is not recorded (needs a multipart upload); keep the hand-built one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Platform tests through HA** `tests/test_platforms.py`:
```python
"""Each platform end to end through HA, with the API mocked from fixtures."""
from __future__ import annotations

from homeassistant.components import conversation, stt, tts
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from .conftest import BASE, load_fixture

AGENT = "conversation.mistral_ai_conversation"


async def test_conversation_text_reply(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(f"{BASE}/chat/completions", text=load_fixture("chat_text.sse"))
    result = await conversation.async_converse(hass, "hoe warm is het?", None, None, "nl", agent_id=AGENT)
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Het is 21 graden in de woonkamer."


async def test_conversation_rate_limit_is_spoken(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(f"{BASE}/chat/completions", status=429, headers={"Retry-After": "0"})
    result = await conversation.async_converse(hass, "hallo", None, None, "nl", agent_id=AGENT)
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.speech["plain"]["speech"] == "Mistral is tijdelijk overbelast. Probeer het zo nog eens."


async def test_tts_batch_returns_mp3(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(f"{BASE}/audio/speech", text=load_fixture("tts_batch.json"))
    engine = hass.states.async_entity_ids("tts")[0]
    extension, data = await tts.async_get_media_source_audio(
        hass, tts.generate_media_source_id(hass, "Test.", engine, "en", {"voice": "en_paul_neutral"}, cache=False)
    )
    assert extension == "mp3"
    assert data.startswith(b"ID3")


async def test_stt_transcribes(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(f"{BASE}/audio/transcriptions", text=load_fixture("transcription.json"))
    engine = stt.async_get_speech_to_text_engine(hass, hass.states.async_entity_ids("stt")[0])
    meta = stt.SpeechMetadata(
        language="nl", format=stt.AudioFormats.WAV, codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16, sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    async def audio():
        yield bytes(320)

    result = await engine.async_process_audio_stream(meta, audio())
    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "doe de keukenlamp aan"
```
Add one test per remaining path that coverage reports as missed (tool-call turn with `chat_tool_call.sse` then `chat_text.sse`; streaming TTS with `tts_stream.sse`; AI Task with `chat_text.sse`; refresh-voices button press), each asserting the user-visible result (reply text, audio bytes start with `b"RIFF"`, returned data, button raises nothing).

- [ ] **Step 4: Run with coverage**

Run: `.venv-ha/Scripts/python -m pytest --cov --cov-report=term-missing`
Expected: all pass. Note the total; add tests for the largest missed blocks until ≥ 90 %. Then `coverage report --include="*/config_flow.py" --fail-under=100` — add config flow tests (user step success / invalid_auth / cannot_connect / unknown / already configured; reauth success and failure; options save) in `tests/test_config_flow.py` until it passes.

- [ ] **Step 5: Add the gates to CI** (latest leg only), after the Pytest step in `tests.yml`:
```yaml
      - name: Coverage gates
        if: matrix.latest
        run: |
          pytest --cov --cov-report=term-missing --cov-fail-under=90
          coverage report --include="*/config_flow.py" --fail-under=100
```

- [ ] **Step 6: Commit** — `git commit -m "MA-11: fixtures, recording script, platform tests and coverage gates"`

---

### Task 4: MA-10 — `MistralClient` and `entry.runtime_data`

**Files:**
- Create: `custom_components/mistral_conversation/api.py` (`git mv _api.py api.py`, then edit)
- Modify: `__init__.py`, `conversation.py`, `ai_task.py`, `stt.py`, `tts.py`, `button.py`, `config_flow.py`, tests importing `_api`
- Test: `tests/test_api.py` (existing, adapted), `tests/test_init.py`

**Interfaces:**
- Produces (used by Tasks 5–10):
  - `class MistralClient` in `api.py`:
    - `__init__(self, hass: HomeAssistant, entry: ConfigEntry, session: aiohttp.ClientSession, api_key: str, errors: deque[dict[str, Any]]) -> None`
    - `request(self, method: str, path: str, *, timeout: float, source: str, multipart: bool = False, data_factory: Callable[[], Any] | None = None, log_context: str = "", log_level: int = logging.ERROR, **kwargs: Any) -> AbstractAsyncContextManager[aiohttp.ClientResponse]`
    - `chat_completions(self, payload: dict[str, Any], *, source: str) -> AbstractAsyncContextManager[aiohttp.ClientResponse]`
    - `start_conversation(self, payload: dict[str, Any], *, model: str) -> AbstractAsyncContextManager[...]` / `append_conversation(self, conv_id: str, payload: dict[str, Any], *, model: str)` / `delete_conversation(self, conv_id: str)`
    - `speech(self, payload: dict[str, Any], *, voice: str, timeout: float)`
    - `transcribe(self, data_factory: Callable[[], aiohttp.FormData])`
    - `async list_voices(self, offset: int, limit: int) -> dict[str, Any]`
    - `async list_models(self) -> list[dict[str, Any]]`
    - `@staticmethod async validate_key(session: aiohttp.ClientSession, api_key: str) -> tuple[str | None, str]` → `(None, "")`, `("invalid_auth", detail)` or `("cannot_connect", detail)`
    - `record(self, source: str, err: BaseException, status: int | None = None) -> None`
  - Module functions unchanged: `mistral_error`, `is_unrecoverable`, `describe_error`, `read_json`, `translate_stream`, `async_spoken_error`, `_sleep`, `RETRY_DELAYS`, `MAX_RETRY_AFTER`.
  - In `__init__.py`: `@dataclass class MistralRuntimeData` with fields `client: MistralClient`, `errors: deque[dict[str, Any]]`, `web_search_convs`, `tts_entity: Any | None = None`, `models: list[dict[str, Any]] | None = None`; `type MistralConfigEntry = ConfigEntry[MistralRuntimeData]`.
  - Error record shape: `{"time": str (ISO UTC), "source": str, "status": int | None, "error": str (translation key or exception class name)}`.

- [ ] **Step 1: Write failing tests** in `tests/test_api.py` (add; existing tests are adapted in Step 4):
```python
from collections import deque

from custom_components.mistral_conversation.api import MistralClient


def _client(hass, entry, session):
    return MistralClient(hass, entry, session, "sk-secret-key", deque(maxlen=10))


async def test_client_records_rate_limit(hass, mock_config_entry, monkeypatch) -> None:
    monkeypatch.setattr(api, "_sleep", AsyncMock())
    session = _Session(_Response(429), _Response(429), _Response(429))
    client = _client(hass, mock_config_entry, session)
    with pytest.raises(HomeAssistantError):
        async with client.request("get", "/models", timeout=5, source="setup"):
            pass
    [record] = client.errors
    assert record["source"] == "setup"
    assert record["status"] == 429
    assert record["error"] == "rate_limited"
    assert "sk-secret-key" not in str(record)


async def test_client_does_not_record_errors_from_caller_code(hass, mock_config_entry) -> None:
    client = _client(hass, mock_config_entry, _Session(_Response(200)))
    with pytest.raises(ValueError):
        async with client.request("get", "/models", timeout=5, source="conversation"):
            raise ValueError("tool failed")
    assert not client.errors


async def test_client_multipart_sends_only_auth_header(hass, mock_config_entry) -> None:
    session = _Session(_Response(200))
    client = _client(hass, mock_config_entry, session)
    async with client.request("post", "/audio/transcriptions", timeout=5, source="stt", multipart=True):
        pass
    _method, kwargs = session.calls[0]
    assert kwargs["headers"] == {"Authorization": "Bearer sk-secret-key"}


async def test_validate_key_results(aioclient_mock, hass) -> None:
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
    session = async_get_clientsession(hass)
    aioclient_mock.get("https://api.mistral.ai/v1/models", status=401)
    error, _ = await MistralClient.validate_key(session, "k")
    assert error == "invalid_auth"
    aioclient_mock.clear_requests()
    aioclient_mock.get("https://api.mistral.ai/v1/models", exc=TimeoutError())
    error, detail = await MistralClient.validate_key(session, "k")
    assert error == "cannot_connect"
    assert "k" not in detail.split()
```
(`_Session`/`_Response` are the existing fakes in this file; `api` is `from custom_components.mistral_conversation import api`.)

- [ ] **Step 2: Run** `pytest tests/test_api.py -v` — Expected: FAIL (`ModuleNotFoundError: ...api` / `MistralClient`).

- [ ] **Step 3: Implement** — `git mv custom_components/mistral_conversation/_api.py custom_components/mistral_conversation/api.py`, keep every function, remove `mistral_request`, and add:
```python
from collections import deque
from contextlib import AbstractAsyncContextManager

from homeassistant.util import dt as dt_util

from .const import MISTRAL_API_BASE


class MistralClient:
    """All calls to the Mistral API for one config entry.

    ``request`` keeps the old ``mistral_request`` behaviour: network errors and
    timeouts → ``cannot_connect``; 401 → reauth + ``invalid_auth``; 429 retried
    up to twice → ``rate_limited``; other errors → ``api_error`` with the body
    only in the log. Errors raised while *sending* are recorded for
    diagnostics; errors from the caller's own code inside ``async with`` are not.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        session: aiohttp.ClientSession,
        api_key: str,
        errors: deque[dict[str, Any]],
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._session = session
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._json_headers = {**self._auth, "Content-Type": "application/json"}
        self.errors = errors

    def record(self, source: str, err: BaseException, status: int | None = None) -> None:
        """Remember an error for the diagnostics file (never a body or header)."""
        self.errors.append(
            {
                "time": dt_util.utcnow().isoformat(),
                "source": source,
                "status": status,
                "error": getattr(err, "translation_key", None) or type(err).__name__,
            }
        )

    async def _open(
        self,
        method: str,
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        data_factory: Callable[[], Any] | None,
        log_level: int,
        kwargs: dict[str, Any],
    ) -> tuple[AsyncExitStack, Any]:
        """Send with 429 retries; return the open response and its exit stack."""
        attempt = 0
        while True:
            stack = AsyncExitStack()
            if data_factory is not None:
                kwargs["data"] = data_factory()
            try:
                resp = await stack.enter_async_context(
                    self._session.request(
                        method.upper(),
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                        **kwargs,
                    )
                )
            except (aiohttp.ClientError, TimeoutError) as err:
                await stack.aclose()
                _LOGGER.log(
                    log_level, "Mistral request to %s failed: %s", url, describe_error(err)
                )
                raise mistral_error("cannot_connect") from err

            delay = None
            if resp.status == 429 and attempt < len(RETRY_DELAYS):
                delay = _retry_delay(resp, attempt)
            if delay is None:
                return stack, resp
            await stack.aclose()
            attempt += 1
            _LOGGER.debug("Mistral rate limit; retry %d in %.1fs", attempt, delay)
            await _sleep(delay)

    @asynccontextmanager
    async def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        source: str,
        multipart: bool = False,
        data_factory: Callable[[], Any] | None = None,
        log_context: str = "",
        log_level: int = logging.ERROR,
        **kwargs: Any,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Send a request to ``MISTRAL_API_BASE + path`` and yield the response."""
        url = f"{MISTRAL_API_BASE}{path}"
        headers = self._auth if multipart else self._json_headers
        status: int | None = None
        try:
            stack, resp = await self._open(
                method, url, timeout=timeout, headers=headers,
                data_factory=data_factory, log_level=log_level, kwargs=kwargs,
            )
            status = resp.status
            try:
                await _raise_for_status(
                    self._hass, self._entry, resp, url, log_context, log_level
                )
            except BaseException:
                await stack.aclose()
                raise
        except HomeAssistantError as err:
            self.record(source, err, status)
            raise
        async with stack:
            yield resp

    def chat_completions(
        self, payload: dict[str, Any], *, source: str
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "post", "/chat/completions", json=payload, timeout=90, source=source,
            log_context=f"model={payload.get('model')}",
        )

    def start_conversation(
        self, payload: dict[str, Any], *, model: str
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "post", "/conversations", json=payload, timeout=90,
            source="conversation", log_context=f"model={model}",
        )

    def append_conversation(
        self, conv_id: str, payload: dict[str, Any], *, model: str
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "post", f"/conversations/{conv_id}", json=payload, timeout=90,
            source="conversation", log_context=f"model={model}",
        )

    def delete_conversation(
        self, conv_id: str
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "delete", f"/conversations/{conv_id}", timeout=15,
            source="conversation", log_level=logging.DEBUG,
        )

    def speech(
        self, payload: dict[str, Any], *, voice: str, timeout: float
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "post", "/audio/speech", json=payload, timeout=timeout,
            source="tts", log_context=f"voice={voice}",
        )

    def transcribe(
        self, data_factory: Callable[[], aiohttp.FormData]
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self.request(
            "post", "/audio/transcriptions", timeout=60, source="stt",
            multipart=True, data_factory=data_factory,
        )

    async def list_voices(self, offset: int, limit: int) -> dict[str, Any]:
        async with self.request(
            "get", "/audio/voices", params={"limit": limit, "offset": offset},
            timeout=10, source="tts", log_level=logging.WARNING,
        ) as resp:
            return await read_json(resp)

    async def list_models(self) -> list[dict[str, Any]]:
        async with self.request(
            "get", "/models", timeout=10, source="models", log_level=logging.DEBUG,
        ) as resp:
            data = await read_json(resp)
        models = data.get("data")
        return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []

    @staticmethod
    async def validate_key(
        session: aiohttp.ClientSession, api_key: str
    ) -> tuple[str | None, str]:
        """Check an API key before an entry (or its client) exists.

        Returns ``(None, "")`` when valid, else ``("invalid_auth" |
        "cannot_connect", safe detail for logs/messages)``.
        """
        try:
            async with session.get(
                f"{MISTRAL_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401:
                    return "invalid_auth", "HTTP 401"
                if resp.status != 200:
                    return "cannot_connect", f"HTTP {resp.status}"
        except (aiohttp.ClientError, TimeoutError) as err:
            return "cannot_connect", describe_error(err)
        return None, ""
```
Update the module docstring: replace "goes through ``mistral_request``" with "goes through ``MistralClient.request``". Remove `from .const import DOMAIN` only if unused (it is still used by `mistral_error`).

In `__init__.py`:
```python
from collections import deque

from .api import MistralClient

type MistralConfigEntry = ConfigEntry[MistralRuntimeData]


@dataclass
class MistralRuntimeData:
    """Shared runtime data for a config entry."""

    client: MistralClient
    errors: deque[dict[str, Any]]
    web_search_convs: WebSearchConversations = field(
        default_factory=WebSearchConversations
    )
    tts_entity: Any | None = field(default=None)
    # Last successful GET /v1/models result (MA-14); None until fetched.
    models: list[dict[str, Any]] | None = field(default=None)


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    api_key = entry.data[CONF_API_KEY]
    session = async_get_clientsession(hass)
    error, detail = await MistralClient.validate_key(session, api_key)
    if error == "invalid_auth":
        raise ConfigEntryAuthFailed("Invalid Mistral AI API key")
    if error:
        raise ConfigEntryNotReady(f"Cannot connect to Mistral AI: {detail}")

    errors: deque[dict[str, Any]] = deque(maxlen=10)
    entry.runtime_data = MistralRuntimeData(
        client=MistralClient(hass, entry, session, api_key, errors), errors=errors
    )
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload a config entry (HA drops runtime_data after a successful unload)."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
```
(Keep the old docstring's point: platforms unload before runtime data is gone — HA guarantees this for `runtime_data`.)

Call sites — replace each `mistral_request(self.hass, self._entry, <method>, f"{MISTRAL_API_BASE}<path>", ...)`:
- `conversation.py` `_stream_and_collect`: `async with self._runtime.client.chat_completions(payload, source="conversation") as resp:`
- `conversation.py` `_conversations_chat`: `cm = client.append_conversation(mistral_conv_id, payload, model=model) if mistral_conv_id else client.start_conversation(payload, model=model)` then `async with cm as resp:`; payload still passed through `_sanitize`.
- `conversation.py` `_delete_mistral_conversation`: `async with self._runtime.client.delete_conversation(mistral_id):`
- `ai_task.py`: `async with self._runtime.client.chat_completions(payload, source="ai_task") as resp:`
- `tts.py` batch: `self._runtime.client.speech(payload, voice=voice, timeout=30)`; stream: `...speech(payload, voice=voice, timeout=60)`; voices loop: `data = await self._runtime.client.list_voices(offset, limit)`.
- `stt.py`: `async with self._entry.runtime_data.client.transcribe(build_form) as resp:` (drop `auth_header`).
- `config_flow.py` `_test_api_key`: `error, _detail = await MistralClient.validate_key(async_get_clientsession(self.hass), api_key); return error` inside the existing `try/except Exception → "unknown"`.
- Every `self.hass.data[DOMAIN][self._entry.entry_id]` / `hass.data.get(DOMAIN, {}).get(...)` → `self._entry.runtime_data`. In `tts.async_will_remove_from_hass` use `getattr(self._entry, "runtime_data", None)`.
- Remove now-unused `MISTRAL_API_BASE`, `DOMAIN`, `aiohttp` imports per file (ruff will flag them).

- [ ] **Step 4: Adapt existing tests** — replace `_api` imports with `api`; tests that put a runtime in `hass.data[DOMAIN][entry_id]` now set `entry.runtime_data = MistralRuntimeData(client=MistralClient(hass, entry, session, API_KEY, errors), errors=errors)`; tests that called `mistral_request(hass, entry, "get", URL, ...)` call `client.request("get", "/x", ..., source="test")` and assert on the same outcomes (URL in the fake session's calls is now `https://api.mistral.ai/v1/x`, which equals the old `URL` constant).

- [ ] **Step 5: Run** `pytest -v` and `ruff check .` — Expected: all pass.

- [ ] **Step 6: Update CLAUDE.md "Code conventions"**: "Calls to the Mistral API go through `MistralClient` in `api.py` (`entry.runtime_data.client`)…"; replace the "only exceptions" bullet with: "API keys are checked with `MistralClient.validate_key()` (setup, config flow, reauth, reconfigure); no code calls the session directly." Update the Layout block (`api.py`, runtime_data).

- [ ] **Step 7: Commit** — `git commit -m "MA-10: MistralClient in api.py and entry.runtime_data instead of hass.data"`

---

### Task 5: MA-10 — `MistralEntity` base, `PARALLEL_UPDATES`, mypy

**Files:**
- Create: `custom_components/mistral_conversation/entity.py`, `tests/test_entity.py`
- Modify: `conversation.py`, `ai_task.py`, `stt.py`, `tts.py`, `button.py`, `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `MistralConfigEntry`, `MistralRuntimeData`, `MistralClient` (Task 4).
- Produces: `class MistralEntity(Entity)` with `__init__(self, entry: MistralConfigEntry, unique_suffix: str)`, attributes `_entry`, properties `_runtime -> MistralRuntimeData`, `_client -> MistralClient`, class attribute `_device: DeviceSpec`, overridable `_device_model() -> str`; `DeviceSpec` instances `CONVERSATION_DEVICE`, `STT_DEVICE`, `TTS_DEVICE`. Entity constructors become `Entity(entry)`.

- [ ] **Step 1: Failing test** `tests/test_entity.py`:
```python
"""Device and entity registry entries must survive the refactor unchanged."""
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er

from custom_components.mistral_conversation.const import DOMAIN


async def test_device_identifiers_and_unique_ids_unchanged(
    hass: HomeAssistant, setup_integration
) -> None:
    entry = setup_integration
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    assert {next(iter(d.identifiers)) for d in devices} == {
        (DOMAIN, f"{entry.entry_id}_conversation"),
        (DOMAIN, f"{entry.entry_id}_stt"),
        (DOMAIN, f"{entry.entry_id}_tts"),
    }
    assert {d.name for d in devices} == {
        "Mistral AI Conversation", "Mistral AI STT", "Mistral AI TTS",
    }
    unique_ids = {e.unique_id for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)}
    assert unique_ids == {
        f"{entry.entry_id}_{s}"
        for s in ("conversation", "ai_task", "stt", "tts", "refresh_voices")
    }


async def test_conversation_device_model_follows_options(hass: HomeAssistant, setup_integration) -> None:
    entry = setup_integration
    device = dr.async_get(hass).async_get_device({(DOMAIN, f"{entry.entry_id}_conversation")})
    assert device.model == "ministral-8b-latest"
```
Run it first against the current code: it must PASS before the refactor (it pins today's behaviour). Then keep it green.

- [ ] **Step 2: Implement** `entity.py`:
```python
"""Base class for all Mistral entities: config entry, client and device info."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity

from .const import DOMAIN

if TYPE_CHECKING:
    from . import MistralConfigEntry, MistralRuntimeData
    from .api import MistralClient


@dataclass(frozen=True)
class DeviceSpec:
    """One of the three service devices this integration creates."""

    suffix: str
    name: str
    model: str
    configuration_url: str


class MistralEntity(Entity):
    """Shared plumbing: ``_entry``, ``_runtime``, ``_client`` and ``device_info``."""

    _attr_has_entity_name = True
    _device: DeviceSpec

    def __init__(self, entry: MistralConfigEntry, unique_suffix: str) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"

    @property
    def _runtime(self) -> MistralRuntimeData:
        return self._entry.runtime_data

    @property
    def _client(self) -> MistralClient:
        return self._entry.runtime_data.client

    def _device_model(self) -> str:
        return self._device.model

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_{self._device.suffix}")},
            name=self._device.name,
            manufacturer="Mistral AI",
            model=self._device_model(),
            entry_type=DeviceEntryType.SERVICE,
            configuration_url=self._device.configuration_url,
        )
```
Add in `const.py` nothing; define the specs at the bottom of `entity.py`:
```python
from .const import STT_MODEL, TTS_MODEL  # noqa: E402 (place with the other imports)

CONVERSATION_DEVICE = DeviceSpec(
    "conversation", "Mistral AI Conversation", "", "https://console.mistral.ai"
)
STT_DEVICE = DeviceSpec(
    "stt", "Mistral AI STT", STT_MODEL,
    "https://docs.mistral.ai/capabilities/audio_transcription",
)
TTS_DEVICE = DeviceSpec(
    "tts", "Mistral AI TTS", TTS_MODEL,
    "https://docs.mistral.ai/capabilities/audio_generation",
)
```
(put the `STT_MODEL, TTS_MODEL` import in the normal import block.)

Entities:
```python
# conversation.py
PARALLEL_UPDATES = 0

class MistralConversationEntity(MistralEntity, ConversationEntity):
    _attr_name = None
    _attr_supports_streaming = True
    _device = CONVERSATION_DEVICE

    def __init__(self, entry: MistralConfigEntry) -> None:
        super().__init__(entry, "conversation")
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = ConversationEntityFeature.CONTROL

    def _device_model(self) -> str:
        return self._entry.options.get(CONF_MODEL, DEFAULT_MODEL)
```
Same pattern: `MistralAITaskEntity(MistralEntity, AITaskEntity)` suffix `"ai_task"`, `_device = CONVERSATION_DEVICE`, same `_device_model`; `MistralSTTEntity(MistralEntity, SpeechToTextEntity)` suffix `"stt"`, `_device = STT_DEVICE`; `MistralTTSEntity(MistralEntity, TextToSpeechEntity)` suffix `"tts"`, `_device = TTS_DEVICE`; `MistralRefreshVoicesButton(MistralEntity, ButtonEntity)` suffix `"refresh_voices"`, `_device = TTS_DEVICE`. Delete each entity's `hass`/`_entry`/`_attr_unique_id` assignments, its `_runtime` property, its `device_info`, and `tts_device_info()`. Keep `_attr_name` values as they are. Each platform's `async_setup_entry` becomes `async_add_entities([Entity(config_entry)])` with `config_entry: MistralConfigEntry`. Add `PARALLEL_UPDATES = 0` at module level in `conversation.py`, `ai_task.py`, `stt.py`, `tts.py`, `button.py`.

- [ ] **Step 3: Update tests** that construct entities: `Entity(hass, entry)` → `Entity(entry)` and set `entity.hass = hass` where the test uses it.

- [ ] **Step 4: mypy** — run `.venv-ha/Scripts/python -m mypy`; fix reported errors with real types (no `# type: ignore` unless the error is inside HA's own stubs, then with an error code and a comment). Add to `tests.yml` after Ruff:
```yaml
      - name: Mypy
        if: matrix.latest
        run: mypy
```

- [ ] **Step 5: Run** `pytest -v`, `ruff check .`, `mypy` — Expected: all pass.

- [ ] **Step 6: Commit** — `git commit -m "MA-10: MistralEntity base class, PARALLEL_UPDATES and mypy in CI"`

---

### Task 6: (folded) — coverage gate verification after MA-10

- [ ] **Step 1:** Run `pytest --cov --cov-fail-under=90` and `coverage report --include="*/config_flow.py" --fail-under=100`. If the refactor lowered coverage, add tests for the missed lines in the existing test files. Expected: both gates pass.
- [ ] **Step 2:** Commit only if tests were added: `git commit -m "MA-11: keep coverage gates green after the refactor"`

---

### Task 7: MA-14 — diagnostics

**Files:**
- Create: `custom_components/mistral_conversation/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `entry.runtime_data.errors`, `.models`, `.tts_entity`, `MistralClient.record` (Task 4); `model_available` (Task 8 — define it in this task in `_models.py` so the order works; Task 8 extends the file).
- Produces: `async_get_config_entry_diagnostics(hass, entry) -> dict[str, Any]`; `_models.model_available(model: str, models: list[dict[str, Any]]) -> bool`.

- [ ] **Step 1: Failing tests** `tests/test_diagnostics.py`:
```python
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.mistral_conversation.api import mistral_error
from custom_components.mistral_conversation.diagnostics import (
    async_get_config_entry_diagnostics,
)

from .conftest import API_KEY


async def test_diagnostics_content(hass: HomeAssistant, setup_integration) -> None:
    entry = setup_integration
    entry.runtime_data.client.record("tts", mistral_error("api_error", status="404"), 404)
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["entry"]["data"]["api_key"] == "**REDACTED**"
    assert diag["entry"]["options"] == dict(entry.options)
    assert diag["versions"]["home_assistant"]
    assert diag["versions"]["integration"]
    assert diag["recent_errors"][-1]["source"] == "tts"
    assert diag["recent_errors"][-1]["status"] == 404
    assert diag["recent_errors"][-1]["error"] == "api_error"
    assert diag["voices_loaded"] == 2


async def test_diagnostics_never_contains_api_key(hass: HomeAssistant, setup_integration) -> None:
    entry = setup_integration
    entry.runtime_data.client.record("setup", HomeAssistantError(f"bad {API_KEY}"))
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert API_KEY not in str(diag)


async def test_diagnostics_keeps_last_ten_errors(hass: HomeAssistant, setup_integration) -> None:
    entry = setup_integration
    for i in range(12):
        entry.runtime_data.client.record("conversation", mistral_error("rate_limited"), 429)
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert len(diag["recent_errors"]) == 10
```
Run: `pytest tests/test_diagnostics.py -v` — Expected: FAIL (module not found).

- [ ] **Step 2: Implement** `_models.py` (first part):
```python
"""Which Mistral models exist, and what to suggest when one is retired (MA-14)."""
from __future__ import annotations

from typing import Any


def model_available(model: str, models: list[dict[str, Any]]) -> bool:
    """True when *model* is an id or alias in a GET /v1/models list."""
    for item in models:
        aliases = item.get("aliases")
        if model == item.get("id") or (isinstance(aliases, list) and model in aliases):
            return True
    return False
```
`diagnostics.py`:
```python
"""Diagnostics download for a Mistral config entry (MA-14)."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY, __version__ as HA_VERSION
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from . import MistralConfigEntry
from ._models import model_available
from .const import CONF_MODEL, DEFAULT_MODEL, DOMAIN

TO_REDACT = {CONF_API_KEY}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: MistralConfigEntry
) -> dict[str, Any]:
    """Settings, versions and the last errors — never the API key or bodies."""
    runtime = entry.runtime_data
    integration = await async_get_integration(hass, DOMAIN)
    model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    tts_entity = runtime.tts_entity
    api_key = entry.data.get(CONF_API_KEY, "")
    errors = [
        {k: (v.replace(api_key, "**REDACTED**") if isinstance(v, str) and api_key else v)
         for k, v in record.items()}
        for record in runtime.errors
    ]
    return {
        "entry": {
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": dict(entry.options),
        },
        "versions": {
            "home_assistant": HA_VERSION,
            "integration": str(integration.version),
        },
        "model": {
            "configured": model,
            "available": None if runtime.models is None else model_available(model, runtime.models),
        },
        "voices_loaded": len(tts_entity.async_get_supported_voices("en")) if tts_entity else 0,
        "recent_errors": errors,
    }
```
(The API key never enters a record — `record()` stores only the translation key or class name — the replace is a second safety net.)

- [ ] **Step 3: Run** `pytest tests/test_diagnostics.py -v` — Expected: PASS. (The `voices_loaded == 2` assertion needs the background voice fetch finished; `setup_integration` already waits with `async_block_till_done`.)

- [ ] **Step 4: Commit** — `git commit -m "MA-14: diagnostics with redacted key and the last 10 errors"`

---

### Task 8: MA-14 — retired-model repair

**Files:**
- Modify: `custom_components/mistral_conversation/_models.py`, `__init__.py`, `strings.json`, `translations/{en,nl,fr}.json`
- Create: `custom_components/mistral_conversation/repairs.py`, `tests/test_repairs.py`

**Interfaces:**
- Consumes: `MistralClient.list_models()` (Task 4), `model_available` (Task 7), `CHAT_MODELS`, `DEFAULT_MODEL`, `CONF_MODEL`.
- Produces: `suggest_replacement(model: str, models: list[dict[str, Any]]) -> str`; `async async_check_model(hass: HomeAssistant, entry: MistralConfigEntry) -> None`; `issue_id(entry_id: str) -> str` (= `f"model_retired_{entry_id}"`); `repairs.async_create_fix_flow`; constant `MODEL_CHECK_INTERVAL = timedelta(hours=24)`.

- [ ] **Step 1: Failing tests** `tests/test_repairs.py`:
```python
import json

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation._models import (
    async_check_model,
    issue_id,
    suggest_replacement,
)
from custom_components.mistral_conversation.const import DOMAIN

from .conftest import BASE, load_fixture

MODELS = json.loads(load_fixture("models.json"))["data"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ministral-8b-2410", "ministral-8b-latest"),     # rule 1: -latest of same name
        ("ministral-9b-latest", "ministral-8b-latest"),   # rule 2: first CHAT_MODELS entry of family
        ("mistral-tiny-2312", "mistral-small-latest"),    # rule 2, family "mistral"
        ("open-mistral-nemo", "ministral-8b-latest"),     # rule 3: DEFAULT_MODEL
    ],
)
def test_suggest_replacement(model: str, expected: str) -> None:
    assert suggest_replacement(model, MODELS) == expected


async def _set_model(hass, entry, model):
    hass.config_entries.async_update_entry(entry, options={**entry.options, "model": model})
    await hass.async_block_till_done()


async def test_retired_model_creates_issue_and_available_model_clears_it(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock
) -> None:
    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={"model": "open-mistral-nemo"})
    await hass.async_block_till_done()
    await async_check_model(hass, entry)
    issue = ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id))
    assert issue is not None and issue.is_fixable
    assert issue.translation_placeholders == {"model": "open-mistral-nemo", "replacement": "ministral-8b-latest"}

    hass.config_entries.async_update_entry(entry, options={"model": "ministral-8b-latest"})
    await hass.async_block_till_done()
    await async_check_model(hass, entry)
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id)) is None


async def test_model_check_failure_changes_nothing(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock
) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=429, headers={"Retry-After": "0"})
    await async_check_model(hass, entry)
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id)) is None


async def test_model_check_ignores_malformed_list(
    hass: HomeAssistant, setup_integration: MockConfigEntry, aioclient_mock
) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", json={"object": "list"})
    await async_check_model(hass, entry)
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id)) is None


async def test_fix_flow_applies_replacement_and_clears_issue(
    hass: HomeAssistant, setup_integration: MockConfigEntry, hass_client
) -> None:
    from homeassistant.components.repairs import DOMAIN as REPAIRS
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.components.repairs import (
        process_repair_fix_flow, start_repair_fix_flow,
    )

    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={"model": "open-mistral-nemo"})
    await hass.async_block_till_done()
    await async_check_model(hass, entry)
    assert await async_setup_component(hass, REPAIRS, {})
    client = await hass_client()
    flow = await start_repair_fix_flow(client, DOMAIN, issue_id(entry.entry_id))
    assert flow["step_id"] == "confirm"
    result = await process_repair_fix_flow(client, flow["flow_id"], json={})
    assert result["type"] == "create_entry"
    await hass.async_block_till_done()
    assert entry.options["model"] == "ministral-8b-latest"
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id)) is None
```
If `pytest_homeassistant_custom_component.components.repairs` is missing in 0.13.285, call the repairs HTTP endpoints directly: `POST /api/repairs/issues/fix` with `{"handler": DOMAIN, "issue_id": ...}` then `POST /api/repairs/issues/fix/{flow_id}` with `{}`.

Note: with an empty `models` result the check must also skip — `{"object": "list"}` gives `[]` from `list_models()`; treat an empty list as "unknown", not "everything retired".

Run: `pytest tests/test_repairs.py -v` — Expected: FAIL (ImportError).

- [ ] **Step 2: Implement** — append to `_models.py`:
```python
import logging
import re
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir

from .api import describe_error
from .const import CHAT_MODELS, CONF_MODEL, DEFAULT_MODEL, DOMAIN

_LOGGER = logging.getLogger(__name__)

MODEL_CHECK_INTERVAL = timedelta(hours=24)
_VERSION_SUFFIX = re.compile(r"-(\d{4}|latest)$")


def issue_id(entry_id: str) -> str:
    return f"model_retired_{entry_id}"


def suggest_replacement(model: str, models: list[dict[str, Any]]) -> str:
    """Replacement for a retired *model* (see the spec's replacement rule)."""
    latest = f"{_VERSION_SUFFIX.sub('', model)}-latest"
    if latest != model and model_available(latest, models):
        return latest
    family = model.split("-")[0]
    for candidate in CHAT_MODELS:
        if candidate.split("-")[0] == family and model_available(candidate, models):
            return candidate
    return DEFAULT_MODEL


async def async_check_model(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Raise or clear the ``model_retired`` repair. Never raises itself."""
    try:
        models = await entry.runtime_data.client.list_models()
    except HomeAssistantError as err:
        _LOGGER.debug("Mistral model list unavailable: %s", describe_error(err))
        return
    if not models:
        _LOGGER.debug("Mistral model list empty; skipping the model check")
        return
    entry.runtime_data.models = models
    model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    if model_available(model, models):
        ir.async_delete_issue(hass, DOMAIN, issue_id(entry.entry_id))
        return
    replacement = suggest_replacement(model, models)
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id(entry.entry_id),
        is_fixable=True,
        severity=ir.IssueSeverity.WARNING,
        translation_key="model_retired",
        translation_placeholders={"model": model, "replacement": replacement},
        data={"entry_id": entry.entry_id, "replacement": replacement},
    )
```
(`MistralConfigEntry` imported under `TYPE_CHECKING` from `.`, to avoid a circular import.)

`repairs.py`:
```python
"""Fix flow for the retired-model repair (MA-14)."""
from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import data_entry_flow
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

from .const import CONF_MODEL


class ModelRetiredRepairFlow(RepairsFlow):
    """Confirm switching the options to the suggested model."""

    def __init__(self, entry_id: str, replacement: str) -> None:
        self._entry_id = entry_id
        self._replacement = replacement

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if user_input is not None:
            if entry is not None:
                # Writing the options reloads the entry, as any options change does.
                self.hass.config_entries.async_update_entry(
                    entry, options={**entry.options, CONF_MODEL: self._replacement}
                )
            return self.async_create_entry(data={})
        model = entry.options.get(CONF_MODEL, "") if entry else ""
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={"model": model, "replacement": self._replacement},
        )


async def async_create_fix_flow(
    hass: HomeAssistant, issue_id: str, data: dict[str, str | int | float | None] | None
) -> RepairsFlow:
    """Create the fix flow for *issue_id*."""
    assert data is not None
    return ModelRetiredRepairFlow(str(data["entry_id"]), str(data["replacement"]))
```

In `__init__.py` `async_setup_entry`, after `async_forward_entry_setups`:
```python
    entry.async_create_background_task(
        hass, async_check_model(hass, entry), "mistral_check_model"
    )

    async def _periodic_check(_now: datetime) -> None:
        await async_check_model(hass, entry)

    entry.async_on_unload(
        async_track_time_interval(hass, _periodic_check, MODEL_CHECK_INTERVAL)
    )
```
(imports: `from datetime import datetime`, `from homeassistant.helpers.event import async_track_time_interval`, `from ._models import MODEL_CHECK_INTERVAL, async_check_model`.)

`strings.json` — add a top-level `"issues"` section (and the same keys in `translations/en.json`):
```json
"issues": {
  "model_retired": {
    "title": "Mistral model {model} is no longer available",
    "fix_flow": {
      "step": {
        "confirm": {
          "title": "Switch to {replacement}?",
          "description": "Mistral no longer offers the model {model}, so the assistant cannot answer. Select Submit to switch to {replacement}. You can choose another model later in the integration options."
        }
      }
    }
  }
}
```
`translations/nl.json`:
```json
"issues": {
  "model_retired": {
    "title": "Mistral-model {model} is niet meer beschikbaar",
    "fix_flow": {
      "step": {
        "confirm": {
          "title": "Overschakelen naar {replacement}?",
          "description": "Mistral biedt het model {model} niet meer aan, dus de assistent kan geen antwoord geven. Kies Verzenden om over te schakelen naar {replacement}. Je kunt later een ander model kiezen in de opties van de integratie."
        }
      }
    }
  }
}
```
`translations/fr.json`:
```json
"issues": {
  "model_retired": {
    "title": "Le modèle Mistral {model} n'est plus disponible",
    "fix_flow": {
      "step": {
        "confirm": {
          "title": "Passer à {replacement} ?",
          "description": "Mistral ne propose plus le modèle {model}, l'assistant ne peut donc plus répondre. Sélectionnez Valider pour passer à {replacement}. Vous pourrez choisir un autre modèle plus tard dans les options de l'intégration."
        }
      }
    }
  }
}
```

- [ ] **Step 3: Run** `pytest tests/test_repairs.py tests/test_init.py -v` — Expected: PASS. Add to `test_init.py`:
```python
async def test_periodic_model_check_scheduled(hass, setup_integration, aioclient_mock, freezer) -> None:
    from datetime import timedelta
    from pytest_homeassistant_custom_component.common import async_fire_time_changed
    calls = aioclient_mock.call_count
    freezer.tick(timedelta(hours=24, seconds=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert aioclient_mock.call_count == calls + 1
```

- [ ] **Step 4: Commit** — `git commit -m "MA-14: repair issue with a suggested replacement when the model is retired"`

---

### Task 9: MA-14 — reconfigure flow

**Files:**
- Modify: `config_flow.py`, `strings.json`, `translations/{en,nl,fr}.json`
- Test: `tests/test_config_flow.py`

**Interfaces:**
- Consumes: `MistralClient.validate_key` via the existing `_test_api_key` (Task 4).
- Produces: `async_step_reconfigure`; strings `config.step.reconfigure`, `config.abort.reconfigure_successful`.

- [ ] **Step 1: Failing tests** in `tests/test_config_flow.py`:
```python
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from .conftest import BASE


async def _start_reconfigure(hass, entry):
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_RECONFIGURE, "entry_id": entry.entry_id}
    )


async def test_reconfigure_changes_key(hass, setup_integration, aioclient_mock) -> None:
    entry = setup_integration
    result = await _start_reconfigure(hass, entry)
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"api_key": "sk-new"})
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["api_key"] == "sk-new"


async def test_reconfigure_invalid_key_shows_error(hass, setup_integration, aioclient_mock) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=401)
    result = await _start_reconfigure(hass, entry)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"api_key": "sk-bad"})
    assert result["errors"] == {"base": "invalid_auth"}
    assert entry.data["api_key"] == "sk-test-key"


async def test_reconfigure_cannot_connect_keeps_old_key(hass, setup_integration, aioclient_mock) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", exc=TimeoutError())
    result = await _start_reconfigure(hass, entry)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"api_key": "sk-new"})
    assert result["errors"] == {"base": "cannot_connect"}
    assert entry.data["api_key"] == "sk-test-key"
```
Run — Expected: FAIL (unknown step `reconfigure`).

- [ ] **Step 2: Implement** in `MistralConversationConfigFlow` (and reuse the schema in user/reauth to remove duplication):
```python
API_KEY_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        ),
    }
)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the API key without removing the integration."""
        errors: dict[str, str] = {}
        if user_input is not None:
            error = await self._test_api_key(user_input[CONF_API_KEY])
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    self._get_reconfigure_entry(),
                    data_updates={CONF_API_KEY: user_input[CONF_API_KEY]},
                )
        return self.async_show_form(
            step_id="reconfigure", data_schema=API_KEY_SCHEMA, errors=errors
        )
```
(`ConfigFlowResult` from `homeassistant.config_entries`; keep `FlowResult` elsewhere unchanged to stay in scope.)

Strings — `config.step.reconfigure` and `config.abort.reconfigure_successful`:
- en: `{"title": "Change the Mistral AI API key", "description": "Enter a new API key. The integration restarts with it; your settings stay the same.", "data": {"api_key": "API Key"}}`, abort `"The API key has been changed."`
- nl: `{"title": "Mistral AI API-sleutel wijzigen", "description": "Voer een nieuwe API-sleutel in. De integratie start opnieuw met deze sleutel; je instellingen blijven hetzelfde.", "data": {"api_key": "API-sleutel"}}`, abort `"De API-sleutel is gewijzigd."`
- fr: `{"title": "Modifier la clé API Mistral AI", "description": "Saisissez une nouvelle clé API. L'intégration redémarre avec cette clé ; vos paramètres restent inchangés.", "data": {"api_key": "Clé API"}}`, abort `"La clé API a été modifiée."`
(Match the existing wording of `api_key` in each translation file if it differs.)

- [ ] **Step 3: Run** `pytest tests/test_config_flow.py -v` and the config_flow coverage gate — Expected: PASS, 100 %.

- [ ] **Step 4: Commit** — `git commit -m "MA-14: reconfigure flow to change the API key"`

---

### Task 10: MA-30 — web search follows the model

**Files:**
- Modify: `const.py`, `conversation.py`, `config_flow.py`, `__init__.py`, `strings.json`, `translations/{en,nl,fr}.json`
- Test: `tests/test_config_flow.py`, `tests/test_init.py`, `tests/test_const.py`

**Interfaces:**
- Produces: `const.supports_web_search(model: str) -> bool`; options flow steps `init` (model) and `settings`; `__init__._async_fix_web_search(hass, entry) -> None`.

- [ ] **Step 1: Failing tests**

`tests/test_const.py`:
```python
from custom_components.mistral_conversation.const import supports_web_search


def test_supports_web_search() -> None:
    assert supports_web_search("mistral-small-latest")
    assert supports_web_search("mistral-large-latest")
    assert not supports_web_search("ministral-14b-latest")
    assert not supports_web_search("ministral-8b-latest")
```

`tests/test_config_flow.py`:
```python
async def _options(hass, entry, model, settings=None):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["step_id"] == "init"
    result = await hass.config_entries.options.async_configure(result["flow_id"], {"model": model})
    assert result["step_id"] == "settings"
    if settings is None:
        return result
    return await hass.config_entries.options.async_configure(result["flow_id"], settings)


def _fields(result) -> set[str]:
    return {str(k) for k in result["data_schema"].schema}


def _default(result, name):
    for key in result["data_schema"].schema:
        if str(key) == name:
            return key.default()
    raise KeyError(name)


async def test_unsupported_model_hides_web_search(hass, setup_integration) -> None:
    result = await _options(hass, setup_integration, "ministral-14b-latest")
    assert not {"web_search", "web_search_mode", "web_search_trigger"} & _fields(result)


async def test_unsupported_model_saves_web_search_off(hass, setup_integration) -> None:
    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={"model": "mistral-small-latest", "web_search": True})
    await hass.async_block_till_done()
    result = await _options(hass, entry, "ministral-14b-latest", {})
    assert result["type"] == "create_entry"
    assert entry.options["web_search"] is False
    assert entry.options["model"] == "ministral-14b-latest"


async def test_unsupported_model_keeps_mode_and_trigger(hass, setup_integration) -> None:
    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={
        "model": "mistral-small-latest", "web_search": True,
        "web_search_mode": "always", "web_search_trigger": "zoek op",
    })
    await hass.async_block_till_done()
    await _options(hass, entry, "ministral-14b-latest", {})
    assert entry.options["web_search_mode"] == "always"
    assert entry.options["web_search_trigger"] == "zoek op"


async def test_switch_to_supported_model_turns_web_search_on(hass, setup_integration) -> None:
    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={"model": "ministral-14b-latest", "web_search": False})
    await hass.async_block_till_done()
    result = await _options(hass, entry, "mistral-small-latest")
    assert _default(result, "web_search") is True


async def test_manual_off_on_supported_model_is_kept(hass, setup_integration) -> None:
    entry = setup_integration
    hass.config_entries.async_update_entry(entry, options={"model": "mistral-small-latest", "web_search": False})
    await hass.async_block_till_done()
    result = await _options(hass, entry, "mistral-large-latest")
    assert _default(result, "web_search") is False
```

`tests/test_init.py`:
```python
async def test_migration_turns_web_search_off(hass, mock_config_entry, aioclient_mock, caplog) -> None:
    hass.config_entries.async_update_entry(
        mock_config_entry, options={"model": "ministral-14b-latest", "web_search": True}
    )
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.options["web_search"] is False
    assert caplog.text.count("Web search turned off: model ministral-14b-latest does not support it") == 1


async def test_migration_does_not_loop(hass, mock_config_entry, aioclient_mock) -> None:
    hass.config_entries.async_update_entry(
        mock_config_entry, options={"model": "ministral-14b-latest", "web_search": True}
    )
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    # One setup: one key check + one background model check.
    model_calls = [c for c in aioclient_mock.mock_calls if str(c[1]).endswith("/models")]
    assert len(model_calls) == 2
```
(Add `from .conftest import load_fixture` to the imports of `test_init.py`.)

Run — Expected: FAIL.

- [ ] **Step 2: Implement**

`const.py` (below `AGENT_CAPABLE_MODELS`):
```python
def supports_web_search(model: str) -> bool:
    """True when *model* can use web search (Agents/Conversations API)."""
    return any(model.startswith(m) for m in AGENT_CAPABLE_MODELS)
```
`conversation.py`: `web_search_available = web_search and supports_web_search(model)`; drop the `AGENT_CAPABLE_MODELS` import.

`__init__.py`:
```python
def _async_fix_web_search(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Turn web search off when the saved model cannot use it (MA-30).

    Runs before the update listener is registered, so this write does not
    trigger a reload.
    """
    model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    if entry.options.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH) and not supports_web_search(model):
        _LOGGER.warning("Web search turned off: model %s does not support it", model)
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, CONF_WEB_SEARCH: False}
        )
```
Call it in `async_setup_entry` right after the key check, before `async_forward_entry_setups`.

`config_flow.py` — replace `MistralOptionsFlow`:
```python
class MistralOptionsFlow(config_entries.OptionsFlow):
    """Two steps: the model first, then settings that depend on it (MA-30)."""

    def __init__(self) -> None:
        self._model: str = DEFAULT_MODEL

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        opts = self.config_entry.options
        if user_input is not None:
            self._model = user_input[CONF_MODEL]
            return await self.async_step_settings()
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL, default=opts.get(CONF_MODEL, DEFAULT_MODEL)
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=CHAT_MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            translation_key="model",
                        )
                    ),
                }
            ),
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        opts = self.config_entry.options
        capable = supports_web_search(self._model)
        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            data = {**user_input, CONF_MODEL: self._model}
            if not capable:
                # Saved as off; mode and trigger phrases keep their values so
                # they come back when a supporting model is chosen again.
                data[CONF_WEB_SEARCH] = False
                data[CONF_WEB_SEARCH_MODE] = opts.get(CONF_WEB_SEARCH_MODE, DEFAULT_WEB_SEARCH_MODE)
                data[CONF_WEB_SEARCH_TRIGGER] = opts.get(CONF_WEB_SEARCH_TRIGGER, DEFAULT_WEB_SEARCH_TRIGGER)
            return self.async_create_entry(title="", data=data)

        schema: dict[Any, Any] = {
            # prompt, llm_hass_api, temperature, max_tokens: moved unchanged
            # from the old single-step form (same keys, defaults, selectors).
        }
        if capable:
            previous = opts.get(CONF_MODEL, DEFAULT_MODEL)
            web_default = (
                opts.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)
                if supports_web_search(previous)
                else True
            )
            schema[vol.Optional(CONF_WEB_SEARCH, default=web_default)] = selector.BooleanSelector()
            # web_search_mode and web_search_trigger: moved unchanged.
        # tts_mode: moved unchanged, last.
        return self.async_show_form(step_id="settings", data_schema=vol.Schema(schema))
```
The three "moved unchanged" comments mean: cut the exact `vol.Optional(...): selector...` entries (with their explanatory comments) from the old `async_step_init` and paste them in that order — prompt, llm_hass_api, temperature, max_tokens, [web_search, web_search_mode, web_search_trigger], tts_mode. Do not leave the placeholder comments in the final code.

Strings — `options.step`:
- `init`: keep `"title"`; `data` and `data_description` keep only `model`; add `"description"`:
  - en: `"Choose the model first. The next step shows the settings for that model; web search is only offered for models that support it (Mistral Small, Medium and Large)."`
  - nl: `"Kies eerst het model. De volgende stap toont de instellingen voor dat model; zoeken op het web is alleen beschikbaar bij modellen die dat ondersteunen (Mistral Small, Medium en Large)."`
  - fr: `"Choisissez d'abord le modèle. L'étape suivante affiche les paramètres de ce modèle ; la recherche web n'est proposée que pour les modèles qui la prennent en charge (Mistral Small, Medium et Large)."`
- `settings`: new step; `"title"` en `"Settings for this model"` / nl `"Instellingen voor dit model"` / fr `"Paramètres pour ce modèle"`; move every other `data` and `data_description` key from `init` into it unchanged, in each of the four files.

- [ ] **Step 3: Run** `pytest -v`, `ruff check .`, `mypy`, both coverage gates — Expected: all pass. Update any existing options-flow test to the two-step flow (same saved values asserted).

- [ ] **Step 4: Commit** — `git commit -m "MA-30: web search follows the selected model (two-step options, migration)"`

---

### Task 11: Docs, backlog, final verification, PR

**Files:**
- Modify: `CLAUDE.md`, `CONTEXT.md`, `CHANGELOG.md`, `docs/backlog-and-review/backlog.md`

- [ ] **Step 1: New issue for mypy strict**
```bash
gh issue create --title "MA-31 · mypy --strict" --label backlog --label type:tech --label priority:could --label release:2-ha-native --label effort:M --body-file - <<'EOF'
As a maintainer, I want the integration to pass `mypy --strict`, so type errors are caught before they reach users.

**Background**

MA-10 (#52) added plain `mypy` to CI; `--strict` was split off to keep that PR reviewable.

**Acceptance criteria**

- [ ] `mypy --strict` passes on `custom_components/mistral_conversation`.
- [ ] CI runs it on the latest HA leg.
- [ ] No `# type: ignore` without an error code and a reason.

**Estimate:** Claude 1–2 h + test 15 min · ≈ 1–3 M tokens
EOF
```

- [ ] **Step 2: Docs**
  - `CHANGELOG.md` under `### Unreleased`:
    - `Added: diagnostics download (settings, versions, last 10 errors; API key redacted). (MA-14)`
    - `Added: a repair notification when the configured model is retired by Mistral, with a one-click switch to a suggested model. (MA-14)`
    - `Added: reconfigure to change the API key without removing the integration. (MA-14)`
    - `Changed: the options are now two steps (model, then settings); web search is only shown for models that support it and is switched on when you move to such a model. (MA-30)`
    - `Changed: an existing setup with web search on and a model that cannot use it gets web search switched off at startup, with one log line. (MA-30)`
    - `Internal: tests run on real Home Assistant (minimum and latest), with coverage gates; one Mistral client and one entity base class. (MA-11, MA-10)`
  - `CLAUDE.md`: Layout block lists `api.py`, `entity.py`, `_models.py`, `repairs.py`, `diagnostics.py`; Commands as in Task 2 Step 5; conventions from Task 4 Step 6.
  - `CONTEXT.md`: add **Repair** ("an HA notification under Settings → Repairs; here: the retired-model issue with a fix button") and **Diagnostics** ("the downloadable JSON under the integration's ⋮ menu").
  - `docs/backlog-and-review/backlog.md`: add MA-30 and MA-31 entries in the existing format; set MA-10, MA-11, MA-14 status to "in PR".

- [ ] **Step 3: Final verification**
```bash
ruff check .
mypy
pytest --cov --cov-report=term-missing --cov-fail-under=90
coverage report --include="*/config_flow.py" --fail-under=100
```
Expected: all pass. Then run the minimum leg in a second venv with `requirements_test_min.txt` (Python 3.13 if available; otherwise rely on CI and say so in the PR).

- [ ] **Step 4: Commit and push** — `git commit -m "Docs for the release 2 bundle"`; `git push -u origin feat/release-2-ha-native`.

- [ ] **Step 5: PR** — `gh pr create` with title `Release 2: real HA tests, HA patterns, diagnostics/repairs/reconfigure, web search follows the model (MA-11, MA-10, MA-14, MA-30)`. Body: per story the problem and what changed in plain language; a test checklist:
  1. Install the pre-release; the integration loads, devices and entities are unchanged (same names, dashboards still work).
  2. Options: step 1 model, step 2 settings. With `ministral-14b-latest` there are no web search fields; with `mistral-small-latest` web search is on by default.
  3. Log after restart shows once "Web search turned off: model ministral-14b-latest does not support it" (for the current setup).
  4. ⋮ → Download diagnostics: no API key in the file; recent errors listed.
  5. ⋮ → Reconfigure: a wrong key shows "Invalid API key"; the right key saves and reloads.
  6. Repairs: temporarily set the model to a retired one via Developer tools isn't possible from the UI — covered by tests; check Settings → Repairs stays empty with a valid model.
  7. CI: both matrix legs, mypy and coverage gates green.

  End with `Closes #53`, `Closes #52`, `Closes #56`, `Closes #81` and the attribution line `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
