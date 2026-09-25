# CLAUDE.md

Home Assistant custom integration (HACS) that connects Mistral AI to HA: conversation agent, AI Task, speech-to-text (Voxtral) and text-to-speech. Domain: `mistral_conversation`. Maintainer: @SnarfNL (not a Python developer — keep changes small and explain them in plain language).

## Layout

```
custom_components/mistral_conversation/
  __init__.py      setup/unload, MistralRuntimeData (entry.runtime_data: client, errors, ...)
  api.py           MistralClient (every Mistral call), error helpers
  conversation.py  conversation entity, chat_log conversion, SSE parser, web search
  ai_task.py       AI Task entity (generate_data, image attachments)
  stt.py           Voxtral STT entity
  tts.py           TTS entity: batch (mp3) and pipelined streaming (wav)
  _streaming.py    stdlib-only helpers: sentence splitter, TTS SSE parser
  config_flow.py   config flow, reauth, options flow
  const.py         config keys, defaults, model and voice lists
  strings.json + translations/{en,nl,fr}.json
tests/             pytest on real HA (pytest-homeassistant-custom-component); API mocked with aioclient_mock + tests/fixtures
```

## Commands

```bash
pip install -r requirements_test.txt
python scripts/component_requirements.py > requirements_components.txt
pip install -r requirements_components.txt
ruff check .        # must pass
pytest              # must pass; add --cov for the coverage gates
mypy                # must pass
```

HA's test runner needs Linux or macOS; on Windows the tests run in CI.
CI (GitHub Actions) runs hassfest, HACS validation, ruff, mypy and pytest on every PR, against the minimum (2025.10) and the latest HA version.

## Workflow rules

- One story per branch and PR, unless the maintainer asks to bundle stories; then one branch and one PR that closes each issue (`Closes #N` per issue). Branch names: `fix/ma-XX-short-name`, `feat/ma-XX-...`, `ci/...`, or a descriptive name for a bundle (e.g. `fix/release-1-stable`). Always branch from the latest `main`.
- Never push to `main`, never merge the PR yourself.
- Stay inside the story's scope. No drive-by refactors, renames or formatting changes in unrelated code; list other problems you notice in the PR instead.
- Every behaviour change gets a unit test. Existing tests must keep passing without loosening their assertions.
- User-visible changes get a line under `### Unreleased` in CHANGELOG.md.
- Do not bump `version` in manifest.json (done at release time).
- Do not add entries to `requirements` in manifest.json without asking.

## Language

Everything that goes into the repo or to GitHub is written in English: code, comments, log messages, docstrings, docs, commit messages, PR descriptions, issues and issue comments. The maintainer sometimes talks to agents in Dutch; that does not change the language of the repo. The only Dutch text is the UI translation file `translations/nl.json`.

## PR description

Write it in English, with: the problem, what changed (plain language, no jargon), and a test checklist of concrete steps the maintainer can do in Home Assistant or on GitHub before merging.

## Code conventions

- Any text shown in the UI goes in `strings.json` AND all three files in `translations/` (en, nl, fr). Keep keys in sync.
- Calls to the Mistral API go through `MistralClient` in `api.py` (`entry.runtime_data.client`); use its methods (`chat_completions`, `speech`, `transcribe`, ...) or `request()`. It handles timeouts, reauth on 401, retries on 429 and translated errors in one place. Read the response with `read_json()` or wrap a stream in `translate_stream()`; code inside the `async with` block is not translated, so errors from HA tools are never blamed on Mistral. Raise user-facing errors with `mistral_error("<key>")`; every key lives in the `exceptions` section of strings.json and all three translations.
- API keys are checked with `MistralClient.validate_key()` (setup, config flow, reauth, reconfigure), which runs before the entry's runtime data exists; no code calls the session directly. It catches `(aiohttp.ClientError, TimeoutError)` — aiohttp raises `TimeoutError`, which is not a `ClientError`.
- Never log or format an aiohttp exception with `%r` or `!r`: the repr of `ClientResponseError` contains the request headers, including the API key. Use `describe_error(err)`.
- Never put raw API response bodies in errors shown to users; log them instead.
- Target Python 3.13 and the minimum HA version in hacs.json.

## Hard-won HA / Mistral facts (do not "fix" these)

- `chat_log.async_add_delta_content_stream` expects dicts. First yield `{"role": "assistant"}`, then separate `{"content": str}` and `{"tool_calls": [llm.ToolInput]}` deltas — never combined in one dict. Without the role delta, TTS streaming in the voice pipeline breaks.
- Tool schemas: HA 2026.9+ uses `probatio`. Use `_schema_to_openapi()` (probatio first, voluptuous_openapi fallback, always returns a dict).
- Mistral's built-in `web_search` tool is rejected by `/v1/chat/completions`; it only works via the Agents/Conversations API. That is why web search is a synthetic function tool serviced by the integration.
- STT language comes from `metadata.language` (the voice pipeline), not from an integration option.
- `GET /v1/audio/voices` is paginated (`limit`/`offset`); synthesis uses the voice `id` (UUID), the picker shows `name`.
- Streaming TTS is WAV 24 kHz / 16-bit / mono; per-sentence requests are stitched into one stream (one RIFF header, then raw PCM).

## Planning and vocabulary
- Backlog: GitHub Issues with label `backlog` (source: docs/backlog-and-review/backlog.md). Reference the issue in every PR (`Closes #N`).
- Vocabulary: CONTEXT.md. Background: docs/backlog-and-review/2026-09-review.md.

## Agent skills

### Issue tracker
Issues are tracked in GitHub Issues (SnarfNL/HA_MistralAI) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels
Default vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
