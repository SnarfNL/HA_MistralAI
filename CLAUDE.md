# CLAUDE.md

Home Assistant custom integration (HACS) that connects Mistral AI to HA: conversation agent, AI Task, speech-to-text (Voxtral) and text-to-speech. Domain: `mistral_conversation`. Maintainer: @SnarfNL (not a Python developer — keep changes small and explain them in plain language).

## Layout

```
custom_components/mistral_conversation/
  __init__.py      setup/unload, MistralRuntimeData (session + headers)
  conversation.py  conversation entity, chat_log conversion, SSE parser, web search
  ai_task.py       AI Task entity (generate_data, image attachments)
  stt.py           Voxtral STT entity
  tts.py           TTS entity: batch (mp3) and pipelined streaming (wav)
  _streaming.py    stdlib-only helpers: sentence splitter, TTS SSE parser
  config_flow.py   config flow, reauth, options flow
  const.py         config keys, defaults, model and voice lists
  strings.json + translations/{en,nl,fr}.json
tests/             unittest-style tests; HA is stubbed via tests/_ha_stubs.py
```

## Commands

```bash
pip install ruff pytest
ruff check .        # must pass
pytest tests        # must pass
```

CI (GitHub Actions) runs hassfest, HACS validation, ruff and pytest on every PR.

## Workflow rules

- One story per branch and PR. Branch names: `fix/ma-XX-short-name`, `feat/ma-XX-...`, `ci/...`. Always branch from the latest `main`.
- Never push to `main`, never merge the PR yourself.
- Stay inside the story's scope. No drive-by refactors, renames or formatting changes in unrelated code; list other problems you notice in the PR instead.
- Every behaviour change gets a unit test. Existing tests must keep passing without loosening their assertions.
- User-visible changes get a line under `### Unreleased` in CHANGELOG.md.
- Do not bump `version` in manifest.json (done at release time).
- Do not add entries to `requirements` in manifest.json without asking.

## PR description

Write it in English, with: the problem, what changed (plain language, no jargon), and a testchecklist of concrete steps the maintainer can do in Home Assistant or on GitHub before merging.

## Code conventions

- Code, comments, log messages and docstrings in English.
- Any text shown in the UI goes in `strings.json` AND all three files in `translations/` (en, nl, fr). Keep keys in sync.
- Network calls: catch `(aiohttp.ClientError, TimeoutError)` — aiohttp raises `TimeoutError`, which is not a `ClientError`.
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
