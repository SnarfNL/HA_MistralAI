# Release 2 bundle: real HA tests, HA patterns, diagnostics/repairs/reconfigure, web search follows the model

Date: 2026-09-26
Branch: `feat/release-2-ha-native` (one PR)
Closes: #53 (MA-11), #52 (MA-10), #56 (MA-14), #81 (MA-30)

## Goal

- **MA-11:** tests run against real Home Assistant code, so an HA release that breaks the integration fails CI.
- **MA-10:** one Mistral client and one entity base class, so an API change is fixed in one place.
- **MA-14:** a diagnostics file for bug reports, a repair notification when the configured model is retired, and a reconfigure flow to change the API key.
- **MA-30:** web search is only offered, and only on, when the selected model supports it.

Behaviour for the end user stays the same except for the MA-14 and MA-30 additions and the two-step options form.

## Decisions (agreed with the maintainer, 2026-09-26)

| # | Topic | Decision |
|---|-------|----------|
| Q1 | mypy | Plain `mypy` in CI in this PR. `--strict` moves to a new backlog issue. |
| Q2 | Minimum HA | Stays **2025.10.0**. Test matrix: Python 3.13 + HA 2025.10, Python 3.14 + HA 2026.9. The pre-probatio fallback stays. |
| Q3 | SSE fixtures | Hand-built fixtures (based on the formats the current tests use) **and** a recording script the maintainer can run with their own key. |
| Q4 | Coverage | Hard CI gates: config flow / reauth / reconfigure / options flow 100 %, total ≥ 90 %. |
| Q5 | Options form | Two steps: step 1 model, step 2 the other settings. Web search fields only appear for a model that supports it. |
| Q6 | Web search default | Switched on only when the model changes from unsupported to supported. A manual "off" on a supported model is kept. |
| Q7 | Existing entries | At setup, web search on + unsupported model is saved as off, with one warning in the log. |
| Q8 | Last errors | Last 10 errors in memory: time, source, HTTP status, translation key. No bodies, no key. |
| Q9 | Retired model | Checked at setup and every 24 h. Suggested replacement per the rule below; the repair's fix applies it. |
| Q10 | Reconfigure | API key only. |

## Order of work (commits)

1. **MA-11** test base: new test infrastructure, all existing tests ported with the same expectations, stubs removed.
2. **MA-10** refactor, verified by the MA-11 tests.
3. **MA-14** diagnostics, repairs, reconfigure.
4. **MA-30** two-step options form and web search capability.
5. Docs: CLAUDE.md, CONTEXT.md, CHANGELOG, backlog.md (adds MA-30 and the new mypy-strict issue).

Each step is one or more commits whose message names the story, so the PR can be reviewed per story.

## MA-11: tests against real HA

**Packages** (test-only, never in `manifest.json`):
- `requirements_test_min.txt`: `pytest-homeassistant-custom-component` release pinned to HA 2025.10.x (0.13.28x line, Python 3.13).
- `requirements_test.txt`: the release pinned to the latest HA (currently 0.13.366 → HA 2026.9.3, Python 3.14).
- Both add `pytest-cov`, `mypy`, `ruff`.

**CI** (`tests.yml`): a matrix job with two legs (min, latest). Both run ruff and pytest. The latest leg also runs mypy and the coverage gates:
- `pytest --cov=custom_components.mistral_conversation --cov-fail-under=90`
- `coverage report --include="*/config_flow.py" --fail-under=100`

**Test layout:**
- `tests/conftest.py`: `enable_custom_integrations` autouse, a `mock_config_entry` fixture, a `setup_integration` fixture that mocks `GET /v1/models` and sets the entry up through HA.
- The Mistral API is mocked with `aioclient_mock`. Streamed answers come from files in `tests/fixtures/` (`*.sse` for chat and TTS streams, `*.json` for models, voices, transcription and conversations).
- Existing test files are ported one by one. Assertions keep their meaning; only the way HA and the API are set up changes. A test that only tested a stub disappears, and the PR lists which ones.
- `tests/_ha_stubs.py` is removed.

**Recording script** `scripts/record_fixtures.py`: reads `MISTRAL_API_KEY` from the environment, makes the handful of calls the fixtures need (about 8 requests), writes them to `tests/fixtures/`, and strips IDs and anything account-specific. Not run in CI.

**Local runs:** HA does not officially support Windows. The plan's first task checks whether the test package runs on the maintainer's Windows machine; if not, local runs use WSL or rely on CI, and CLAUDE.md says which.

## MA-10: HA patterns

**`api.py`** (replaces `_api.py`; the module-level helpers `mistral_error`, `describe_error`, `read_json`, `translate_stream`, `async_spoken_error` move with it unchanged).

`MistralClient(hass, entry, session, api_key)`:
- `request(method, path, *, timeout, ...)`: the current `mistral_request` logic (retries, reauth, translated errors) as an async context manager. Paths are relative to `MISTRAL_API_BASE`.
- Thin methods on top of `request`: `chat_stream(payload)`, `start_conversation(payload)` / `append_conversation(conv_id, payload)`, `delete_conversation(conv_id)`, `speech(payload, *, stream)`, `transcribe(form_factory, ...)`, `list_voices(offset, limit)`, `list_models()`.
- Every translated error is also recorded in the runtime's error log (MA-14).
- `static async validate_key(session, api_key) -> None | "invalid_auth" | "cannot_connect"` (the config flow keeps its current `unknown` fallback for unexpected exceptions): used by setup, the config flow, reauth and reconfigure. This removes the two direct session calls CLAUDE.md now lists as exceptions.

**Runtime data:** `type MistralConfigEntry = ConfigEntry[MistralRuntimeData]`; `entry.runtime_data` replaces all 9 uses of `hass.data[DOMAIN]`. `MistralRuntimeData` holds `client`, `web_search_convs`, `tts_entity`, `errors` (a `deque(maxlen=10)`) and `models` (the last `/v1/models` result, for MA-14).

**`entity.py`:** `MistralEntity` base class with `_attr_has_entity_name = True`, `entry`, `runtime` and `client` properties, and `device_info` built from a small per-device description (conversation, STT, TTS). **Device identifiers, names and entity unique IDs stay exactly as they are**, so no device or entity is recreated.

**`PARALLEL_UPDATES = 0`** in every platform (cloud service, nothing polls).

**mypy:** plain mode, configured in `pyproject.toml`, must pass in CI.

## MA-14: diagnostics, repairs, reconfigure

**Diagnostics** (`diagnostics.py`, `async_get_config_entry_diagnostics`):
- entry data with `api_key` redacted (`async_redact_data`), options, HA version, integration version (from the loaded integration's manifest), number of voices loaded, whether the configured model is in the last models list, and the last 10 errors.
- An error record: ISO time, source (`conversation`, `ai_task`, `stt`, `tts`, `setup`), HTTP status or `null`, translation key. Never a response body, URL query or header.

**Retired model repair** (`repairs.py`):
- At setup (after platforms load, in the background so startup is not delayed) and every 24 h (`async_track_time_interval`, cancelled on unload), the client fetches `/v1/models`. A failed fetch only logs at debug level and changes nothing.
- The configured model counts as available when it matches a model's `id` or one of its `aliases`.
- Not available → a fixable repair issue `model_retired` (translation placeholders: old model, suggested model). Available again → the issue is deleted.
- **Replacement rule:** (1) the `-latest` alias of the same model name (`ministral-8b-2410` → `ministral-8b-latest`), if available; else (2) the first entry of `CHAT_MODELS` in the same family (text before the first `-`, e.g. `ministral`, `mistral`) that is available; else (3) `DEFAULT_MODEL`.
- The fix flow is a confirm step; confirming writes the replacement into the options (the entry reloads as with any options change) and removes the issue.

**Reconfigure** (`async_step_reconfigure` in the config flow): one field, the API key. Validated with `MistralClient.validate_key`; on success `async_update_reload_and_abort`, on failure the form shows `invalid_auth` or `cannot_connect`.

## MA-30: web search follows the model

**One helper:** `supports_web_search(model: str) -> bool` in `const.py`, replacing the `startswith` check in `conversation.py`. Still based on `AGENT_CAPABLE_MODELS`; switching to `/v1/models` capabilities is left to MA-23.

**Options flow, two steps:**
1. `init`: the model only.
2. `settings`: prompt, LLM API, temperature, max tokens, TTS mode, and — only when `supports_web_search(model)` — web search, web search mode, trigger phrases.

Saving (step 2):
- Unsupported model: `web_search` is saved as `false`. Mode and trigger phrases keep their stored values, so they come back when the model is switched back.
- Supported model, previous model unsupported (or no previous model): the web search field starts as **on**.
- Supported model, previous model also supported: the field starts at the stored value (a manual "off" stays off).

**Existing entries:** in `async_setup_entry`, before the update listener is registered (so no reload loop), web search on + unsupported model → `async_update_entry` with `web_search: false` and one warning: "Web search turned off: model <m> does not support it".

**Texts:** new options step `settings`, the step 1 description, and the repair issue texts in `strings.json` and `en`, `nl`, `fr`.

## Error handling

No change to how errors reach the user; MA-10 moves the code, it does not change it. New failure paths: the models check (MA-14) never raises; diagnostics never calls Mistral; the migration (MA-30) only writes options.

## Testing

All on the MA-11 base, following the Q4 gates:
- Ported tests keep their expectations (MA-10 acceptance criterion).
- New: runtime_data setup/unload; client methods (retries, reauth, error recording); device identifiers unchanged; diagnostics redaction and error list; repair created, deleted and fixed; each replacement rule branch; reconfigure success and both errors; options flow both steps for supported/unsupported models, the Q6 default rules, and the Q7 migration with its log line.

## Out of scope

- `mypy --strict` (new issue).
- Model capabilities from `/v1/models` for web search (MA-23).
- Changing the minimum HA version.
- Any new user-facing features beyond the four issues.

## Risks

- **Windows:** HA's test package may not run natively on Windows (see MA-11, local runs).
- **Two HA versions:** an API that exists in 2026.9 but not 2025.10 (for example around config subentries or the reconfigure helpers) fails the min leg. Fix with a small version check, not by raising the minimum.
- **PR size:** mitigated by commits per story and a PR description per story.
