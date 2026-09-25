# Changelog

All notable changes to this project are documented here.

Versions before 2026.05 used `vX.Y.Z` numbering; from 2026.05 on the format is `YYYY.MM.NN`.


### Unreleased
- **Changed:** web search no longer creates a Mistral *agent* (MA-04). Each search starts a conversation with the model and the `web_search` tool directly, so restarting Home Assistant no longer adds another "HA Mistral Web Search" agent to your Mistral account, and no house data (entity list, date, your prompt) is stored on Mistral's side. Agents with that name created by earlier versions are unused now and can be deleted in Mistral Studio; they are not removed automatically.
- **Changed:** searches the model asks for are stateless (`store: false`). On the trigger-phrase and *Always search* routes the Mistral conversation kept for follow-ups now expires 5 minutes after last use, holds at most 50 entries, and is deleted on Mistral's side when it expires.
- **Changed:** answers on the trigger-phrase and *Always search* routes use a generic instruction (short, factual, suited to speech) plus your language instead of your full system prompt, so they no longer follow the assistant's persona.
- **Removed:** the *Continue conversation* option (MA-05). Home Assistant core already sets `continue_conversation` when a reply ends in a question mark (also when the option was off), so the switch never controlled anything on the standard path. The stored option value is ignored. Web-search replies from the Agents API path no longer continue the conversation until they go through the chat log (MA-02).
- **Fixed:** With web search enabled, *every* utterance was routed through the Agents API — including plain device commands. That path never passed Home Assistant tools, so device control silently failed on it (the model replies "I can't perform physical actions"). Web search is now opt-in per turn. Fixes #29.
- **Added:** `web_search_mode` option. Default `model` advertises web search to the model as a tool and services the call against the Agents API, so a turn keeps its HA tools and can search *and* control devices. `always` preserves the previous behaviour.
- **Added:** `web_search_trigger` option — optional comma-separated phrases (empty by default). When set it takes precedence over `web_search_mode`: only utterances starting with a phrase search (phrase stripped, longest match wins, case-insensitive); all others never search.
- **Fixed:** Assistant turns carrying neither text nor tool calls are no longer sent to Mistral — they were rejected with HTTP 400 `Assistant message must have either content or tool_calls, but not none.` (code 3240).
- **Added:** 29 unit tests for trigger resolution, tool-call interception, and the content-less-assistant-turn guard.
- **Fixed:** streaming TTS stays playable when the first sentence fails (MA-01).
- **Fixed:** STT now offers the 13 languages Voxtral documents (was 60) and TTS the 9 documented ones (was 11, including four that are not supported). The stale `stt_language` texts are removed. MA-06.
- **Changed:** The voice picker lists only the voices of your Mistral account (presets and custom voices), with readable names such as `Paul – Angry (English)`. The static preset list is gone; the picker is empty until Home Assistant has fetched the account voices. MA-13.
- **Added:** A **Refresh voices** button on the Mistral AI TTS device re-fetches the account voices without restarting Home Assistant; a failed refresh keeps the previous list. MA-13.
- **Changed:** The *Text-to-speech voice* option is removed from the integration options. Choose the voice under Settings → Voice assistants, or pass `voice` in `tts.speak`. Without a voice the built-in default (`en_paul_neutral`) is used; a voice saved in the old option is no longer used. Fixes #75.

---

### 2026.05 — 2026-05-04
- **Changed:** Version numbering from digits to year.month.version format.
- **Added:** New AI task entity that lets Mistral generate structured data, handle image attachments, and respect output schemas defined in HA automations.
- **Added:** Streaming TTS via Mistral's SSE WAV endpoint (`response_format=wav`, `stream=true`). End-of-speech-to-first-audio drops from ~5 s to ~1 s for long responses — audio chunks arrive while synthesis is in progress instead of waiting for the full MP3.
- **Added:** Sentence-level pipelining — incoming LLM tokens are segmented into complete sentences and dispatched to Mistral in parallel (up to 5 concurrent requests bounded by an `asyncio.Semaphore`). Audio is reassembled in strict order with the WAV header from sentence 0 followed by raw PCM samples from sentences 1..N.
- **Added:** `tts_mode` integration option — choose between `stream` (default) and `batch`. Direct `tts.speak` service calls always use the batch path regardless of this setting.
- **Added:** `tests/` directory with 64 unit tests covering `_streaming` helpers (sentence segmenter, SSE parser), const sanity (defaults reference valid values, voice naming convention, WAV header pin), `_pcm_to_wav` (round-trips RIFF/WAVE, fmt subchunk, data subchunk), and `_async_stream_delta` (chat-completions SSE: text deltas, `[DONE]` termination, tool-call accumulation, frame-split tolerance, malformed JSON resilience). Run with `python -m unittest discover -t . -s tests`.
- **Added:** MIT `LICENSE` file at the repo root.
- **Fixed:** `async_unload_entry` now unloads platforms *before* clearing runtime data — previously the order was reversed, which could race with in-flight streaming TTS that resolves `self._runtime` lazily on each chunk.
- **Changed:** Minimum Home Assistant version bumped to **2025.10.0** — required for `async_stream_tts_audio` (HA streaming TTS API, shipped 2025.7).

---

### v0.3.6 — 2026-04-10
- **Changed** :Setting default voices and language has been clarified

### v0.3.5 — 2026-04-10
- **Fixed:** : Update available TTS voices and set new default by @kalon33 in #12
- **Changed:** :Translate in French by @kalon33 in #13

### v0.3.1.2 — 2026-04-09
- **Fixed:** TTS voice list corrected — previous lists contained non-existent voice IDs. Replaced with the complete official list of 20 voices from Mistral TTS documentation, covering English (casual, cheerful, neutral), French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi. Default changed to `neutral_female`.

---

### v0.3.1.1 — 2026-04-09
- **Fixed:** Mistral TTS returned HTTP 400 `Invalid model` — corrected model name from `mistral-tts-latest` to `voxtral-mini-tts-2603`.
- **Fixed:** TTS API returns base64-encoded audio in a JSON `audio_data` field, not raw bytes — the response is now decoded correctly via `base64.b64decode()`.
- **Fixed:** Voice parameter renamed from `voice` to `voice_id` to match the Mistral API spec.
- **Updated:** Voice list replaced with actual Mistral TTS voices in `language_name_style` format (e.g. `gb_oliver_excited`). New default: `s3_rachel`. Added voices for EN-GB, EN-US, FR, DE, ES, NL, IT, PT.

---

### v0.3.1 — 2026-04-09
- **Fixed (HA 2026.4):** `TypeError: can only concatenate str (not "list") to str` — HA 2026.4 changed `chat_log.async_add_delta_content_stream` to expect the generator to yield plain types directly: `str` for text deltas, `llm.ToolInput` for completed tool calls. Our generator was still yielding wrapper dicts (`{"content": ..., "tool_calls": [...]}`), causing HA to attempt concatenating a list onto a string. Fixed `_async_stream_delta` to yield `str` and `llm.ToolInput` objects directly. Tool calls are still buffered until all arguments are streamed before being yielded.
- **Added:** Text-to-speech (TTS) platform using Mistral TTS (`mistral-tts-latest`) via `/v1/audio/speech`. Returns MP3 audio. Registers as a third separate HA device alongside Conversation and STT.
- **Added:** Six selectable TTS voices: nova (default), alloy, echo, fable, onyx, shimmer — all multilingual.
- **Added:** TTS voice selector in the integration options (Settings → Configure).

---

### v0.2.2.3 — 2026-03-05
- **Fixed:** `422 Unprocessable Entity` from Mistral API — HA tool parameters were being sent in HA's own intermediate list format `[{"type": "string", "name": "area", ...}]` instead of the OpenAI-compatible JSON Schema format Mistral requires (`{"type": "object", "properties": {...}, "required": [...]}`). Added `_ha_params_to_json_schema()` which performs the full conversion, including: `string/integer/float/boolean` primitives, `select` → `enum`, `multi_select` → array of enum, `list` → string array, `dict` → object. The `required` list is only populated for parameters that have `required: true` and no `optional: true`.

---

### v0.2.2.2 — 2026-03-05
- **Fixed:** `TypeError: Type is not JSON serializable: function` — voluptuous validators (`str`, `int`, `bool`, etc.) are Python callables and were ending up as values inside tool parameter schemas. Two-part fix:
  1. `_format_tool` now uses `voluptuous_serialize.convert()` with HA's `cv.custom_serializer` — the same approach used by HA's own OpenAI and Gemini integrations — to produce a proper JSON Schema dict from `tool.parameters`.
  2. `_sanitize` extended to handle non-serializable values (functions, types, voluptuous validators): anything that is not a JSON scalar, dict, or list is now converted to `repr(obj)` instead of being passed through, so a single unexpected value can never crash serialization.

---

### v0.2.2.1 — 2026-03-05
Community contributions merged with priority fix applied.

- **Fixed (priority):** `TypeError: Dict key must be a type serializable with OPT_NON_STR_KEYS` — root cause identified as voluptuous schema objects (`vol.Required`, `vol.Optional`) being used as dict keys in tool parameter schemas from HA's LLM API. A recursive `_sanitize()` helper now converts all dict keys to plain strings before any payload is passed to aiohttp. Applied to messages, tools, and all nested structures.
- **Fixed:** `_convert_chat_log_to_messages` now explicitly casts all `id`, `tool_name`, `content` values to `str`, and `tool_result`/`tool_args` dicts are also sanitized before `json.dumps`.
- **Added (community):** `MistralRuntimeData` dataclass in `__init__.py` — shared `aiohttp.ClientSession` and auth headers stored in `hass.data`, avoiding repeated header construction per request.
- **Added (community):** Re-authentication flow (`async_step_reauth`) — when the API key becomes invalid, HA now shows a re-auth notification instead of leaving the integration broken.
- **Added (community):** Native HA LLM API integration via `CONF_LLM_HASS_API` — replaces the custom `CONF_CONTROL_HA` approach. Device control now uses HA's standard `Assist` API, identical to how Google Gemini and OpenAI integrations work.
- **Added (community):** Streaming responses via `chat_log.async_add_delta_content_stream` — words appear progressively in the HA UI.
- **Added (community):** Tool-call loop (max 10 iterations) for multi-step HA device control commands.
- **Added (community):** Web search option (Beta) — uses Mistral's Agents/Conversations API. Requires `mistral-medium-latest` or `mistral-large-latest`.
- **Added (community):** STT now uses the shared runtime session from `hass.data` instead of creating a new client per request.
- **Kept:** `continue_conversation` (Experimental) — re-integrated into the new streaming architecture. Reads the final speech text from `ConversationResult` and sets `continue_conversation=True` when a `?` is detected.

---

### v0.2.2 — 2026-03-05
- **Fixed:** `TypeError: Dict key must be a type serializable with OPT_NON_STR_KEYS` — caused by a community contribution that passed HA `ChatLog` objects into the aiohttp JSON payload. The `_async_handle_message` method now intentionally ignores the `chat_log` argument and manages its own rolling history using `_make_message()`, which explicitly casts all keys and values to plain Python strings before serialization.
- **Fixed:** `service_data` keys returned by the model are also explicitly cast to `str` as an additional safeguard against non-string keys in nested payload structures.

---

### v0.2.1 — 2026-02-23
- **Fixed:** Service confirmation responses are now fully dynamic and language-aware. The AI generates the confirmation text itself (in whatever language the user is speaking) via a `"confirmation"` field in the JSON action payload. The hardcoded English `_SERVICE_PAST_TENSE` dictionary has been removed entirely.
- **Fixed:** `volume_set` service call was incorrectly blocked — added `volume_set`, `volume_mute`, `select_source`, `select_sound_mode`, `media_next_track`, `media_previous_track` to the media_player allowlist.
- **Fixed:** Service calls with extra parameters (e.g. `volume_level`, `temperature`) now work correctly via a `"service_data"` field in the JSON payload.
- **Improved:** Extended allowlist with `cover.set_cover_position`, `fan.set_percentage`, `fan.set_preset_mode`, `climate.set_temperature`, `climate.set_hvac_mode`, `input_boolean`, `input_number`, and `number` domains.

---

### v0.2.0 — 2026-02-23
**Breaking:** Removed Agent mode — integration now uses Model mode only.

- **Removed:** Agent mode and all Mistral Console agent configuration. All configuration is now done directly in Home Assistant.
- **Added:** `continue_conversation` option — when enabled, the assistant automatically keeps the microphone open after responses containing a question. Implemented natively via HA's `ConversationResult.continue_conversation` flag (no external automation required). Labelled *Experimental*.
- **Updated:** Model list — removed deprecated `ministral-7b-latest` and `open-codestral-mamba`. Added `ministral-8b-latest` (new default) and `ministral-3b-latest`. `ministral-8b-latest` is the recommended model for home automation: fast, cost-effective, and excellent at structured instruction-following.
- **Fixed:** All hardcoded Dutch strings in Python code replaced with English fallbacks. UI labels remain available in both English and Dutch via translation files.
- **Fixed:** Service confirmation messages no longer start with "Done!" / "Klaar!". Format is now e.g. *"Kitchen light has been turned off."*
- **Fixed:** Wrong GitHub URL in documentation corrected from `SnarfNL/mistral_conversation` to `SnarfNL/HA_MistralAI`.
- **Fixed:** Removed "(only in Model mode)" labels from all UI options since Agent mode no longer exists.
- **Optimised:** `_post_chat` error handling consolidated; `HomeAssistantError` and `aiohttp.ClientError` caught in a single handler. Error messages are now in English.
- **Optimised:** History trimming now preserves exactly the last 40 messages (20 turns) using a single slice operation.

---

### v0.1.8 — 2026-02-21
- **Added:** `icon.png` (128 px) and `icon@2x.png` (256 px) — Mistral M-logo on orange rounded-square background.
- **Added:** `images/` folder with 256 px and 512 px versions for submission to the home-assistant/brands repository.
- **Added:** Comprehensive `README.md` modelled after the BlaXun integration.
- **Fixed:** STT and conversation entities now have **separate `DeviceInfo`** with distinct `identifiers`, matching the pattern used by the Google Gemini integration.
- **Fixed:** `MistralSTTEntity` was missing `DeviceInfo` entirely — caused WebSocket handler errors (`Received binary message for non-existing handler`).
- **Fixed:** PCM-to-WAV wrapping now always applied regardless of `metadata.format`, fixing 400 errors on the Voxtral endpoint.
- **Fixed:** Full HTTP response body now logged on any 4xx/5xx from the chat API, making debugging possible.

---

### v0.1.7 — 2026-02-21
- **Fixed:** STT 400 error: HA always delivers raw PCM bytes; the WAV wrapper was incorrectly skipped when `metadata.format == WAV`.
- **Fixed:** Conversation 400 error: error response body was silently discarded; now logged at ERROR level.
- **Fixed:** `HomeAssistantError` raised inside `_post_chat` was not caught by the `aiohttp.ClientError` handler — added combined except clause.
- **Fixed:** `DeviceInfo` added to `MistralSTTEntity` to allow correct HA device registration.

---

### v0.1.6 — 2026-02-21
- **Added:** Speech-to-text (STT) platform using Mistral's **Voxtral Mini** (`voxtral-mini-latest`).
- **Added:** Agent mode — use a pre-configured agent from Mistral Console via `agent_id`.
- **Added:** STT language selector (dropdown with 60+ languages + Auto-detect).
- **Changed:** Conversation and STT entities registered as separate HA devices.

---

### v0.1.5 — 2026-02-21
- **Added:** `icon.png` and `icon@2x.png` in the component directory.
- **Added:** Full `README.md` with installation guide, option descriptions, automation examples and FAQ.

---

### v0.1.4 — 2026-02-21
- **Fixed:** Mistral API rejects `temperature` values above 1.0 — clamped to `0.0–1.0`.
- **Fixed:** Removed `top_p` from API payload (cannot be sent together with `temperature`).
- **Added:** `ConversationEntityFeature.CONTROL` to enable device control.
- **Improved:** JSON extraction from AI response now handles markdown code fences.

---

### v0.1.3 — 2026-02-21
- **Fixed:** `MistralOptionsFlow.__init__` tried to set `self.config_entry` which is a read-only property in HA 2024.x — removed `__init__`.
- **Added:** `_async_handle_message` (HA 2024.6+ API) with `async_process` fallback for older versions.

---

### v0.1.2 — 2026-02-21
- **Fixed:** Conversation agent did not appear in the Voice Assistants dropdown because entities were registered directly instead of via the `conversation` platform.
- **Changed:** Switched to `async_forward_entry_setups` with `PLATFORMS = ["conversation"]`.

---

### v0.1.1 — 2026-02-21
- **Fixed:** 500 error in config flow caused by incorrect OptionsFlow structure.
- **Changed:** Deprecated `conversation.async_set_agent()` replaced by proper platform setup.

---

### v0.1.0 — 2026-02-21
- Initial release.
- Mistral AI selectable as conversation agent in HA Assist.
- Configurable model, system prompt, temperature and max tokens via the UI.
- Home Assistant device control via spoken commands.
- Conversation history per session.
