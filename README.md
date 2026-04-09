<a name="readme-top"></a>

<div align="center">
  <img src="custom_components/mistral_conversation/icon@2x.png" alt="Mistral AI Conversation" width="128" height="128">

  <h1>Mistral AI Conversation</h1>
  <p><strong>Home Assistant custom integration — Mistral AI as conversation agent, Voxtral for speech-to-text, and Mistral TTS for text-to-speech.</strong></p>

  <p><em>⚠️ Please note this is not an officially supported integration and is not affiliated with Mistral AI in any way.</em></p>

  [![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge)](https://github.com/hacs/integration)
  [![HA Version](https://img.shields.io/badge/Home%20Assistant-2023.5%2B-blue?style=for-the-badge&logo=home-assistant)](https://www.home-assistant.io/)
  [![Mistral AI](https://img.shields.io/badge/Mistral%20AI-Powered-orange?style=for-the-badge)](https://mistral.ai/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
</div>

---

## Table of Contents

1. [About](#about)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
   - [Creating an API key](#creating-an-api-key)
   - [Setting up the integration](#setting-up-the-integration)
   - [Selecting as voice assistant](#selecting-as-voice-assistant)
6. [Options](#options)
   - [Available models](#available-models)
   - [System prompt](#system-prompt)
   - [Continue conversation (Experimental)](#continue-conversation-experimental)
7. [Controlling devices](#controlling-devices)
8. [Using as a service action](#using-as-a-service-action)
9. [Speech recognition (STT)](#speech-recognition-stt)
10. [Text-to-speech (TTS)](#text-to-speech-tts)
11. [FAQ](#faq)
12. [Release Notes](#release-notes)
13. [License](#license)

---

## About

This integration makes **Mistral AI** available as a fully-featured conversation agent inside Home Assistant's built-in Assist voice pipeline. It also registers **Voxtral** (Mistral's own speech-to-text model) as a native HA STT provider — creating two separate devices, one for conversation and one for transcription, just like the official Google Gemini integration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Features

| Feature | Status | Description |
|---|---|---|
| Conversation agent in HA Assist | ✅ | Selectable as agent in Voice Assistants |
| Smart home control | ✅ | Control lights, switches, covers, locks, etc. |
| Speech recognition (STT) | ✅ | Voxtral Mini via `/v1/audio/transcriptions` |
| Text-to-speech (TTS) | ✅ | Mistral TTS via `/v1/audio/speech` with multiple voices |
| Conversation memory | ✅ | Context kept per session (20 turns) |
| Jinja2 system prompt | ✅ | Templates with `{{ now() }}`, `{{ ha_name }}` etc. |
| Multilingual | ✅ | Responds in the user's language |
| Continue conversation | ✅ | Keeps microphone open after questions (Experimental) |
| Separate devices | ✅ | Conversation and STT appear as separate HA devices |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Requirements

| Requirement | Minimum version |
|---|---|
| Home Assistant Core | 2023.5 |
| Python | 3.11 |
| Mistral AI account + API key | — |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Installation

### Via HACS (recommended)

1. HACS → **Integrations** → ⋮ → **Custom repositories**
2. URL: `https://github.com/SnarfNL/HA_MistralAI` — category: **Integration**
3. Search "Mistral AI Conversation" → **Download**
4. **Fully restart** Home Assistant

### Manual

1. Copy `custom_components/mistral_conversation/` to `/config/custom_components/`
2. Remove old `__pycache__` directories if updating from a previous version:
   ```bash
   rm -rf /config/custom_components/mistral_conversation/__pycache__
   ```
3. **Fully restart** Home Assistant (not just reload)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration

### Creating an API key

1. Sign up at [mistral.ai](https://mistral.ai/)
2. Go to [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys)
3. Click **Create new key** and copy it immediately

### Setting up the integration

1. **Settings → Devices & Services → + Add Integration**
2. Search for **Mistral AI Conversation**
3. Enter your API key → **Submit**

### Selecting as voice assistant

1. **Settings → Voice Assistants** → click your assistant
2. Set **Conversation agent** to **Mistral AI Conversation**
3. Optionally set **Speech-to-text** to **Mistral AI STT (Voxtral)**
4. Save

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Options

Click the integration → **Configure** to change settings.

| Option | Default | Description |
|---|---|---|
| **AI model** | `ministral-8b-latest` | Which Mistral model to use |
| **System prompt** | See below | Jinja2 template with AI instructions |
| **Temperature** | `0.7` | Creativity: 0.0 = deterministic, 1.0 = creative |
| **Max tokens** | `1024` | Maximum response length |
| **Control HA** | On | Allow the AI to control exposed devices |
| **Continue conversation** | Off | Keep listening after questions (Experimental) |
| **STT language** | Auto-detect | Language for Voxtral transcription |

### Available models

| Model | Speed | Cost | Best for |
|---|---|---|---|
| `ministral-8b-latest` ⭐ | ★★★★★ | $ | Home automation commands — fast, accurate, cheap |
| `ministral-3b-latest` | ★★★★★ | $ | Ultra-simple commands, lowest latency |
| `mistral-small-latest` | ★★★★ | $$ | Balanced: quality and speed |
| `mistral-large-latest` | ★★★ | $$$$ | Complex reasoning, long conversations |
| `open-mistral-nemo` | ★★★★ | $ | Open-source alternative |

> **Recommendation:** Start with `ministral-8b-latest`. It has excellent instruction-following, handles structured JSON output reliably (needed for device control), and costs a fraction of larger models.

### System prompt

The system prompt supports Jinja2 templates:

```jinja2
You are a helpful voice assistant for {{ ha_name }}.
Answer in the same language the user speaks.
Today is {{ now().strftime('%A, %B %d, %Y') }} and the time is {{ now().strftime('%H:%M') }}.
Be concise and friendly.
```

**Available template variables:**

| Variable | Description |
|---|---|
| `{{ ha_name }}` | Your Home Assistant location name |
| `{{ now() }}` | Current datetime object |
| `{{ now().strftime(…) }}` | Formatted date/time string |

### Continue conversation (Experimental)

When enabled, the assistant automatically keeps the microphone open after any response that contains a question (`?`). This is implemented using the native `continue_conversation` flag in HA's `ConversationResult` — no separate automation is needed.

> **Note:** This feature requires a satellite device that supports `assist_satellite.start_conversation`. Behaviour may vary between satellite types.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Controlling devices

Enable **Allow AI to control Home Assistant devices** in the options, then expose the entities you want via **Settings → Voice Assistants → Exposed devices**.

### Example commands

| What you say | What happens |
|---|---|
| "Turn off the kitchen light" | `light.turn_off` |
| "Open the blinds" | `cover.open_cover` |
| "Lock the front door" | `lock.lock` |
| "Play something in the living room" | `media_player.media_play` |
| "Activate the movie scene" | `scene.turn_on` |

### Supported domains

`light` · `switch` · `cover` · `media_player` · `fan` · `climate` · `lock` · `alarm_control_panel` · `scene` · `script` · `automation` · `homeassistant`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Using as a service action

Use `conversation.process` in automations or scripts:

```yaml
action: conversation.process
data:
  agent_id: conversation.mistral_ai_conversation
  text: "What is the temperature in the living room?"
response_variable: result
```

The response text is in `result.response.speech.plain.speech`.

### Example: Smart doorbell notification

```yaml
alias: Smart doorbell notification
sequence:
  - action: conversation.process
    data:
      agent_id: conversation.mistral_ai_conversation
      text: >
        The doorbell rang at {{ now().strftime('%H:%M') }}.
        Write a short, friendly notification message.
    response_variable: ai_result
  - action: notify.mobile_app
    data:
      title: "Doorbell 🔔"
      message: "{{ ai_result.response.speech.plain.speech }}"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Speech recognition (STT)

A `stt.mistral_ai_stt_voxtral` entity is registered automatically.

### Voxtral specifications

| Property | Value |
|---|---|
| Model | `voxtral-mini-latest` |
| Supported format | WAV (16-bit, 16 kHz, mono PCM) |
| Languages | 60+ with auto-detect |
| Pricing | ~$0.003 per minute |

### Setting STT language

In the options, select a language from the dropdown for best accuracy, or leave it on **Auto-detect**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


---

## Text-to-speech (TTS)

When the integration is installed, a **Mistral AI TTS** entity is registered automatically as a separate HA TTS provider. It uses Mistral's  endpoint and returns MP3 audio.

### Selecting a voice

In **Settings → Devices & Services → Mistral AI Conversation → Configure**, choose from the available voices:

| Voice | Character |
|---|---|
| nova | Neutral, clear — recommended default |
| alloy | Warm, conversational |
| echo | Balanced, slightly deeper |
| fable | Expressive, British accent |
| onyx | Deep, authoritative |
| shimmer | Soft, friendly |

All voices support all languages.

### Using TTS in automations



<p align=right>(<a href=#readme-top>back to top</a>)</p>
---

## FAQ

**Q: The integration does not appear in the Voice Assistants dropdown.**
A: Make sure you performed a full restart (not just reload) and cleared any `__pycache__` directories.

**Q: I get a 400 Bad Request error.**
A: Check the HA logs for the full error body. A common cause is an invalid model name or a temperature value outside 0.0–1.0.

**Q: Can I use TTS with this integration?**
A: Now that Mistral has a TTS API, yes. Please refer to Text-to-speech (TTS) section for details.

**Q: How much does it cost?**
A: With `ministral-8b-latest` and typical home use, expect less than €1–2 per month. Voxtral STT adds ~€0.003/minute. See [mistral.ai/pricing](https://mistral.ai/pricing/).

**Q: Does continue conversation work on all satellites?**
A: It requires a satellite that supports the `assist_satellite` integration and `start_conversation`. It has been tested with ESPHome voice satellites. Behaviour on other devices may vary.

**Q: Are my conversations stored?**
A: Mistral AI processes requests via their servers. See their [privacy policy](https://mistral.ai/privacy-policy) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Release Notes

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<div align="center">
  Made with ❤️ for the Home Assistant community<br>
  Inspired by the work of <a href="https://github.com/BlaXun/home_assistant_mistral_ai">BlaXun</a>
</div>
