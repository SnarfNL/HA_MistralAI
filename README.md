<a name="readme-top"></a>

<div align="center">
  <img src="custom_components/mistral_conversation/icon@2x.png" alt="Mistral AI Conversation" width="128" height="128">

  <h1>Mistral AI Conversation</h1>
  <p><strong>Home Assistant custom integration — Mistral AI as conversation agent, Voxtral for speech-to-text, and Mistral TTS for text-to-speech.</strong></p>

  <p><em>⚠️ Please note this is not an officially supported integration and is not affiliated with Mistral AI in any way.</em></p>

  [![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge)](https://github.com/hacs/integration)
  [![HA Version](https://img.shields.io/badge/Home%20Assistant-2025.10%2B-blue?style=for-the-badge&logo=home-assistant)](https://www.home-assistant.io/)
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
   - [Web search (Beta)](#web-search-beta)
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
| Streaming TTS | ✅ | Low-latency SSE WAV with sentence-level pipelining (≥ v0.4.0) |
| Conversation memory | ✅ | Context kept per session until 5 min idle (HA timeout). |
| Jinja2 system prompt | ✅ | Templates with `{{ now() }}`, `{{ ha_name }}` etc. |
| Multilingual | ✅ | Responds in the user's language |
| Continue conversation | ✅ | Keeps microphone open after questions (Experimental) |
| Web search | ✅ | Model-decided web search via the Conversations API, or trigger phrases (Beta) |
| Separate devices | ✅ | Conversation and STT appear as separate HA devices |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Requirements

| Requirement | Minimum version |
|---|---|
| Home Assistant Core | 2025.10 |
| Python | 3.13 |
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
| **Web search** | Off | Allow the AI to search the web (Beta) |
| **Web search routing** | `Let the model decide` | How web search is triggered — see below |
| **Web search trigger phrases** | *(empty)* | Optional phrases that force a web search — see below |
| **STT language** | Auto-detect | Language for Voxtral transcription |
| **TTS mode** | `Streaming` | `Streaming` (SSE WAV with sentence-level pipelining) or `Batch` (single MP3 request) |
### Available models

Verified against Mistral's model lineup, September 2026. All entries use Mistral's `-latest` aliases, which track the current model automatically as Mistral ships new versions (e.g. `mistral-large-latest` now serves Mistral Large 3).

| Model | Speed | Cost | Free tier | Best for |
|---|---|---|---|---|
| `ministral-8b-latest` ⭐ | ★★★★★ | $ | ✅ | Home automation commands — fast, accurate, cheap |
| `ministral-3b-latest` | ★★★★★ | $ | ✅ | Ultra-simple commands, lowest latency |
| `ministral-14b-latest` 🆕 | ★★★★ | $ | ✅ | New (Dec 2025). Strongest of the small edge models |
| `mistral-small-latest` | ★★★★ | $$ | ✅ | Balanced: quality and speed |
| `mistral-medium-latest` | ★★★ | $$$ | ⚠️ | Required for web search; heavier, often rate-limited on free API keys |
| `mistral-large-latest` | ★★★ | $$$$ | ⚠️ | Complex reasoning, long conversations; heavier, often rate-limited on free API keys |

> **Recommendation:** Start with `ministral-8b-latest`. It has excellent instruction-following, handles structured JSON output reliably (needed for device control), and costs a fraction of larger models.
>
> **About the "Free tier" column:** Mistral's free/Experiment API tier gives rate-limited trial access to all models — Mistral doesn't publish an official per-model block list. In practice, though, Medium and Large are the ones most likely to hit those limits or get rejected on a free-tier key, since they're the most expensive per token. If you're on a free key and see errors on Medium or Large, that's most likely why — check your usage tier at [admin.mistral.ai](https://admin.mistral.ai/plateforme/limits).
>
> **Removed:** `open-mistral-nemo` (Mistral Nemo 12B) has been deprecated and retired by Mistral and no longer works — switch to `ministral-8b-latest` or `ministral-3b-latest` if your config still uses it.

### System prompt

The system prompt supports Jinja2 templates:

```jinja2
You are a helpful voice assistant for {{ ha_name }}.
Answer in the same language the user speaks.
Today is {{ now().strftime('%A, %B %d, %Y') }} and the time is {{ now().strftime('%H:%M') }}.
Be concise and friendly.
For straightforward home control commands, briefly confirm the action taken without asking follow-up questions.
Your responses are read aloud by text-to-speech, so reply in plain text.
Do not use markdown formatting that cannot be read aloud, such as asterisks for bold, underscores for italics, backticks, bullet lists, emojis, or headers.
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

### Web search (Beta)

Web search is only available through Mistral's **Conversations API**, which is a separate,
slower endpoint than the regular chat completions call and **cannot carry Home
Assistant tools**. A turn answered by the Conversations API therefore cannot control
your devices. Requires a model that supports Mistral's built-in web search (`mistral-small-latest`,
`mistral-medium-latest` or `mistral-large-latest`).

**Web search routing** controls when that endpoint is used:

| Mode | Behaviour |
|---|---|
| **Let the model decide** (default) | Web search is offered to the model as a tool. It searches only when it judges a search is needed, and Home Assistant tools stay available — so one turn can search *and* control devices. |
| **Always search** | Legacy behaviour: every request goes to the Conversations API. Slower, and device control does not work on those turns. |

> **Performance note:** enabling web search with **Always search** routes *every*
> utterance — including "turn on the lamp" — through the Conversations API. Use
> **Let the model decide**, or set trigger phrases, to keep ordinary commands on
> the fast path.

**Web search trigger phrases** (optional) is a comma-separated list, for example:

```
search for, look up, google
```

When it is non-empty it **takes precedence over the routing mode**: only utterances
that *start with* one of the phrases search the web (the phrase itself is stripped
from the query), and every other utterance never searches. If several phrases
match, the longest one wins. Matching is case-insensitive. Leave the field empty
to let the routing mode decide.

Trigger phrases are language-specific, so nothing is shipped by default. Dutch
users might use `zoek op, zoek online, google`; German users `suche, google`.

> Note: with trigger phrases set, a matching turn goes straight to the Conversations API
> and so cannot control devices — that is the same trade-off as **Always search**,
> just limited to utterances you opt into.

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

When the integration is installed, a **Mistral AI TTS** entity is registered automatically as a separate HA TTS provider. It uses Mistral's `/v1/audio/speech` endpoint and supports two operating modes (see [TTS modes](#tts-modes) below): low-latency streaming WAV (default, since v0.4.0) and single-shot MP3.

### TTS modes

Selectable via **Settings → Devices & Services → Mistral AI Conversation → Configure → Text-to-speech mode**:

| Mode | Behaviour | First audio | Best for |
|---|---|---|---|
| **Streaming** *(default)* | Server-Sent Events with chunked WAV. Sentences are extracted from the LLM token stream and dispatched to Mistral in parallel (up to 5 concurrent), with audio reassembled in strict order. | ~0.5–1 s | HA Voice satellites (Voice PE, ESPHome) |
| **Batch** | Single non-streaming POST returns the full MP3 in a JSON body before any audio is yielded. | After full synthesis (~3–5 s for long replies) | Direct `tts.speak` service calls; environments where chunked HTTP is problematic |

> **Note:** Direct `tts.speak` service calls always use the batch path regardless of this setting (the service expects a single audio file, not a stream).

### Voxtral TTS specifications

| Property | Value |
|---|---|
| Model | `voxtral-mini-tts-2603` |
| Stream container | WAV (24 kHz, 16-bit, mono PCM) |
| Batch container | MP3 (base64-wrapped JSON response) |
| Voices | EN-US (Paul), GB (Jane, Oliver), FR (Marie) — emotion variants |

### Selecting a voice

In **Settings → Devices & Services → Mistral AI Conversation → Configure**, choose from the available voices.
The available voices are retrieved dynamically. Currently there are only voices for EN, GB and FR available. 

<p align=right>(<a href=#readme-top>back to top</a>)</p>
---

## FAQ

**Q: The integration does not appear in the Voice Assistants dropdown.**
A: Make sure you performed a full restart (not just reload) and cleared any `__pycache__` directories.

**Q: I get a 400 Bad Request error.**
A: Check the HA logs for the full error body. A common cause is an invalid model name or a temperature value outside 0.0–1.0.

**Q: Can I use TTS with this integration?**
A: Now that Mistral has a TTS model, yes. Please refer to Text-to-speech (TTS) section for details.

**Q: How much does it cost?**
A: With `ministral-8b-latest` and typical home use, expect less than €1–2 per month. Voxtral STT adds ~€0.003/minute. See [mistral.ai/pricing](https://mistral.ai/pricing/).

**Q: Does continue conversation work on all satellites?**
A: It requires a satellite that supports the `assist_satellite` integration and `start_conversation`. It has been tested with ESPHome voice satellites. Behaviour on other devices may vary.

**Q: Are my conversations stored?**
A: Mistral AI processes requests via their servers. See their [privacy policy](https://mistral.ai/privacy-policy) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

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
