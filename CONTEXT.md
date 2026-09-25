# CONTEXT.md

Shared vocabulary for HA_MistralAI. Use these terms consistently in code, issues,
PRs and conversations with agents. Dutch equivalents in brackets, because the
maintainer and PR descriptions use Dutch.

## Home Assistant side

- **Assist** — Home Assistant's voice/text assistant framework. Users talk to Assist; Assist hands the text to a *conversation agent*.
- **Voice pipeline** [spraakpipeline] — the chain wake word → STT → conversation agent → TTS, configured under *Settings → Voice assistants*. One pipeline per assistant.
- **Satellite** [satelliet] — a physical voice device (ESPHome/Voice PE, Wyoming) that streams audio to a pipeline and plays the TTS reply.
- **Conversation agent** [gespreksagent] — the entity that turns user text into a reply. In this integration: `MistralConversationEntity`.
- **Chat log** — HA's per-conversation history object (`conversation.ChatLog`). Source of truth for messages sent to Mistral. Identified by `conversation_id`.
- **Delta stream** — the async generator of dicts fed to `chat_log.async_add_delta_content_stream` (`{"role"}`, `{"content"}`, `{"tool_calls"}`).
- **LLM API** — HA's tool layer (e.g. "Assist") that exposes entities and intents to the model as *tools*. Selected in the option *Control Home Assistant*.
- **Tool / tool call** [tool-aanroep] — a function the model asks to run. *HA tools* are executed by HA; an *external tool call* (`ToolInput(external=True)`) is executed by the provider or by this integration (web search).
- **Exposed entity** [gedeelde entiteit] — an entity the user has made visible to Assist; only these may be controlled or named to the model.
- **AI Task** — HA's non-conversational LLM entity (`ai_task.generate_data`, later `generate_image`), used from automations.
- **Config entry** — one configured instance of the integration (holds the API key). **Options** are its editable settings. **Subentry** — a child configuration under a config entry (planned: one per conversation agent / AI Task, MA-12).
- **Runtime data** — per-entry objects created at setup (HTTP session, headers, caches). Currently in `hass.data`, moving to `entry.runtime_data` (MA-10).
- **Continue conversation** [doorluisteren] — the satellite keeps listening after a reply, without a new wake word.
- **Repair issue** — a user-facing HA notification with an optional fix flow.

## Mistral side

- **Model** — a Mistral chat model ID, preferably a `-latest` alias (e.g. `ministral-8b-latest`). *Agent-capable models* can use the Agents/Conversations API.
- **Chat Completions API** — `/v1/chat/completions`; the default path for every conversation turn.
- **Agents / Conversations API** — `/v1/agents`, `/v1/conversations`; the only place Mistral's built-in `web_search` works.
- **Web-search agent** — the single Mistral agent this integration creates to service web searches ("HA Mistral Web Search").
- **Voxtral (STT)** — Mistral's speech-to-text models. *Batch* = upload a whole WAV; *realtime* = WebSocket streaming (MA-24).
- **Context bias** — up to 100 words/phrases sent to Voxtral to improve recognition of names (MA-18).
- **Voice** [stem] — a Mistral TTS voice. Has an `id` (UUID, used for synthesis) and a `name` (shown in pickers). *Custom voice* = created by the user in Mistral Studio or by cloning.
- **Free tier / Experiment tier** — rate-limited Mistral API keys; Medium and Large are throttled first.

## Integration internals

- **Batch TTS** — one request, full mp3 back. Used by `tts.speak`.
- **Streaming TTS** — per-sentence WAV requests pipelined in parallel and stitched into one stream (one RIFF header, then raw PCM, with silence between sentences).
- **Sentence splitter** [zinssplitser] — `pop_complete_sentences()` in `_streaming.py`; decides when a sentence is complete enough to send to TTS.
- **Web search mode** — `model` (the model decides via a synthetic `web_search` tool), `always` (legacy), or *trigger phrases*.
- **Story** — a backlog item `MA-xx`, tracked as a GitHub issue with label `backlog`. See `docs/backlog.md`.
