# Mistral TTS stays a native Home Assistant TTS entity, not a Wyoming server

We considered moving text-to-speech into a separate Wyoming server (in the style of `wyoming-microsoft-tts`, deployed as an add-on or Docker container) so that voice selection would follow the Wyoming pattern. We decided against it: the integration's TTS entity already gets the native voice picker under Settings → Voice assistants, so a Wyoming server would only move the default-voice setting into add-on configuration, replace a refresh button with a server restart, and add a second project (image, add-on repository, releases) to maintain.

Voice selection therefore lives in the assistant settings (or the `voice` option of `tts.speak`). The integration has no voice option of its own; it uses a built-in default voice when none is given, and the picker lists the account's voices, refreshed on demand (MA-13).

Revisit if we want Mistral TTS/STT usable by non-HA Wyoming clients. That would be a separate server, not a replacement for the entity.
