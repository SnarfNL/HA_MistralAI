"""Each platform end to end through HA, with the API mocked from fixtures."""
from __future__ import annotations

import voluptuous as vol
from homeassistant.components import ai_task, conversation, stt, tts
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, selector
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

from .conftest import BASE, load_fixture

AGENT = "conversation.mistral_ai_conversation"


async def test_conversation_text_reply(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(f"{BASE}/chat/completions", text=load_fixture("chat_text.sse"))
    result = await conversation.async_converse(
        hass, "hoe warm is het?", None, None, "nl", agent_id=AGENT
    )
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Het is 21 graden in de woonkamer."


async def test_conversation_rate_limit_is_spoken(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(
        f"{BASE}/chat/completions", status=429, headers={"Retry-After": "0"}
    )
    result = await conversation.async_converse(
        hass, "hallo", None, None, "nl", agent_id=AGENT
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert (
        result.response.speech["plain"]["speech"]
        == "Mistral is tijdelijk overbelast. Probeer het zo nog eens."
    )


async def test_tts_batch_returns_mp3(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(f"{BASE}/audio/speech", text=load_fixture("tts_batch.json"))
    # Call the entity directly: HA's own TTS layer would re-encode with FFmpeg.
    entity = tts.get_engine_instance(hass, hass.states.async_entity_ids("tts")[0])
    assert entity is not None
    extension, data = await entity.async_get_tts_audio(
        "Test.", "en", {"voice": "en_paul_neutral"}
    )
    assert extension == "mp3"
    assert data.startswith(b"ID3")


async def test_stt_transcribes(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(
        f"{BASE}/audio/transcriptions", text=load_fixture("transcription.json")
    )
    engine = stt.async_get_speech_to_text_engine(
        hass, hass.states.async_entity_ids("stt")[0]
    )
    assert engine is not None
    meta = stt.SpeechMetadata(
        language="nl",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    async def audio():
        yield bytes(320)

    result = await engine.async_process_audio_stream(meta, audio())
    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "doe de keukenlamp aan"


def _sequence(*bodies: str):
    """side_effect for aioclient_mock: return *bodies* one per request."""
    queue = list(bodies)

    async def respond(method, url, data):
        return AiohttpClientMockResponse(method, url, text=queue.pop(0))

    return respond


async def test_conversation_tool_call_then_reply(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """A tool call round (HA runs the tool) followed by the spoken reply."""
    hass.config_entries.async_update_entry(
        mock_config_entry, options={"llm_hass_api": ["assist"]}
    )
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "intent", {})
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    aioclient_mock.post(
        f"{BASE}/chat/completions",
        side_effect=_sequence(
            load_fixture("chat_tool_call.sse"), load_fixture("chat_text.sse")
        ),
    )
    result = await conversation.async_converse(
        hass, "doe de keukenlamp aan", None, None, "nl", agent_id=AGENT
    )
    assert result.response.speech["plain"]["speech"] == "Het is 21 graden in de woonkamer."
    second_payload = aioclient_mock.mock_calls[-1][2]
    roles = [m["role"] for m in second_payload["messages"]]
    assert roles[-2:] == ["assistant", "tool"]


async def test_conversation_model_requested_web_search(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """The model calls web_search; the result is fed back for the final reply."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={"model": "mistral-small-latest", "web_search": True},
    )
    assert await async_setup_component(hass, "homeassistant", {})
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    search_call = load_fixture("chat_tool_call.sse").replace(
        '"name":"HassTurnOn","arguments":"{\\"name\\": \\"keukenlamp\\"}"',
        '"name":"web_search","arguments":"{\\"query\\": \\"weer morgen Amsterdam\\"}"',
    )
    aioclient_mock.post(
        f"{BASE}/chat/completions",
        side_effect=_sequence(search_call, load_fixture("chat_text.sse")),
    )
    aioclient_mock.post(
        f"{BASE}/conversations", text=load_fixture("conversation_web_search.json")
    )
    result = await conversation.async_converse(
        hass, "wat voor weer wordt het morgen?", None, None, "nl", agent_id=AGENT
    )
    assert result.response.speech["plain"]["speech"] == "Het is 21 graden in de woonkamer."
    final_payload = aioclient_mock.mock_calls[-1][2]
    assert "Morgen wordt het 18 graden" in final_payload["messages"][-1]["content"]


async def test_tts_stream_returns_wav(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(f"{BASE}/audio/speech", text=load_fixture("tts_stream.sse"))
    entity = tts.get_engine_instance(hass, hass.states.async_entity_ids("tts")[0])
    assert entity is not None

    async def message():
        yield "Dit is de eerste zin van het antwoord."

    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest(language="nl", options={}, message_gen=message())
    )
    audio = b"".join([chunk async for chunk in response.data_gen])
    assert response.extension == "wav"
    assert audio.startswith(b"RIFF")
    assert len(audio) == 44 + 480


async def test_ai_task_text(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.post(f"{BASE}/chat/completions", text=load_fixture("chat_text.sse"))
    result = await ai_task.async_generate_data(
        hass,
        task_name="test",
        entity_id=hass.states.async_entity_ids("ai_task")[0],
        instructions="Hoe warm is het?",
    )
    assert result.data == "Het is 21 graden in de woonkamer."


async def test_ai_task_structured(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    body = load_fixture("chat_text.sse").replace(
        "Het is 21 graden", '{\\"room\\": \\"Keuken\\", \\"temperature\\": 21'
    ).replace(" in de woonkamer.", "}")
    aioclient_mock.post(f"{BASE}/chat/completions", text=body)
    result = await ai_task.async_generate_data(
        hass,
        task_name="test structure",
        entity_id=hass.states.async_entity_ids("ai_task")[0],
        instructions="Geef een kamer.",
        structure=vol.Schema(
            {
                vol.Required("room"): selector.TextSelector(),
                vol.Required("temperature"): selector.NumberSelector(),
            }
        ),
    )
    assert result.data == {"room": "Keuken", "temperature": 21}
    payload = aioclient_mock.mock_calls[-1][2]
    assert payload["response_format"]["type"] == "json_schema"


async def test_refresh_voices_button(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    button_id = hass.states.async_entity_ids("button")[0]
    await hass.services.async_call(
        "button", "press", {"entity_id": button_id}, blocking=True
    )
