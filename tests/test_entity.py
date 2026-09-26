"""Device and entity registry entries must survive the refactor unchanged."""
from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import DOMAIN


async def test_device_identifiers_and_unique_ids_unchanged(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    assert {next(iter(d.identifiers)) for d in devices} == {
        (DOMAIN, f"{entry.entry_id}_conversation"),
        (DOMAIN, f"{entry.entry_id}_stt"),
        (DOMAIN, f"{entry.entry_id}_tts"),
    }
    assert {d.name for d in devices} == {
        "Mistral AI Conversation",
        "Mistral AI STT",
        "Mistral AI TTS",
    }
    unique_ids = {
        e.unique_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    }
    assert unique_ids == {
        f"{entry.entry_id}_{suffix}"
        for suffix in ("conversation", "ai_task", "stt", "tts", "refresh_voices")
    }


async def test_entity_ids_unchanged(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    ids = {
        e.entity_id
        for e in er.async_entries_for_config_entry(
            er.async_get(hass), setup_integration.entry_id
        )
    }
    assert ids == EXPECTED_ENTITY_IDS


async def test_conversation_device_model_follows_options(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    models = {
        next(iter(d.identifiers))[1]: d.model
        for d in dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    }
    assert models[f"{entry.entry_id}_conversation"] == "ministral-8b-latest"
    assert models[f"{entry.entry_id}_stt"] == "voxtral-mini-latest"
    assert models[f"{entry.entry_id}_tts"] == "voxtral-mini-tts-2603"


EXPECTED_ENTITY_IDS = {
    "ai_task.mistral_ai_conversation",
    "button.mistral_ai_tts_refresh_voices",
    "conversation.mistral_ai_conversation",
    "stt.mistral_ai_stt_mistral_ai_stt_voxtral",
    "tts.mistral_ai_tts_mistral_ai_tts",
}
