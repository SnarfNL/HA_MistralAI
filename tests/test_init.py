"""Setup and unload through real Home Assistant."""
from __future__ import annotations

from datetime import timedelta

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from .conftest import BASE, load_fixture


async def test_setup_and_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("conversation.mistral_ai_conversation") is not None
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_setup_invalid_key_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{BASE}/models", status=401)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress()
    assert any(f["context"]["source"] == "reauth" for f in flows)


async def test_setup_timeout_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{BASE}/models", exc=TimeoutError())
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_model_check_runs_daily(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    def model_calls() -> int:
        return sum(1 for call in aioclient_mock.mock_calls if str(call[1]).endswith("/models"))

    before = model_calls()
    freezer.tick(timedelta(hours=24, seconds=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert model_calls() == before + 1


async def test_model_check_stops_after_unload(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    assert await hass.config_entries.async_unload(setup_integration.entry_id)
    calls = aioclient_mock.call_count
    freezer.tick(timedelta(hours=24, seconds=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert aioclient_mock.call_count == calls


async def _setup_with_options(hass, entry, aioclient_mock, **options) -> None:
    hass.config_entries.async_update_entry(entry, options=options)
    assert await async_setup_component(hass, "homeassistant", {})
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def test_migration_turns_web_search_off(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    caplog,
) -> None:
    await _setup_with_options(
        hass,
        mock_config_entry,
        aioclient_mock,
        model="ministral-14b-latest",
        web_search=True,
        web_search_trigger="zoek op",
    )
    assert mock_config_entry.options["web_search"] is False
    assert mock_config_entry.options["web_search_trigger"] == "zoek op"
    message = "Web search turned off: model ministral-14b-latest does not support it"
    assert caplog.text.count(message) == 1


async def test_migration_does_not_loop(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    await _setup_with_options(
        hass,
        mock_config_entry,
        aioclient_mock,
        model="ministral-14b-latest",
        web_search=True,
    )
    assert mock_config_entry.state is ConfigEntryState.LOADED
    # One setup: the key check plus one background model check, no reload.
    model_calls = [c for c in aioclient_mock.mock_calls if str(c[1]).endswith("/models")]
    assert len(model_calls) == 2


async def test_supported_model_keeps_web_search(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    await _setup_with_options(
        hass,
        mock_config_entry,
        aioclient_mock,
        model="mistral-small-latest",
        web_search=True,
    )
    assert mock_config_entry.options["web_search"] is True
