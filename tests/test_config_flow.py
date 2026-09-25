"""Config flow, reauth and options flow on real HA (100 % coverage gate)."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.mistral_conversation.const import DOMAIN

from .conftest import API_KEY, BASE


async def test_user_step_creates_entry(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{BASE}/models", json={"data": []})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    with patch(
        "custom_components.mistral_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"api_key": API_KEY}
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {"api_key": API_KEY}


@pytest.mark.parametrize(
    ("mock_kwargs", "error"),
    [
        ({"status": 401}, "invalid_auth"),
        ({"status": 500}, "cannot_connect"),
        ({"exc": TimeoutError()}, "cannot_connect"),
    ],
)
async def test_user_step_errors(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_kwargs: dict,
    error: str,
) -> None:
    aioclient_mock.get(f"{BASE}/models", **mock_kwargs)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-bad"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}


async def test_user_step_unexpected_error(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{BASE}/models", exc=ValueError("boom"))
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-bad"}
    )
    assert result["errors"] == {"base": "unknown"}


async def test_user_step_already_configured(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{BASE}/models", json={"data": []})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": API_KEY}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_updates_key(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    result = await entry.start_reauth_flow(hass)
    assert result["step_id"] == "reauth_confirm"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-new"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data["api_key"] == "sk-new"


async def test_reauth_invalid_key_shows_error(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=401)
    result = await entry.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-bad"}
    )
    assert result["errors"] == {"base": "invalid_auth"}
    assert entry.data["api_key"] == API_KEY


async def test_options_flow_saves(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    result = await _options(
        hass,
        entry,
        "ministral-3b-latest",
        {"llm_hass_api": [], "temperature": 0.3},
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["model"] == "ministral-3b-latest"
    assert entry.options["temperature"] == 0.3
    assert "llm_hass_api" not in entry.options


# ---------------------------------------------------------------------------
# Reconfigure (MA-14): change the API key without removing the integration
# ---------------------------------------------------------------------------


async def _start_reconfigure(hass: HomeAssistant, entry: MockConfigEntry):
    return await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )


async def test_reconfigure_changes_key(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    result = await _start_reconfigure(hass, entry)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-new"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["api_key"] == "sk-new"


async def test_reconfigure_invalid_key_shows_error(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=401)
    result = await _start_reconfigure(hass, entry)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-bad"}
    )
    assert result["errors"] == {"base": "invalid_auth"}
    assert entry.data["api_key"] == API_KEY


async def test_reconfigure_cannot_connect_keeps_old_key(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", exc=TimeoutError())
    result = await _start_reconfigure(hass, entry)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_key": "sk-new"}
    )
    assert result["errors"] == {"base": "cannot_connect"}
    assert entry.data["api_key"] == API_KEY


# ---------------------------------------------------------------------------
# Options in two steps; web search follows the model (MA-30)
# ---------------------------------------------------------------------------


async def _options(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    model: str,
    settings: dict | None = None,
):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["step_id"] == "init"
    assert _fields(result) == {"model"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"model": model}
    )
    assert result["step_id"] == "settings"
    if settings is None:
        return result
    return await hass.config_entries.options.async_configure(
        result["flow_id"], settings
    )


def _fields(result) -> set[str]:
    return {str(key) for key in result["data_schema"].schema}


def _default(result, name: str):
    for key in result["data_schema"].schema:
        if str(key) == name:
            return key.default()
    raise KeyError(name)


WEB_FIELDS = {"web_search", "web_search_mode", "web_search_trigger"}


async def _set_options(hass: HomeAssistant, entry: MockConfigEntry, **options) -> None:
    hass.config_entries.async_update_entry(entry, options=options)
    await hass.async_block_till_done()


async def test_unsupported_model_hides_web_search(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _options(hass, setup_integration, "ministral-14b-latest")
    assert not WEB_FIELDS & _fields(result)
    assert {"prompt", "temperature", "max_tokens", "tts_mode"} <= _fields(result)


async def test_supported_model_shows_web_search(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _options(hass, setup_integration, "mistral-small-latest")
    assert WEB_FIELDS <= _fields(result)


async def test_unsupported_model_saves_web_search_off(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_options(hass, entry, model="mistral-small-latest", web_search=True)
    result = await _options(hass, entry, "ministral-14b-latest", {})
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["web_search"] is False
    assert entry.options["model"] == "ministral-14b-latest"


async def test_unsupported_model_keeps_mode_and_trigger(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_options(
        hass,
        entry,
        model="mistral-small-latest",
        web_search=True,
        web_search_mode="always",
        web_search_trigger="zoek op",
    )
    await _options(hass, entry, "ministral-14b-latest", {})
    await hass.async_block_till_done()
    assert entry.options["web_search_mode"] == "always"
    assert entry.options["web_search_trigger"] == "zoek op"


async def test_switch_to_supported_model_turns_web_search_on(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_options(hass, entry, model="ministral-14b-latest", web_search=False)
    result = await _options(hass, entry, "mistral-small-latest")
    assert _default(result, "web_search") is True


async def test_first_options_with_supported_model_turn_web_search_on(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _options(hass, setup_integration, "mistral-large-latest")
    assert _default(result, "web_search") is True


async def test_manual_off_on_supported_model_is_kept(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_options(hass, entry, model="mistral-small-latest", web_search=False)
    result = await _options(hass, entry, "mistral-large-latest")
    assert _default(result, "web_search") is False
