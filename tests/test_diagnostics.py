"""Diagnostics download (MA-14): settings, versions, last errors — never the key."""
from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.api import mistral_error
from custom_components.mistral_conversation.diagnostics import (
    async_get_config_entry_diagnostics,
)

from .conftest import API_KEY


async def test_diagnostics_content(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    entry.runtime_data.client.record(
        "tts", mistral_error("api_error", status="404"), 404
    )
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["entry"]["data"]["api_key"] == "**REDACTED**"
    assert diag["entry"]["options"] == dict(entry.options)
    assert diag["versions"]["home_assistant"]
    assert diag["versions"]["integration"]
    assert diag["model"]["configured"] == "ministral-8b-latest"
    assert diag["recent_errors"][-1]["source"] == "tts"
    assert diag["recent_errors"][-1]["status"] == 404
    assert diag["recent_errors"][-1]["error"] == "api_error"
    assert diag["voices_loaded"] == 2


async def test_diagnostics_never_contains_api_key(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    entry.runtime_data.client.record("setup", HomeAssistantError(f"bad {API_KEY}"))
    entry.runtime_data.errors.append(
        {"time": "t", "source": "x", "status": None, "error": f"leak {API_KEY}"}
    )
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert API_KEY not in str(diag)


async def test_diagnostics_keeps_last_ten_errors(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    for _ in range(12):
        entry.runtime_data.client.record(
            "conversation", mistral_error("rate_limited"), 429
        )
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert len(diag["recent_errors"]) == 10
