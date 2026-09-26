"""Shared fixtures: a Mistral config entry and a set-up integration on real HA."""
from __future__ import annotations

from pathlib import Path

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.mistral_conversation.const import DOMAIN

API_KEY = "sk-test-key"
BASE = "https://api.mistral.ai/v1"
FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    """Return the text of tests/fixtures/<name>."""
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let HA load custom_components/mistral_conversation in every test."""


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A Mistral entry with default options, added to HA but not set up."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mistral AI Conversation",
        data={CONF_API_KEY: API_KEY},
        options={},
        unique_id=DOMAIN,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def setup_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> MockConfigEntry:
    """Set the entry up with /models and /audio/voices mocked.

    The core ``homeassistant`` component goes first: ``conversation`` (a
    dependency of this integration) needs its exposed-entities store.
    """
    assert await async_setup_component(hass, "homeassistant", {})
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
