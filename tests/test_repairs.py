"""Retired-model repair (MA-14): detection, suggested replacement, fix flow."""
from __future__ import annotations

import json

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.mistral_conversation._models import (
    async_check_model,
    issue_id,
    suggest_replacement,
)
from custom_components.mistral_conversation.const import DOMAIN

from .conftest import BASE, load_fixture

MODELS = json.loads(load_fixture("models.json"))["data"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # rule 1: the -latest alias of the same model name
        ("ministral-8b-2410", "ministral-8b-latest"),
        # rule 2: first recommended model of the same family
        ("ministral-9b-latest", "ministral-8b-latest"),
        ("mistral-tiny-2312", "mistral-small-latest"),
        # rule 3: the default model
        ("open-mistral-nemo", "ministral-8b-latest"),
    ],
)
def test_suggest_replacement(model: str, expected: str) -> None:
    assert suggest_replacement(model, MODELS) == expected


def _issue(hass: HomeAssistant, entry: MockConfigEntry) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, issue_id(entry.entry_id))


async def _set_model(hass: HomeAssistant, entry: MockConfigEntry, model: str) -> None:
    hass.config_entries.async_update_entry(entry, options={"model": model})
    await hass.async_block_till_done()


async def test_retired_model_creates_issue_and_available_model_clears_it(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_model(hass, entry, "open-mistral-nemo")
    await async_check_model(hass, entry)
    issue = _issue(hass, entry)
    assert issue is not None
    assert issue.is_fixable
    assert issue.translation_placeholders == {
        "model": "open-mistral-nemo",
        "replacement": "ministral-8b-latest",
    }

    await _set_model(hass, entry, "ministral-14b-latest")
    await async_check_model(hass, entry)
    assert _issue(hass, entry) is None


async def test_check_runs_in_the_background_at_setup(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    hass.config_entries.async_update_entry(
        mock_config_entry, options={"model": "open-mistral-nemo"}
    )
    assert await async_setup_component(hass, "homeassistant", {})
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))
    aioclient_mock.get(f"{BASE}/audio/voices", text=load_fixture("voices.json"))
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert _issue(hass, mock_config_entry) is not None
    assert mock_config_entry.runtime_data.models == MODELS


async def test_model_check_failure_changes_nothing(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    # The reload after this options change already ran the check in the
    # background, so the issue exists; a useless model list must keep it as is.
    await _set_model(hass, entry, "open-mistral-nemo")
    before = _issue(hass, entry)
    assert before is not None
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=429, headers={"Retry-After": "0"})
    await async_check_model(hass, entry)
    assert _issue(hass, entry) == before
    assert entry.runtime_data.models == MODELS


async def test_model_check_ignores_malformed_list(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    entry = setup_integration
    # The reload after this options change already ran the check in the
    # background, so the issue exists; a useless model list must keep it as is.
    await _set_model(hass, entry, "open-mistral-nemo")
    before = _issue(hass, entry)
    assert before is not None
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", json={"object": "list"})
    await async_check_model(hass, entry)
    assert _issue(hass, entry) == before
    assert entry.runtime_data.models == MODELS


async def test_fix_flow_applies_replacement_and_clears_issue(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entry = setup_integration
    await _set_model(hass, entry, "open-mistral-nemo")
    await async_check_model(hass, entry)
    assert await async_setup_component(hass, "repairs", {})
    manager = hass.data["repairs"]["flow_manager"]

    result = await manager.async_init(
        DOMAIN, data={"issue_id": issue_id(entry.entry_id)}
    )
    assert result["step_id"] == "confirm"
    assert result["description_placeholders"] == {
        "model": "open-mistral-nemo",
        "replacement": "ministral-8b-latest",
    }
    result = await manager.async_configure(result["flow_id"], {})
    await hass.async_block_till_done()

    assert result["type"] == "create_entry"
    assert entry.options["model"] == "ministral-8b-latest"
    assert _issue(hass, entry) is None


async def test_fix_flow_for_a_removed_entry_just_closes(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    from custom_components.mistral_conversation.repairs import async_create_fix_flow

    flow = await async_create_fix_flow(
        hass, "model_retired_gone", {"entry_id": "gone", "replacement": "x"}
    )
    flow.hass = hass
    result = await flow.async_step_confirm({})
    assert result["type"] == "create_entry"
