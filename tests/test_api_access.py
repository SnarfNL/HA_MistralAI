"""Repair issue for an account Mistral keeps refusing (HTTP 429, code 1300)."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.mistral_conversation.api import (
    REFUSED_ISSUE_FAILURES,
    api_access_issue_id,
)
from custom_components.mistral_conversation.const import DOMAIN

from .conftest import BASE, load_fixture

REFUSED_BODY = {
    "object": "error",
    "message": "Rate limit exceeded",
    "type": "rate_limited",
    "param": None,
    "code": "1300",
    "raw_status_code": 429,
}
# Longer than MAX_RETRY_AFTER, so each request fails at once without waiting.
LONG_RETRY_AFTER = {"Retry-After": "30"}


def _issue(hass: HomeAssistant, entry: MockConfigEntry) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(
        DOMAIN, api_access_issue_id(entry.entry_id)
    )


def _refuse(aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.clear_requests()
    aioclient_mock.get(
        f"{BASE}/models", status=429, headers=LONG_RETRY_AFTER, json=REFUSED_BODY
    )


def _accept(aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", text=load_fixture("models.json"))


async def _request(entry: MockConfigEntry, *, ok: bool = False) -> None:
    client = entry.runtime_data.client
    if ok:
        async with client.request("get", "/models", timeout=5, source="test"):
            return
    with pytest.raises(HomeAssistantError) as err:
        async with client.request("get", "/models", timeout=5, source="test"):
            pass
    assert err.value.translation_key == "rate_limited"


async def _fail(entry: MockConfigEntry, times: int) -> None:
    for _ in range(times):
        await _request(entry)


async def test_issue_is_raised_after_five_failures_over_two_minutes(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
    assert _issue(hass, entry) is None
    freezer.tick(timedelta(minutes=2))
    await _fail(entry, 1)

    issue = _issue(hass, entry)
    assert issue is not None
    assert issue.is_fixable is False
    assert issue.severity is ir.IssueSeverity.WARNING
    assert issue.translation_key == "api_access_refused"
    assert (
        issue.learn_more_url
        == "https://github.com/SnarfNL/HA_MistralAI#mistral-account-and-billing"
    )


async def test_five_failures_within_two_minutes_raise_nothing(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    await _fail(entry, 2)
    freezer.tick(timedelta(minutes=1, seconds=59))
    await _fail(entry, REFUSED_ISSUE_FAILURES)  # plenty of failures, span just short
    assert _issue(hass, entry) is None
    # Only the span was missing: one more failure once two minutes have passed.
    freezer.tick(timedelta(seconds=1))
    await _fail(entry, 1)
    assert _issue(hass, entry) is not None


async def test_too_few_failures_raise_nothing_however_long_they_take(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    for _ in range(REFUSED_ISSUE_FAILURES - 1):
        await _fail(entry, 1)
        freezer.tick(timedelta(minutes=10))
    assert _issue(hass, entry) is None


async def test_a_success_in_between_resets_the_count(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
    freezer.tick(timedelta(minutes=3))
    _accept(aioclient_mock)
    await _request(entry, ok=True)
    _refuse(aioclient_mock)
    await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
    assert _issue(hass, entry) is None


async def test_a_success_deletes_the_issue_and_starts_over(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
    freezer.tick(timedelta(minutes=2))
    await _fail(entry, 1)
    assert _issue(hass, entry) is not None

    _accept(aioclient_mock)
    await _request(entry, ok=True)
    assert _issue(hass, entry) is None

    # The count started over: one more failure is nowhere near enough.
    _refuse(aioclient_mock)
    freezer.tick(timedelta(minutes=2))
    await _fail(entry, 1)
    assert _issue(hass, entry) is None


async def test_the_issue_is_raised_once_and_deleted_only_when_it_was_raised(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    target = "custom_components.mistral_conversation.api.ir"
    _accept(aioclient_mock)
    with patch(f"{target}.async_delete_issue") as delete:
        await _request(entry, ok=True)
        await _request(entry, ok=True)
    delete.assert_not_called()  # no work per request while there is no issue

    _refuse(aioclient_mock)
    with patch(f"{target}.async_create_issue") as create:
        await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
        freezer.tick(timedelta(minutes=2))
        await _fail(entry, 3)
    create.assert_called_once()

    _accept(aioclient_mock)
    with patch(f"{target}.async_delete_issue") as delete:
        await _request(entry, ok=True)
        await _request(entry, ok=True)
    delete.assert_called_once_with(
        hass, DOMAIN, api_access_issue_id(entry.entry_id)
    )


async def test_other_errors_do_not_count(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    client = entry.runtime_data.client
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE}/models", status=500)
    for _ in range(REFUSED_ISSUE_FAILURES + 1):
        with pytest.raises(HomeAssistantError):
            async with client.request("get", "/models", timeout=5, source="test"):
                pass
        freezer.tick(timedelta(minutes=1))
    assert _issue(hass, entry) is None


async def test_unloading_the_entry_deletes_the_issue(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
    freezer,
) -> None:
    entry = setup_integration
    _refuse(aioclient_mock)
    await _fail(entry, REFUSED_ISSUE_FAILURES - 1)
    freezer.tick(timedelta(minutes=2))
    await _fail(entry, 1)
    assert _issue(hass, entry) is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert _issue(hass, entry) is None


async def test_removing_the_entry_deletes_the_issue(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Also for an entry that never loaded, where unload does not run."""
    issue_id = api_access_issue_id(mock_config_entry.entry_id)
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="api_access_refused",
    )
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None

    await hass.config_entries.async_remove(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None
