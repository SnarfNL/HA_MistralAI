"""Tests for ``_api.py``: shared request and error handling (MA-03, MA-07).

No real Home Assistant and no network: the HTTP session is a fake that serves
queued responses or raises queued exceptions, and ``asyncio.sleep`` is patched
so retries are instant and their waits can be checked.
"""
# ruff: noqa: I001 - `_ha_stubs` must run before the `mistral_conversation` import.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

import aiohttp

from homeassistant.exceptions import HomeAssistantError
from mistral_conversation import _api
from mistral_conversation._api import (
    MAX_RETRY_AFTER,
    async_spoken_error,
    mistral_error,
    mistral_request,
)
from mistral_conversation.const import DOMAIN

COMPONENT = Path(__file__).resolve().parent.parent / "custom_components" / "mistral_conversation"
URL = "https://api.mistral.ai/v1/x"


class _Response:
    def __init__(self, status: int, headers: dict | None = None, body: str = "secret body") -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def text(self) -> str:
        return self._body

    async def json(self):
        return {"ok": True}


class _Session:
    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _setup(*outcomes):
    session = _Session(*outcomes)
    runtime = SimpleNamespace(session=session, headers={"Authorization": "Bearer k"})
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", async_start_reauth=MagicMock())
    return hass, entry, session


class MistralRequestTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *outcomes, body=None):
        """Run one request; return (result, error, sleeps, entry, session)."""
        hass, entry, session = _setup(*outcomes)
        sleep = AsyncMock()
        result = error = None
        with patch.object(_api.asyncio, "sleep", sleep):
            try:
                async with mistral_request(hass, entry, "post", URL, json={}, timeout=5) as resp:
                    result = await resp.json()
                    if body is not None:
                        raise body
            except HomeAssistantError as err:
                error = err
        sleeps = [c.args[0] for c in sleep.await_args_list]
        return result, error, sleeps, entry, session

    async def test_success_yields_the_response(self) -> None:
        result, error, sleeps, _, session = await self._run(_Response(200))
        self.assertEqual(result, {"ok": True})
        self.assertIsNone(error)
        self.assertEqual(sleeps, [])
        self.assertEqual(session.calls[0]["headers"], {"Authorization": "Bearer k"})

    async def test_401_starts_reauth(self) -> None:
        _, error, _, entry, _ = await self._run(_Response(401))
        self.assertEqual(error.translation_key, "invalid_auth")
        entry.async_start_reauth.assert_called_once()

    async def test_429_is_retried_with_backoff_then_succeeds(self) -> None:
        result, error, sleeps, _, _ = await self._run(_Response(429), _Response(429), _Response(200))
        self.assertIsNone(error)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(sleeps, [1.0, 2.0])

    async def test_429_gives_up_after_two_retries(self) -> None:
        _, error, sleeps, _, session = await self._run(_Response(429), _Response(429), _Response(429))
        self.assertEqual(error.translation_key, "rate_limited")
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(len(session.calls), 3)

    async def test_short_retry_after_is_respected(self) -> None:
        _, error, sleeps, _, _ = await self._run(_Response(429, {"Retry-After": "3"}), _Response(200))
        self.assertIsNone(error)
        self.assertEqual(sleeps, [3.0])

    async def test_long_retry_after_fails_at_once(self) -> None:
        long_wait = str(int(MAX_RETRY_AFTER) + 25)
        _, error, sleeps, _, session = await self._run(_Response(429, {"Retry-After": long_wait}))
        self.assertEqual(error.translation_key, "rate_limited")
        self.assertEqual(sleeps, [])
        self.assertEqual(len(session.calls), 1)

    async def test_other_error_status_hides_the_body(self) -> None:
        with self.assertLogs(_api._LOGGER, "ERROR") as logs:
            _, error, _, _, _ = await self._run(_Response(500, body="secret body"))
        self.assertEqual(error.translation_key, "api_error")
        self.assertEqual(error.translation_placeholders, {"status": "500"})
        self.assertNotIn("secret body", str(error))
        self.assertIn("secret body", "".join(logs.output))

    async def test_network_error_and_timeout_become_cannot_connect(self) -> None:
        for failure in (aiohttp.ClientError("down"), TimeoutError()):
            with self.subTest(failure=repr(failure)):
                _, error, _, _, _ = await self._run(failure)
                self.assertEqual(error.translation_key, "cannot_connect")

    async def test_timeout_while_reading_becomes_cannot_connect(self) -> None:
        _, error, _, _, _ = await self._run(_Response(200), body=TimeoutError())
        self.assertEqual(error.translation_key, "cannot_connect")

    async def test_errors_raised_by_the_caller_pass_through(self) -> None:
        own = mistral_error("empty_response")
        _, error, _, _, _ = await self._run(_Response(200), body=own)
        self.assertIs(error, own)


def _fake_translations(table: dict[str, dict[str, str]]):
    async def fake(hass, language, category, integrations):
        return {
            f"component.{DOMAIN}.exceptions.{key}.message": text
            for key, text in table.get(language, {}).items()
        }

    return fake


TABLE = {
    "en": {"rate_limited": "Busy.", "api_error": "Error {status}.", "unexpected_error": "Oops."},
    "nl": {"rate_limited": "Druk."},
}


class SpokenErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _spoken(self, err, language):
        with patch.object(_api, "async_get_translations", _fake_translations(TABLE)):
            return await async_spoken_error(None, err, language)

    async def test_uses_the_pipeline_language(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "nl"), "Druk.")

    async def test_region_suffix_is_ignored(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "nl-NL"), "Druk.")

    async def test_falls_back_to_english(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "de"), "Busy.")
        self.assertEqual(await self._spoken(mistral_error("api_error", status="500"), "nl"), "Error 500.")

    async def test_error_without_translation_gets_generic_message(self) -> None:
        self.assertEqual(await self._spoken(HomeAssistantError("raw text"), "en"), "Oops.")


class StringsTests(unittest.TestCase):
    KEYS = (
        "invalid_auth", "rate_limited", "cannot_connect", "api_error",
        "empty_response", "json_parse_error", "structure_mismatch", "unexpected_error",
    )

    def test_every_error_key_exists_in_every_language(self) -> None:
        files = [COMPONENT / "strings.json", *sorted((COMPONENT / "translations").glob("*.json"))]
        self.assertGreaterEqual(len(files), 4)  # strings + en, nl, fr
        for path in files:
            exceptions = json.loads(path.read_text(encoding="utf-8"))["exceptions"]
            for key in self.KEYS:
                self.assertTrue(exceptions[key]["message"], f"{path.name}: {key}")
            self.assertIn("{status}", exceptions["api_error"]["message"], path.name)


if __name__ == "__main__":
    unittest.main()
