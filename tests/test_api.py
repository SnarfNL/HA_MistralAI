"""Tests for ``_api.py``: shared request and error handling (MA-03, MA-07).

No real Home Assistant and no network: the HTTP session is a fake that serves
queued responses or raises queued exceptions, and ``_api._sleep`` is patched
so retries are instant and their waits can be checked.
"""
from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from custom_components.mistral_conversation import _api
from custom_components.mistral_conversation._api import (
    MAX_RETRY_AFTER,
    async_spoken_error,
    describe_error,
    is_unrecoverable,
    mistral_error,
    mistral_request,
    read_json,
    translate_stream,
)
from custom_components.mistral_conversation.const import DOMAIN

COMPONENT = Path(__file__).resolve().parent.parent / "custom_components" / "mistral_conversation"
URL = "https://api.mistral.ai/v1/x"
API_KEY = "sk-secret-key"


class _Response:
    def __init__(
        self,
        status: int,
        headers: dict | None = None,
        body: str | BaseException = "secret body",
        payload=None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._payload = {"ok": True} if payload is None else payload
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    async def text(self) -> str:
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body

    async def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _setup(*outcomes):
    session = _Session(*outcomes)
    runtime = SimpleNamespace(session=session, headers={"Authorization": f"Bearer {API_KEY}"})
    hass = SimpleNamespace(data={DOMAIN: {"entry1": runtime}})
    entry = SimpleNamespace(entry_id="entry1", async_start_reauth=MagicMock())
    return hass, entry, session


class MistralRequestTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *outcomes, body=None, method="post", **kwargs):
        """Run one request; return (result, error, sleeps, entry, session)."""
        hass, entry, session = _setup(*outcomes)
        sleep = AsyncMock()
        result = error = None
        with patch.object(_api, "_sleep", sleep):
            try:
                async with mistral_request(hass, entry, method, URL, timeout=5, **kwargs) as resp:
                    result = await resp.json()
                    if body is not None:
                        raise body
            except Exception as err:  # noqa: BLE001 - the test inspects whatever came out
                error = err
        sleeps = [c.args[0] for c in sleep.await_args_list]
        return result, error, sleeps, entry, session

    async def test_success_yields_the_response(self) -> None:
        result, error, sleeps, _, session = await self._run(_Response(200))
        self.assertEqual(result, {"ok": True})
        self.assertIsNone(error)
        self.assertEqual(sleeps, [])
        method, kwargs = session.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(kwargs["headers"], {"Authorization": f"Bearer {API_KEY}"})

    async def test_method_is_case_insensitive(self) -> None:
        _, error, _, _, session = await self._run(_Response(200), method="Delete")
        self.assertIsNone(error)
        self.assertEqual(session.calls[0][0], "DELETE")

    async def test_response_is_closed_after_the_block(self) -> None:
        resp = _Response(200)
        await self._run(resp)
        self.assertTrue(resp.closed)

    async def test_401_starts_reauth(self) -> None:
        _, error, _, entry, _ = await self._run(_Response(401))
        self.assertEqual(error.translation_key, "invalid_auth")
        entry.async_start_reauth.assert_called_once()

    async def test_429_is_retried_with_backoff_then_succeeds(self) -> None:
        first, second = _Response(429), _Response(429)
        result, error, sleeps, _, _ = await self._run(first, second, _Response(200))
        self.assertIsNone(error)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertTrue(first.closed and second.closed)

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

    async def test_data_factory_builds_a_new_body_per_attempt(self) -> None:
        bodies = iter(["form-1", "form-2"])
        _, error, _, _, session = await self._run(
            _Response(429), _Response(200), data_factory=lambda: next(bodies)
        )
        self.assertIsNone(error)
        self.assertEqual([kw["data"] for _, kw in session.calls], ["form-1", "form-2"])

    async def test_other_error_status_hides_the_body_and_logs_context(self) -> None:
        with self.assertLogs(_api._LOGGER, "ERROR") as logs:
            _, error, _, _, _ = await self._run(
                _Response(500, body="secret body"), log_context="model=m1"
            )
        self.assertEqual(error.translation_key, "api_error")
        self.assertEqual(error.translation_placeholders, {"status": "500"})
        self.assertNotIn("secret body", str(error))
        output = "".join(logs.output)
        self.assertIn("secret body", output)
        self.assertIn("model=m1", output)

    async def test_log_level_can_be_lowered(self) -> None:
        with self.assertLogs(_api._LOGGER, "DEBUG") as logs:
            await self._run(_Response(500), log_level=logging.WARNING)
        self.assertTrue(any(line.startswith("WARNING") for line in logs.output))
        self.assertFalse(any(line.startswith("ERROR") for line in logs.output))

    async def test_unreadable_error_body_still_reports_the_status(self) -> None:
        with self.assertLogs(_api._LOGGER, "ERROR"):
            _, error, _, _, _ = await self._run(_Response(503, body=TimeoutError()))
        self.assertEqual(error.translation_key, "api_error")
        self.assertEqual(error.translation_placeholders, {"status": "503"})

    async def test_network_error_and_timeout_become_cannot_connect(self) -> None:
        for failure in (aiohttp.ClientError("down"), TimeoutError()):
            with self.subTest(failure=repr(failure)):
                _, error, _, _, _ = await self._run(failure)
                self.assertEqual(error.translation_key, "cannot_connect")

    async def test_errors_raised_inside_the_block_pass_through(self) -> None:
        """A failing HA tool must not be reported as a Mistral problem."""
        for own in (TimeoutError(), aiohttp.ClientError("tool"), mistral_error("empty_response")):
            with self.subTest(error=repr(own)):
                _, error, _, _, _ = await self._run(_Response(200), body=own)
                self.assertIs(error, own)


class _LeakyError(aiohttp.ClientError):
    """Mimics ClientResponseError: repr() includes the request headers."""

    def __str__(self) -> str:
        return "502, message='Bad Gateway', url='https://api.mistral.ai/v1/x'"

    def __repr__(self) -> str:
        return f"ClientResponseError(headers={{'Authorization': 'Bearer {API_KEY}'}})"


class NoKeyInLogsTests(unittest.IsolatedAsyncioTestCase):
    def test_describe_error_uses_str_not_repr(self) -> None:
        text = describe_error(_LeakyError())
        self.assertNotIn(API_KEY, text)
        self.assertIn("502", text)

    async def test_failed_request_does_not_log_the_key(self) -> None:
        hass, entry, _ = _setup(_LeakyError())
        with self.assertLogs(_api._LOGGER, "DEBUG") as logs, self.assertRaises(HomeAssistantError):
            async with mistral_request(hass, entry, "post", URL, timeout=5):
                pass
        self.assertNotIn(API_KEY, "".join(logs.output))

    async def test_failed_read_does_not_log_the_key(self) -> None:
        with self.assertLogs(_api._LOGGER, "DEBUG") as logs, self.assertRaises(HomeAssistantError):
            await read_json(_Response(200, payload=_LeakyError()))
        self.assertNotIn(API_KEY, "".join(logs.output))


class ReadJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_object_is_returned(self) -> None:
        self.assertEqual(await read_json(_Response(200, payload={"a": 1})), {"a": 1})

    async def test_invalid_json_is_invalid_response(self) -> None:
        for bad in (json.JSONDecodeError("bad", "x", 0), ValueError("not json"), ["a", "list"]):
            with self.subTest(bad=repr(bad)), self.assertRaises(HomeAssistantError) as ctx:
                await read_json(_Response(200, payload=bad))
            self.assertEqual(ctx.exception.translation_key, "invalid_response")

    async def test_network_error_while_reading_is_cannot_connect(self) -> None:
        for failure in (TimeoutError(), aiohttp.ClientError("cut")):
            with self.subTest(failure=repr(failure)), self.assertRaises(HomeAssistantError) as ctx:
                await read_json(_Response(200, payload=failure))
            self.assertEqual(ctx.exception.translation_key, "cannot_connect")


class TranslateStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_items_pass_through(self) -> None:
        async def source():
            yield 1
            yield 2

        self.assertEqual([i async for i in translate_stream(source())], [1, 2])

    async def test_error_while_reading_becomes_cannot_connect(self) -> None:
        async def source():
            yield 1
            raise TimeoutError

        with self.assertRaises(HomeAssistantError) as ctx:
            async for _ in translate_stream(source()):
                pass
        self.assertEqual(ctx.exception.translation_key, "cannot_connect")

    async def test_error_raised_by_the_consumer_is_untouched(self) -> None:
        async def source():
            yield 1
            yield 2

        with self.assertRaises(TimeoutError):
            async for _ in translate_stream(source()):
                raise TimeoutError


class UnrecoverableTests(unittest.TestCase):
    def test_auth_and_rate_limit_are_unrecoverable(self) -> None:
        self.assertTrue(is_unrecoverable(mistral_error("invalid_auth")))
        self.assertTrue(is_unrecoverable(mistral_error("rate_limited")))
        self.assertFalse(is_unrecoverable(mistral_error("cannot_connect")))
        self.assertFalse(is_unrecoverable(HomeAssistantError("other")))


def _fake_translations(table: dict[str, dict[str, str]], calls: list[str]):
    """Mimics HA: every language is merged over English."""

    async def fake(hass, language, category, integrations):
        calls.append(language)
        merged = {**table["en"], **table.get(language, {})}
        return {
            f"component.{DOMAIN}.exceptions.{key}.message": text
            for key, text in merged.items()
        }

    return fake


TABLE = {
    "en": {"rate_limited": "Busy.", "api_error": "Error {status}.", "unexpected_error": "Oops."},
    "nl": {"rate_limited": "Druk.", "unexpected_error": "Oeps."},
}


class SpokenErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _spoken(self, err, language, table=TABLE):
        calls: list[str] = []
        with patch.object(_api, "async_get_translations", _fake_translations(table, calls)):
            message = await async_spoken_error(None, err, language)
        self.assertEqual(len(calls), 1, "one translation lookup per error")
        return message

    async def test_uses_the_pipeline_language(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "nl"), "Druk.")

    async def test_region_suffix_is_ignored(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "nl-NL"), "Druk.")

    async def test_missing_key_or_language_uses_hass_english_fallback(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("rate_limited"), "de"), "Busy.")
        self.assertEqual(await self._spoken(mistral_error("api_error", status="500"), "nl"), "Error 500.")

    async def test_error_without_translation_gets_generic_message(self) -> None:
        self.assertEqual(await self._spoken(HomeAssistantError("raw text"), "nl"), "Oeps.")
        self.assertEqual(await self._spoken(ValueError("boom"), "en"), "Oops.")

    async def test_unknown_key_gets_generic_message(self) -> None:
        self.assertEqual(await self._spoken(mistral_error("no_such_key"), "en"), "Oops.")

    async def test_no_translations_at_all_gives_a_fixed_sentence(self) -> None:
        message = await self._spoken(mistral_error("rate_limited"), "en", table={"en": {}})
        self.assertTrue(message)


class StringsTests(unittest.TestCase):
    KEYS = (
        "invalid_auth", "rate_limited", "cannot_connect", "api_error", "invalid_response",
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
