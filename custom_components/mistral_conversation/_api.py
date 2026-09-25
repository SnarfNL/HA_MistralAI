"""Shared request handling for calls to the Mistral API.

Every platform sends its requests through ``mistral_request`` so errors are
handled the same way everywhere:

* network errors and timeouts become ``cannot_connect``;
* a 401 starts the reauth flow and becomes ``invalid_auth``;
* a 429 is retried at most twice (short backoff, or ``Retry-After`` when it is
  short enough), then becomes ``rate_limited``;
* any other error status becomes ``api_error``. The response body is logged,
  never shown to the user.

All errors are ``HomeAssistantError``s with a ``translation_key`` from the
``exceptions`` section of strings.json, so HA shows them in the user's language
and the conversation agent can speak them (see ``async_spoken_error``).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.translation import async_get_translations

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Waits before retry 1 and retry 2 after a 429.
RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)
# A Retry-After longer than this is not worth waiting for in a voice turn:
# fail straight away instead.
MAX_RETRY_AFTER = 5.0


def mistral_error(key: str, **placeholders: str) -> HomeAssistantError:
    """Return a translated HomeAssistantError for *key* (see strings.json)."""
    return HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key=key,
        translation_placeholders=placeholders or None,
    )


def _retry_delay(resp: Any, attempt: int) -> float | None:
    """Seconds to wait before the next retry, or None to give up now."""
    headers = getattr(resp, "headers", None) or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = None
        if seconds is not None:
            return seconds if seconds <= MAX_RETRY_AFTER else None
    return RETRY_DELAYS[attempt]


async def _raise_for_status(
    hass: HomeAssistant, entry: ConfigEntry, resp: Any, url: str
) -> None:
    if resp.status < 400:
        return
    if resp.status == 401:
        entry.async_start_reauth(hass)
        raise mistral_error("invalid_auth")
    if resp.status == 429:
        raise mistral_error("rate_limited")
    body = await resp.text()
    _LOGGER.error("Mistral API HTTP %s for %s: %s", resp.status, url, body)
    raise mistral_error("api_error", status=str(resp.status))


@asynccontextmanager
async def mistral_request(
    hass: HomeAssistant,
    entry: ConfigEntry,
    method: str,
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Send a request to Mistral and yield the successful response.

    Use as ``async with mistral_request(...) as resp:``. Network errors and
    timeouts while reading the response inside the block are translated too.
    """
    runtime = hass.data[DOMAIN][entry.entry_id]
    send = getattr(runtime.session, method)
    attempt = 0
    while True:
        try:
            async with send(
                url,
                headers=runtime.headers if headers is None else headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                **kwargs,
            ) as resp:
                delay = None
                if resp.status == 429 and attempt < len(RETRY_DELAYS):
                    delay = _retry_delay(resp, attempt)
                if delay is None:
                    await _raise_for_status(hass, entry, resp, url)
                    yield resp
                    return
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.warning("Mistral request to %s failed: %r", url, err)
            raise mistral_error("cannot_connect") from err
        attempt += 1
        _LOGGER.debug("Mistral rate limit; retry %d in %.1fs", attempt, delay)
        await asyncio.sleep(delay)


async def async_spoken_error(
    hass: HomeAssistant, err: HomeAssistantError, language: str | None
) -> str:
    """Return the message for *err* in *language*, for speaking to the user.

    Uses this integration's translations; languages without a translation
    get English. An error without a translation key gets a generic message.
    """
    key = getattr(err, "translation_key", None)
    if getattr(err, "translation_domain", None) != DOMAIN or not key:
        key = "unexpected_error"
    placeholders = getattr(err, "translation_placeholders", None) or {}

    lang = (language or "en").split("-")[0].lower()
    prefix = f"component.{DOMAIN}.exceptions.{key}.message"
    for candidate in (lang, "en"):
        translations = await async_get_translations(
            hass, candidate, "exceptions", {DOMAIN}
        )
        message = translations.get(prefix)
        if message:
            try:
                return message.format(**placeholders)
            except (KeyError, IndexError, ValueError):
                return message
    return "Something went wrong talking to Mistral."
