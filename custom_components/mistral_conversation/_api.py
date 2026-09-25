"""Shared request handling for calls to the Mistral API.

Every call to Mistral made while the integration is running goes through
``mistral_request`` so errors are handled the same way everywhere:

* network errors and timeouts while sending become ``cannot_connect``;
* a 401 starts the reauth flow and becomes ``invalid_auth``;
* a 429 is retried at most twice (short backoff, or ``Retry-After`` when it is
  short enough), then becomes ``rate_limited``;
* any other error status becomes ``api_error``. The response body is logged,
  never shown to the user.

Only the request itself is covered. Code inside the ``async with`` block (for
example HA running an Assist tool while a reply streams) raises its own errors
untouched, so a failing tool is never reported as a Mistral problem. Reading
the response is translated separately with ``read_json`` and
``translate_stream``.

All errors are ``HomeAssistantError``s with a ``translation_key`` from the
``exceptions`` section of strings.json, so HA shows them in the user's language
and the conversation agent can speak them (see ``async_spoken_error``).

Never log an aiohttp exception with ``%r``: the repr of ``ClientResponseError``
includes the request headers, and those hold the API key. Use
``describe_error``.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
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
# Errors that another call to Mistral in the same turn cannot fix.
UNRECOVERABLE_KEYS = frozenset({"invalid_auth", "rate_limited"})
# Last resort when even the translations cannot be loaded.
_FALLBACK_MESSAGE = "Something went wrong talking to Mistral."

# Module-level alias so tests can patch retry waits without touching asyncio.
_sleep = asyncio.sleep


def mistral_error(key: str, **placeholders: str) -> HomeAssistantError:
    """Return a translated HomeAssistantError for *key* (see strings.json)."""
    return HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key=key,
        translation_placeholders=placeholders or None,
    )


def is_unrecoverable(err: HomeAssistantError) -> bool:
    """True if retrying against Mistral in this turn is pointless."""
    return getattr(err, "translation_key", None) in UNRECOVERABLE_KEYS


def describe_error(err: BaseException) -> str:
    """Safe one-line description of an exception, for logs and messages.

    ``str()`` of aiohttp errors holds the status and URL; ``repr()`` of a
    ``ClientResponseError`` also holds the request headers, API key included.
    """
    return f"{type(err).__name__}: {err}"


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
    hass: HomeAssistant,
    entry: ConfigEntry,
    resp: Any,
    url: str,
    log_context: str,
    log_level: int,
) -> None:
    if resp.status < 400:
        return
    if resp.status == 401:
        entry.async_start_reauth(hass)
        raise mistral_error("invalid_auth")
    if resp.status == 429:
        raise mistral_error("rate_limited")
    try:
        body = await resp.text()
    except (aiohttp.ClientError, TimeoutError) as err:
        # The status is what matters; a body we cannot read must not turn a
        # real HTTP error into "cannot connect".
        body = f"<unreadable: {describe_error(err)}>"
    _LOGGER.log(
        log_level,
        "Mistral API HTTP %s for %s%s: %s",
        resp.status,
        url,
        f" ({log_context})" if log_context else "",
        body,
    )
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
    data_factory: Callable[[], Any] | None = None,
    log_context: str = "",
    log_level: int = logging.ERROR,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Send a request to Mistral and yield the successful response.

    Use as ``async with mistral_request(...) as resp:``.

    *data_factory* builds the request body per attempt, for bodies that can
    only be sent once (``aiohttp.FormData``). *log_context* (e.g. the model or
    voice) is added to the log line of an error status; *log_level* is the
    level of that line and of network failures.
    """
    runtime = hass.data[DOMAIN][entry.entry_id]
    attempt = 0
    while True:
        stack = AsyncExitStack()
        if data_factory is not None:
            kwargs["data"] = data_factory()
        try:
            resp = await stack.enter_async_context(
                runtime.session.request(
                    method.upper(),
                    url,
                    headers=runtime.headers if headers is None else headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    **kwargs,
                )
            )
        except (aiohttp.ClientError, TimeoutError) as err:
            await stack.aclose()
            _LOGGER.log(
                log_level, "Mistral request to %s failed: %s", url, describe_error(err)
            )
            raise mistral_error("cannot_connect") from err

        delay = None
        if resp.status == 429 and attempt < len(RETRY_DELAYS):
            delay = _retry_delay(resp, attempt)
        if delay is None:
            break
        await stack.aclose()
        attempt += 1
        _LOGGER.debug("Mistral rate limit; retry %d in %.1fs", attempt, delay)
        await _sleep(delay)

    async with stack:
        await _raise_for_status(hass, entry, resp, url, log_context, log_level)
        yield resp


async def read_json(resp: Any) -> dict[str, Any]:
    """Read a JSON object from a Mistral response, or raise a translated error."""
    try:
        data = await resp.json()
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.error("Reading the Mistral response failed: %s", describe_error(err))
        raise mistral_error("cannot_connect") from err
    except ValueError as err:  # JSONDecodeError, or a non-JSON content type
        _LOGGER.error("Mistral response is not valid JSON: %s", describe_error(err))
        raise mistral_error("invalid_response") from err
    if not isinstance(data, dict):
        _LOGGER.error("Mistral response is not a JSON object: %.200s", data)
        raise mistral_error("invalid_response")
    return data


async def translate_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """Pass *stream* through, translating network errors while it is read.

    Only errors raised while fetching the next item are translated; whatever
    the consumer does between items is not affected.
    """
    try:
        async for item in stream:
            yield item
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.error("Mistral stream failed: %s", describe_error(err))
        raise mistral_error("cannot_connect") from err


async def async_spoken_error(
    hass: HomeAssistant, err: BaseException, language: str | None
) -> str:
    """Return the message for *err* in *language*, for speaking to the user.

    HA's translations already fall back to English for a language or key
    that has no translation. An error without one of this integration's
    translation keys gets the generic ``unexpected_error`` message.
    """
    key = getattr(err, "translation_key", None)
    if getattr(err, "translation_domain", None) != DOMAIN or not key:
        key = "unexpected_error"
    placeholders = getattr(err, "translation_placeholders", None) or {}

    lang = (language or "en").split("-")[0].lower()
    prefix = f"component.{DOMAIN}.exceptions.{{}}.message"
    translations = await async_get_translations(hass, lang, "exceptions", {DOMAIN})
    message = translations.get(prefix.format(key)) or translations.get(
        prefix.format("unexpected_error")
    )
    if not message:
        return _FALLBACK_MESSAGE
    try:
        return message.format(**placeholders)
    except (KeyError, IndexError, ValueError):
        return message
