"""Shared request handling for calls to the Mistral API.

Every call to Mistral made while the integration is running goes through
``MistralClient.request`` (reached as ``entry.runtime_data.client``), so
errors are handled the same way everywhere:

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
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util

from .const import DOMAIN, MISTRAL_API_BASE

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


class MistralClient:
    """All calls to the Mistral API for one config entry.

    ``request`` handles retries, reauth and translated errors (see the module
    docstring). Errors raised while *sending* a request are also recorded in
    ``errors`` for the diagnostics file; errors from the caller's own code
    inside ``async with`` are not.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        session: aiohttp.ClientSession,
        api_key: str,
        errors: deque[dict[str, Any]],
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._session = session
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._json_headers = {**self._auth, "Content-Type": "application/json"}
        self.errors = errors

    def record(
        self, source: str, err: BaseException, status: int | None = None
    ) -> None:
        """Remember an error for diagnostics: never a body, URL or header."""
        self.errors.append(
            {
                "time": dt_util.utcnow().isoformat(),
                "source": source,
                "status": status,
                "error": getattr(err, "translation_key", None) or type(err).__name__,
            }
        )

    async def _open(
        self,
        method: str,
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        data_factory: Callable[[], Any] | None,
        log_level: int,
        kwargs: dict[str, Any],
    ) -> tuple[AsyncExitStack, Any]:
        """Send with 429 retries; return the open response and its exit stack."""
        attempt = 0
        while True:
            stack = AsyncExitStack()
            if data_factory is not None:
                kwargs["data"] = data_factory()
            try:
                resp = await stack.enter_async_context(
                    self._session.request(
                        method.upper(),
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                        **kwargs,
                    )
                )
            except (aiohttp.ClientError, TimeoutError) as err:
                await stack.aclose()
                _LOGGER.log(
                    log_level,
                    "Mistral request to %s failed: %s",
                    url,
                    describe_error(err),
                )
                raise mistral_error("cannot_connect") from err

            delay = None
            if resp.status == 429 and attempt < len(RETRY_DELAYS):
                delay = _retry_delay(resp, attempt)
            if delay is None:
                return stack, resp
            await stack.aclose()
            attempt += 1
            _LOGGER.debug("Mistral rate limit; retry %d in %.1fs", attempt, delay)
            await _sleep(delay)

    @asynccontextmanager
    async def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        source: str,
        multipart: bool = False,
        data_factory: Callable[[], Any] | None = None,
        log_context: str = "",
        log_level: int = logging.ERROR,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Send a request to ``MISTRAL_API_BASE + path`` and yield the response.

        Use as ``async with client.request(...) as resp:``. *multipart* sends
        only the Authorization header (aiohttp sets the multipart
        Content-Type). *data_factory* builds the body per attempt, for bodies
        that can only be sent once (``aiohttp.FormData``). *log_context*
        (e.g. the model or voice) is added to the log line of an error
        status; *log_level* is the level of that line and of network failures.
        """
        url = f"{MISTRAL_API_BASE}{path}"
        headers = self._auth if multipart else self._json_headers
        status: int | None = None
        try:
            stack, resp = await self._open(
                method,
                url,
                timeout=timeout,
                headers=headers,
                data_factory=data_factory,
                log_level=log_level,
                kwargs=kwargs,
            )
            status = resp.status
            try:
                await _raise_for_status(
                    self._hass, self._entry, resp, url, log_context, log_level
                )
            except BaseException:
                await stack.aclose()
                raise
        except HomeAssistantError as err:
            self.record(source, err, status)
            raise
        async with stack:
            yield resp

    def chat_completions(
        self, payload: dict[str, Any], *, source: str
    ) -> AbstractAsyncContextManager[Any]:
        """Streamed chat completion (conversation and AI Task)."""
        return self.request(
            "post",
            "/chat/completions",
            json=payload,
            timeout=90,
            source=source,
            log_context=f"model={payload.get('model')}",
        )

    def start_conversation(
        self, payload: dict[str, Any], *, model: str
    ) -> AbstractAsyncContextManager[Any]:
        """New Conversations API conversation (web search)."""
        return self.request(
            "post",
            "/conversations",
            json=payload,
            timeout=90,
            source="conversation",
            log_context=f"model={model}",
        )

    def append_conversation(
        self, conv_id: str, payload: dict[str, Any], *, model: str
    ) -> AbstractAsyncContextManager[Any]:
        """Follow-up in an existing Conversations API conversation."""
        return self.request(
            "post",
            f"/conversations/{conv_id}",
            json=payload,
            timeout=90,
            source="conversation",
            log_context=f"model={model}",
        )

    def delete_conversation(self, conv_id: str) -> AbstractAsyncContextManager[Any]:
        """Best-effort cleanup of a Conversations API conversation."""
        return self.request(
            "delete",
            f"/conversations/{conv_id}",
            timeout=15,
            source="conversation",
            log_level=logging.DEBUG,
        )

    def speech(
        self, payload: dict[str, Any], *, voice: str, timeout: float
    ) -> AbstractAsyncContextManager[Any]:
        """Text-to-speech, batch (mp3) or streamed (wav)."""
        return self.request(
            "post",
            "/audio/speech",
            json=payload,
            timeout=timeout,
            source="tts",
            log_context=f"voice={voice}",
        )

    def transcribe(
        self, data_factory: Callable[[], aiohttp.FormData]
    ) -> AbstractAsyncContextManager[Any]:
        """Speech-to-text upload (multipart)."""
        return self.request(
            "post",
            "/audio/transcriptions",
            timeout=60,
            source="stt",
            multipart=True,
            data_factory=data_factory,
        )

    async def list_voices(self, offset: int, limit: int) -> dict[str, Any]:
        """One page of the account's voices."""
        async with self.request(
            "get",
            "/audio/voices",
            params={"limit": limit, "offset": offset},
            timeout=10,
            source="tts",
            # A failed fetch is not fatal: the last good list stays.
            log_level=logging.WARNING,
        ) as resp:
            return await read_json(resp)

    async def list_models(self) -> list[dict[str, Any]]:
        """The models on this account (GET /v1/models); [] if the shape is odd."""
        async with self.request(
            "get", "/models", timeout=10, source="models", log_level=logging.DEBUG
        ) as resp:
            data = await read_json(resp)
        models = data.get("data")
        if not isinstance(models, list):
            return []
        return [m for m in models if isinstance(m, dict)]

    @staticmethod
    async def validate_key(
        session: aiohttp.ClientSession, api_key: str
    ) -> tuple[str | None, str]:
        """Check an API key before an entry (or its client) exists.

        Returns ``(None, "")`` when valid, else ``("invalid_auth" |
        "cannot_connect", detail)``; the detail never contains the key.
        """
        try:
            async with session.get(
                f"{MISTRAL_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401:
                    return "invalid_auth", "HTTP 401"
                if resp.status != 200:
                    return "cannot_connect", f"HTTP {resp.status}"
        except (aiohttp.ClientError, TimeoutError) as err:
            # Never repr() an aiohttp error: it would include the API key header.
            return "cannot_connect", describe_error(err)
        return None, ""


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
