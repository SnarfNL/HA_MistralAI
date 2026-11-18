"""Shared request handling for calls to the Mistral API.

Every call to Mistral made while the integration is running goes through
``MistralClient.request`` (reached as ``entry.runtime_data.client``), so
errors are handled the same way everywhere:

* network errors and timeouts while sending become ``cannot_connect``;
* a 401 starts the reauth flow and becomes ``invalid_auth``;
* a 429 is retried at most twice (short backoff, or ``Retry-After`` when it is
  short enough), then becomes ``rate_limited``. Mistral's own error code and
  message are logged: since November 2026 an account without API access (Free
  plan, or no credits) gets a 429 with code 1300 on every call. Many 429s in a
  row raise the ``api_access_refused`` repair issue;
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
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util

from .const import DEFAULT_API_REGION, DEFAULT_MODEL, DOMAIN, REGION_BASES, api_base

_LOGGER = logging.getLogger(__name__)

# Waits before retry 1 and retry 2 after a 429.
RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)
# A Retry-After longer than this is not worth waiting for in a voice turn:
# fail straight away instead.
MAX_RETRY_AFTER = 5.0
# Errors that another call to Mistral in the same turn cannot fix.
UNRECOVERABLE_KEYS = frozenset({"invalid_auth", "rate_limited"})
# The repair issue for an account Mistral keeps refusing: this many 429s in a
# row (after retries, no success in between) spread over at least this long.
# A real overload passes on its own within seconds.
REFUSED_ISSUE_FAILURES = 5
REFUSED_ISSUE_MIN_SPAN = timedelta(minutes=2)
REFUSED_ISSUE_LEARN_MORE = (
    "https://github.com/SnarfNL/HA_MistralAI#mistral-account-and-billing"
)
# Parsed fields of a 429 body are cut to this length before they are logged.
_MAX_LOGGED_FIELD = 200
# Model for the key check's test request when the default is not on the account.
_PROBE_FALLBACK_MODEL = "ministral-3b-latest"
# Last resort when even the translations cannot be loaded.
_FALLBACK_MESSAGE = "Something went wrong talking to Mistral."

# Module-level alias so tests can patch retry waits without touching asyncio.
_sleep = asyncio.sleep


class MistralAPIError(HomeAssistantError):
    """A translated error from this integration.

    ``mistral_code`` is the short error code Mistral sent with a 429 (for
    example ``"1300"``), or None. It is kept for the diagnostics file.
    """

    mistral_code: str | None = None


def mistral_error(key: str, **placeholders: str) -> MistralAPIError:
    """Return a translated HomeAssistantError for *key* (see strings.json)."""
    return MistralAPIError(
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


def api_access_issue_id(entry_id: str) -> str:
    """Repair issue id for the "Mistral keeps refusing requests" issue."""
    return f"api_access_refused_{entry_id}"


def _clean(value: Any) -> str | None:
    """A short, single-line version of a parsed body field, or None."""
    if not isinstance(value, str | int) or isinstance(value, bool):
        return None
    text = " ".join(str(value).split())
    return text[:_MAX_LOGGED_FIELD] or None


async def _read_error_fields(resp: Any) -> tuple[str | None, str | None, str | None]:
    """``(code, message, type)`` from a Mistral error body; None when unavailable.

    Reads the body the same safe way as any other error status. Only these
    three parsed fields are returned, never the raw body.
    """
    try:
        body = await resp.text()
    except (aiohttp.ClientError, TimeoutError):
        return None, None, None
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None
    return (
        _clean(data.get("code")),
        _clean(data.get("message")),
        _clean(data.get("type")),
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
        code, message, error_type = await _read_error_fields(resp)
        headers = getattr(resp, "headers", None) or {}
        retry_after = _clean(headers.get("Retry-After"))
        _LOGGER.log(
            # A lowered level (best-effort calls) stays lowered.
            min(log_level, logging.WARNING),
            "Mistral API HTTP 429 for %s%s: code=%s type=%s message=%s"
            " retry-after=%s",
            url,
            f" ({log_context})" if log_context else "",
            code or "<unavailable>",
            error_type or "<unavailable>",
            message or "<unavailable>",
            retry_after or "<none>",
        )
        err = mistral_error("rate_limited")
        err.mistral_code = code
        raise err
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
        region: str | None = None,
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._session = session
        # "global", "eu" or "us"; a missing or unknown value means global.
        self.region = region if region in REGION_BASES else DEFAULT_API_REGION
        self._base = api_base(self.region)
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._json_headers = {**self._auth, "Content-Type": "application/json"}
        self.errors = errors
        # Consecutive rate_limited failures (see _note_rate_limited).
        self._refused_count = 0
        self._refused_since: datetime | None = None
        self._refused_issue_raised = False

    def _api_url(self, path: str, version: str) -> str:
        """Use the same (regional) host for every API version."""
        return f"{self._base.removesuffix('/v1')}/{version}{path}"

    def record(
        self, source: str, err: BaseException, status: int | None = None
    ) -> None:
        """Remember an error for diagnostics: never a body, URL or header.

        ``code`` is Mistral's short error code when it sent one (a 429).
        """
        self.record_note(
            source,
            getattr(err, "translation_key", None) or type(err).__name__,
            status,
            getattr(err, "mistral_code", None),
        )

    def record_note(
        self,
        source: str,
        note: str,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        """Remember a problem that is not an exception (a short fixed label)."""
        self.errors.append(
            {
                "time": dt_util.utcnow().isoformat(),
                "source": source,
                "status": status,
                "error": note,
                "code": code,
            }
        )

    def _note_success(self) -> None:
        """A request succeeded: forget earlier 429s and clear the repair issue."""
        self._refused_count = 0
        self._refused_since = None
        if self._refused_issue_raised:
            self._refused_issue_raised = False
            ir.async_delete_issue(
                self._hass, DOMAIN, api_access_issue_id(self._entry.entry_id)
            )

    def _note_rate_limited(self) -> None:
        """A request ended in rate_limited (after its retries).

        Raise the repair issue once there are enough in a row, spread over
        enough time: a real overload does not last minutes.
        """
        now = dt_util.utcnow()
        if self._refused_since is None:
            self._refused_since = now
        self._refused_count += 1
        if (
            not self._refused_issue_raised
            and self._refused_count >= REFUSED_ISSUE_FAILURES
            and now - self._refused_since >= REFUSED_ISSUE_MIN_SPAN
        ):
            self._refused_issue_raised = True
            ir.async_create_issue(
                self._hass,
                DOMAIN,
                api_access_issue_id(self._entry.entry_id),
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="api_access_refused",
                learn_more_url=REFUSED_ISSUE_LEARN_MORE,
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
        api_version: str = "v1",
        multipart: bool = False,
        data_factory: Callable[[], Any] | None = None,
        log_context: str = "",
        log_level: int = logging.ERROR,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Send a request to the region's base URL + *path* and yield the response.

        Use as ``async with client.request(...) as resp:``. *multipart* sends
        only the Authorization header (aiohttp sets the multipart
        Content-Type). *data_factory* builds the body per attempt, for bodies
        that can only be sent once (``aiohttp.FormData``). *log_context*
        (e.g. the model or voice) is added to the log line of an error
        status; *log_level* is the level of that line and of network failures.
        *api_version* changes only the version path, keeping the same host.
        """
        url = self._api_url(path, api_version)
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
            if getattr(err, "translation_key", None) == "rate_limited":
                self._note_rate_limited()
            raise
        self._note_success()
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

    async def list_voices(
        self, page_token: str | None, page_size: int
    ) -> dict[str, Any]:
        """One cursor-paginated page of preset and custom voices (v2 API)."""
        params: dict[str, str | int] = {"page_size": page_size, "type": "all"}
        if page_token is not None:
            params["page_token"] = page_token
        async with self.request(
            "get",
            "/audio/voices",
            api_version="v2",
            params=params,
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
        session: aiohttp.ClientSession,
        api_key: str,
        region: str | None = None,
        *,
        probe_chat: bool = True,
    ) -> tuple[str | None, str]:
        """Check an API key on *region*'s host before an entry (or its client) exists.

        Returns ``(None, "")`` when valid, else ``("invalid_auth" |
        "no_api_access" | "cannot_connect", detail)``; the detail never
        contains the key.

        ``GET /v1/models`` accepts a key whose account has no API access (the
        Free plan since November 2026, or no credits). So when it succeeds,
        one minimal chat completion (``max_tokens`` 1) is sent as well, to the
        same host: a 429 there means ``no_api_access``. It costs a fraction of
        a cent, once per setup, reauth or reconfigure, and is skipped when the
        account lists neither the default model nor ``ministral-3b-latest``.
        Pass ``probe_chat=False`` to skip it (the check at every Home
        Assistant start does, so a restart costs nothing).
        """
        # Imported here: _models imports this module.
        from ._models import model_available

        base = api_base(region)
        auth = {"Authorization": f"Bearer {api_key}"}
        try:
            async with session.get(
                f"{base}/models",
                headers=auth,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401:
                    return "invalid_auth", "HTTP 401"
                if resp.status != 200:
                    return "cannot_connect", f"HTTP {resp.status}"
                if not probe_chat:
                    return None, ""
                try:
                    listing = await resp.json(content_type=None)
                except (ValueError, aiohttp.ClientError, TimeoutError):
                    listing = None
            models = listing.get("data") if isinstance(listing, dict) else None
            if not isinstance(models, list):
                return None, ""
            models = [m for m in models if isinstance(m, dict)]
            model = next(
                (
                    m
                    for m in (DEFAULT_MODEL, _PROBE_FALLBACK_MODEL)
                    if model_available(m, models)
                ),
                None,
            )
            if model is None:
                return None, ""
            async with session.post(
                f"{base}/chat/completions",
                headers={**auth, "Content-Type": "application/json"},
                json={
                    "model": model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 401:
                    return "invalid_auth", "HTTP 401"
                if resp.status == 429:
                    return "no_api_access", "HTTP 429"
                if not 200 <= resp.status < 300:
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
