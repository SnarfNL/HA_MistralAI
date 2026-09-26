"""Config flow for Mistral AI Conversation."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.helpers import llm, selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import MistralClient
from .const import (
    CHAT_MODELS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TTS_MODE,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_MODE,
    CONF_WEB_SEARCH_TRIGGER,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TTS_MODE,
    DEFAULT_WEB_SEARCH,
    DEFAULT_WEB_SEARCH_MODE,
    DEFAULT_WEB_SEARCH_TRIGGER,
    DOMAIN,
    TTS_MODES,
    WEB_SEARCH_MODES,
    supports_web_search,
)

_LOGGER = logging.getLogger(__name__)

API_KEY_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        ),
    }
)


class MistralConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the initial setup config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._test_api_key(user_input[CONF_API_KEY])
            if error:
                errors["base"] = error
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Mistral AI Conversation",
                    data={CONF_API_KEY: user_input[CONF_API_KEY]},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=API_KEY_SCHEMA,
            errors=errors,
            description_placeholders={
                "api_key_url": "https://console.mistral.ai/api-keys"
            },
        )

    async def async_step_reauth(
        self, entry_data: dict[str, Any]
    ) -> config_entries.ConfigFlowResult:
        """Handle reauth when API key becomes invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Dialog to re-enter the API key."""
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._test_api_key(user_input[CONF_API_KEY])
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    self._get_reauth_entry(),
                    data_updates={CONF_API_KEY: user_input[CONF_API_KEY]},
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=API_KEY_SCHEMA,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Change the API key without removing the integration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await self._test_api_key(user_input[CONF_API_KEY])
            if error:
                errors["base"] = error
            else:
                entry = self._get_reconfigure_entry()
                data_updates = {CONF_API_KEY: user_input[CONF_API_KEY]}
                if entry.state is not config_entries.ConfigEntryState.LOADED:
                    return self.async_update_reload_and_abort(
                        entry, data_updates=data_updates
                    )
                # A loaded entry reloads through its update listener; letting
                # async_update_reload_and_abort reload as well would repeat the
                # key check, which a free-tier key can fail with a rate limit.
                self.hass.config_entries.async_update_entry(
                    entry, data={**entry.data, **data_updates}
                )
                return self.async_abort(reason="reconfigure_successful")

        return self.async_show_form(
            step_id="reconfigure", data_schema=API_KEY_SCHEMA, errors=errors
        )

    async def _test_api_key(self, api_key: str) -> str | None:
        try:
            error, _detail = await MistralClient.validate_key(
                async_get_clientsession(self.hass), api_key
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error testing API key")
            return "unknown"
        return error

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> MistralOptionsFlow:
        return MistralOptionsFlow()


class MistralOptionsFlow(config_entries.OptionsFlow):
    """Options in two steps: the model first, then the settings for it (MA-30).

    Home Assistant cannot show or hide a field while the form is open, so the
    web search fields are only offered in step 2, and only for a model that
    supports web search. HA injects self.config_entry as a read-only property.
    """

    def __init__(self) -> None:
        self._model: str = DEFAULT_MODEL

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 1: the model."""
        if user_input is not None:
            self._model = user_input[CONF_MODEL]
            return await self.async_step_settings()

        opts = self.config_entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL,
                        default=opts.get(CONF_MODEL, DEFAULT_MODEL),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=CHAT_MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            translation_key="model",
                        )
                    ),
                }
            ),
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 2: everything else; web search only for a supporting model."""
        opts = self.config_entry.options
        capable = supports_web_search(self._model)

        if user_input is not None:
            # Clean up empty LLM API selection
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            data = {**user_input, CONF_MODEL: self._model}
            if not capable:
                # Saved as off. Mode and trigger phrases keep their values, so
                # they come back when a supporting model is chosen again.
                data[CONF_WEB_SEARCH] = False
                data[CONF_WEB_SEARCH_MODE] = opts.get(
                    CONF_WEB_SEARCH_MODE, DEFAULT_WEB_SEARCH_MODE
                )
                data[CONF_WEB_SEARCH_TRIGGER] = opts.get(
                    CONF_WEB_SEARCH_TRIGGER, DEFAULT_WEB_SEARCH_TRIGGER
                )
            return self.async_create_entry(title="", data=data)

        # Build LLM API options list
        hass_apis = [
            selector.SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        schema: dict[Any, Any] = {
            # ── System prompt ─────────────────────────────────────
            vol.Optional(
                CONF_PROMPT,
                default=opts.get(CONF_PROMPT, DEFAULT_PROMPT),
            ): selector.TemplateSelector(),
            # ── LLM API (Home Assistant device control) ───────────
            vol.Optional(
                CONF_LLM_HASS_API,
                description={
                    "suggested_value": opts.get(CONF_LLM_HASS_API),
                },
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=hass_apis,
                    multiple=True,
                )
            ),
            # ── Temperature ───────────────────────────────────────
            vol.Optional(
                CONF_TEMPERATURE,
                default=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
            # ── Max tokens ────────────────────────────────────────
            vol.Optional(
                CONF_MAX_TOKENS,
                default=opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=64,
                    max=8192,
                    step=64,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }

        if capable:
            # Switching from a model without web search to one with it turns
            # web search on; a manual "off" on a supporting model is kept.
            previous = opts.get(CONF_MODEL, DEFAULT_MODEL)
            web_default = (
                opts.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)
                if supports_web_search(previous)
                else True
            )
            schema.update(
                {
                    # ── Web search (beta) ─────────────────────────────────
                    vol.Optional(
                        CONF_WEB_SEARCH, default=web_default
                    ): selector.BooleanSelector(),
                    # ── Web search routing ────────────────────────────────
                    # 'model': the model calls a web_search tool when it needs
                    # one, keeping HA tools available on the same turn.
                    # 'always': legacy — every turn goes to the Conversations API,
                    # which is slower and carries no HA tools.
                    vol.Optional(
                        CONF_WEB_SEARCH_MODE,
                        default=opts.get(
                            CONF_WEB_SEARCH_MODE, DEFAULT_WEB_SEARCH_MODE
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=WEB_SEARCH_MODES,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            translation_key="web_search_mode",
                        )
                    ),
                    # ── Web search trigger phrases (optional) ─────────────
                    # Comma-separated. When set, these take precedence over the
                    # routing mode: only utterances starting with one of them
                    # search (phrase stripped), everything else never does.
                    # Empty = let the mode above decide.
                    vol.Optional(
                        CONF_WEB_SEARCH_TRIGGER,
                        default=opts.get(
                            CONF_WEB_SEARCH_TRIGGER, DEFAULT_WEB_SEARCH_TRIGGER
                        ),
                    ): selector.TextSelector(),
                }
            )

        # ── TTS mode (stream vs batch) ────────────────────────
        # 'stream' uses Mistral's SSE WAV endpoint with sentence
        # pipelining for low time-to-first-audio. 'batch' issues a
        # single mp3 request. Direct tts.speak service calls
        # always use batch regardless of this setting.
        schema[
            vol.Optional(
                CONF_TTS_MODE,
                default=opts.get(CONF_TTS_MODE, DEFAULT_TTS_MODE),
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=TTS_MODES,
                mode=selector.SelectSelectorMode.DROPDOWN,
                translation_key="tts_mode",
            )
        )

        return self.async_show_form(step_id="settings", data_schema=vol.Schema(schema))
