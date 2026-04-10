"""Config flow for Mistral AI Conversation."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import llm, selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CHAT_MODELS,
    CONF_CONTINUE_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TTS_VOICE,
    CONF_WEB_SEARCH,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TTS_VOICE,
    DEFAULT_WEB_SEARCH,
    DOMAIN,
    MISTRAL_API_BASE,
    TTS_VOICES,
)
from .stt import LANGUAGE_OPTIONS

_LOGGER = logging.getLogger(__name__)


class MistralConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the initial setup config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
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
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD
                        )
                    ),
                }
            ),
            errors=errors,
            description_placeholders={
                "api_key_url": "https://console.mistral.ai/api-keys"
            },
        )

    async def async_step_reauth(
        self, entry_data: dict[str, Any]
    ) -> FlowResult:
        """Handle reauth when API key becomes invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
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
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def _test_api_key(self, api_key: str) -> str | None:
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(
                f"{MISTRAL_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401:
                    return "invalid_auth"
                if resp.status != 200:
                    return "cannot_connect"
        except aiohttp.ClientConnectorError:
            return "cannot_connect"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error testing API key")
            return "unknown"
        return None

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "MistralOptionsFlow":
        return MistralOptionsFlow()


class MistralOptionsFlow(config_entries.OptionsFlow):
    """Options flow — HA injects self.config_entry as a read-only property."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            # Clean up empty LLM API selection
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(title="", data=user_input)

        opts = self.config_entry.options

        # Build LLM API options list
        hass_apis = [
            selector.SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    # ── Model ─────────────────────────────────────────────
                    vol.Optional(
                        CONF_MODEL,
                        default=opts.get(CONF_MODEL, DEFAULT_MODEL),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=CHAT_MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
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
                    # ── Continue conversation (experimental) ──────────────
                    vol.Optional(
                        CONF_CONTINUE_CONVERSATION,
                        default=opts.get(
                            CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION
                        ),
                    ): selector.BooleanSelector(),
                    # ── Web search (beta) ─────────────────────────────────
                    vol.Optional(
                        CONF_WEB_SEARCH,
                        default=opts.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH),
                    ): selector.BooleanSelector(),
                    # ── TTS voice (fallback default) ──────────────────────
                    # Primary voice selection is in Settings → Voice Assistants.
                    # This setting is used as fallback when no voice is chosen
                    # there, or when TTS is called directly from an automation.
                    vol.Optional(
                        CONF_TTS_VOICE,
                        default=opts.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=TTS_VOICES,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )
