"""Which Mistral models exist, and what to suggest when one is retired (MA-14)."""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir

from .api import describe_error
from .const import CHAT_MODELS, CONF_MODEL, DEFAULT_MODEL, DOMAIN

if TYPE_CHECKING:
    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)

# Mistral retires models without warning, and HA can run for weeks.
MODEL_CHECK_INTERVAL = timedelta(hours=24)
# A dated or -latest suffix: ministral-8b-2410, mistral-small-latest.
_VERSION_SUFFIX = re.compile(r"-(\d{4}|latest)$")


def issue_id(entry_id: str) -> str:
    """Repair issue id for the retired-model issue of one config entry."""
    return f"model_retired_{entry_id}"


def model_available(model: str, models: list[dict[str, Any]]) -> bool:
    """True when *model* is an id or alias in a GET /v1/models list."""
    for item in models:
        aliases = item.get("aliases")
        if model == item.get("id") or (isinstance(aliases, list) and model in aliases):
            return True
    return False


def suggest_replacement(model: str, models: list[dict[str, Any]]) -> str:
    """The model to suggest instead of the retired *model*.

    1. The ``-latest`` alias of the same model name, if it exists.
    2. Else the first recommended model (``CHAT_MODELS``) of the same family,
       the text before the first ``-`` (``ministral``, ``mistral``).
    3. Else ``DEFAULT_MODEL``.
    """
    latest = f"{_VERSION_SUFFIX.sub('', model)}-latest"
    if latest != model and model_available(latest, models):
        return latest
    family = model.split("-")[0]
    for candidate in CHAT_MODELS:
        if candidate.split("-")[0] == family and model_available(candidate, models):
            return candidate
    return DEFAULT_MODEL


async def async_check_model(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Raise or clear the ``model_retired`` repair issue. Never raises itself.

    A failed or empty model list says nothing about the configured model, so
    it changes nothing (free-tier keys are often rate-limited).
    """
    try:
        models = await entry.runtime_data.client.list_models()
    except HomeAssistantError as err:
        _LOGGER.debug("Mistral model list unavailable: %s", describe_error(err))
        return
    if not models:
        _LOGGER.debug("Mistral model list empty; skipping the model check")
        return
    entry.runtime_data.models = models

    model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    if model_available(model, models):
        ir.async_delete_issue(hass, DOMAIN, issue_id(entry.entry_id))
        return
    replacement = suggest_replacement(model, models)
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id(entry.entry_id),
        is_fixable=True,
        severity=ir.IssueSeverity.WARNING,
        translation_key="model_retired",
        translation_placeholders={"model": model, "replacement": replacement},
        data={"entry_id": entry.entry_id, "replacement": replacement},
    )
