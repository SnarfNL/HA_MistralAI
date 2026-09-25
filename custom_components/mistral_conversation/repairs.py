"""Fix flow for the retired-model repair issue (MA-14)."""
from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import data_entry_flow
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

from .const import CONF_MODEL


class ModelRetiredRepairFlow(RepairsFlow):
    """Confirm switching the options to the suggested model."""

    def __init__(self, entry_id: str, replacement: str) -> None:
        self._entry_id = entry_id
        self._replacement = replacement

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if user_input is not None:
            if entry is not None:
                # Writing the options reloads the entry, as any options change does.
                self.hass.config_entries.async_update_entry(
                    entry, options={**entry.options, CONF_MODEL: self._replacement}
                )
            return self.async_create_entry(data={})
        model = entry.options.get(CONF_MODEL, "") if entry is not None else ""
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={"model": model, "replacement": self._replacement},
        )


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create the fix flow for *issue_id*."""
    assert data is not None
    return ModelRetiredRepairFlow(str(data["entry_id"]), str(data["replacement"]))
