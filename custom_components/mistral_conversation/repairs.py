"""Fix flow for the retired-model repair issue (MA-14)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    # HA 2026.9+ validates with probatio; ``voluptuous`` is only an alias.
    import probatio as vol
except ImportError:  # pragma: no cover - HA < 2026.9 ships voluptuous only
    import voluptuous as vol  # type: ignore[no-redef]
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

from .const import CONF_MODEL, DEFAULT_MODEL

if TYPE_CHECKING:
    # Only for type checking: RepairsFlowResult exists from HA 2026.9.
    from homeassistant.components.repairs import RepairsFlowResult


class ModelRetiredRepairFlow(RepairsFlow):
    """Confirm switching the options to the suggested model."""

    def __init__(self, entry_id: str, model: str, replacement: str) -> None:
        self._entry_id = entry_id
        self._model = model
        self._replacement = replacement

    def _is_stale(self) -> bool:
        """True when the entry is gone or no longer uses the retired model."""
        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        return entry is None or entry.options.get(CONF_MODEL, DEFAULT_MODEL) != self._model

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> RepairsFlowResult:
        # A model picked by hand since the issue was raised is never
        # overwritten: the issue is simply closed.
        if self._is_stale():
            return self.async_create_entry(data={})
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> RepairsFlowResult:
        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if user_input is not None:
            if entry is not None and not self._is_stale():
                # Writing the options reloads the entry, as any options change does.
                self.hass.config_entries.async_update_entry(
                    entry, options={**entry.options, CONF_MODEL: self._replacement}
                )
            return self.async_create_entry(data={})
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "model": self._model,
                "replacement": self._replacement,
            },
        )


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create the fix flow for *issue_id*."""
    assert data is not None
    return ModelRetiredRepairFlow(
        str(data["entry_id"]), str(data.get("model")), str(data["replacement"])
    )
