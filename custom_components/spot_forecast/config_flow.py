"""Config flow (choose the zone) and options flow (surcharge, peak threshold, backfill, updates)."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)
import voluptuous as vol

from .const import (
    CONF_AUTO_UPDATE,
    CONF_BACKFILL_DAYS,
    CONF_PEAK_THRESHOLD,
    CONF_SURCHARGE,
    CONF_ZONE,
    DEFAULT_AUTO_UPDATE,
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_SURCHARGE,
    DOMAIN,
    MAX_BACKFILL_DAYS,
    MAX_SURCHARGE,
)
from .core.zones import ZONES


class SpotForecastConfigFlow(ConfigFlow, domain=DOMAIN):
    """One entry per bidding zone."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Pick the zone."""
        if user_input is not None:
            zone = user_input[CONF_ZONE]
            await self.async_set_unique_id(zone)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=f"Spot Forecast {zone}", data={CONF_ZONE: zone})

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ZONE, default=next(iter(ZONES))): SelectSelector(
                        SelectSelectorConfig(options=list(ZONES), mode=SelectSelectorMode.DROPDOWN)
                    ),
                }
            ),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SpotForecastOptionsFlow:
        """Options flow for an existing entry."""
        return SpotForecastOptionsFlow(config_entry)


class SpotForecastOptionsFlow(OptionsFlow):
    """Surcharge, peak threshold, backfill window and automatic model updates."""

    def __init__(self, entry: ConfigEntry) -> None:
        # Own reference: OptionsFlow.config_entry is read-only from HA 2024.12 on.
        self._entry = entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Show and save the options."""
        if user_input is not None:
            options: dict[str, Any] = {
                CONF_BACKFILL_DAYS: int(user_input[CONF_BACKFILL_DAYS]),
                CONF_AUTO_UPDATE: bool(user_input[CONF_AUTO_UPDATE]),
            }
            surcharge = round(float(user_input[CONF_SURCHARGE]), 3)
            if surcharge:  # 0: no total-price sensor
                options[CONF_SURCHARGE] = surcharge
            threshold = round(float(user_input[CONF_PEAK_THRESHOLD]), 1)
            # the model's own value is not stored: it then follows a newer (downloaded) model
            if threshold != self._model_peak_threshold():
                options[CONF_PEAK_THRESHOLD] = threshold
            return self.async_create_entry(data=options)

        options = self._entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SURCHARGE,
                        default=options.get(CONF_SURCHARGE, DEFAULT_SURCHARGE),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0,
                            max=MAX_SURCHARGE,
                            step=0.001,
                            unit_of_measurement="EUR/kWh",
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Required(
                        CONF_PEAK_THRESHOLD,
                        default=options.get(CONF_PEAK_THRESHOLD, self._model_peak_threshold()),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0,
                            max=100,
                            step=0.1,
                            unit_of_measurement="ct/kWh",
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Required(
                        CONF_BACKFILL_DAYS,
                        default=options.get(CONF_BACKFILL_DAYS, DEFAULT_BACKFILL_DAYS),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=MAX_BACKFILL_DAYS, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Required(
                        CONF_AUTO_UPDATE,
                        default=options.get(CONF_AUTO_UPDATE, DEFAULT_AUTO_UPDATE),
                    ): BooleanSelector(),
                }
            ),
        )

    def _model_peak_threshold(self) -> float:
        """The active model's threshold in ct/kWh; never reads the model file itself."""
        runtime_data = getattr(self._entry, "runtime_data", None)
        if runtime_data is None:
            return DEFAULT_PEAK_THRESHOLD
        return round(runtime_data.model.peak_threshold / 10, 1)  # EUR/MWh -> ct/kWh
