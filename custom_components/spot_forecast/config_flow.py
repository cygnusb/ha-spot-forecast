"""Config flow (choose the zone) and options flow (total price, threshold, backfill, updates)."""

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
    SelectOptionDict,
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
    CONF_VAT,
    CONF_ZONE,
    DEFAULT_AUTO_UPDATE,
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_SURCHARGE,
    DOMAIN,
    MAX_BACKFILL_DAYS,
    MAX_SURCHARGE,
    MAX_VAT,
)
from .core.zones import ZONES


def _zone_options() -> list[SelectOptionDict]:
    """The zones as "DE-LU – Germany and Luxembourg"; the value stays the zone code.

    Not translated: translation keys of selector options must be lower case, the stored
    zone is its code.
    """
    return [
        SelectOptionDict(value=code, label=f"{code} \u2013 {zone.name}")
        for code, zone in ZONES.items()
    ]


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
                        SelectSelectorConfig(
                            options=_zone_options(), mode=SelectSelectorMode.DROPDOWN
                        )
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
    """Surcharge and VAT, peak threshold, backfill window and automatic model updates."""

    def __init__(self, entry: ConfigEntry) -> None:
        # Own reference: OptionsFlow.config_entry is read-only from HA 2024.12 on.
        self._entry = entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Show and save the options."""
        if user_input is not None:
            options: dict[str, Any] = {
                CONF_BACKFILL_DAYS: int(user_input[CONF_BACKFILL_DAYS]),
                CONF_AUTO_UPDATE: bool(user_input[CONF_AUTO_UPDATE]),
                CONF_VAT: round(float(user_input[CONF_VAT]), 1),
            }
            surcharge = round(float(user_input[CONF_SURCHARGE]), 4)
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
                            step="any",  # HA allows 0.001 at the finest; saved to 0.0001
                            unit_of_measurement="EUR/kWh",
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Required(
                        CONF_VAT,
                        default=options.get(CONF_VAT, self._zone_vat()),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0,
                            max=MAX_VAT,
                            step=0.1,
                            unit_of_measurement="%",
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

    def _zone_vat(self) -> float:
        """The VAT default of the entry's zone (19 % for DE-LU, 21 % for NL)."""
        zone = ZONES.get(self._entry.data.get(CONF_ZONE, ""))
        return zone.default_vat if zone is not None else 0.0

    def _model_peak_threshold(self) -> float:
        """The active model's threshold in ct/kWh; never reads the model file itself."""
        runtime_data = getattr(self._entry, "runtime_data", None)
        if runtime_data is None:
            return DEFAULT_PEAK_THRESHOLD
        return round(runtime_data.model.peak_threshold / 10, 1)  # EUR/MWh -> ct/kWh
