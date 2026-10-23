"""Price sensor, total-price sensor, diagnostic sensors and the per-day sensors."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from . import SpotForecastConfigEntry
from .const import DOMAIN
from .coordinator import BACKFILL_STATES, SpotForecastCoordinator
from .core.output import current_price, day_summaries, total_price, with_surcharge
from .entity import SpotEntity

UNIT = "EUR/kWh"
PRECISION = 4  # EUR/kWh to 0.01 ct, as the surcharge option is kept
WEATHER_ATTRIBUTION = "Open-Meteo (CC BY 4.0)"
DAYS = 7


async def async_setup_entry(
    hass: HomeAssistant, entry: SpotForecastConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator = entry.runtime_data.coordinator
    total: list[SensorEntity] = []
    if coordinator.surcharge:
        total.append(TotalPriceSensor(coordinator))
    else:
        # The surcharge was cleared: drop the sensor rather than leave it unavailable.
        registry = er.async_get(hass)
        unique_id = f"{entry.entry_id}_{TotalPriceSensor.KEY}"
        if entity_id := registry.async_get_entity_id("sensor", DOMAIN, unique_id):
            registry.async_remove(entity_id)
    async_add_entities(
        [
            PriceSensor(coordinator),
            *total,
            LastRunSensor(coordinator),
            ModelVersionSensor(coordinator),
            ForecastErrorSensor(coordinator),
            BackfillSensor(coordinator),
            *(DaySensor(coordinator, offset) for offset in range(DAYS)),
        ]
    )


class _SpotSensor(SpotEntity, SensorEntity):
    """A sensor of the zone's device."""

    _entity_domain = "sensor"


class PriceSensor(_SpotSensor):
    """Current market price; the ``prices`` attribute holds market then forecast slots."""

    _attr_native_unit_of_measurement = UNIT
    _attr_suggested_display_precision = PRECISION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _unrecorded_attributes = frozenset({"prices"})  # up to ~300 entries, above 16 KB

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "price")
        self._attr_attribution = (
            f"Prices: {coordinator.zone.price_attribution}; weather: {WEATHER_ATTRIBUTION}"
        )

    @property
    def native_value(self) -> float | None:
        return current_price(self.coordinator.data.market, dt_util.utcnow())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "prices": data.entries,
            "unit": UNIT,
            "issued_at": None if data.issued_at is None else data.issued_at.isoformat(),
            "anchor_day": data.anchor.isoformat(),
            "model_version": data.model.version,
            "stale": data.stale,
            "surcharge": self.coordinator.surcharge,  # for the cards' total-price view
            "vat": self.coordinator.vat,
        }


class TotalPriceSensor(_SpotSensor):
    """Market price with VAT plus the surcharge; ``prices`` likewise, for SBP or automations.

    The surcharge is gross (VAT included), so VAT is added to the market price only.
    Only while a surcharge is set. Statistics, error figures and the risk sensors stay
    on the market price.
    """

    KEY = "total_price"
    _attr_native_unit_of_measurement = UNIT
    _attr_suggested_display_precision = PRECISION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _unrecorded_attributes = frozenset({"prices"})

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, self.KEY)
        self._attr_attribution = (
            f"Prices: {coordinator.zone.price_attribution}; weather: {WEATHER_ATTRIBUTION}"
        )

    @property
    def native_value(self) -> float | None:
        price = current_price(self.coordinator.data.market, dt_util.utcnow())
        if price is None:
            return None
        return total_price(price, self.coordinator.surcharge, self.coordinator.vat)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "prices": with_surcharge(
                data.entries, self.coordinator.surcharge, self.coordinator.vat
            ),
            "unit": UNIT,
            "surcharge": self.coordinator.surcharge,
            "vat": self.coordinator.vat,
            "issued_at": None if data.issued_at is None else data.issued_at.isoformat(),
            "anchor_day": data.anchor.isoformat(),
            "model_version": data.model.version,
            "stale": data.stale,
        }


class LastRunSensor(_SpotSensor):
    """When the last forecast run succeeded."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "last_run")

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.last_success


class ModelVersionSensor(_SpotSensor):
    """Version of the active model, with its source, training cutoff and backtest metrics."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "model_version")

    @property
    def native_value(self) -> str:
        return self.coordinator.data.model.version

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "source": data.model_source,
            "train_cutoff": data.model.train_cutoff.isoformat(),
            "metrics": data.model.metrics,
        }


class ForecastErrorSensor(_SpotSensor):
    """Mean absolute error of the median forecast over the last 30 days, all horizons."""

    _attr_native_unit_of_measurement = UNIT
    _attr_suggested_display_precision = PRECISION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "forecast_error")

    @property
    def native_value(self) -> float | None:
        return self.coordinator.data.accuracy.get("mae")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        accuracy = self.coordinator.data.accuracy
        return {k: accuracy.get(k) for k in ("mae_h", "peak_mae", "n")}


class BackfillSensor(_SpotSensor):
    """Whether past forecasts were reconstructed for the look-back, and how many."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = list(BACKFILL_STATES)
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "backfill", "backfill_status")

    @property
    def native_value(self) -> str:
        return self.coordinator.backfill_state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        result = self.coordinator.backfill_result
        return {
            "days": self.coordinator.backfill_done_days,
            "imported": None if result is None else result.imported,
            "recomputed": None if result is None else result.recomputed,
            "finalized_days": None if result is None else result.finalized_days,
            "last_error": self.coordinator.backfill_error,
        }


class DaySensor(_SpotSensor):
    """Mean price of the local day ``offset`` days from today; min, max, P90/P10 as attributes."""

    _attr_native_unit_of_measurement = UNIT
    _attr_suggested_display_precision = PRECISION
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator: SpotForecastCoordinator, offset: int) -> None:
        super().__init__(coordinator, f"day_{offset}")
        self._offset = offset

    def _summary(self) -> dict[str, Any]:
        tz = self.coordinator.zone.tz
        today = dt_util.utcnow().astimezone(ZoneInfo(tz)).date()
        return day_summaries(self.coordinator.data.entries, tz, today, DAYS)[self._offset]

    @property
    def native_value(self) -> float | None:
        return self._summary()["mean"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {k: v for k, v in self._summary().items() if k != "mean"}
