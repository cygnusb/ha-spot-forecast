"""Peak-risk and negative-price flags for the next 48 h."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from . import SpotForecastConfigEntry
from .coordinator import SpotForecastCoordinator
from .core.output import flag_window
from .entity import SpotEntity


async def async_setup_entry(
    hass: HomeAssistant, entry: SpotForecastConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator = entry.runtime_data.coordinator
    async_add_entities([PeakRiskSensor(coordinator), NegativePriceSensor(coordinator)])


class _FlagSensor(SpotEntity, BinarySensorEntity):
    """On if any slot in [now, now + 48 h) crosses the threshold: market price or forecast key."""

    _key: str  # forecast entry key compared instead of the median
    _above: bool

    @property
    def threshold(self) -> float:
        """EUR/kWh."""
        raise NotImplementedError

    def _flagged(self) -> tuple[list[dict], bool]:
        return flag_window(
            self.coordinator.data.entries,
            dt_util.utcnow(),
            key=self._key,
            threshold=self.threshold,
            above=self._above,
        )

    @property
    def is_on(self) -> bool:
        return bool(self._flagged()[0])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        slots, certain = self._flagged()
        return {"slots": slots, "certain": certain, "threshold": self.threshold}


class PeakRiskSensor(_FlagSensor):
    """A price at or above the peak threshold is known (market) or likely (P90) within 48 h."""

    _key = "p90"
    _above = True

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "peak_risk")

    @property
    def threshold(self) -> float:
        return round(self.coordinator.peak_threshold / 100, 5)


class NegativePriceSensor(_FlagSensor):
    """A negative price is known (market) or likely (P10) within 48 h."""

    _key = "p10"
    _above = False

    def __init__(self, coordinator: SpotForecastCoordinator) -> None:
        super().__init__(coordinator, "negative_price")

    @property
    def threshold(self) -> float:
        return 0.0
