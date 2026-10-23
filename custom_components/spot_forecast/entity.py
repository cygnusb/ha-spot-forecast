"""Base entity: one device per entry, names from the translations."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SpotForecastCoordinator


class SpotEntity(CoordinatorEntity[SpotForecastCoordinator]):
    """An entity of the zone's device, e.g. "Spot Forecast DE-LU Price"."""

    _attr_has_entity_name = True
    _entity_domain: str  # "sensor" / "binary_sensor", set by the platform's classes

    def __init__(
        self, coordinator: SpotForecastCoordinator, key: str, object_id: str | None = None
    ) -> None:
        super().__init__(coordinator)
        entry_id = coordinator.config_entry.entry_id
        # Fixed, English entity ids whatever the instance's language: HA would otherwise
        # build them from the translated name (sensor.spot_forecast_de_lu_preis on a
        # German instance), and the cards and Smart Battery Pilot refer to these ids.
        # Only a suggestion: a renamed or taken id is left to the entity registry.
        self.entity_id = (
            f"{self._entity_domain}.spot_forecast_{coordinator.zone.slug}_{object_id or key}"
        )
        self._attr_unique_id = f"{entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name=f"Spot Forecast {coordinator.zone.code}",
            manufacturer="Spot Forecast",
            model=coordinator.model.version,
            entry_type=DeviceEntryType.SERVICE,
        )
