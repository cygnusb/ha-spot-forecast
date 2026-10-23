"""Spot Forecast: multi-day day-ahead price forecast for a bidding zone."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from .const import CONF_ZONE, DOMAIN
from .coordinator import SpotForecastCoordinator
from .core.model import ModelFile, ModelMismatch
from .core.zones import ZONES
from .model_loader import load_active
from .model_update import ModelUpdater, model_dir, remove_models
from .store import ForecastStore

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


@dataclass
class SpotForecastRuntimeData:
    """What a loaded entry keeps in ``entry.runtime_data``."""

    coordinator: SpotForecastCoordinator
    updater: ModelUpdater

    @property
    def model(self) -> ModelFile:
        """The active model (bundled or downloaded); it can change while loaded."""
        return self.coordinator.model

    @property
    def backfill(self) -> bytes:
        """Raw gzip bytes of the active model's backfill file."""
        return self.coordinator.backfill


type SpotForecastConfigEntry = ConfigEntry[SpotForecastRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> bool:
    """Load the zone's newest model, run (or restore) the forecast and add the entities."""
    zone = entry.data[CONF_ZONE]
    if zone not in ZONES:
        raise ConfigEntryError(f"unsupported zone {zone}")
    try:
        model, backfill, source = await hass.async_add_executor_job(
            load_active, zone, model_dir(hass)
        )
    except (OSError, ModelMismatch) as e:
        raise ConfigEntryError(f"bundled model for {zone} cannot be loaded: {e}") from e
    coordinator = SpotForecastCoordinator(hass, entry, model, backfill, source)
    await coordinator.async_config_entry_first_refresh()
    updater = ModelUpdater(hass, entry, coordinator)
    entry.runtime_data = SpotForecastRuntimeData(coordinator=coordinator, updater=updater)
    coordinator.async_start()
    updater.async_start()
    entry.async_on_unload(entry.add_update_listener(_async_reload_on_update))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_reload_on_update(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> None:
    """Options changed: set the entry up again with them."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> bool:
    """Remove the entities; timers, Store flush and listener go via ``async_on_unload``."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> None:
    """Delete the entry's Store file and its zone's downloaded models.

    The models stay while another entry has the same zone; the long-term statistics
    stay in the recorder.
    """
    await ForecastStore(hass, entry.entry_id).async_remove()
    zone = entry.data[CONF_ZONE]
    if zone not in ZONES or any(
        other.entry_id != entry.entry_id and other.data.get(CONF_ZONE) == zone
        for other in hass.config_entries.async_entries(DOMAIN)
    ):
        return
    await hass.async_add_executor_job(remove_models, ZONES[zone].slug, model_dir(hass))
