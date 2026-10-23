"""Spot Forecast: multi-day day-ahead price forecast for a bidding zone."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from .const import CONF_ZONE
from .coordinator import SpotForecastCoordinator
from .core.model import ModelFile, ModelMismatch
from .core.zones import ZONES
from .model_loader import load_bundled
from .store import ForecastStore

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


@dataclass
class SpotForecastRuntimeData:
    """What a loaded entry keeps in ``entry.runtime_data``."""

    model: ModelFile
    backfill: bytes  # raw gzip bytes of the bundled backfill file
    coordinator: SpotForecastCoordinator


type SpotForecastConfigEntry = ConfigEntry[SpotForecastRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> bool:
    """Load the zone's bundled model, run (or restore) the forecast and add the entities."""
    zone = entry.data[CONF_ZONE]
    if zone not in ZONES:
        raise ConfigEntryError(f"unsupported zone {zone}")
    try:
        model, backfill = await hass.async_add_executor_job(load_bundled, zone)
    except (OSError, ModelMismatch) as e:
        raise ConfigEntryError(f"bundled model for {zone} cannot be loaded: {e}") from e
    coordinator = SpotForecastCoordinator(hass, entry, model)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = SpotForecastRuntimeData(
        model=model, backfill=backfill, coordinator=coordinator
    )
    coordinator.async_start()
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
    """Delete the entry's Store file; its long-term statistics stay in the recorder."""
    await ForecastStore(hass, entry.entry_id).async_remove()
