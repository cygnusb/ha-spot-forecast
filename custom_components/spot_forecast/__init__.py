"""Spot Forecast: multi-day day-ahead price forecast for a bidding zone."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from .const import CONF_ZONE
from .core.model import ModelFile, ModelMismatch
from .core.zones import ZONES
from .model_loader import load_bundled


@dataclass
class SpotForecastRuntimeData:
    """What a loaded entry keeps in ``entry.runtime_data``."""

    model: ModelFile
    backfill: bytes  # raw gzip bytes of the bundled backfill file


type SpotForecastConfigEntry = ConfigEntry[SpotForecastRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> bool:
    """Load the zone's bundled model."""
    zone = entry.data[CONF_ZONE]
    if zone not in ZONES:
        raise ConfigEntryError(f"unsupported zone {zone}")
    try:
        model, backfill = await hass.async_add_executor_job(load_bundled, zone)
    except (OSError, ModelMismatch) as e:
        raise ConfigEntryError(f"bundled model for {zone} cannot be loaded: {e}") from e
    entry.runtime_data = SpotForecastRuntimeData(model=model, backfill=backfill)
    entry.async_on_unload(entry.add_update_listener(_async_reload_on_update))
    return True


async def _async_reload_on_update(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> None:
    """Options changed: set the entry up again with them."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> bool:
    """Nothing to tear down yet; the update listener goes via ``async_on_unload``."""
    return True
