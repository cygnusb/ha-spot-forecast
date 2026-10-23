"""Spot Forecast: multi-day day-ahead price forecast for a bidding zone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from .const import (
    CONF_PEAK_THRESHOLD,
    CONF_SURCHARGE,
    CONF_VAT,
    CONF_ZONE,
    DEFAULT_SURCHARGE,
    DOMAIN,
)
from .coordinator import SpotForecastCoordinator
from .core.model import ModelFile, ModelMismatch
from .core.zones import ZONES
from .model_loader import load_active
from .model_update import ModelUpdater, model_dir, remove_models
from .store import ForecastStore

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]
# options that only change what is shown: applied without setting the entry up again
IN_PLACE_OPTIONS = frozenset({CONF_PEAK_THRESHOLD, CONF_SURCHARGE, CONF_VAT})


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
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_options_updated(hass: HomeAssistant, entry: SpotForecastConfigEntry) -> None:
    """Options changed: apply them in place where they only change what is shown.

    Anything else sets the entry up again, and so does a surcharge set or cleared: the
    total-price sensor exists only with one.
    """
    coordinator = entry.runtime_data.coordinator
    old, new = coordinator.options, dict(entry.options)
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if changed <= IN_PLACE_OPTIONS and _has_surcharge(old) == _has_surcharge(new):
        await coordinator.async_apply_options(new)
    else:
        await hass.config_entries.async_reload(entry.entry_id)


def _has_surcharge(options: Mapping[str, Any]) -> bool:
    return float(options.get(CONF_SURCHARGE, DEFAULT_SURCHARGE)) != 0


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
