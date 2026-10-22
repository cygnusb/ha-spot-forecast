"""Constants of the Spot Forecast integration (stdlib only: the pure tests read it)."""

from __future__ import annotations

from datetime import timedelta

DOMAIN = "spot_forecast"

CONF_ZONE = "zone"  # zone code, e.g. "DE-LU"
CONF_PEAK_THRESHOLD = "peak_threshold"  # ct/kWh
CONF_BACKFILL_DAYS = "backfill_days"

DEFAULT_BACKFILL_DAYS = 60
MAX_BACKFILL_DAYS = 180
DEFAULT_PEAK_THRESHOLD = 15.0  # ct/kWh, used only while no model is loaded

STALE_AFTER = timedelta(hours=36)
MODEL_MAX_AGE_DAYS = 60

STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}.{{entry_id}}"  # STORE_KEY.format(entry_id=entry.entry_id)

PRICE_HISTORY_DAYS = 40
