"""Constants of the Spot Forecast integration (stdlib only: the pure tests read it)."""

from __future__ import annotations

DOMAIN = "spot_forecast"

CONF_ZONE = "zone"  # zone code, e.g. "DE-LU"
CONF_PEAK_THRESHOLD = "peak_threshold"  # ct/kWh
CONF_BACKFILL_DAYS = "backfill_days"
CONF_AUTO_UPDATE = "auto_update"

DEFAULT_BACKFILL_DAYS = 60
MAX_BACKFILL_DAYS = 180
DEFAULT_PEAK_THRESHOLD = 15.0  # ct/kWh, used only while no model is loaded

MODEL_MAX_AGE_DAYS = 60

DEFAULT_AUTO_UPDATE = True
# rolling release the monthly training publishes to (see training/manifest.py)
MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
MODEL_DIR = "spot_forecast"  # downloaded models: <config>/.storage/spot_forecast/
MAX_DOWNLOADED_MODELS = 3  # versions kept per zone

STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}.{{entry_id}}"  # STORE_KEY.format(entry_id=entry.entry_id)

PRICE_HISTORY_DAYS = 40
