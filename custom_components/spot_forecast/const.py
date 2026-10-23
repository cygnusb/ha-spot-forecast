"""Constants of the Spot Forecast integration (stdlib only: the pure tests read it)."""

from __future__ import annotations

DOMAIN = "spot_forecast"

CONF_ZONE = "zone"  # zone code, e.g. "DE-LU"
CONF_PEAK_THRESHOLD = "peak_threshold"  # ct/kWh
CONF_BACKFILL_DAYS = "backfill_days"
CONF_AUTO_UPDATE = "auto_update"
CONF_SURCHARGE = "surcharge"  # EUR/kWh of grid fees and taxes, for the total price
CONF_VAT = "vat"  # percent, charged on the exchange part of the total price

DEFAULT_BACKFILL_DAYS = 60
MAX_BACKFILL_DAYS = 180
DEFAULT_PEAK_THRESHOLD = 15.0  # ct/kWh, used only while no model is loaded
DEFAULT_SURCHARGE = 0.0  # no total-price sensor
MAX_SURCHARGE = 1.0  # EUR/kWh
DEFAULT_VAT = 19.0  # percent, Germany
MAX_VAT = 30.0  # percent

MODEL_MAX_AGE_DAYS = 60

DEFAULT_AUTO_UPDATE = True
# rolling release the monthly training publishes to (see training/manifest.py)
MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
MODEL_DIR = "spot_forecast"  # downloaded models: <config>/.storage/spot_forecast/
MAX_DOWNLOADED_MODELS = 3  # versions kept per zone
# download limits (bytes); the published files are ~2 KB, ~1.6 MB and ~0.3 MB
MAX_MANIFEST_BYTES = 256 * 1024
MAX_MODEL_BYTES = 32 * 1024 * 1024
MAX_BACKFILL_BYTES = 16 * 1024 * 1024
# data sources; the largest bodies (a backfill's prices, a month of previous runs) are ~2 MB
MAX_PRICES_BYTES = 16 * 1024 * 1024
MAX_WEATHER_BYTES = 16 * 1024 * 1024

STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}.{{entry_id}}"  # STORE_KEY.format(entry_id=entry.entry_id)

PRICE_HISTORY_DAYS = 40
