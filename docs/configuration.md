# Configuration

## Installation

Spot Forecast is a HACS custom repository for now.

1. In HACS: ⋮ → *Custom repositories*, URL
   `https://github.com/cygnusb/ha-spot-forecast`, type *Integration*. Or use
   [this link](https://my.home-assistant.io/redirect/hacs_repository/?owner=cygnusb&repository=ha-spot-forecast&category=integration).
2. Download **Spot Forecast** and restart Home Assistant.
3. Settings → Devices & services → *Add integration* → **Spot Forecast**, or
   [start the setup directly](https://my.home-assistant.io/redirect/config_flow_start/?domain=spot_forecast).

Requirements: Home Assistant 2024.11 or newer, and the `recorder`
integration (on by default), which holds the long-term statistics. The only
Python requirement is `numpy`, which a default Home Assistant install already
has.

## Setup

The setup has one step: the **bidding zone**:

| Zone | Country | Default VAT |
|---|---|---|
| DE-LU | Germany and Luxembourg | 19 % |
| NL | Netherlands | 21 % |

Each zone can be set up once; several zones run side by side, each with its own
model, entities (`sensor.spot_forecast_nl_price`, …) and statistics.

The integration ships with a trained model and its backfill file, so it works
right after setup and without access to GitHub.

## Options

*Configure* on the integration opens the options.

| Option | Default | Range | What it does |
|---|---|---|---|
| Grid fees and taxes | 0 | 0–1 EUR/kWh, to 0.0001 | Gross amount per kWh, VAT included, added for the [total-price sensor](entities.md#total-price) and the "mit Abgaben" view of the [forecast card](cards.md#forecast-card). 0 means no total-price sensor. |
| VAT on the market price | the zone's (19 % DE-LU, 21 % NL) | 0–30 %, in steps of 0.1 | Value-added tax charged on the exchange price for the total price. |
| Peak price threshold | the model's, 21.2 ct/kWh | 0–100 ct/kWh | Wholesale price at or above which an hour counts as a price peak, for the [peak-risk sensor](entities.md#peak-risk). |
| Backfill window (days) | 60 | 0–180 | How many past days of forecasts are reconstructed on first start, so the look-back card and the calibration have history. 0 turns the backfill off. |
| Update models automatically | on | | Check for a newer model for the zone after setup and once a day between 03:00 and 06:00, and download it if it passes its checks. Off stops further downloads; a model downloaded before stays in use (see [model updates](model.md#model-updates)). |

The default peak threshold is the 97th percentile of hourly prices in the
model's last training year. As long as you leave it at that value, it follows
the model: a newer model brings its own threshold. Once you set another value,
yours is kept.

All other prices are the bare exchange price, without grid fees, taxes or
supplier margin. The surcharge is one fixed amount per kWh: enter what you pay
on top of the exchange price, value-added tax included. VAT is also charged on
the exchange price itself, so the total price is

```text
total price = exchange price × (1 + VAT / 100) + grid fees and taxes
```

The default VAT is the zone's standard rate: Germany's 19 % for DE-LU, the
Netherlands' 21 % for NL. Luxembourg, the other half of DE-LU, charges a
different rate on electricity: enter the one on your bill. An entry without the
option set always uses its zone's default. A
negative exchange price lowers the total by its amount plus VAT. The
statistics, the look-back, the error figures and the peak-risk and
negative-price sensors stay on the exchange price, so they keep comparing like
with like. A consumer such as [Smart Battery Pilot](smart-battery-pilot.md) can
read either sensor; with the total-price sensor, set its own offset to 0 so the
fees do not count twice.

## First start and the backfill

On the first start the integration:

1. fetches the known prices and the weather forecast and makes the first
   forecast;
2. imports the bundled backfill: out-of-sample forecasts, made in training by
   models that had not seen those days, for target days before the model's
   training cutoff;
3. recomputes the days from the training cutoff to today on the device, from
   archived weather forecasts (Open-Meteo previous runs) and Energy-Charts
   prices, with pauses between requests.

From these it writes the long-term statistics for the
[look-back card](cards.md#look-back-card) and calibrates the peak-risk and
low-price bands. `sensor.spot_forecast_de_lu_backfill_status` shows the
progress (`pending`, `running`, `done`, `failed`) and how many days were
imported or recomputed. A backfill that stops on a source error is retried
later.

Widening the backfill window later adds the missing days on the next start.

## Schedule

A forecast run happens:

* on start and whenever the entry is reloaded;
* after each new ECMWF run becomes available on Open-Meteo, four times a day;
* from 13:00 (Europe/Berlin) every 15 minutes until tomorrow's auction prices
  are published, at the latest until 16:00, then with the new anchor day;
* just after midnight, when the day rolls over.

That is about 6 to 8 runs a day, each with one request to Energy-Charts and
two to Open-Meteo.

## When a source fails

A failed request keeps the last forecast. `stale: true` on the price sensor
and its `issued_at` show that it is old. A failed run is tried again 20 minutes
later; each further failure in a row doubles the wait, up to 3 hours, and a
regular run that comes sooner always happens. Only the first failure of an
outage is logged as a warning, the retries at debug level, and the first
success after it at info level. After 36 hours without a successful
run the forecast part is dropped and only the known prices remain, so an old
forecast never looks like a fresh one.

## Troubleshooting

*Download diagnostics* in the ⋮ menu of the integration entry dumps the active model (version,
source, training cutoff), the run state, the ledger sizes, the backfill state
and the recent accuracy. Attach it to an
[issue](https://github.com/cygnusb/ha-spot-forecast/issues).

For more detail, turn on debug logging (⋮ menu of the integration entry →
*Enable debug logging*, or in `configuration.yaml`):

```yaml
logger:
  logs:
    custom_components.spot_forecast: debug
```

A repairs issue appears when the active model was trained on data more than
60 days old. Turn on automatic model updates, or update the integration.
