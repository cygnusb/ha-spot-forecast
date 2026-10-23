# Entities

Every price is the bare day-ahead exchange price in **EUR/kWh**, without grid
fees, taxes or supplier margin, except the [total price](#total-price). The entities belong to one device per zone,
*Spot Forecast DE-LU*.

Entity ids are fixed and English on every instance language, so the cards and
other integrations can rely on them: a German instance also gets
`sensor.spot_forecast_de_lu_price`, only the display names are translated.

## Price

`sensor.spot_forecast_de_lu_price`

* **State:** the price of the current slot (a known market price).
* **`prices`:** every slot from today 00:00 to the end of the forecast, known
  slots first, forecast slots right after them, without a gap or overlap.
* `unit`: `EUR/kWh`.
* `issued_at`: when the shown forecast was made (ISO time).
* `anchor_day`: the latest day whose prices were known when it was made; the
  forecast covers the six days after it.
* `model_version`: the model that made it, e.g. `de_lu-2026-10`.
* `stale`: `true` when the last run failed or is more than 7 hours old.
* `surcharge`: the option [grid fees and taxes](configuration.md#options) in
  EUR/kWh, 0 when unset; the forecast card reads it for its "mit Abgaben" view.
* `attribution`: the data sources and their licences.

An entry of `prices`:

```json
{"time": "2026-10-17T18:00:00+02:00", "price": 0.2134, "p90": 0.3812, "p10": 0.142, "source": "forecast", "h": 2}
```

| Key | On | Meaning |
|---|---|---|
| `time` | every entry | Slot start, local time with offset. The slot ends where the next one starts. |
| `price` | every entry | Known price (`market`) or expected price, the median forecast (`forecast`). |
| `source` | every entry | `market` for a published price, `forecast` for a forecast hour. |
| `p90` | forecast | Peak risk: a price this high or higher is expected in about 1 hour in 10. Never below `price`. |
| `p10` | forecast | Low-price value: a price this low or lower is expected in about 1 hour in 10. Never above `price`. |
| `h` | forecast | Horizon in days after `anchor_day` (1 … 6). |

Known slots keep their native resolution: 15 minutes since 1 October 2025,
hourly before. Forecast slots are hourly. The attribute can hold around 300
entries, so it is not recorded (it would exceed the recorder's 16 KB limit);
its history lives in the [statistics](#long-term-statistics) instead.

A template for the lowest expected price among the forecast hours, as an
example:

```jinja
{% set prices = state_attr('sensor.spot_forecast_de_lu_price', 'prices') or [] %}
{% set ahead = prices | selectattr('source', 'eq', 'forecast') | map(attribute='price') | list %}
{{ ahead | min if ahead else none }}
```

## Total price

`sensor.spot_forecast_de_lu_total_price`

Only while the option [grid fees and taxes](configuration.md#options) is set;
setting it back to 0 removes the sensor.

* **State:** the current exchange price plus the surcharge.
* **`prices`:** the same entries as on the [price sensor](#price), with the
  surcharge added to `price`, `p90` and `p10`. Smart Battery Pilot reads it like
  the price sensor.
* `surcharge`, `unit`, `issued_at`, `anchor_day`, `model_version`, `stale`: as on
  the price sensor.

## Peak risk

`binary_sensor.spot_forecast_de_lu_peak_risk`

On when a price at or above the [peak threshold](configuration.md#options) is
certain or likely in the next 48 hours: a known market price above it, or a
forecast hour whose peak risk (P90) is above it. Counting both keeps the
meaning stable across the day-ahead publication: when tomorrow's peak becomes
a known price, the sensor stays on.

Attributes: `slots` (the flagged entries of `prices`), `threshold` (EUR/kWh)
and `certain` (`true` when at least one flagged slot is a known price).

## Negative price

`binary_sensor.spot_forecast_de_lu_negative_price`

On when a negative price is certain or likely in the next 48 hours: a known
price below zero, or a forecast hour whose low-price value (P10) is below zero.
Attributes as for peak risk.

## Days

`sensor.spot_forecast_de_lu_day_0` … `sensor.spot_forecast_de_lu_day_6`,
**disabled by default**. Day 0 is today. State: the mean price of the day,
weighted by slot length. Attributes: `date`, `min`, `max`, `p90_max` and
`p10_min` (forecast slots only) and `source`. They make automations such as
"is Thursday cheaper than Wednesday?" easy.

## Diagnostics

| Entity | State | Attributes |
|---|---|---|
| `sensor.spot_forecast_de_lu_last_run` | Time of the last successful run | |
| `sensor.spot_forecast_de_lu_model_version` | Active model version | `source` (`bundled` or `downloaded`), `train_cutoff`, `metrics` (the model's backtest report) |
| `sensor.spot_forecast_de_lu_forecast_error` | Mean absolute error of the expected price over the last 30 days, all horizons, EUR/kWh | `mae_h` (per horizon), `peak_mae` (hours above the threshold), `n` |
| `sensor.spot_forecast_de_lu_backfill_status` | `off`, `pending`, `running`, `done` or `failed` | `days`, `imported`, `recomputed`, `finalized_days`, `last_error` |

The forecast error is computed from the integration's own stored forecasts,
backfill included, against the prices that came. Compare it with the figures
on the [accuracy page](accuracy.md).

## Long-term statistics

Every forecast the integration makes is kept until its target day is over and
its prices are known. Then, per completed day, it writes hourly external
statistics (EUR/kWh):

| Statistic id | Content |
|---|---|
| `spot_forecast:de_lu_price` | Actual day-ahead price, hourly mean |
| `spot_forecast:de_lu_forecast_h1` … `_h6` | Expected price, forecast with anchor = target day − h |
| `spot_forecast:de_lu_peak_risk_h1` … `_h6` | Calibrated P90 of the same forecast |
| `spot_forecast:de_lu_low_h1` … `_h6` | Calibrated P10 of the same forecast |

If several runs share an anchor day, the last one wins: the forecast you saw
last. Backfilled days are written the same way. A newer model only affects new
runs; what was written stays.

The statistics appear in *Developer tools → Statistics* and can be used by any
card that reads statistics, such as the [look-back card](cards.md#look-back-card)
or the built-in statistics graph card.
