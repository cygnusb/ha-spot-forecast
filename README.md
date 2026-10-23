# Spot Forecast

<p align="center">
  <img src="https://raw.githubusercontent.com/cygnusb/ha-spot-forecast/main/assets/logo.svg" width="320" alt="Spot Forecast">
</p>

[![GitHub Release][release-badge]][release-url]
[![GitHub Downloads (all assets, all releases)][downloads-badge]][release-url]
[![HACS Custom][hacs-badge]][hacs-url]
[![HA Version][ha-badge]][ha-url]
[![License][license-badge]][license-url]
[![GitHub commit activity][commits-badge]][commits-url]
[![Validate][validate-badge]][validate-url]
[![GitHub Stars][stars-badge]][stars-url]

**Day-ahead electricity prices for Germany and Luxembourg, six days further than the auction.**

[Project website](https://cygnusb.github.io/ha-spot-forecast/) · [Documentation](https://cygnusb.github.io/ha-spot-forecast/docs/) · [Releases](https://github.com/cygnusb/ha-spot-forecast/releases)

The day-ahead auction publishes prices for today and, from about 13:00, for
tomorrow. Anyone who plans a home battery, an electric car or a heat pump
further ahead has to guess. Spot Forecast is a Home Assistant integration that
fills the six days after the last published price with an hourly forecast for
the DE-LU bidding zone, and says how far to trust it: an expected price per
hour, a **peak-risk** value that is exceeded only about one hour in ten, and a
**low-price** value for likely negative prices.

It runs on open data (Energy-Charts and Open-Meteo), needs no account and no
API key, and is light enough for a Raspberry Pi: the models are trained every
month on GitHub and only evaluated on your device, with numpy.

<p align="center">
  <a href="assets/screenshots/forecast_light.png">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/forecast_dark.png">
      <img src="assets/screenshots/forecast_light.png" width="100%" alt="The forecast card on 14 October 2026: known prices for 13 to 15 October as a black step line, then six forecast days in blue with a hatched red peak-risk band above and a hatched green low-price band below. Above each day its maximum and, on the forecast days, a peak-risk warning such as “bis 37”.">
    </picture>
  </a>
  <br>
  <em>The forecast card on 14 October 2026, two weeks after the bundled model's training cutoff: an out-of-sample forecast from the shipped model on real data. Card labels are German.</em>
</p>

## How it works

1. **Known prices.** The last 40 days of DE-LU day-ahead prices from
   Energy-Charts, in one request.
2. **Weather forecast.** ECMWF IFS (0.25°) from Open-Meteo for 10 points in
   Germany and 12 in neighbouring countries: wind at 100 m, solar radiation,
   temperature, cloud cover and precipitation. Weather carries most of the
   skill: without it the error stays close to the naive baseline's.
3. **Three tree models.** Gradient-boosted trees predict, per hour, the
   expected price (median), a peak-risk value (P90) and a low-price value
   (P10), relative to the recent price level, for every hour of the six days
   after the latest known day.
4. **Calibration.** Every forecast is kept. The P90 and P10 bands are shifted
   by how often they were crossed over the last 56 days, so the peak risk stays
   near one hour in ten. On install, a backfill supplies that history.
5. **Output.** One price sensor with known and forecast slots, two warning
   sensors, and long-term statistics of every forecast against the actual
   price for the look-back card.

A run repeats after each new weather model run (four times a day) and again
once the auction result for tomorrow is published. Before that, the forecast
covers tomorrow to today + 6; after it, today + 2 to today + 7. Known prices
always come first and the forecast follows without a gap. Details:
[how the model works](docs/model.md).

## How good is it?

Out of sample, walk-forward: the models were refit month by month on past data
only and forecast days they had never seen, with weather forecasts at least
one run older than a live install uses. The figures are the shipped model's
own report (`de_lu-2026-10`), mean absolute error in ct/kWh:

| Horizon | Apr 2025 – Sep 2026 | Jul – Sep 2026 | Naive | Better than naive | Above peak risk |
|---|---|---|---|---|---|
| D+1 | 1.79 | 2.36 | 3.39 | 47% | 12% |
| D+2 | 1.88 | 2.49 | 3.39 | 45% | 12% |
| D+3 | 1.96 | 2.64 | 3.39 | 42% | 12% |
| D+4 | 2.12 | 2.86 | 3.39 | 38% | 12% |
| D+5 | 2.36 | 3.18 | 3.39 | 30% | 13% |
| D+6 | 2.48 | 3.20 | 3.39 | 27% | 12% |

*Naive* repeats last week's price for the same hour. *Better than naive* is
1 − error ÷ naive error, which depends far less on how volatile the market
was. *Above peak risk* is how often the actual price exceeded the calibrated
P90; the target is about 10 %.

> **Peaks come out too low.** The forecast gets the timing of price peaks
> right, not their height. In hours above 21.2 ct/kWh it is on average
> 8.4 ct too low one day ahead and 10.3 ct six days ahead. For peaks, use the
> peak-risk band (P90) and the peak-risk sensor that switches on it, not the
> expected price.

> **The error depends on the market.** From July 2026 gas and electricity
> prices were higher and more volatile, and the error rose to 2.4 ct (D+1) …
> 3.2 ct (D+6). A frozen model degrades within months when the market shifts,
> so new models are trained every month and reach your install without an
> integration update (see [the model channel](docs/model.md#model-updates)).

The low-price value catches negative prices: in 85 % (D+1) … 81 % (D+6) of
the hours that turned out negative, P10 was below zero. More in
[accuracy](docs/accuracy.md); the [website](https://cygnusb.github.io/ha-spot-forecast/#accuracy)
has the forecast against the actual price, per horizon, as an interactive
chart.

## Installation (HACS)

Spot Forecast is a HACS **custom repository** for now:

[![Open your Home Assistant instance and open this repository in HACS.][my-hacs-badge]][my-hacs-url]

1. Use the button above, or in HACS: ⋮ → *Custom repositories*, URL
   `https://github.com/cygnusb/ha-spot-forecast`, type *Integration*.
2. Download **Spot Forecast** in HACS and restart Home Assistant.
3. Add the integration:

   [![Open your Home Assistant instance and start setting up Spot Forecast.][my-flow-badge]][my-flow-url]

   or Settings → Devices & services → *Add integration* → **Spot Forecast**.

The integration ships with a trained model and works right after setup,
without access to GitHub.

## Configuration

Setup asks for one thing: the bidding zone (DE-LU in this version). The
options (*Configure* on the integration):

| Option | Default | What it does |
|---|---|---|
| Peak price threshold | from the model, 21.2 ct/kWh | Wholesale price above which an hour counts as a peak, for the peak-risk sensor. |
| Backfill window (days) | 60 | Past days of forecasts reconstructed on first start, so the look-back and the calibration have history (0–180, 0 = off). |
| Update models automatically | on | Once a day, download a newer model for the zone if one is published and passes its checks. |

Details: [docs/configuration.md](docs/configuration.md).

## Entities

All prices are the bare exchange price in EUR/kWh, without grid fees, taxes or
supplier margin. Entity ids are the same on every instance language.

| Entity | Description |
|---|---|
| `sensor.spot_forecast_de_lu_price` | Current price. Attribute `prices`: every slot from today 00:00 to the end of the forecast |
| `binary_sensor.spot_forecast_de_lu_peak_risk` | On if a price above the threshold is known or likely (P90) in the next 48 h |
| `binary_sensor.spot_forecast_de_lu_negative_price` | On if a negative price is known or likely (P10) in the next 48 h |
| `sensor.spot_forecast_de_lu_day_0` … `_day_6` | Mean price per day, with min, max, P90 max and P10 min (disabled by default) |
| `sensor.spot_forecast_de_lu_last_run` | Last successful forecast run (diagnostic) |
| `sensor.spot_forecast_de_lu_model_version` | Active model, its source and training cutoff (diagnostic) |
| `sensor.spot_forecast_de_lu_forecast_error` | Mean absolute error over the last 30 days, per horizon as attributes (diagnostic) |
| `sensor.spot_forecast_de_lu_backfill_status` | Whether the backfill ran, and how many days it added (diagnostic) |

An entry of `prices`:

```json
{"time": "2026-10-17T18:00:00+02:00", "price": 0.2134, "p90": 0.3812, "p10": 0.142, "source": "forecast", "h": 2}
```

Known slots (`source: market`) keep their native resolution (15 minutes since
October 2025); forecast slots are hourly. The integration also writes hourly
long-term statistics: `spot_forecast:de_lu_price` (actual price) and
`spot_forecast:de_lu_forecast_h1` … `_h6` (the forecast made 1 … 6 days
before), plus `_peak_risk_h1…6` and `_low_h1…6`. All entities and attributes:
[docs/entities.md](docs/entities.md).

## Dashboard cards

Two cards for the [Plotly Graph Card](https://github.com/dbuezas/lovelace-plotly-graph-card),
written for v4.0.2. Install it from HACS, add a *Manual* card and paste the
YAML:

* [`examples/plotly_forecast.yaml`](examples/plotly_forecast.yaml): known
  prices and the six-day forecast with the peak-risk and low-price bands, the
  four dearest and cheapest hours per day, and a warning above each day whose
  peak risk crosses your threshold (the image at the top).
* [`examples/plotly_lookback.yaml`](examples/plotly_lookback.yaml): the
  forecast made D+1 … D+6 days ahead against the actual price, with the mean
  error over the shown range.

<p align="center">
  <a href="assets/screenshots/lookback_light.png"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/lookback_dark.png"><img src="assets/screenshots/lookback_light.png" width="600" alt="The look-back card: the actual price from 15 September to 14 October 2026 as a dark line and the forecast made two days ahead in orange, the gap between them filled red; buttons for D+1 to D+6 or all, and 7 days to 6 months."></picture></a>
</p>

Both screenshots are real: [`tools/gen_card_screenshots.py`](tools/gen_card_screenshots.py)
renders the shipped YAML in the released card against states and statistics
that the integration's own code computes from the model and research data.
See [docs/cards.md](docs/cards.md).

## Using it with Smart Battery Pilot

[Smart Battery Pilot](https://github.com/cygnusb/ha-smart-battery-pilot) reads
the forecast as it is: select `sensor.spot_forecast_de_lu_price` as its price
entity. Its ENTSO-E adapter reads the `prices` attribute (`time`, `price`);
hours beyond the published market prices are marked `source: forecast` and
carry `p90` and `p10`, which the adapter ignores. Neither project imports the
other. See [docs/smart-battery-pilot.md](docs/smart-battery-pilot.md).

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Day-ahead prices DE-LU | [Energy-Charts](https://www.energy-charts.info/) (Fraunhofer ISE), price data from Bundesnetzagentur \| SMARD.de | CC BY 4.0 |
| Weather forecasts (ECMWF IFS 0.25°, and its previous runs for the backfill) | [Open-Meteo](https://open-meteo.com/) | CC BY 4.0, free for non-commercial use |

No keys, no accounts. The price sensor names both sources in its
`attribution` attribute.

## Languages

The setup flow, options and entity names are in English and German; the
project website too. The card YAML is labelled in German (with English
comments), and the documentation is in English.

## License

MIT

[release-badge]: https://img.shields.io/github/v/release/cygnusb/ha-spot-forecast?include_prereleases
[release-url]: https://github.com/cygnusb/ha-spot-forecast/releases
[downloads-badge]: https://img.shields.io/github/downloads/cygnusb/ha-spot-forecast/total
[hacs-badge]: https://img.shields.io/badge/HACS-Custom-orange.svg
[hacs-url]: https://hacs.xyz/docs/faq/custom_repositories/
[my-hacs-badge]: https://my.home-assistant.io/badges/hacs_repository.svg
[my-hacs-url]: https://my.home-assistant.io/redirect/hacs_repository/?owner=cygnusb&repository=ha-spot-forecast&category=integration
[my-flow-badge]: https://my.home-assistant.io/badges/config_flow_start.svg
[my-flow-url]: https://my.home-assistant.io/redirect/config_flow_start/?domain=spot_forecast
[ha-badge]: https://img.shields.io/badge/HA-2024.11.0+-blue.svg
[ha-url]: https://www.home-assistant.io/
[license-badge]: https://img.shields.io/github/license/cygnusb/ha-spot-forecast
[license-url]: https://github.com/cygnusb/ha-spot-forecast/blob/main/LICENSE
[commits-badge]: https://img.shields.io/github/commit-activity/y/cygnusb/ha-spot-forecast
[commits-url]: https://github.com/cygnusb/ha-spot-forecast/commits/main
[validate-badge]: https://img.shields.io/github/actions/workflow/status/cygnusb/ha-spot-forecast/validate.yml?label=validate&logo=github
[validate-url]: https://github.com/cygnusb/ha-spot-forecast/actions/workflows/validate.yml
[stars-badge]: https://img.shields.io/github/stars/cygnusb/ha-spot-forecast?style=flat
[stars-url]: https://github.com/cygnusb/ha-spot-forecast/stargazers
