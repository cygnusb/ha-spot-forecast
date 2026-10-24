# Spot Forecast documentation

Spot Forecast is a Home Assistant integration that forecasts the day-ahead
electricity price of the DE-LU, NL, DK1, DK2 or NO2 bidding zone, hour by hour, for the six days
after the last published auction result. Per hour it gives an expected price,
a peak-risk value (P90) and a low-price value (P10). It runs on open data from
Energy-Charts and Open-Meteo, without an account or API key.

| Page | What it covers |
|---|---|
| [Configuration](configuration.md) | Installing through HACS, the setup step, the options and what happens on first start. |
| [Entities](entities.md) | Every sensor, its attributes, the `prices` format and the long-term statistics. |
| [Dashboard cards](cards.md) | The forecast and look-back cards for the Plotly Graph Card. |
| [How the model works](model.md) | Data, features, the three tree models, calibration, the schedule and the monthly model updates. |
| [Accuracy](accuracy.md) | The backtest figures per horizon, the peak and regime caveats, and how to check the error on your own install. |
| [Smart Battery Pilot](smart-battery-pilot.md) | Using the forecast as the price source of Smart Battery Pilot. |
| [Changelog](changelog.md) | Every release, from the GitHub release notes. |

Installation and an overview are also in the
[README](https://github.com/cygnusb/ha-spot-forecast#readme); the
[project website](https://cygnusb.github.io/ha-spot-forecast/) shows the
forecast against the actual price as an interactive chart.
