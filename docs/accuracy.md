# Accuracy

## How it is measured

All figures are **out of sample**, from a walk-forward backtest: for every
test month the models were refit on the data before it only, then forecast
that month's days. The weather forecast for target day X+*h* is the archived
ECMWF run from *h* + 1 days before: never fresher than, and up to a day older
than, what a live install uses, so the figures are on the conservative side. Prices are hourly means
(15-minute prices since October 2025 are averaged to hours).

The figures below are the report of the bundled model, `de_lu-2026-10`
(training cutoff 1 October 2026), as its `metrics` hold them. Each model
version carries its own report, shown in the `metrics` attribute of
`sensor.spot_forecast_de_lu_model_version`.

## Per horizon

Mean absolute error (MAE) of the expected price in ct/kWh, over the full
backtest, April 2025 to September 2026:

| Horizon | MAE | Naive MAE | Skill | Above peak risk | P10 below 0 in negative hours |
|---|---|---|---|---|---|
| D+1 | 1.70 | 3.39 | 50% | 12% | 85% |
| D+2 | 1.78 | 3.39 | 48% | 12% | 83% |
| D+3 | 1.89 | 3.39 | 44% | 12% | 85% |
| D+4 | 2.04 | 3.39 | 40% | 12% | 83% |
| D+5 | 2.31 | 3.39 | 32% | 12% | 79% |
| D+6 | 2.43 | 3.39 | 28% | 12% | 81% |

* **Naive** is "same as last week": the price of the same hour seven days
  earlier, the usual baseline for day-ahead prices.
* **Skill** ("better than naive") is 1 − MAE ÷ naive MAE. It depends far
  less on how volatile the market was than the error itself does, so it is the
  figure to compare across periods.
* **Above peak risk** is how often the actual price exceeded the calibrated
  P90. The target is about 10 %.
* **P10 below 0 in negative hours** is the share of hours that turned out
  negative for which the low-price value was below zero, i.e. what the
  negative-price sensor would have caught.

### Last three months

July to September 2026, the trailing three months of the report:

| Horizon | MAE | Naive MAE | Skill |
|---|---|---|---|
| D+1 | 1.99 | 4.29 | 54% |
| D+2 | 2.14 | 4.29 | 50% |
| D+3 | 2.29 | 4.29 | 47% |
| D+4 | 2.50 | 4.29 | 42% |
| D+5 | 2.92 | 4.29 | 32% |
| D+6 | 3.03 | 4.29 | 29% |

The MAE is the model report's (`mae_3m`). The report has the naive MAE for the
full period only, so here it and the skill are computed from the bundled
backfill and the Energy-Charts prices by
[`tools/site_data.py`](https://github.com/cygnusb/ha-spot-forecast/blob/main/tools/site_data.py),
on the same hours (the script checks that its MAE equals the report's).

## Peaks

The expected price gets the **timing** of peaks right, not their **height**.
In the hours above the peak threshold of 21.2 ct/kWh (the top 3 % of hours of
the last training year):

| Horizon | Error in peak hours | Mean bias in peak hours |
|---|---|---|
| D+1 | 7.7 ct | −7.5 ct |
| D+3 | 8.4 ct | −8.0 ct |
| D+6 | 9.9 ct | −9.7 ct |

Nearly all of that error is the forecast being too low. Even a model given the
actual load, wind and solar of the target hour left peaks about 4 ct too low
in the research backtest: part of a peak's height is not explained by German
weather and demand (imports, outages, scarcity bidding).

So for peaks, read the **peak-risk band**: the P90 value is above the actual
price in about nine hours out of ten, and the
[peak-risk sensor](entities.md#peak-risk) switches on it. It raises false
alarms by design: better a warning that turns out unneeded than a peak that
comes as a surprise.

## Market regime

The error is strongly regime-dependent. From July 2026, with high gas prices,
electricity prices were higher and more volatile, and the error rose from
1.70–2.43 ct over the full period to 1.99–3.03 ct. The naive forecast got
worse as well, from 3.39 to 4.29 ct, so the skill changed much less: 54–29 %
against 50–28 % over the full period. Expect the error on your install to move
with the market in the same way, and the skill to stay steadier.

A model that is not retrained loses its edge when the market shifts: in the
research backtest a model frozen in April 2025 stayed close to the monthly
refit for about a year, then fell to the naive baseline and below it during
the 2026 gas price rise. That is why models are retrained monthly and
[updated automatically](model.md#model-updates).

## On your install

`sensor.spot_forecast_de_lu_forecast_error` shows the mean absolute error of
the last 30 days, per horizon in its attributes, computed from your own stored
forecasts. Of several runs with the same anchor day, the integration keeps the
last one, made on the anchor day itself before the next day's prices appear.
Its weather is the ECMWF run fetched around 08:15 UTC that day (10:15 in
summer, 09:15 in winter, local time), close to the training's convention of
the weather run available at 06:00 on the anchor day. Your figures therefore
measure about the same thing as the backtest, with a slightly fresher weather
forecast. The [look-back card](cards.md#look-back-card) shows the same over
any range up to six months, and the
[project website](https://cygnusb.github.io/ha-spot-forecast/#accuracy) shows
the bundled model's backtest as an interactive chart.
