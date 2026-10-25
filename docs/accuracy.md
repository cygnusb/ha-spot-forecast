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
`sensor.spot_forecast_de_lu_model_version`. The NL model's figures are in
[their own section](#netherlands-nl).

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

## Netherlands (NL)

The report of the bundled NL model, `nl-2026-10`, with the same cutoff, walk-forward window
and monthly refit as DE-LU's (`sensor.spot_forecast_nl_model_version` shows it). MAE in
ct/kWh, April 2025 to September 2026, and for the last three months (July to September
2026):

| Horizon | MAE | MAE last three months | Naive MAE | Skill | Above peak risk | P10 below 0 in negative hours |
|---|---|---|---|---|---|---|
| D+1 | 1.75 | 2.24 | 3.10 | 44% | 12% | 87% |
| D+2 | 1.85 | 2.36 | 3.10 | 40% | 12% | 86% |
| D+3 | 1.93 | 2.53 | 3.10 | 38% | 12% | 88% |
| D+4 | 2.05 | 2.72 | 3.10 | 34% | 12% | 87% |
| D+5 | 2.22 | 2.76 | 3.10 | 28% | 12% | 82% |
| D+6 | 2.31 | 2.88 | 3.10 | 25% | 12% | 82% |

The naive MAE is lower than DE-LU's (3.10 against 3.39 ct), and so is the skill, most of
all at D+5 and D+6 (28 % and 25 % against 32 % and 28 %). Peaks come out too low as for
DE-LU: in the hours above 20.8 ct/kWh (NL's peak threshold), on average 6.8 ct one day ahead
and 8.4 ct six days ahead. The project website's look-back chart shows DE-LU only.

## Denmark West (DK1) and Denmark East (DK2)

The reports of the bundled DK models, `dk1-2026-10` and `dk2-2026-10`, with the same cutoff,
walk-forward window and monthly refit as DE-LU's. Each is the [blend of its own model and
the DE-LU model](model.md#dk1-and-dk2-blended-with-de-lu), and every figure is the blend's.
MAE in ct/kWh, April 2025 to September 2026, and for the last three months (July to
September 2026).

### DK1

| Horizon | MAE | MAE last three months | Naive MAE | Skill | Above peak risk | P10 below 0 in negative hours |
|---|---|---|---|---|---|---|
| D+1 | 1.67 | 2.01 | 3.68 | 55% | 12% | 90% |
| D+2 | 1.75 | 2.18 | 3.68 | 52% | 12% | 90% |
| D+3 | 1.89 | 2.39 | 3.68 | 49% | 12% | 93% |
| D+4 | 2.08 | 2.62 | 3.68 | 43% | 12% | 90% |
| D+5 | 2.38 | 2.97 | 3.68 | 35% | 12% | 88% |
| D+6 | 2.51 | 3.19 | 3.68 | 32% | 13% | 88% |

### DK2

| Horizon | MAE | MAE last three months | Naive MAE | Skill | Above peak risk | P10 below 0 in negative hours |
|---|---|---|---|---|---|---|
| D+1 | 1.82 | 2.02 | 3.80 | 52% | 12% | 89% |
| D+2 | 1.92 | 2.22 | 3.80 | 50% | 12% | 88% |
| D+3 | 2.02 | 2.38 | 3.80 | 47% | 12% | 89% |
| D+4 | 2.21 | 2.55 | 3.80 | 42% | 12% | 88% |
| D+5 | 2.50 | 2.95 | 3.80 | 34% | 12% | 86% |
| D+6 | 2.62 | 3.16 | 3.80 | 31% | 12% | 88% |

Over all horizons the blend has an MAE of 2.05 ct for DK1 and 2.18 ct for DK2. Its two halves
alone would have 2.17 (own model) and 2.17 (DE-LU model) for DK1, 2.29 and 2.28 for DK2:
they err in different hours, so their average is better than either. The own models come
out about 0.7 ct too low on average, the DE-LU model hardly at all, so the blend is about
0.3 ct too low. Peaks come out too low as for DE-LU: in the hours above DK1's peak threshold
of 20.9 ct/kWh on average 7.6 ct one day ahead and 9.9 ct six days ahead, above DK2's 21.7
ct/kWh 8.8 and 11.3 ct.

## Southern Norway (NO2)

The report of the bundled NO2 model, `no2-2026-10`, with the same cutoff, walk-forward window
and monthly refit as DE-LU's. NO2 has its own model with the [hydro
features](model.md#no2-hydro-features). MAE in ct/kWh, April 2025 to September 2026, and
for the last three months (July to September 2026):

| Horizon | MAE | MAE last three months | Naive MAE | Skill | Above peak risk | P10 below 0 in negative hours |
|---|---|---|---|---|---|---|
| D+1 | 1.50 | 1.85 | 2.25 | 33% | 12% | 63% |
| D+2 | 1.58 | 1.98 | 2.25 | 30% | 12% | 68% |
| D+3 | 1.66 | 2.19 | 2.25 | 26% | 12% | 71% |
| D+4 | 1.73 | 2.27 | 2.25 | 23% | 13% | 67% |
| D+5 | 1.79 | 2.32 | 2.26 | 20% | 13% | 69% |
| D+6 | 1.85 | 2.49 | 2.26 | 18% | 13% | 67% |

NO2's prices move less from one week to the next than the other zones' (naive MAE 2.25 ct
against 3.10–3.80), so the same error is a smaller skill: 1.69 ct over all horizons is 25 %
better than the naive forecast, 33 % one day ahead and 18 % six days ahead. From October 2025,
the window of the research backtest, the model has an MAE of 1.77 ct over all horizons and a
skill of 28 %; the research predictions have 1.75 ct on the same rows. The forecast is
about 0.6 ct too low on average, and peaks come out too low as elsewhere: in the hours above
NO2's peak threshold of 16.9 ct/kWh on average 4.7 ct one day ahead and 5.8 ct six days
ahead. Fewer negative hours are covered by a P10 below zero (67 %) than in the other zones:
since October 2025 fewer than 1 % of NO2's hours were negative, at −0.3 ct/kWh on average
(NL: 5 %, at −1.6 ct/kWh).

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
