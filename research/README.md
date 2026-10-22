# Backtest: multi-day DE-LU day-ahead price forecast

Feasibility study before building the Home Assistant integration.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python fetch.py      # cached downloads into data/raw/
.venv/bin/python backtest.py   # walk-forward backtest, prints report
.venv/bin/python plot_months.py  # monthly charts -> plots/<YYYY-MM>.png, plots/all_months.pdf
```

## Setup

- **Data (no API key):** day-ahead prices DE-LU and installed capacity from Energy-Charts
  (prices from Bundesnetzagentur | SMARD.de, CC BY 4.0); archived ECMWF IFS 0.25 forecasts
  from the Open-Meteo Previous Runs API (wind 100 m, radiation, temperature, clouds at 10
  points across Germany; wind, radiation, temperature, precipitation at 12 points in
  NL, BE, FR, DK, PL, the Alps, NO, SE, FI). The forecast archive starts March 2024.
- **Issue time:** 06:00 Europe/Berlin on day D. Known: prices through end of D. Targets:
  every hour of D+1 … D+6.
- **No look-ahead:** weather for D+h uses `previous_day{h+1}`, i.e. a model run at least
  one day older than live operation would use, so these numbers are conservative.
  Generation/load actuals are not used.
- **Model:** one gradient-boosted tree model (scikit-learn `HistGradientBoostingRegressor`,
  absolute-error loss) with horizon as a feature, predicting the deviation from the last
  7-day mean price. Refit monthly on all data before the test month (expanding window).
- **Resolution:** hourly; 15-min prices (since 2025-10-01) are averaged to hours.

## Result (test period 2025-04 … 2026-10, ct/kWh)

| horizon | model MAE | naive "same as day D" | naive "same as last week" | cheapest-4h hit | priciest-4h hit |
|---|---|---|---|---|---|
| D+1 | 1.86 | 2.84 | 3.48 | 81 % | 78 % |
| D+2 | 1.95 | 3.62 | 3.48 | 80 % | 78 % |
| D+3 | 2.04 | 3.84 | 3.48 | 79 % | 77 % |
| D+4 | 2.18 | 3.89 | 3.48 | 77 % | 74 % |
| D+5 | 2.45 | 3.84 | 3.48 | 76 % | 73 % |
| D+6 | 2.57 | 3.65 | 3.48 | 75 % | 71 % |

Last three months (Aug–Oct 2026, higher and more volatile prices): MAE 2.8 (D+1) … 3.8 (D+6).
Without weather features the model gets 3.0–3.3, i.e. the weather forecasts carry the skill.
Run-to-run differences from feature changes of about ±0.05 ct/kWh are noise.

**Data quality:** in the Open-Meteo previous-runs archive, `shortwave_radiation` at lead
offsets 6 and 7 days is shifted/scaled (0.5–1.6× the shorter leads) before 2025-07; cloud
cover, wind and temperature are consistent. Those radiation values are masked and a
clear-sky × cloud-cover irradiance estimate is used as an always-available substitute.

### Ablation: neighbouring countries (mean MAE D+1..D+6, ct/kWh)

| variant | all | Aug–Oct 2026 | regret cheapest-4h |
|---|---|---|---|
| Germany only | 2.21 | 3.32 | 0.23 |
| + neighbour weather (aggregates, **default**) | 2.16 | 3.27 | 0.21 |
| + neighbour weather (all raw points) | 2.16 | 3.23 | 0.22 |
| + neighbour prices (FR, NL, BE, AT, CH, PL, CZ, DK1/2, NO2, SE4) | 2.27 | 3.49 | 0.22 |
| + both | 2.21 | 3.25 | 0.21 |

(Ablation run before the radiation fix; relative differences still hold.)
Neighbour weather helps a little (~2–4 %, mostly D+1..D+3). Lagged neighbour prices do
not help and are off by default (`--drop nbp_`).

Known gaps: no gas/CO2 prices (only absorbed via recent price level), no hydro/reservoir
data, issue time before the day-ahead auction only.

## Peaks (top 3 % of hours, > 21 ct/kWh)

The mean MAE hides the peaks: they are 3 % of the hours but 12 % of the absolute error.

| variant | MAE all | peak MAE D+1 | peak bias D+1 |
|---|---|---|---|
| default (absolute-error loss → median) | 2.18 | 7.6 | ≈ −7 |
| squared-error loss | 2.29 | 6.2 | −4.8 |
| squared-error + **oracle** actual load/wind/solar of the target hour (diagnostic, `--oracle`) | 1.77 | 5.7 | −3.7 |

Even perfect knowledge of German load and renewables leaves peaks ~4 ct too low, so part of
the peak height is not explained by DE residual load (imports, outages, scarcity bidding).

Further oracle/variant runs, peak MAE D+1 (squared-error loss):

| variant | peak MAE D+1 | peak bias D+1 |
|---|---|---|
| + actual DE cross-border flows | 5.73 | −3.8 |
| + actual load/residual load of FR, NL, AT, CH, PL, DK | 5.76 | −3.8 |
| smaller leaves (`--min-leaf 10`, no oracle) | 6.11 | −4.4 |
| volatility-scaled target (`--target z`, no oracle) | 6.45 | −5.1 |

None of these closes the gap: realised neighbour load and flows add nothing for peak height.

**Decision value:** peak *timing* is caught. Regret of discharging in the forecast's 4 dearest
hours instead of the true ones (`regret dear4`): 0.29 ct (D+1) … 0.59 ct (D+6), vs.
0.84–1.05 for the naive baselines; the true 4 dearest hours average 5.9 ct above the daily mean.

**Peak warning:** a P90 quantile model (`--quantile 0.9`) is under-dispersed out of sample
(25 % exceedance). With a rolling conformal offset (trailing 56 days' 90 % quantile of
`price − P90`, known at issue time) exceedance is 12 %, and "P90 > 21 ct" flags 80 % (D+1) …
67 % (D+6) of peak hours with 39 … 26 % precision (median forecast: 27 … 17 % recall).

### Gas price (TTF) and scaling the target

TTF front-month closes (Yahoo `TTF=F`, unofficial, research only, `fetch_gas()`) rose from
~32 €/MWh (Sep 2025) to 75 (Sep 2026) and 82 (Oct 2026), exactly the months with the
largest errors. Peak = hour above 21.2 ct; bias = mean (forecast − actual) in peak hours.

| variant (median model) | MAE all | MAE Aug–Oct 2026 | peak MAE D+1 | peak bias D+1 | peak bias Aug–Oct |
|---|---|---|---|---|---|
| default | 2.18 | 3.27 | 7.48 | −7.07 | −5.48 |
| + key-free proxy features (`gq_`: 14-day evening price median, 14-day mean of daily P90) | 2.19 | 3.36 | 7.50 | −7.12 | −5.68 |
| + TTF as a feature | 2.17 | 3.23 | 7.38 | −6.89 | −5.14 |
| target relative to TTF (`--target gas`) | 2.16 | 3.25 | 6.77 | −6.05 | −3.70 |
| target relative to evening price level (`--target gq`, key-free) | 2.14 | 3.27 | 7.16 | −6.60 | −4.81 |
| target relative to TTF, squared error | 2.29 | 3.52 | 6.13 | −3.04 | −1.00 |

As a plain feature TTF does nothing: trees cannot extrapolate to a gas price they never saw.
Scaling the target by it lets spikes grow with fuel cost and roughly halves the peak bias in
the high-gas months. For the calibrated P90 peak warning the gain is small:

| P90 variant (calibrated) | exceedance | recall D+1 / D+3 / D+6 | precision D+1 / D+6 | recall Aug–Oct | alarm h/month |
|---|---|---|---|---|---|
| default | 12 % | 0.80 / 0.79 / 0.68 | 0.40 / 0.27 | 0.87 | 53 |
| relative to TTF | 12 % | 0.84 / 0.80 / 0.71 | 0.38 / 0.27 | 0.93 | 58 |
| relative to evening price level | 12 % | 0.82 / 0.78 / 0.70 | 0.41 / 0.27 | 0.90 | 54 |

The key-free evening-price scaling gets most of the TTF gain without an unofficial data
source.

### How long does a model last without retraining? (`--freeze`)

Evening-price-scaled model (`--target gq`), MAE ct/kWh by target quarter: monthly refit vs.
one model trained once before the start and never refit.

| quarter | monthly refit | frozen 2025-04 | frozen 2025-10 | naive last week |
|---|---|---|---|---|
| 2025 Q2 | 1.76 | 1.84 | – | 2.71 |
| 2025 Q3 | 1.70 | 1.71 | – | 2.65 |
| 2025 Q4 | 1.78 | 1.90 | 1.86 | 3.48 |
| 2026 Q1 | 2.01 | 2.03 | 1.98 | 3.13 |
| 2026 Q2 | 2.43 | 2.87 | 2.59 | 4.05 |
| 2026 Q3 | 2.80 | 4.30 | 3.70 | 4.29 |
| 2026 Q4 | 3.65 | 7.14 | 5.68 | 5.93 |

In a stable market a frozen model holds for 6–9 months. When the regime shifts (gas price
doubling in 2026) it falls to the naive baseline within about two quarters and then below it.
Retraining is required, at least monthly.

### Holidays and negative prices

National holidays are features (`holiday`, `offday`). 1 May 2026 (Friday, sunny): the model
caught the negative hours (forecast negative 10–16 h, actual 9–16 h) but not their depth
(forecast minimum −4.6 ct, actual −49.9 ct). Of all negative hours (7.3 %), the median
forecast is negative for 55 % (D+1) … 41 % (D+6) and below 2 ct for 80 % … 61 %; mean bias
in negative hours is only +1.4 ct (D+1), so days like 1 May are rare.

Population-weighted regional holidays (`hol_share`, e.g. Corpus Christi 0.64), bridge days
and the Christmas week (`hol_` features) changed nothing (MAE 2.14 → 2.15, national holiday
bias +0.93 → +0.73, 1 May unchanged), so they stay off (`--drop` default includes `hol_`).
With ~25 holidays in the data, holiday × strong sun is too rare to learn.
