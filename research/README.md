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

A P90 quantile model (`--quantile 0.9`) works as a peak warning: "P90 > 21 ct" flags
65 % (D+1) … 45 % (D+6) of peak hours with 46 … 30 % precision; the median forecast flags only
27 … 17 %. P90 is under-dispersed out of sample (25 % of hours exceed it instead of 10 %)
and needs recalibration.
