# Backtest: multi-day DE-LU day-ahead price forecast

Feasibility study before building the Home Assistant integration.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python fetch.py      # cached downloads into data/raw/
.venv/bin/python backtest.py   # walk-forward backtest, prints report
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
| D+1 | 1.80 | 2.84 | 3.48 | 81 % | 79 % |
| D+2 | 1.90 | 3.62 | 3.48 | 80 % | 78 % |
| D+3 | 2.01 | 3.84 | 3.48 | 79 % | 76 % |
| D+4 | 2.17 | 3.89 | 3.48 | 78 % | 74 % |
| D+5 | 2.52 | 3.84 | 3.48 | 77 % | 72 % |
| D+6 | 2.57 | 3.65 | 3.48 | 76 % | 71 % |

Last three months (Aug–Oct 2026, higher and more volatile prices): MAE 2.8 (D+1) … 3.8 (D+6).
Without weather features the model gets 3.0–3.3, i.e. the weather forecasts carry the skill.

### Ablation: neighbouring countries (mean MAE D+1..D+6, ct/kWh)

| variant | all | Aug–Oct 2026 | regret cheapest-4h |
|---|---|---|---|
| Germany only | 2.21 | 3.32 | 0.23 |
| + neighbour weather (aggregates, **default**) | 2.16 | 3.27 | 0.21 |
| + neighbour weather (all raw points) | 2.16 | 3.23 | 0.22 |
| + neighbour prices (FR, NL, BE, AT, CH, PL, CZ, DK1/2, NO2, SE4) | 2.27 | 3.49 | 0.22 |
| + both | 2.21 | 3.25 | 0.21 |

Neighbour weather helps a little (~2–4 %, mostly D+1..D+3). Lagged neighbour prices do
not help and are off by default (`--drop nbp_`).

Known gaps: no gas/CO2 prices (only absorbed via recent price level), no hydro/reservoir
data, issue time before the day-ahead auction only.
