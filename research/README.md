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
  points across Germany). The forecast archive starts March 2024.
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
| D+1 | 1.88 | 2.84 | 3.48 | 81 % | 78 % |
| D+2 | 1.98 | 3.62 | 3.48 | 79 % | 78 % |
| D+3 | 2.08 | 3.84 | 3.48 | 78 % | 76 % |
| D+4 | 2.20 | 3.89 | 3.48 | 77 % | 75 % |
| D+5 | 2.53 | 3.84 | 3.48 | 76 % | 72 % |
| D+6 | 2.60 | 3.65 | 3.48 | 76 % | 71 % |

Last three months (Aug–Oct 2026, higher and more volatile prices): MAE 2.8 (D+1) … 3.8 (D+6).
Without weather features the model gets 3.0–3.3, i.e. the weather forecasts carry the skill.

Known gaps: no gas/CO2 prices (only absorbed via recent price level), no neighbouring
countries' weather, no hydro/reservoir data, issue time before the day-ahead auction only.
