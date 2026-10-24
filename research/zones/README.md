# Other bidding zones and relative price levels

Research of 2026-10-23: how far the DE-LU model carries to other bidding zones (NL, DK1,
DK2, SE1–SE4, NO1–NO5, FI), and the `rel` variant (price levels relative to the 7-day
mean) that became feature schema 2. The DE-LU results are in [`../README.md`](../README.md).

| File | What |
|---|---|
| `zonecfg.py` | zone catalogue: weather points and their roles, timezone, holiday calendar, licence |
| `fetch_prices.py`, `fetch_weather.py` | cached downloads (Energy-Charts, Open-Meteo previous runs) into `research/data/zones/` |
| `ready.py`, `queue.sh` | wait until a zone's data is complete, then run `bt.py` jobs one after the other |
| `genfeat.py` | zone-generic feature builder on the `core` helpers, mirroring `core.features` of schema 1 |
| `bt.py` | walk-forward backtest of one zone and feature variant, as `training.train` does it |
| `metrics.py`, `tables.py` | metrics per run (`training.report.evaluate`) and the report tables |
| `parity.py` | `genfeat` against `training.dataset` on DE-LU (equal on schema 1; since schema 2 the price columns differ by design) |
| `coupling.py`, `volume.py`, `diag.py`, `blend.py`, `reuse_de.py` | price coupling, data volume, bias diagnostics, reusing the DE-LU forecast for another zone |
| `dk_options.py` | DK1/DK2 in the production setting: own model, own model with DE-LU inputs (`desetprel`), the DE-LU forecast reused, and 50/50 blends |

Raw data: DE-LU and the research weather points come from `research/data/raw` (the
training data), everything else from `research/data/zones/` (git-ignored, as the outputs in
`research/data/zones/out/`). Prices of the other zones are for research only (see `licence`
in `zonecfg.py`).

Run from this folder with the repo's virtualenv:

```sh
../../.venv/bin/python -I fetch_prices.py NL DK1        # prices of the zones not cached yet
../../.venv/bin/python -I fetch_weather.py 1            # weather points up to tier 1
REFIT_MONTHS=1 ../../.venv/bin/python -I bt.py NL rel 600   # monthly refit, 600 iterations
../../.venv/bin/python -I metrics.py ../data/zones/out/pred_NL_rel_600.pkl
```

`bt.py` variants: `base` (the generic features), `rel` (`base` with relative price levels),
`hydro`, `hydnd`, `hydrel` (hydro features), `plusde` / `deset` / `desetp` (DE-LU weather and
prices), `noprecip`, `noweather`, `nodoy`, and for DE-LU `core` (the training pipeline's
features). `WF_FROM`, `WF_TO` and `TAG` select another walk-forward window, e.g.
`WF_FROM=2025-04-01 WF_TO=2025-10-01 TAG=_2025q2q3` for the second DE-LU window.

The `core` and `core_rel` results were run on feature schema 1 (v0.1.2). Since schema 2
`core` itself has the relative levels and the wall-clock lags, so `core_rel` is refused.

### DK1 and DK2: own model, DE-LU inputs or the DE-LU forecast

Production setting (monthly refit, 600 iterations), issue days 2025-10-01 … 2026-10-21, the same
55,080 rows in every run, MAE ct/kWh (`dk_options.py`). `desetprel` is the generic layout plus
the ten DE-LU weather points (raw) and DE-LU price features (`de_p_lastday, de_p_lastweek,
de_D_mean, de_D_mean7, de_D_gq_evening14, de_D_max`, relative to the zone's own 7-day mean in
its scale units). "DE" is the DE-LU production model's forecast (`core_rel`) used as is.

| zone | own (`rel`) | own + DE inputs (`desetprel`) | DE-LU forecast | `rel` + DE, 50/50 | `desetprel` + DE, 50/50 |
|---|---|---|---|---|---|
| DK1 | 2.50 (bias −1.07) | 2.34 (−0.92) | 2.29 (+0.03) | 2.20 | **2.18** (−0.45) |
| DK2 | 2.77 (−0.85) | 2.48 (−0.93) | 2.45 (−0.17) | 2.43 | **2.35** (−0.55) |

DE-LU inputs help the own model a lot (0.16 / 0.29 ct) but alone it does not beat the DE-LU
forecast; blending both is best for both zones. Every own DK model is biased by about −0.9 ct
whatever its features, while the DE-LU forecast is not. The scale floor (50 EUR/MWh) is not the
cause: since 2025-10 the 14-day evening median never falls below it in DE-LU, DK1, DK2 or NL.
