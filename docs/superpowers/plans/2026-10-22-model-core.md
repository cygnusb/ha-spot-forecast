# Spot Forecast, plan 1 of 4: model core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Home-Assistant-free model core (features, tree inference, forecast assembly) plus the offline training pipeline that produces the model file, the out-of-sample backfill file and a report, matching the research backtest.

**Architecture:** `custom_components/spot_forecast/core/` is a self-contained package (numpy only, relative imports only, no `homeassistant`, no pandas) used by the integration at run time and imported by `training/` (pandas + scikit-learn) at training time, so features exist exactly once. Training fits scikit-learn `HistGradientBoostingRegressor` models and exports their trees to gzip JSON; `core/model.py` walks those trees.

**Tech Stack:** Python 3.12, numpy (runtime); pandas, scikit-learn 1.9, pyarrow, requests, holidays (training and tests only); pytest; ruff.

**Spec:** `docs/superpowers/specs/2026-10-22-spot-forecast-design.md` (sections "How it works", "Model and inference", "Peak risk calibration", "Keeping the forecasts" → backfill, "Releases and the model"). Research reference: `research/backtest.py` run as `--target gq --drop nbp_ gas_` (MAE 2.14 ct, `research/README.md`).

**Later plans** (written after this one lands, against its real interfaces):
2. HA integration: sources (Energy-Charts, Open-Meteo live and previous runs), coordinator and anchor schedule, entities, Store + external statistics, backfill, P90 calibration, repairs, translations, brand.
3. Model channel and CI: `train.yml`, data archive release, quality gate, `models` release + manifest, model download/verification in the integration, `validate.yml`, `release.yml`.
4. Plotly cards (variant A), README, website, MkDocs docs, HACS metadata.

## Global Constraints

- Runtime code under `custom_components/spot_forecast/core/` imports only the standard library and numpy; `manifest.json` will have `requirements: []`.
- `core/` uses relative imports only, so it is importable both as `custom_components.spot_forecast.core` and as top-level `core` (tests, training).
- Python ≥ 3.12 (`target-version = "py312"`), ruff config copied from `~/git/ha-smart-battery-pilot/ruff.toml`.
- Prices inside `core` and `training` are EUR/MWh; conversion to EUR/kWh happens only in the integration (plan 2).
- Zone DE-LU, time zone `Europe/Berlin`, horizons h = 1…6 relative to the anchor day X (prices known through the end of X).
- Weather for target day X+h in training comes from Open-Meteo `previous_day{h+1}`; `shortwave_radiation` at offsets 6 and 7 is NaN before 2025-07-01.
- Target: `y = (price − D_mean7) / scale`, `scale = max(D_gq_evening14, 50) / 100`; prediction = `model(X) · scale + D_mean7`.
- Model hyperparameters (both models): `learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0`, default `early_stopping` (as in research). Median: `loss="absolute_error"`; P90: `loss="quantile", quantile=0.9`. **No categorical features** (dow and issue_dow are plain numbers), so the export only needs numeric splits.
- P90 conformal offset: 90 % quantile of `actual − p90_raw` over the trailing 56 target days per horizon; fewer than 200 values → default offset. Reported peak risk = `max(p90_raw + offset, median)`.
- Peak threshold default: 97th percentile of hourly prices in the 365 days before the training cutoff.
- `FEATURE_SCHEMA = 1`.

## Review Focus

1. DST days: an anchor day or target day with 23 or 25 hours must give the same features as research (local date grouping, `hour` from local time, lags shifted in UTC). Golden anchors 2025-10-25, 2025-10-26 and 2026-03-28 pin this (Task 2).
2. NaN inputs (weather gap, missing price hour): features become NaN, inference routes NaN like scikit-learn (`missing_go_to_left`), nothing raises. Pinned in Task 3 (`test_nan_routing_matches_sklearn`) and Task 2 (`test_missing_weather_gives_nan_not_error`).
3. Too little price history (< 30 complete days before the anchor end): `build_features` raises `InsufficientHistory`, not a silent NaN forecast. Pinned in Task 2.
4. Evening level below the floor (negative or near-zero evening prices): scale clamps at 0.5. Pinned in Task 2 (`test_scale_floor`).
5. A model file whose feature list or schema differs from the code: `load_model` raises `ModelMismatch` instead of predicting with shuffled columns. Pinned in Task 3.

---

### Task 1: Scaffold, zone definition, holidays

**Files:**
- Create: `custom_components/spot_forecast/core/__init__.py` (empty docstring module), `core/zones.py`, `core/calendar.py`
- Create: `pyproject.toml` (pytest only: `pythonpath = ["custom_components/spot_forecast", "."]`, `testpaths = ["tests"]`), `ruff.toml` (copy from SBP, drop its per-file ignores), `tests/requirements.txt` (`pytest`, `numpy`), `training/requirements.txt` (`numpy`, `pandas`, `pyarrow`, `requests`, `holidays`, `scikit-learn==1.9.1`, `pytest`)
- Modify: `.gitignore` (add `training/data/`, `dist/`, `__pycache__/`, `.venv/`)
- Test: `tests/core/test_calendar.py`, `tests/core/test_zones.py`

**Interfaces:**
- Produces:
  - `core.zones.Zone` (frozen dataclass): `code: str`, `tz: str`, `points: dict[str, tuple[float, float]]`, `neighbor_points: dict[str, tuple[float, float]]`, `weather_vars: tuple[str, ...]`, `neighbor_vars: tuple[str, ...]`, `wind_points`, `solar_points`, `nb_wind`, `nb_hydro`, `nb_nordic: tuple[str, ...]`, `evening_hours: tuple[int, int] = (18, 21)`, `scale_floor: float = 50.0`, `scale_divisor: float = 100.0`.
  - `core.zones.DE_LU: Zone`, with values copied verbatim from `research/fetch.py` (`POINTS`, `NEIGHBOR_POINTS`, `WEATHER_VARS`, `NEIGHBOR_VARS`) and `research/backtest.py` (`WIND_POINTS`, `SOLAR_POINTS`, `NB_WIND`, `NB_HYDRO`, `NB_NORDIC`). The order of `points` and `weather_vars` is the research order; it fixes the feature order.
  - `core.calendar.de_holidays(year: int) -> frozenset[date]`: German nationwide holidays (Neujahr, Karfreitag, Ostermontag, 1. Mai, Christi Himmelfahrt, Pfingstmontag, 3. Oktober, 25. and 26. Dezember), Easter by the anonymous Gregorian algorithm.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_calendar.py
import pytest
from core.calendar import de_holidays

holidays = pytest.importorskip("holidays")

@pytest.mark.parametrize("year", range(2023, 2031))
def test_matches_holidays_package(year):
    assert de_holidays(year) == frozenset(holidays.Germany(years=year).keys())

# tests/core/test_zones.py
from core.zones import DE_LU

def test_de_lu_matches_research():
    assert list(DE_LU.points) == ["north_sea", "baltic", "schleswig", "ostfriesland", "uckermark",
                                  "sachsen_anhalt", "nrw", "hessen", "bawue", "bayern"]
    assert DE_LU.points["bayern"] == (48.4, 11.8)
    assert len(DE_LU.neighbor_points) == 12
    assert DE_LU.weather_vars == ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
```

- [ ] **Step 2: Run** `pytest tests/core -v` → FAIL (`ModuleNotFoundError: core.calendar`)
- [ ] **Step 3: Implement** `core/zones.py` and `core/calendar.py` with the interfaces above.
- [ ] **Step 4: Run** `pytest tests/core -v` → PASS (calendar test runs where `holidays` is installed, that is in the training venv)
- [ ] **Step 5: Commit** `git commit -m "Add model core scaffold: DE-LU zone, German holidays"`

---

### Task 2: Features (`core/features.py`) with golden parity to research

**Files:**
- Create: `custom_components/spot_forecast/core/features.py`
- Create: `training/make_golden.py` (script; reads `research/data/raw` through `research/backtest.py`, writes the fixture)
- Create: `tests/fixtures/golden_features.json.gz` (generated, committed)
- Test: `tests/core/test_features.py`

**Interfaces:**
- Consumes: `core.zones.Zone`, `core.calendar.de_holidays`.
- Produces:
  - `Hourly` (frozen dataclass): `start: datetime` (UTC, on the hour), `values: np.ndarray` (float64, one per hour, NaN = missing). Method `at(t: datetime) -> float` (NaN outside the range).
  - `DailyStats` (frozen dataclass): `dates: list[date]` (local dates, consecutive), plus `float64` arrays aligned to `dates`: `mean, min, max, std, mean7, std7, mean30, min7, max7, gq_evening14, gq_q90_14`.
  - `daily_stats(prices: Hourly, zone: Zone) -> DailyStats`: local-date grouping; `std` with ddof=1; rolling windows over consecutive dates as in `research/backtest.py::build_samples` (`std7` = 7-day mean of daily std; `gq_evening14` = 14-day rolling median of the daily median of hours 18–21 local; `gq_q90_14` = 14-day rolling mean of the daily 0.9 quantile, linear interpolation).
  - `FEATURE_NAMES: tuple[str, ...]`: the 82 names below, in this order. `FEATURE_SCHEMA = 1`.
  - `FeatureMatrix` (frozen dataclass): `X: np.ndarray` (n × 82), `times: list[datetime]` (UTC), `h: np.ndarray` (int), `mean7: np.ndarray`, `scale: np.ndarray`.
  - `build_features(anchor: date, prices: Hourly, daily: DailyStats, weather: Mapping[int, Mapping[str, Hourly]], capacity: Mapping[str, float], zone: Zone) -> FeatureMatrix`. `weather[h]` maps `"{point}__{var}"` to the series used for target day anchor+h (the caller picks the run). Rows are all UTC hours whose local date is anchor+h, h = 1…6, h-major then time.
  - `exception InsufficientHistory(ValueError)`: raised when `mean30` or `gq_evening14` of the anchor day is NaN.
  - `decode(y: np.ndarray, fm: FeatureMatrix) -> np.ndarray` = `y · fm.scale + fm.mean7`.

Feature order (copy into `FEATURE_NAMES`):
`h, hour, dow, doy, holiday, offday, issue_dow, p_lastday, p_lastweek, D_mean, D_min, D_max, D_std, D_mean7, D_std7, D_mean30, D_min7, D_max7, D_gq_evening14, D_gq_q90_14`, then for each DE point in zone order its four `weather_vars` (`{point}__{var}`), then `solar_est_mean, wind_mean, wind_power_idx, solar_mean, temp_mean, cloud_mean, nb_wind_power_idx, nb_temp_fr, nb_temp_nordic, nb_precip_hydro, nb_solar_fr, wind_power_idx_day, solar_mean_day, solar_est_mean_day, temp_mean_day, nb_wind_power_idx_day, nb_temp_fr_day, nb_temp_nordic_day, nb_precip_hydro_day, cap_solar, cap_wind_on, cap_wind_off`.

Definitions follow `research/backtest.py` exactly: `p_lastday` = price at t − h days and `p_lastweek` = price at t − 7 days, both shifted in UTC; `hour`, `dow`, `doy`, `holiday` from local target time; `offday = holiday or dow >= 5`; `issue_dow` = anchor weekday; the `D_*` features are the anchor day's `DailyStats`; weather aggregates as `weather_features()` (Haurwitz clear sky at the hour midpoint, wind power curve, `_day` = local-day mean, `nb_precip_hydro_day` = local-day sum); `scale = max(gq_evening14, zone.scale_floor) / zone.scale_divisor`.

- [ ] **Step 1: Write `training/make_golden.py`**: for anchors `2025-06-15, 2025-10-25, 2025-10-26, 2026-03-28, 2026-09-20` take research `build_samples(..., load_gas())` rows with `issue_date == anchor`. Write gzip JSON with the following and run it once:
  - the price `Hourly` window from anchor − 45 days to the end of the anchor day;
  - the weather `Hourly` per h (offset h+1, radiation masked as in research) for the target days;
  - the capacity row;
  - the expected `X` in `FEATURE_NAMES` order, the times, `D_mean7` and the scale.

- [ ] **Step 2: Write the failing tests**

```python
# tests/core/test_features.py
def test_golden_parity(golden):            # one case per golden anchor
    fm = build_features(case.anchor, case.prices, daily_stats(case.prices, DE_LU), case.weather, case.capacity, DE_LU)
    assert [t.isoformat() for t in fm.times] == case.times
    np.testing.assert_allclose(fm.X, case.X, rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fm.scale, case.scale, rtol=1e-12)

def test_dst_end_target_day_has_25_rows(golden):      # anchor 2025-10-25 → target 2025-10-26
    fm = ...; assert (fm.h == 1).sum() == 25

def test_insufficient_history_raises(golden):         # prices window cut to 20 days
    with pytest.raises(InsufficientHistory): build_features(...)

def test_scale_floor():                               # evening median 12 €/MWh → scale 0.5
    ...; assert fm.scale.min() == pytest.approx(0.5)

def test_missing_weather_gives_nan_not_error(golden): # drop one weather series entirely
    fm = ...; assert np.isnan(fm.X[:, FEATURE_NAMES.index("bayern__temperature_2m")]).all()
```

- [ ] **Step 3: Run** `pytest tests/core/test_features.py -v` → FAIL (`ImportError`)
- [ ] **Step 4: Implement** `core/features.py` per the interface. Use `zoneinfo.ZoneInfo(zone.tz)` for local conversion. No pandas.
- [ ] **Step 5: Run** `pytest tests/core -v` → PASS
- [ ] **Step 6: Commit** `git commit -m "Add numpy feature builder with golden parity to research backtest"`

---

### Task 3: Model file and tree inference (`core/model.py`)

**Files:**
- Create: `custom_components/spot_forecast/core/model.py`, `training/export.py`
- Test: `tests/core/test_model.py`, `tests/training/test_export.py`

**Interfaces:**
- Consumes: `core.features.FEATURE_NAMES`, `FEATURE_SCHEMA`.
- Produces:
  - `core.model.Trees`: `baseline: float` and flat int/float arrays over all nodes of all trees: `feature, threshold, left, right, missing_left, is_leaf, value`, plus `roots: np.ndarray` (index of each tree's root). `predict(X: np.ndarray) -> np.ndarray` = baseline + sum of leaf values; NaN goes to the `missing_left` side; a non-NaN value goes left when `x <= threshold`.
  - `core.model.ModelFile` (frozen dataclass): `format: int` (=1), `zone: str`, `version: str` (e.g. `"de_lu-2026-11"`), `feature_schema: int`, `features: tuple[str, ...]`, `train_cutoff: date`, `created: datetime`, `median: Trees`, `p90: Trees`, `p90_offsets: dict[int, float]` (EUR/MWh per h), `peak_threshold: float` (EUR/MWh), `capacity: dict[str, float]`, `metrics: dict`.
  - `core.model.load_model(data: bytes) -> ModelFile` (gzip JSON), raising `ModelMismatch(ValueError)` if `format != 1`, `feature_schema != FEATURE_SCHEMA` or `features != FEATURE_NAMES`.
  - `core.model.dump_model(m: ModelFile) -> bytes` (gzip JSON; arrays as lists; floats as written by `json`).
  - `training.export.trees_from_sklearn(model: HistGradientBoostingRegressor) -> Trees`: reads `model._baseline_prediction` and each `model._predictors[i][0].nodes` (`feature_idx, num_threshold, left, right, missing_go_to_left, is_leaf, value`), offsetting child indices into the flat arrays. It asserts that no node `is_categorical`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_export.py  (importorskip sklearn)
def test_export_reproduces_sklearn():
    X, y = synthetic 5000 × 82 with 3 % NaN, seed 0
    for kw in ({"loss": "absolute_error"}, {"loss": "quantile", "quantile": 0.9}):
        m = HistGradientBoostingRegressor(max_iter=50, **kw, random_state=0).fit(X, y)
        np.testing.assert_allclose(trees_from_sklearn(m).predict(X), m.predict(X), atol=1e-6)

def test_nan_routing_matches_sklearn():   # rows that are all-NaN and single-NaN columns
    ...

# tests/core/test_model.py
def test_roundtrip(tiny_model_file):      # dump → load → identical predictions
def test_rejects_wrong_schema(tiny_model_file):
    with pytest.raises(ModelMismatch): load_model(dump_model(replace(m, feature_schema=2)))
def test_rejects_reordered_features(tiny_model_file): ...
def test_predict_144_rows_under_1s(real_sized_trees):   # 600 trees × 31 leaves, synthetic
```

- [ ] **Step 2: Run** `pytest tests/core/test_model.py tests/training/test_export.py -v` → FAIL
- [ ] **Step 3: Implement** `core/model.py` and `training/export.py`. Vectorise `predict` over rows (walk all rows one level per step until every row sits on a leaf, tree by tree).
- [ ] **Step 4: Run** the tests again → PASS
- [ ] **Step 5: Commit** `git commit -m "Add tree export from scikit-learn and numpy inference"`

---

### Task 4: Forecast assembly and calibration (`core/forecast.py`)

**Files:**
- Create: `custom_components/spot_forecast/core/forecast.py`
- Test: `tests/core/test_forecast.py`

**Interfaces:**
- Consumes: `build_features`, `decode`, `daily_stats`, `Hourly`, `ModelFile`, `Trees.predict`.
- Produces:
  - `Slot` (frozen dataclass): `time: datetime` (UTC hour start), `h: int`, `median: float`, `p90_raw: float` (EUR/MWh).
  - `run_forecast(model: ModelFile, anchor: date, prices: Hourly, weather: Mapping[int, Mapping[str, Hourly]], zone: Zone, capacity: Mapping[str, float] | None = None) -> list[Slot]`. `capacity` defaults to `model.capacity`.
  - `conformal_offset(residuals: np.ndarray, default: float, min_n: int = 200, q: float = 0.9) -> float`: NaNs are ignored.
  - `peak_risk(slot: Slot, offset: float) -> float` = `max(slot.p90_raw + offset, slot.median)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_forecast_matches_manual(golden, tiny_model_file):
    slots = run_forecast(m, case.anchor, case.prices, case.weather, DE_LU, case.capacity)
    assert len(slots) == len(case.times) and {s.h for s in slots} == {1, 2, 3, 4, 5, 6}
    fm = build_features(...); np.testing.assert_allclose([s.median for s in slots], decode(m.median.predict(fm.X), fm))

def test_conformal_offset_uses_default_below_200():
    assert conformal_offset(np.ones(199), default=7.0) == 7.0
def test_conformal_offset_quantile():
    assert conformal_offset(np.arange(1000, dtype=float), default=0) == pytest.approx(np.quantile(np.arange(1000), 0.9))
def test_peak_risk_never_below_median():
    assert peak_risk(Slot(t, 1, median=100, p90_raw=80), offset=5) == 100
```

- [ ] **Step 2: Run** `pytest tests/core/test_forecast.py -v` → FAIL
- [ ] **Step 3: Implement** `core/forecast.py`.
- [ ] **Step 4: Run** `pytest tests/core -v` → PASS
- [ ] **Step 5: Commit** `git commit -m "Add forecast assembly and conformal P90 offset"`

---

### Task 5: Training data (`training/fetch.py`, `training/dataset.py`)

**Files:**
- Create: `training/__init__.py`, `training/fetch.py`, `training/dataset.py`
- Test: `tests/training/test_dataset.py`

**Interfaces:**
- Consumes: `core.zones.DE_LU`, `core.features` (all of it).
- Produces:
  - `training/fetch.py`: the DE-LU parts of `research/fetch.py` (prices DE-LU, `installed_power`, weather for `DE_LU.points` + `DE_LU.neighbor_points`, offsets 2–7, `ecmwf_ifs025`), same chunking, caching and 429 backoff. CLI `python -m training.fetch --raw training/data/raw [--end YYYY-MM-DD]`. Points and variables come from `DE_LU`, not from copies.
  - `training.dataset.load_raw(raw: Path) -> RawData` with `prices: Hourly` (hourly means of 15-min values, as research), `weather: dict[int, dict[str, Hourly]]` (offset → series, radiation at offsets 6/7 NaN before 2025-07-01), `capacity: dict[str, dict[str, float]]` (`"YYYY-MM"` → `cap_*`, forward-filled).
  - `training.dataset.build_dataset(raw: RawData, first_anchor: date, last_anchor: date) -> Dataset`: one `build_features` call per anchor with `weather = {h: raw.weather[h + 1]}` and the capacity of the anchor's month. It skips anchors raising `InsufficientHistory` and rows whose actual price is NaN. `Dataset` fields: `X` (n × 82), `y_price`, `times`, `h`, `anchor` (`date` per row), `target_date`, `mean7`, `scale`, `p_lastweek` (for the naive baseline).

- [ ] **Step 1: Write the failing test** (skipped unless `research/data/raw` exists):

```python
def test_dataset_matches_research_rows():
    ds = build_dataset(load_raw(RESEARCH_RAW), date(2025, 9, 1), date(2025, 9, 30))
    ref = research build_samples rows with issue_date in Sept 2025, same feature order
    assert len(ds.y_price) == len(ref)
    np.testing.assert_allclose(ds.X, ref[FEATURE_NAMES].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)
```

- [ ] **Step 2: Run** `pytest tests/training/test_dataset.py -v` → FAIL
- [ ] **Step 3: Implement** `training/fetch.py` and `training/dataset.py`.
- [ ] **Step 4: Run** the test → PASS
- [ ] **Step 5: Commit** `git commit -m "Add training data fetch and dataset built on core features"`

---

### Task 6: Training CLI, backfill file, report (`training/train.py`)

**Files:**
- Create: `training/train.py`, `training/report.py`
- Modify: `research/README.md` (add one line pointing to `training/` as the production pipeline)
- Test: `tests/training/test_train.py`

**Interfaces:**
- Consumes: `build_dataset`, `trees_from_sklearn`, `ModelFile`, `dump_model`, `conformal_offset`.
- Produces (CLI `python -m training.train --raw DIR --out DIR --cutoff YYYY-MM-DD [--walk-forward-from YYYY-MM-DD] [--backfill-days 180]`):
  - `{out}/de_lu-{YYYY-MM}.json.gz`: the `ModelFile`, fitted on all rows with `target_date < cutoff`. Version = cutoff month. `p90_offsets` = `conformal_offset` per h over the walk-forward residuals of the last 56 target days. `peak_threshold` = 97th percentile of hourly prices in the 365 days before the cutoff. `capacity` = last month ≤ cutoff. `metrics` = the report's numbers.
  - `{out}/de_lu-{YYYY-MM}.backfill.json.gz`: `{"format": 1, "model_version": str, "rows": [[time_utc_iso, h, median, p90_raw], ...]}`, the walk-forward out-of-sample predictions for target days in `[cutoff − backfill_days, cutoff)`. There is one monthly refit per month in that range, trained on `target_date < month start`, exactly as `research/backtest.py::walk_forward`.
  - `{out}/de_lu-{YYYY-MM}.report.md` and `.metrics.json`. Per h: MAE over the whole walk-forward range and over the trailing 3 months, the naive "same as last week" MAE, skill `1 − MAE/naive`, peak MAE and peak bias (actual > `peak_threshold`), and calibrated P90 exceedance. All in ct/kWh in the Markdown, EUR/MWh in the JSON.
  - `training.train.fit(ds: Dataset, quantile: float | None) -> HistGradientBoostingRegressor` with the Global Constraints hyperparameters and target.
  - The walk-forward range starts at `--walk-forward-from` (default: cutoff − backfill_days).

- [ ] **Step 1: Write the failing tests**

```python
def test_train_end_to_end_small(tmp_path, research_raw):   # skipped without research data
    run(["--raw", research_raw, "--out", tmp_path, "--cutoff", "2025-10-01", "--backfill-days", "31"], max_iter=30)
    m = load_model((tmp_path / "de_lu-2025-10.json.gz").read_bytes())
    assert m.train_cutoff == date(2025, 10, 1) and set(m.p90_offsets) == {1, 2, 3, 4, 5, 6}
    bf = json.loads(gzip.decompress((tmp_path / "de_lu-2025-10.backfill.json.gz").read_bytes()))
    assert min(r[0] for r in bf["rows"]) >= "2025-08-31" and max(r[0] for r in bf["rows"]) < "2025-10-01"
    assert all(r[0] < "2025-10-01" for r in bf["rows"])        # never in sample

def test_backfill_rows_are_out_of_sample(...):   # each row's refit month cutoff ≤ its target day
```

(`run` accepts a test-only `max_iter` override to keep the test fast.)

- [ ] **Step 2: Run** `pytest tests/training/test_train.py -v` → FAIL
- [ ] **Step 3: Implement** `training/train.py` and `training/report.py`.
- [ ] **Step 4: Run** the tests → PASS
- [ ] **Step 5: Parity run** (full size, in the background, about 15 minutes):
  `python -m training.train --raw research/data/raw --out dist --cutoff 2026-10-01 --walk-forward-from 2025-04-01`
  Expected: the report's walk-forward MAE over all h is 2.14 ± 0.05 ct/kWh and the calibrated P90 exceedance is 11–13 %. This is the research `--target gq` result; the only intended difference is numeric instead of categorical weekday. If it misses, stop and report. Do not tune.
- [ ] **Step 6: Commit** code and the parity report excerpt in the commit message: `git commit -m "Add training CLI: model file, out-of-sample backfill, report"`

---

## Self-review notes

- Spec coverage for this plan: features, inference, export, P90 calibration, peak threshold, out-of-sample backfill, report with full-period and trailing-3-month numbers, and the 1e-6 export acceptance are all covered. Run-time data sources, anchor switching, entities, statistics, model download, CI and cards are plans 2–4.
- Names used across tasks: `Hourly`, `DailyStats`, `daily_stats`, `build_features`, `FeatureMatrix`, `FEATURE_NAMES`, `FEATURE_SCHEMA`, `InsufficientHistory`, `decode`, `Trees`, `ModelFile`, `load_model`, `dump_model`, `ModelMismatch`, `trees_from_sklearn`, `Slot`, `run_forecast`, `conformal_offset`, `peak_risk`, `load_raw`, `RawData`, `build_dataset`, `Dataset`, `fit`.
