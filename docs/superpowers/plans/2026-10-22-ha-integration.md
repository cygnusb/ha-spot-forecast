# Spot Forecast, plan 2 of 4: Home Assistant integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `spot_forecast` integration: it fetches prices and weather key-free, runs the bundled model on a schedule, exposes the SBP-compatible price sensor, the peak-risk and negative-price binary sensors, diagnostics and day sensors, keeps every forecast as long-term statistics and backfills 60 days on install.

**Architecture:** All logic that can live without Home Assistant goes into `core/` (pure, numpy only): ingesting raw API payloads, choosing the anchor and run times, the forecast ledger (stored runs → statistic rows, rolling calibration, accuracy), and the output model (attribute list, binary-sensor evaluation, day summaries). The HA layer (`coordinator.py`, `store.py`, `statistics.py`, `sources/`, platforms) only moves data between HA and `core`. Pure code is tested in `tests/`; HA glue is tested end to end in `tests_ha/` with pytest-homeassistant-custom-component.

**Tech Stack:** Python 3.12+, numpy, aiohttp via `async_get_clientsession`, HA `DataUpdateCoordinator`, `Store`, recorder external statistics, issue registry; pytest, pytest-homeassistant-custom-component (PHCC).

**Spec:** `docs/superpowers/specs/2026-10-22-spot-forecast-design.md` ("How it works", "Entities", "Keeping the forecasts", "Failure behaviour", "Configuration"). Plan 1 (`docs/superpowers/plans/2026-10-22-model-core.md`) defines `core/` and its "Carry-over to plan 2" section is binding here.

**Later plans:** 3: model channel + CI (`train.yml`, `models` release, download/verify in the integration, `validate.yml`, `release.yml`, `hacs.json`). 4: Plotly cards, README, website, docs, brand.

## Global Constraints

- `manifest.json`: `domain "spot_forecast"`, `name "Spot Forecast"`, `codeowners ["@cygnusb"]`, `config_flow true`, `dependencies ["recorder"]`, `documentation "https://github.com/cygnusb/ha-spot-forecast"`, `issue_tracker "https://github.com/cygnusb/ha-spot-forecast/issues"`, `integration_type "service"`, `iot_class "cloud_polling"`, `requirements []`, `version "0.1.0"`, `loggers ["custom_components.spot_forecast"]`.
- Minimum HA 2024.11.0: no APIs newer than that without a fallback. Options flow keeps the entry as `self._entry`, never `self.config_entry`. Use `entry.runtime_data`, never `hass.data`.
- Everything in `core/` stays HA-free: standard library and numpy only, relative imports.
- Market time zone `Europe/Berlin` (from `core.zones.DE_LU.tz`), whatever HA's own time zone is.
- Prices are EUR/MWh inside `core`. Everything exposed to HA (states, attributes, statistics) is **EUR/kWh** = EUR/MWh ÷ 1000, rounded to 5 decimals.
- Data sources, no key: Energy-Charts `https://api.energy-charts.info/price?bzn=DE-LU&start=…&end=…`. Open-Meteo forecast `https://api.open-meteo.com/v1/forecast` with `models=ecmwf_ifs025`, `forecast_days=8`, `timezone=GMT`, all points of one variable set in one multi-location request. Open-Meteo previous runs `https://previous-runs-api.open-meteo.com/v1/forecast` with `{var}_previous_day{h+1}`.
- HTTP: shared `async_get_clientsession(hass)`, `aiohttp.ClientTimeout(total=60, connect=10)`, 3 attempts with doubling back-off from 2 s for 429/5xx/connection errors. On 429 wait at least 30 s.
- Statistic ids: `spot_forecast:de_lu_price`, `spot_forecast:de_lu_forecast_h{1..6}`, `spot_forecast:de_lu_peak_risk_h{1..6}`, `spot_forecast:de_lu_low_h{1..6}`. Unit `EUR/kWh`, hourly rows with `mean = min = max = state = value`, `source "spot_forecast"`.
- Peak threshold option in ct/kWh, default `model.peak_threshold / 10` rounded to 0.1. Backfill window option in days, default 60, range 0–180.
- Stale after 36 h without a successful run: drop forecast slots, keep market slots, `stale: true`. Repairs issue `model_outdated` when `today − model.train_cutoff > 60 days`.
- Translations: `translations/en.json` and `translations/de.json`, identical key sets.
- Pure tests: `.venv/bin/python -m pytest tests`. HA tests: `.venv-ha/bin/python -m pytest tests_ha -o asyncio_mode=auto`. Never collect both in one run. `.venv-ha` is created with `uv venv --python 3.13 .venv-ha && uv pip install --python .venv-ha/bin/python -r tests_ha/requirements.txt` (fallback: `python3.13 -m venv .venv-ha` and pip).

## Review Focus

1. Day-ahead publication arriving late or not at all: the anchor stays today, and the forecast covers tomorrow…today+6 instead of stopping. Pinned in Task 2 (`test_anchor_stays_today_without_tomorrow`).
2. DST days in the attribute list: 23 or 25 hourly forecast slots plus 92 or 100 quarter-hour market slots, with no duplicate or missing local times. Pinned in Task 4 (`test_prices_attribute_dst_fall_back_market`, `test_prices_attribute_dst_fall_back_forecast`).
3. HA restart mid-day while the APIs are down: the Store restores runs and the last market slots, entities come up with the cached forecast, and statistics are not written twice. Pinned in Task 3 (`test_finalize_is_idempotent`) and Task 6 (`test_restart_keeps_forecast_and_does_not_duplicate_statistics`).
6. The price state and the 48 h windows move with the clock between network runs (every quarter-hour, no HTTP). Pinned in Task 6 (`test_state_advances_each_quarter_hour_without_fetching`).
4. An API down for a day (Open-Meteo 5xx, Energy-Charts 429): entities stay available with the last forecast and `stale` flips only after 36 h. Pinned in Task 6 (`test_weather_outage_keeps_last_forecast`).
5. Partial weather (a point or hour missing): those hours carry no forecast (`valid=False` from plan 1's carry-over), the others still do. Pinned in Task 1 (`test_missing_weather_hours_are_invalid`).

---

### Task 1: Ingest raw payloads in `core`, forecast validity (plan 1 carry-over)

**Files:**
- Create: `custom_components/spot_forecast/core/ingest.py`
- Modify: `custom_components/spot_forecast/core/forecast.py` (validity), `training/dataset.py` (use `core.ingest`)
- Test: `tests/core/test_ingest.py`, `tests/core/test_forecast.py`, `tests/training/test_dataset.py` (existing parity tests must keep passing)

**Interfaces:**
- Produces:
  - `MarketSlot` (frozen dataclass): `start: datetime` (UTC), `end: datetime` (UTC), `price: float` (EUR/MWh).
  - `parse_energy_charts(doc: dict) -> list[MarketSlot]`: reads `unix_seconds` and `price`, skips `None` prices, takes each slot's end from the next start, and the last slot keeps its own length.
  - `hourly_prices(slots: Sequence[MarketSlot]) -> Hourly`: the mean per UTC hour (15-min → hourly, as research); hours with no slot are NaN.
  - `merge_hourly(a: Hourly, b: Hourly) -> Hourly`: union; `b` wins on overlap.
  - `parse_open_meteo(docs: Sequence[dict], names: Sequence[str], variables: Sequence[str], suffix: str = "") -> dict[str, Hourly]`: one doc per point, in `names` order; times are GMT; keys `"{name}__{var}"` read from `"{var}{suffix}"`.
  - `weather_by_horizon_from_previous_runs(docs, names, variables) -> dict[int, dict[str, Hourly]]`: for h = 1…6 reads suffix `_previous_day{h+1}`. Applies the research radiation mask (offsets 6 and 7 NaN before 2025-07-01).
  - `RADIATION_BROKEN: dict[int, date]` moves here from `training/dataset.py`.
  - In `core/forecast.py`, `Slot` gains `valid: bool`. It is False when the row's `wind_mean` feature is NaN.
  - `run_forecast` raises `IncompleteAnchorDay(InsufficientHistory)` when any hourly price of the anchor day is NaN.
- `training/dataset.load_raw` must build its `Hourly` series through these functions. Its existing parity tests are the proof.

- [ ] **Step 1: Write the failing tests.** Test names:
  - `test_energy_charts_quarter_hours_to_hourly_mean`: four 15-min slots of 10/20/30/40 give hourly 25.
  - `test_energy_charts_skips_none`
  - `test_parse_open_meteo_multi_location_keys`
  - `test_previous_runs_mask_radiation_before_2025_07`
  - `test_missing_weather_hours_are_invalid`: blank one hour of all wind points, so that slot has `valid is False` and the others are True.
  - `test_incomplete_anchor_day_raises`
  - `test_ingest_matches_training_load_raw`: for one month of research raw files, ingest equals `load_raw` (skip without `SPOT_RESEARCH_DIR` data).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/core/test_ingest.py tests/core/test_forecast.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `core/ingest.py`, the validity flag and `IncompleteAnchorDay`, and refactor `training/dataset.load_raw` onto `core.ingest`.
- [ ] **Step 4: Run** `SPOT_RESEARCH_DIR=$PWD/research .venv/bin/python -m pytest tests -q` and confirm everything passes, including the plan 1 parity tests.
- [ ] **Step 5: Commit** with `git commit -m "Move raw payload ingest into core; mark forecast hours without weather invalid"`.

---

### Task 2: Anchor and schedule (`core/schedule.py`)

**Files:**
- Create: `custom_components/spot_forecast/core/schedule.py`
- Test: `tests/core/test_schedule.py`

**Interfaces:**
- Produces:
  - `choose_anchor(now: datetime, market: Sequence[MarketSlot], tz: str) -> date`: returns tomorrow (local) when market slots cover every quarter-hour or hour of tomorrow, otherwise today.
  - `next_run(now: datetime, anchor_is_tomorrow: bool, tz: str) -> datetime` (UTC): the earliest of the following.
    - **Day-ahead poll:** if the anchor is today and the local time is in [13:00, 16:00), the next quarter-hour mark. If it is before 13:00, 13:00 local.
    - **Weather refresh** at the fixed UTC times `WEATHER_RUNS_UTC = (time(2, 15), time(8, 15), time(14, 15), time(20, 15))`.
    - **Midnight rollover:** local 00:05.
  - `is_stale(last_success: datetime | None, now: datetime) -> bool`: true after 36 h.

- [ ] **Step 1: Write the failing tests:**
  - `test_anchor_tomorrow_when_published`
  - `test_anchor_stays_today_without_tomorrow`
  - `test_anchor_dst_fall_back_tomorrow_has_100_quarters`
  - `test_next_run_polls_every_15_min_between_13_and_16`
  - `test_next_run_after_16_without_tomorrow_waits_for_weather_or_midnight`
  - `test_next_run_hits_weather_times`
  - `test_stale_after_36h`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/core/test_schedule.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `core/schedule.py`.
- [ ] **Step 4: Run** the same command and confirm it PASSES.
- [ ] **Step 5: Commit** with `git commit -m "Add anchor choice and run schedule"`.

---

### Task 3: Forecast ledger (`core/ledger.py`)

**Files:**
- Create: `custom_components/spot_forecast/core/ledger.py`
- Test: `tests/core/test_ledger.py`

**Interfaces:**
- Consumes: `Slot`, `conformal_offset`, `peak_risk`, `low_price` (`core.forecast`), and `MarketSlot`.
- Produces:
  - `Run` (frozen dataclass): `anchor: date`, `issued_at: datetime`, `model_version: str`, `slots: tuple[Slot, ...]`, `backfill: bool`.
  - `Ledger` (dataclass):
    - Fields: `runs: dict[date, Run]` (the last run per anchor wins), `finalized: set[date]` (target days whose statistics are written), and `actual: dict[datetime, float]` (UTC hour to EUR/MWh).
    - `add_run(run)`: replaces the stored run for `run.anchor` only if `run.issued_at` is later.
    - `record_actual(hourly: Hourly)`.
    - `offsets(anchor: date, default_p90: dict[int, float], default_p10: dict[int, float]) -> tuple[dict[int, float], dict[int, float]]`: rolling `conformal_offset` per h over residuals of target days in [anchor − 56 d, anchor), from stored runs only. It never uses rows with a target day ≥ anchor.
    - `finalize(tz: str) -> list[StatRow]`: covers every target day not yet in `finalized` whose 24 (or 23/25) actual hours are all known. It emits the price row and, per h that has a run, the forecast/peak-risk/low rows. Each is calibrated with `offsets(run.anchor, …)` and converted to EUR/kWh. The day is then added to `finalized`, so a second call yields nothing.
    - `accuracy(now: datetime, days: int = 30, peak_threshold: float) -> dict`: `mae` (all h), `mae_h` {h: …}, `peak_mae`, `n`. Values in EUR/kWh.
    - `prune(keep_days: int = 200)`.
    - `to_json() -> dict` / `Ledger.from_json(d) -> Ledger`.
  - `StatRow` (frozen dataclass): `statistic_id: str`, `start: datetime` (UTC hour), `value: float` (EUR/kWh).
  - `runs_from_backfill_file(data: bytes, model_version: str) -> list[Run]`: parses the plan 1 backfill format and groups rows by anchor (= target day − h), with `backfill=True` and `issued_at` = anchor 06:00 local.
- Ruling carried from plan 1: backfilled runs are calibrated only with offsets from earlier rows, which `offsets(anchor)` guarantees.

- [ ] **Step 1: Write the failing tests:**
  - `test_last_run_per_anchor_wins`
  - `test_offsets_ignore_targets_on_or_after_anchor`: residuals +100 on target days ≥ anchor, +1 before. Expect an offset of about 1.
  - `test_offsets_default_below_200`
  - `test_finalize_emits_price_and_19_series_rows`: one day with complete actuals and runs for h = 1…6. Expect 24 price rows and 24 × 6 × 3 forecast rows, with values in EUR/kWh.
  - `test_finalize_is_idempotent`
  - `test_finalize_waits_for_complete_day`
  - `test_peak_risk_row_never_below_forecast_row`
  - `test_accuracy_mae_per_horizon`
  - `test_json_roundtrip`
  - `test_backfill_file_groups_by_anchor`: use a small synthetic file in the plan 1 format.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/core/test_ledger.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `core/ledger.py`.
- [ ] **Step 4: Run** the same command and confirm it PASSES.
- [ ] **Step 5: Commit** with `git commit -m "Add forecast ledger: stored runs, statistic rows, rolling calibration, accuracy"`.

---

### Task 4: Output model (`core/output.py`)

**Files:**
- Create: `custom_components/spot_forecast/core/output.py`
- Test: `tests/core/test_output.py`

**Interfaces:**
- Consumes: `MarketSlot`, `Slot`, `peak_risk`, `low_price`.
- Produces:
  - `prices_attribute(market: Sequence[MarketSlot], forecast: Sequence[Slot], p90_off: dict[int, float], p10_off: dict[int, float], tz: str, start: datetime) -> list[dict]`.
    - It lists market slots from `start` (local midnight today) at native resolution, then every **valid** forecast slot after the last market slot.
    - Each entry is `{"time": iso_local, "price": eur_kwh, "source": "market"|"forecast"}`. Forecast entries add `"p90"`, `"p10"` and `"h"`.
    - Prices are rounded to 5 decimals, and the list is sorted with unique times.
    - SBP's ENTSO-E adapter must parse this list: `time` and `price` keys, ISO strings with offset.
  - `current_price(market, now) -> float | None` (EUR/kWh).
  - `flag_window(entries: list[dict], now: datetime, hours: int = 48, *, key: str, threshold: float, above: bool) -> tuple[list[dict], bool]`:
    - The flagged entries come back first. For a market entry the price is compared; for a forecast entry `key` ("p90" or "p10") is.
    - The bool is `certain`, true if any flagged entry is a market entry.
    - Peak: `key="p90", threshold=peak_threshold_eur_kwh, above=True`. Negative: `key="p10", threshold=0.0, above=False`.
  - `day_summaries(entries: list[dict], tz: str, today: date, days: int = 7) -> list[dict]`: for each day offset 0…6, `{"date", "mean", "min", "max", "p90_max", "p10_min", "source"}`. `source` is "market", "forecast" or "mixed". Days without entries get `None` values.

- [ ] **Step 1: Write the failing tests:**
  - `test_prices_attribute_market_then_forecast_no_gap_no_overlap`
  - `test_prices_attribute_skips_invalid_forecast_slots`
  - `test_prices_attribute_dst_fall_back_market`: the market covers 2026-10-25 (the fall-back day), 100 quarter-hours with unique local ISO times.
  - `test_prices_attribute_dst_fall_back_forecast`: the market ends 2026-10-24, the forecast covers 2026-10-25 with 25 hourly entries, with unique local ISO times.
  - `test_prices_attribute_is_parsed_by_sbp_entsoe_adapter`: feed the mixed list (quarter-hour market, then hourly forecast) plus `unit: "EUR/kWh"` to SBP's `detect_adapter` and parse. Copy `slots_from_entries`, `merge_future_slots` and `price_factor_from_attrs` from `~/git/ha-smart-battery-pilot/custom_components/smart_battery_pilot/price_adapters/base.py` and the ENTSO-E `_matches`/`_parse` into `tests/core/sbp_adapter_copy.py`, with a header naming the source file. Assert: the adapter matches, factor is 1.0, slot count equals entry count, the last market slot lasts 15 min, the first forecast slot lasts 60 min, and prices match.
  - `test_flag_window_peak_counts_market_and_forecast`
  - `test_flag_window_negative_uses_p10`
  - `test_day_summaries_seven_days`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/core/test_output.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `core/output.py`.
- [ ] **Step 4: Run** the same command and confirm it PASSES.
- [ ] **Step 5: Commit** with `git commit -m "Add output model: SBP-compatible price list, risk windows, day summaries"`.

---

### Task 5: Integration scaffold: manifest, config flow, sources, bundled model, HA test harness

**Files:**
- Create in `custom_components/spot_forecast/`:
  - `manifest.json`, `const.py`, `__init__.py` (setup, unload and an update listener that reloads; no platforms yet)
  - `config_flow.py`
  - `sources/__init__.py`, `sources/http.py`, `sources/energy_charts.py`, `sources/open_meteo.py`
  - `model_loader.py`
  - `translations/en.json`, `translations/de.json`
- Create: `custom_components/spot_forecast/models/de_lu.json.gz` and `models/de_lu.backfill.json.gz`, copied from `dist/de_lu-2026-10.json.gz` and `dist/de_lu-2026-10.backfill.json.gz` in the main checkout.
- Create: `tests_ha/conftest.py`, `tests_ha/requirements.txt` (`pytest-homeassistant-custom-component`), `tests_ha/fixtures/` (recorded API payloads)
- Modify: `pyproject.toml` (keep `testpaths = ["tests"]`), `.gitignore` (`.venv-ha/`), `docs/superpowers/specs/2026-10-22-spot-forecast-design.md`. In the spec's data table, change the weather row from "7 days" to "8 days" (plan 1 carry-over).
- Test: `tests_ha/test_config_flow.py`, `tests_ha/test_sources.py`, `tests/test_translations.py`

**Interfaces:**
- `const.py`:
  - `DOMAIN = "spot_forecast"`
  - `CONF_ZONE = "zone"` (value `"DE-LU"`)
  - `CONF_PEAK_THRESHOLD = "peak_threshold"` (ct/kWh)
  - `CONF_BACKFILL_DAYS = "backfill_days"`
  - `DEFAULT_BACKFILL_DAYS = 60`, `MAX_BACKFILL_DAYS = 180`
  - `STALE_AFTER = timedelta(hours=36)`, `MODEL_MAX_AGE_DAYS = 60`
  - `STORE_VERSION = 1`, `STORE_KEY = f"{DOMAIN}.{entry_id}"`
  - `PRICE_HISTORY_DAYS = 40`
- `config_flow.py`:
  - User step: a zone selector with DE-LU as the only option. `unique_id` is the zone, abort `already_configured`, and the title is "Spot Forecast DE-LU".
  - Options step with `NumberSelector` fields: peak threshold 0–100 ct/kWh in steps of 0.1 (default from `entry.runtime_data.model.peak_threshold`, falling back to 15 ct when not loaded; never reload the gzip in the flow) and backfill days 0–180.
- `sources/energy_charts.py`: `async fetch_prices(session, zone: str, start: date, end: date) -> list[MarketSlot]`.
- `sources/open_meteo.py`:
  - `async fetch_forecast(session, zone: Zone) -> dict[str, Hourly]`: two requests, one for the DE points with `weather_vars` and one for the neighbour points with `neighbor_vars`.
  - `async fetch_previous_runs(session, zone, start: date, end: date) -> dict[int, dict[str, Hourly]]`: chunked per month, with 3 s between requests.
- `sources/http.py`: `async get_json(session, url, params) -> Any`, implementing the retry policy from Global Constraints. It raises `SourceError` (with `SourceRateLimited` as a subclass).
- `model_loader.py`: `load_bundled(zone: str) -> tuple[ModelFile, bytes]`, returning the model and the raw backfill bytes. Reading happens in the executor; HA calls it through `hass.async_add_executor_job`.

- [ ] **Step 1: Create the HA test harness.**
  - `tests_ha/conftest.py` puts the repo root on `sys.path` and sets `pytest_plugins = "pytest_homeassistant_custom_component"`.
  - Its fixture `hass_with_recorder(recorder_mock, enable_custom_integrations, hass)` keeps `recorder_mock` before `hass`.
  - Record fixture payloads with one live call each into `tests_ha/fixtures/`: `energy_charts_price.json` (3 days), `open_meteo_de.json`, `open_meteo_nb.json` (forecast) and `open_meteo_prev_de.json` (one month, previous runs).
  - Make the fixtures deterministic for the test date 2026-10-22: `energy_charts_price.json` must cover 2026-10-19 through 2026-10-22 only (today's day-ahead), and `energy_charts_price_tomorrow.json` must cover through 2026-10-23 23:45 local. If the live recording lacks 2026-10-23, synthesize those rows by copying 2026-10-22's prices one day later. Open-Meteo fixtures must start 2026-10-22 00:00 GMT; shift timestamps if recorded on another day. Put a `README.md` in `tests_ha/fixtures/` saying what was synthesized.
- [ ] **Step 2: Write the failing tests:**
  - `test_user_flow_creates_entry`
  - `test_second_entry_aborts`
  - `test_options_flow_saves_threshold_and_backfill`
  - `test_options_flow_does_not_set_config_entry_attribute`: an `ast` check, as in `ha-netze-bw`.
  - `test_fetch_prices_parses_fixture`
  - `test_fetch_forecast_two_requests_keys`: uses `aioclient_mock`.
  - `test_get_json_retries_429_then_succeeds`
  - `test_bundled_model_loads`
  - In `tests/test_translations.py`: `test_translation_files_have_same_keys` and `test_every_config_field_has_label`.
- [ ] **Step 3: Run** `.venv-ha/bin/python -m pytest tests_ha -o asyncio_mode=auto -q` and confirm it FAILS.
- [ ] **Step 4: Implement** the files listed above.
- [ ] **Step 5: Run** both suites and confirm they PASS. Run `ruff check` and `ruff format --check`.
- [ ] **Step 6: Commit** with `git commit -m "Add integration scaffold: config flow, key-free sources, bundled model"`.

---

### Task 6: Coordinator, Store, statistics, entities

**Files:**
- Create in `custom_components/spot_forecast/`: `coordinator.py`, `store.py`, `statistics.py`, `entity.py`, `sensor.py`, `binary_sensor.py`
- Modify: `__init__.py` (platforms `["sensor", "binary_sensor"]`, `runtime_data`), `translations/*.json`
- Test: `tests_ha/test_setup.py`

**Interfaces:**
- `store.py`: `ForecastStore(hass, entry_id)` with `async_load() -> tuple[Ledger, list[MarketSlot], datetime | None]` and `async_save(ledger, market, last_success)`. It persists the ledger, the market slots of the last 3 days onward and `last_success`, so a restart during an API outage can rebuild `SpotData` (forecast = the ledger's run for the latest anchor). It uses `Store(hass, STORE_VERSION, STORE_KEY)` and `async_delay_save(…, 10)`.
- `statistics.py`: `async_write(hass, rows: list[StatRow])` groups rows per `statistic_id` and calls `async_add_external_statistics` with one metadata object per id: `has_mean=True`, `has_sum=False`, `name`, `source=DOMAIN`, `statistic_id`, `unit_of_measurement="EUR/kWh"`. Two compatibility shims:
  - Import `StatisticData` and `StatisticMetaData` with the `ha-netze-bw` fallback (`recorder.statistics`, falling back to `recorder.models`).
  - Add `mean_type=StatisticMeanType.ARITHMETIC` and `unit_class=None` only when `StatisticMeanType` is importable.
- `coordinator.py`: `SpotForecastCoordinator(DataUpdateCoordinator[SpotData])` with `update_interval=None`. It schedules itself with `async_track_point_in_utc_time` at `core.schedule.next_run`. Each run does the following, in order:
  1. Fetch the market prices (PRICE_HISTORY_DAYS back through tomorrow) and record the actuals in the ledger.
  2. Choose the anchor.
  3. Fetch the weather forecast.
  4. `run_forecast`, then `ledger.add_run`.
  5. Write `ledger.finalize()` rows through `statistics.async_write`.
  6. Save the store.
  7. Build `SpotData`.
  - A source error (`SourceError`) or a data error (`InsufficientHistory`, which covers `IncompleteAnchorDay`) keeps the previous `SpotData`, with `stale` set by `is_stale`. On startup the previous `SpotData` is rebuilt from the Store. It raises `UpdateFailed` only when there is neither a stored nor a fresh result.
  - **Clock tick:** `async_track_time_change(hass, …, minute=(0, 15, 30, 45), second=0)` re-derives `entries`, the current price, windows and `stale` from the cached data with no network, then calls `async_update_listeners()`. Entities read `now` through `dt_util.utcnow()` at render time.
  - `WEATHER_RUNS_UTC` is a provisional constant (ECMWF data may land on Open-Meteo later than x:15); treat the values as tunable, not spec.
  - `SpotData`: `market: list[MarketSlot]`, `forecast: list[Slot]`, `p90_off`, `p10_off`, `anchor: date`, `issued_at: datetime | None`, `last_success: datetime | None`, `stale: bool`, `model: ModelFile`, `entries: list[dict]` (from `prices_attribute`), `accuracy: dict`.
- `entity.py`: `SpotEntity(CoordinatorEntity)` with `_attr_has_entity_name = True`, `unique_id = f"{entry_id}_{key}"`, `translation_key = key`, and `DeviceInfo(identifiers={(DOMAIN, entry_id)}, name="Spot Forecast DE-LU", manufacturer="Spot Forecast", model=model.version)`.
- `sensor.py`:
  - **`price`** (entity id `sensor.spot_forecast_de_lu_price`):
    - State is `current_price`.
    - `unit_of_measurement = "EUR/kWh"`, with `state_class` measurement.
    - Attributes: `prices` (unrecorded), `unit: "EUR/kWh"`, `issued_at`, `anchor_day`, `model_version`, `stale`.
    - `attribution: "Prices: Bundesnetzagentur | SMARD.de via Energy-Charts (CC BY 4.0); weather: Open-Meteo (CC BY 4.0)"`.
  - Diagnostic sensors (`EntityCategory.DIAGNOSTIC`): `last_run` (timestamp), `model_version` (string, with attributes `train_cutoff` and `metrics`), and `forecast_error` (EUR/kWh, with attributes `mae_h`, `peak_mae` and `n`).
  - Day sensors `day_0` … `day_6`: disabled by default. State is the mean, with the other `day_summaries` fields as attributes.
- `binary_sensor.py`: `peak_risk` and `negative_price`, using `flag_window` with the semantics from the spec. Attributes are `slots`, `certain` and `threshold`.

- [ ] **Step 1: Write the failing tests** (PHCC, recorded fixtures through `aioclient_mock`, with `freezer` setting the time to 2026-10-22 10:00 Europe/Berlin):
  - `test_setup_creates_entities_with_expected_ids`
  - `test_price_attribute_has_market_then_forecast_and_sbp_keys`
  - `test_anchor_switches_after_publication`: replay at 14:00 with a fixture that contains tomorrow.
  - `test_peak_and_negative_binary_sensors`
  - `test_day_sensors_disabled_by_default`
  - `test_statistics_written_for_completed_day`: query `statistics_during_period` for `spot_forecast:de_lu_price` and `_forecast_h1`.
  - `test_weather_outage_keeps_last_forecast`: Open-Meteo returns 500, the entities stay available, `stale` is false before 36 h and true after.
  - `test_restart_keeps_forecast_and_does_not_duplicate_statistics`: after the first run, reload the entry with every API returning 500. The price sensor is available with the same forecast entries, and the statistic row count for `_forecast_h1` is unchanged.
  - `test_state_advances_each_quarter_hour_without_fetching`: at 10:00 read the state, advance the clock 15 min with `async_fire_time_changed`, assert the state equals the 10:15 market price and `aioclient_mock.call_count` is unchanged.
- [ ] **Step 2: Run** `.venv-ha/bin/python -m pytest tests_ha/test_setup.py -o asyncio_mode=auto -v` and confirm it FAILS.
- [ ] **Step 3: Implement** the files listed above.
- [ ] **Step 4: Run** both suites and ruff, and confirm they PASS.
- [ ] **Step 5: Commit** with `git commit -m "Add coordinator, store, statistics and entities"`.

---

### Task 7: Backfill, repairs, diagnostics

**Files:**
- Create: `custom_components/spot_forecast/backfill.py`, `diagnostics.py`, `repairs.py` (issue helpers only)
- Modify: `coordinator.py` (start backfill once after the first successful run; model-age issue), `translations/*.json` (`issues.model_outdated`)
- Test: `tests_ha/test_backfill.py`, `tests_ha/test_diagnostics_repairs.py`

**Interfaces:**
- `backfill.py`: `async run_backfill(hass, coordinator, days: int) -> BackfillResult` with fields `imported`, `recomputed` and `finalized_days`.
  1. Import the bundled backfill runs (`runs_from_backfill_file`) for target days in [today − days, model.train_cutoff).
  2. Recompute the rest of the window, [max(train_cutoff, today − days), today), with `fetch_previous_runs` and `fetch_prices`. For each anchor X it calls `run_forecast(model, X, prices_through_X, weather_by_h)`, with `backfill=True`.
  3. Call `ledger.finalize()` and write the statistics.
  - It runs as a background task (`entry.async_create_background_task`). The Store records `backfill_done_days` so it runs again only when the option grows.
- `diagnostics.py`: `async_get_config_entry_diagnostics` returns the options, model version and cutoff, `last_success`, `stale`, the anchor, the ledger sizes (runs, finalized days), the backfill result, accuracy and the first 48 `entries`.
- Repairs: `ir.async_create_issue(hass, DOMAIN, "model_outdated", is_fixable=False, severity=WARNING, translation_key="model_outdated", translation_placeholders={"version", "cutoff"})`. The issue is deleted when the condition clears and on unload.

- [ ] **Step 1: Write the failing tests:**
  - `test_backfill_imports_bundled_rows_and_recomputes_after_cutoff`: frozen time 2026-10-22, window 30 days, previous-runs fixture. Statistics must exist for 2026-09-25, from the bundled file, and for 2026-10-05, recomputed.
  - `test_backfill_runs_once_and_again_when_window_grows`
  - `test_backfilled_calibration_uses_only_earlier_rows`
  - `test_model_outdated_issue_created_and_cleared`
  - `test_diagnostics_contains_model_and_ledger`
- [ ] **Step 2: Run** `.venv-ha/bin/python -m pytest tests_ha/test_backfill.py tests_ha/test_diagnostics_repairs.py -o asyncio_mode=auto -v` and confirm it FAILS.
- [ ] **Step 3: Implement** the files listed above.
- [ ] **Step 4: Run** both suites and ruff, and confirm they PASS.
- [ ] **Step 5: Commit** with `git commit -m "Add backfill, model-age repair and diagnostics"`.

---

## Self-review notes

- Spec coverage: data sources, anchor, schedule, model/inference via `core`, calibration, look-back statistics, backfill (out of sample), failure behaviour, all entities, configuration, SBP compatibility (Task 4 test), repairs and diagnostics. Model download, CI, `hacs.json`, cards, brand and docs are in plans 3 and 4.
- Plan 1 carry-over: validity and ingest (Task 1), 8 weather days (Task 5 spec edit and `forecast_days=8`), and backfill calibration from earlier rows only (Task 3 `offsets`, Task 7 test).
- Names used across tasks: `MarketSlot`, `parse_energy_charts`, `hourly_prices`, `merge_hourly`, `parse_open_meteo`, `weather_by_horizon_from_previous_runs`, `IncompleteAnchorDay`, `Slot.valid`, `choose_anchor`, `next_run`, `is_stale`, `Run`, `Ledger`, `StatRow`, `runs_from_backfill_file`, `prices_attribute`, `current_price`, `flag_window`, `day_summaries`, `SpotData`, `SpotForecastCoordinator`, `ForecastStore`, `run_backfill`.
