# Spot Forecast — design

Status: draft for review · 2026-10-22

## Intent

A Home Assistant integration that forecasts the day-ahead electricity price for
the next six days, like energyforecast.de or energypriceforecast.eu, but built
only on open data and shipped through HACS.

The published day-ahead auction covers at most today and tomorrow. Anyone who
plans a home battery, an EV or a heat pump further ahead has to guess. This
integration fills the days after that with a forecast and says how much to
trust it: an expected price per hour, plus a "peak risk" value per hour that is
exceeded only about one hour in ten.

The owner's own consumer is Smart Battery Pilot (SBP,
`~/git/ha-smart-battery-pilot`). The two stay independent: Spot Forecast is
useful on its own, and SBP can read its sensor later without either project
importing the other.

## Evidence it works

A retrospective walk-forward backtest lives in `research/` (see
`research/README.md`). Monthly refit, issue at 06:00 with prices known through
the end of that day, weather from archived forecasts at least one run older
than live operation would use. Test period 2025-04 … 2026-10, DE-LU.

| horizon | MAE ct/kWh | naive "same as last week" |
|---|---|---|
| D+1 | 1.86 | 3.48 |
| D+3 | 2.04 | 3.48 |
| D+6 | 2.57 | 3.48 |

The benchmark the owner set is 4–5 ct/kWh MAE; the backtest beats it at every
horizon. Known weakness, which the product must state openly: price **peaks**
are timed well but their height is underestimated by about 30 %. The
calibrated P90 ("peak risk") compensates: it catches 80 % (D+1) … 67 % (D+6) of
hours above 21 ct at the cost of false alarms.

## Goals

1. Forecast hourly DE-LU day-ahead prices for every hour after the last
   published one, up to six days after the latest known day.
2. Per hour, an expected price (median model), a peak-risk value
   (calibrated P90) and a low-price value (calibrated P10).
3. Key-free by default: no account, no token, no API key.
4. Runs on a Raspberry Pi: inference only, no heavy dependencies.
5. A forecast card and a look-back card for the Plotly Graph Card, in the
   design chosen from the mock-ups (variant A, see below).
6. Every forecast the integration makes is kept, so users can see in Home
   Assistant how good past forecasts were.
7. Output format that SBP's existing ENTSO-E price adapter reads without SBP
   changes.
8. Published like Smart Battery Pilot: HACS, README with badges, project
   website, documentation, CI.

## Non-goals (v1)

- Bidding zones other than DE-LU. The code is structured so a zone is data
  (points, model file, holidays), not code, but only DE-LU ships.
- Grid fees, taxes, supplier margins. Output is the bare exchange price; SBP
  already adds its own `price_offset`.
- 15-minute forecast resolution. Known prices keep their native resolution;
  forecast slots are hourly.
- Training on the device. Models are trained monthly in GitHub Actions and
  delivered through the model channel.
- A bundled custom Lovelace card. The cards are Plotly Graph Card YAML
  examples, as in `ha-netze-bw`.
- ENTSO-E data (outages, load forecast). It needs a token; a later optional
  feature if it is shown to improve peaks clearly.
- Intraday / continuous market prices.
- A gas price feed. Scaling by the TTF price helped peaks a little more than
  the evening-price scaling (D+1 peak error 6.8 vs 7.2 ct, P90 recall 0.84 vs
  0.82). But there is no official key-free TTF source, only Yahoo's
  unofficial API, so it is not worth the dependency in v1 (see
  `research/README.md`). Revisit if an official open source appears.

## Anti-goals

- Coupling to SBP (imports, shared config, SBP-specific entities).
- Requiring an API key, or failing to work without one.
- Shipping scikit-learn, pandas or LightGBM to the Pi.
- Presenting the forecast as more certain than it is. The peak weakness is in
  the README and on the website, with numbers.
- Silent staleness: an old forecast must never look like a fresh one.

## Constraints

- **Data:** Energy-Charts API (Fraunhofer ISE, prices from Bundesnetzagentur |
  SMARD.de, CC BY 4.0) and Open-Meteo (CC BY 4.0, free tier for
  non-commercial use). Energy-Charts rate-limits hard (HTTP 429): few requests,
  large chunks, backoff. Both sources must be credited in the README, the
  website and the integration's attribution attribute.
- **Hardware:** Raspberry Pi 4 class. A forecast run must finish in a few
  seconds and stay well under 100 MB extra memory. `numpy` is available from
  Home Assistant core; no other runtime requirement (`manifest.json`
  `requirements: []`).
- **Home Assistant:** minimum version 2024.11.0, as SBP.
- **Training:** monthly in GitHub Actions (Ubuntu runner, Python 3.12, CPU
  only); the same pipeline runs locally on the owner's machines.
- **Licence:** MIT.

## Identity

- Repository: `cygnusb/ha-spot-forecast` (this repo).
- Display name "Spot Forecast", domain `spot_forecast`. *(My call; easy to
  change before the first release, painful after.)*
- Integration type `service`, `iot_class: cloud_polling`.

## How it works

### Data at run time

| what | source | when |
|---|---|---|
| Day-ahead prices DE-LU, last 35 days through the latest published day | Energy-Charts `/price?bzn=DE-LU` | every run; one request |
| Weather forecast, hourly, 7 days, ECMWF IFS 0.25, at the 10 German and 12 neighbour points of `research/fetch.py` | Open-Meteo forecast API, `models=ecmwf_ifs025`, all points in one multi-location request | every run |
| Installed wind/solar capacity | shipped in the model file (last training value) | — |
| Public holidays DE | computed in the integration (fixed dates plus Easter offsets) | — |

Variables and aggregates are those of the backtest's default run: wind speed
100 m, shortwave radiation, temperature 2 m, cloud cover in Germany;
wind, radiation, temperature and precipitation at the neighbour points. The
same aggregates are used (wind power index, clear-sky × cloud solar estimate,
`nb_*` aggregates, daily aggregates). Lagged neighbour **prices** are not
used: the backtest showed they make it worse.

### The issue anchor

The model forecasts relative to an anchor day **X**: prices are known through
the end of X, and it predicts X+1 … X+6 (horizon h = 1 … 6). This is exactly
the backtest setup.

- Before the day-ahead result for tomorrow is published (around 12:45–13:15
  CET), X = today: forecast for tomorrow … today+6.
- After it is published, X = tomorrow: forecast for today+2 … today+7.

So the integration always shows known prices first and the forecast right
after them, with no gap and no overlap.

**Training/serving match:** in training, weather for target day X+h comes from
the archived run `previous_day{h+1}`, i.e. a run from day X−1. Live, with X =
tomorrow, the newest weather run is from today = X−1, so the setups match.
With X = today (morning), live weather is one day fresher than training
assumed; that only helps. Keep training on `previous_day{h+1}`.

### Schedule

- On start-up, and when the config entry is reloaded.
- Daily shortly after the day-ahead publication: poll Energy-Charts from
  13:00 CET every 15 min until tomorrow's prices appear (stop at 16:00),
  then run the forecast with the new anchor.
- After each new ECMWF run becomes available on Open-Meteo (about four times
  a day): rerun with fresher weather, same anchor.
- Requests stay around 10–20 per day in total.

### Model and inference

- Three gradient-boosted tree models per zone, as in the backtest: a median
  model (absolute-error loss), a P90 model (quantile loss 0.9) and a P10
  model (quantile loss 0.1). Target:
  price minus the last 7-day mean price, divided by the recent evening price
  level (median of 18–21 h prices over the last 14 days, at least
  50 €/MWh, ÷ 100); `--target gq` in `research/backtest.py`. Evening hours are
  when gas plants usually set the price, so this level tracks the fuel cost
  without a gas price feed. It lets spikes scale with it: in the backtest
  it cut the D+1 peak error from 7.5 to 7.2 ct and raised the P90 warning's
  recall slightly, with no loss in mean error.
- Training uses scikit-learn `HistGradientBoostingRegressor` offline. The
  trees are exported to a compact JSON file (gzip) in
  `custom_components/spot_forecast/models/de_lu.json.gz` with: feature list
  and order, tree arrays, categorical handling, training period, model
  version, backtest metrics, default P90 offsets per horizon, installed
  capacity, default peak threshold.
- Inference is a small tree walker over numpy arrays (or pure Python).
  **Acceptance:** for a fixed test set, the exported model reproduces
  scikit-learn's predictions to 1e-6.
- **One feature implementation:** feature engineering lives in one module
  without Home Assistant or pandas imports. The training pipeline imports it
  too. A golden test compares its output with `research/backtest.py`'s
  features for sample days.

### Peak risk calibration

The raw P90 model is under-dispersed (25 % exceedance out of sample). At run
time each horizon gets an offset: the 90 % quantile of `actual − raw P90` over
the trailing 56 target days, from the integration's own stored forecasts.
With fewer than 200 hours of history, use the offset shipped in the model
file. Target: about 10–12 % exceedance, matching the backtest (12 %).

Peak risk is reported as `max(P90, median)` so the band never dips below the
line.

The low-price value (P10) is calibrated the same way, mirrored: offset = the
10 % quantile of `actual − raw P10` over the trailing 56 target days, and it is
reported as `min(P10, median)`. In the backtest the calibrated P10 is
undercut in 10 % of hours. "P10 < 0" catches 84 % (D+1) … 80 % (D+6) of
negative-price hours, against 55 % … 40 % for "median < 0"
(`research/README.md`, "Low-price chance").

### Keeping the forecasts (look-back)

Every run's hourly forecast is kept in an HA `Store` until its target hour is
in the past and its actual price is known. Then, per completed target day,
the integration writes **external long-term statistics**, hourly rows, as
`ha-netze-bw` does:

| statistic_id | content |
|---|---|
| `spot_forecast:de_lu_price` | actual day-ahead price, hourly mean |
| `spot_forecast:de_lu_forecast_h1` … `_h6` | median forecast made with anchor = target day − h |
| `spot_forecast:de_lu_peak_risk_h1` … `_h6` | calibrated P90 of the same forecast |
| `spot_forecast:de_lu_low_h1` … `_h6` | calibrated P10 of the same forecast |

Unit EUR/kWh. If several runs share an anchor, the last run before the anchor
changes wins (the one a user would have seen last).

**Backfill on install:** without it the look-back card is empty for weeks and
the P90 calibration falls back to the shipped default. The backfill must be
**out of sample**: the shipped model was trained on everything up to its
training cutoff, so its own forecasts for those days would look better than
reality and would shrink the P90 offset towards zero (in sample the raw P90 is
exceeded ~10 % of the time, out of sample 25 %). *(My call.)* Two parts:

1. **Up to the training cutoff:** each release ships the walk-forward
   out-of-sample predictions from `training/` for the last 180 days before
   the cutoff (median and raw P90 per hour and horizon, gzip, a few hundred
   KB). The integration imports them on first setup.
2. **From the cutoff to today:** the shipped model has not seen these days, so
   the integration recomputes them on the device. It uses the Open-Meteo
   Previous Runs API (`previous_day{h+1}`, exactly as in training) and
   Energy-Charts prices. This runs once in the background with rate-limit
   pauses.

Default window 60 days, maximum 180. Backfilled rows carry no flag in the
statistics; the diagnostics show how many days came from backfill.

**Model updates:** forecasts already written stay as they are ("what the user
saw"); a new model only affects new runs. The P90 calibration uses the
stored forecasts across model versions. They were all out of sample when
made, which is what the offset has to reflect. *(My call.)*

### Failure behaviour

- A failed weather or price request keeps the last forecast. Attributes
  `issued_at` and `stale: true` make the age visible; after 36 h without a
  successful run the forecast part is dropped and only known prices remain.
- Known prices come from Energy-Charts. A key-free fallback for known DE-LU
  prices (for example aWATTar's public API) is the builder's call.
- Repairs issue if the active model's training cutoff is more than 60 days
  old (see "Releases and the model").

## Entities

All prices in EUR/kWh, the bare exchange price.

### `sensor.spot_forecast_de_lu_price`, the main sensor

- State: price of the current slot (known).
- Attribute `prices`: list from today 00:00 through the end of the forecast,
  in the format SBP's ENTSO-E adapter already detects:

  ```json
  {"time": "2026-09-22T18:00:00+02:00", "price": 0.2134, "p90": 0.3812, "p10": 0.1420, "source": "forecast", "h": 2}
  ```

  `p90`, `p10`, `source` (`"market"` | `"forecast"`) and `h` are additional keys that
  the adapter ignores. Known slots keep their native resolution (15 min since
  2025-10-01); forecast slots are hourly. Slot end is the next slot's start,
  which is how the adapter reads it.
- Attributes `unit: "EUR/kWh"`, `issued_at`, `anchor_day`, `model_version`,
  `stale`, `attribution`.
- `prices` is in `_unrecorded_attributes` (up to ~300 entries, above the
  recorder's 16 KB limit). History lives in the statistics instead.

### `binary_sensor.spot_forecast_de_lu_peak_risk`

- Meaning: "a price above the threshold is certain or likely in the next
  48 h". On if any slot from now to now + 48 h is above the threshold, using
  the known price for market slots and the peak risk (P90) for forecast
  slots. *(My call.)* This keeps the meaning stable across the day-ahead
  publication: when tomorrow's peak becomes certain, the sensor stays on
  instead of dropping out because the hour is no longer a forecast.
- Attributes: the flagged slots (time, price or median, p90, `source`), the
  threshold, and `certain: true` if at least one flagged slot is a market
  price.
- Threshold: configurable in the options flow. Default from the model file =
  the 97th percentile of hourly prices in the last training year (21.2 ct for
  the current backtest).

### `binary_sensor.spot_forecast_de_lu_negative_price`

- Meaning: "a negative price is certain or likely in the next 48 h". On if
  any slot from now to now + 48 h has a known price below 0 or a forecast
  low-price value (P10) below 0. It mirrors the peak-risk sensor.
- Attributes: the flagged slots (time, price or median, p10, `source`), and
  `certain: true` if at least one flagged slot is a market price.

### Diagnostic sensors

- Last successful forecast run (timestamp).
- Model version and training period.
- Forecast error, last 30 days: state = mean absolute error over all
  horizons; attributes = MAE per horizon and peak MAE, computed from the
  stored forecasts.
- Backfill status.

### Days for automations

Day-level sensors (forecast mean, min, max and peak risk per day for the next
seven days) make "is Thursday cheap?" automations easy. *(My call: include
them as disabled-by-default entities.)*

## Dashboard cards (Plotly Graph Card, variant A)

Chosen from three mock-ups built with real backtest data (variant A
"Verlauf"; the others were day tiles and a calendar heatmap).
Shipped as `examples/plotly_forecast.yaml` and `examples/plotly_lookback.yaml`
for `custom:plotly-graph`, with screenshots in light and dark mode in the
README and on the website, as `ha-netze-bw` does.

### Forecast card

- One continuous time axis: yesterday and today (known) and the forecast days.
- Known prices: solid ink step line. Forecast: blue step line (median).
- Peak risk: hatched band from median up to P90, in the risk colour.
- Low-price chance: a lighter hatched band from P10 up to the median, in the
  cheap colour, so likely negative hours show below the zero line.
- Per day, the 4 dearest (▲ orange) and 4 cheapest (▼ green) hours marked.
- Above each day: weekday and date, maximum, and "⚠ bis NN" in the risk
  colour if peak risk exceeds the threshold.
- Nights shaded from `sun.sun`; dotted "now" line with the label "jetzt, ab
  hier Prognose".
- Data from the main sensor's `prices` attribute via `$fn`; ct/kWh on the
  axis (×100), German separators.

### Look-back card

- Actual price (ink) against the forecast made h days before, from the
  statistics.
- Horizon selection D+1 … D+6 and "alle" (all lines, no fill). With one
  horizon, the deviation is filled red between forecast and actual.
- Range buttons 7 T, 30 T, 3 M, 6 M (rangeselector, as in `ha-netze-bw`).
- Above the plot: mean absolute error for the shown horizon, plus the error in
  hours above the peak threshold.
- In the mock-up, the error figure updates while zooming. Whether the Plotly
  Graph Card can do that, or only for the loaded window, the builder checks;
  the loaded window is acceptable. The same goes for `updatemenus` buttons
  versus legend clicks for the horizon switch: use whichever survives the
  card's refresh.

Colours: the validated categorical palette already used in `research/` and
`ha-netze-bw` (D+1 … D+6 = `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4
#008300`), risk `#d03b3b`, both cards readable in HA's light and dark themes.

## Using it from Smart Battery Pilot

Nothing to build in SBP for v1. A user selects
`sensor.spot_forecast_de_lu_price` as SBP's price entity; SBP's ENTSO-E
adapter detects `prices` with `time`/`price` and reads it.
The adapter test in SBP's suite should get a fixture with this exact format
(SBP-side change, later, owner's call).

Whether SBP should plan on forecast slots, or use `source` and `p90` (for
example to hold charge before a likely peak), is an SBP design question for
later. This design only guarantees the data is there.

## Repository layout

```
custom_components/spot_forecast/
  manifest.json, __init__.py, const.py, config_flow.py, coordinator.py
  sources/      energy_charts.py, open_meteo.py
  features.py   shared with training, no HA/pandas imports
  model.py      tree inference
  models/       de_lu.json.gz
  zones.py      per-zone points, model file, holidays
  history.py    store, statistics, backfill
  sensor.py, binary_sensor.py, diagnostics.py, repairs.py
  translations/ en.json, de.json
  brand/        icon.png, icon@2x.png, logo.png, logo@2x.png
training/       data fetch, train, export, backtest report (grows out of research/)
research/       kept as the record of the feasibility study
examples/       plotly_forecast.yaml, plotly_lookback.yaml, screenshots
site/, tools/   project website (as SBP)
docs/           MkDocs documentation (as SBP)
tests/          fast tests without HA, plus tests_ha/ against real HA
```

## Releases and the model

The model must be retrained regularly. A frozen model holds for 6–9 months in a
stable market. After the 2026 gas price shift it fell to the naive baseline
within two quarters and then below it (`research/README.md`, "How long does a
model last"). Most HACS users update rarely, so new models reach them
**separately from integration releases**. The owner decided on monthly
training in GitHub Actions.

### Model channel

- A rolling GitHub release with the tag `models` holds the current model
  files and a manifest at a stable URL:
  `https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json`.
- Manifest per zone: model version (`de_lu-2026-11`), file URL, SHA-256,
  `feature_schema` (integer), training cutoff, creation time, and the
  report's headline metrics. The out-of-sample backfill predictions for that
  model (see "Backfill on install") are listed alongside, with their own
  SHA-256.
- Older model files stay attached so a user can see which model produced
  their forecasts. The diagnostics name the version.

### In the integration

- **Bundled model:** each integration release contains the latest published
  model and its backfill file. It works right after install and without
  access to GitHub.
- **Automatic update:** once a day, at a random minute, the integration reads
  the manifest. It downloads a newer model only if three checks pass:
  `feature_schema` equals the integration's own, the SHA-256 matches, and the
  file parses and predicts on a built-in sample. Downloaded models are stored
  under the HA config directory (`.storage/spot_forecast/`), not in
  `custom_components/`, which HACS overwrites on update. The next forecast run
  uses the new model.
- Fallback order: the newest valid downloaded model, then the bundled model.
  A failed download or check keeps the current model and is logged once.
- Option "Update models automatically", on by default.
- The model file is data only: tree arrays and numbers. The integration never
  executes downloaded code.
- A diagnostic sensor shows the model version, source (bundled or downloaded)
  and training cutoff. A repairs issue appears when the active model's cutoff
  is more than 60 days old.
- `feature_schema` is bumped whenever `features.py` changes in a way that
  changes model inputs. A new schema needs an integration release. Until a
  user installs it, their integration keeps using the last model for its
  schema.

### Training workflow (`train.yml`)

- Runs monthly on a schedule (3rd of the month, night) and on manual
  dispatch.
- **Data:**
  - restore the raw-data archive;
  - fetch only what is new (incremental, with the 429 backoff of
    `research/fetch.py`);
  - store the archive again.

  The GitHub Actions cache evicts entries unused for 7 days, so a monthly job
  cannot rely on it. Keep the archive as an asset of a separate
  `training-data` release, or on a data branch *(builder's call)*.
- **Train:** the median and P90 models on all data through the end of the
  last month, then export, with the same code as `training/` locally.
- **Quality gate**, out of sample on both sides: retrain with data up to the
  month before last and evaluate on last month. Compare it with the
  previously published model on the same month. Publish only if the MAE is
  no more than 0.1 ct worse than the previous model and the skill against the
  naive baseline is positive. If the gate fails, the job opens a GitHub issue
  with the report and publishes nothing.
- **Publish:** upload the model, the backfill predictions and the report
  (Markdown, MAE per horizon, peak MAE, P90 exceedance, trailing 3 months)
  to the `models` release. Then update the manifest and append the report to
  the release notes.
- The training also runs locally (`training/` CLI) for development and for
  the bundled model of an integration release.

### Integration releases

- GitHub workflows as in SBP: `validate.yml` (hassfest, HACS action, pytest,
  HA test matrix at the minimum version and latest), `release.yml` (ZIP
  asset, `zip_release` in `hacs.json`), `pages.yml` (site and docs) and
  `train.yml`.
- `release.yml` copies the current published model and backfill file into
  the ZIP, so the bundled model is never older than the last training run
  before the release.

## README, website, discoverability

Modelled on `ha-smart-battery-pilot`:

- **README:** logo, badge row (release, downloads, HACS Custom, HA version,
  license, commit activity, validate, stars), one-line pitch, links to
  website, documentation and releases, forecast card screenshot (light and
  dark `<picture>`), "How it works", accuracy table with the peak caveat,
  installation via HACS with the my.home-assistant badge, configuration,
  entities, the two Plotly cards, using it with Smart Battery Pilot, data
  sources and licences, languages, licence.
- **Website:** GitHub Pages, `site/index.html.j2` + `tools/build_site.py` +
  `site/i18n/`, same visual language as the SBP site. Content: the forecast
  card as the hero; "how good is it" with an interactive look-back chart
  (variant A) built from the training report's backtest data; install steps;
  data sources. Languages: German and English. *(My call; SBP has ten, but
  the v1 audience is the DE-LU zone.)*
- **Docs:** MkDocs Material under `/docs/`, as SBP: configuration, entities,
  cards, how the model works, accuracy, SBP usage, changelog from releases.
- **GitHub topics:** `home-assistant`, `hacs`, `custom-integration`,
  `home-assistant-integration`, `electricity-price`, `price-forecast`,
  `day-ahead`, `spot-price`, `epex-spot`, `dynamic-pricing`,
  `energy-management`, `open-meteo`, `energy-charts`, `germany`,
  `forecasting`, `machine-learning`.
- Repository description: "Home Assistant integration: six-day day-ahead
  electricity price forecast for DE-LU from open data, with peak risk.
  Key-free, HACS-ready."
- HACS: start as a custom repository; apply for HACS default once the
  integration has run a few weeks live and the live error matches the
  backtest.

## Configuration

Config flow, one step: bidding zone (only DE-LU selectable in v1). Options:
peak threshold (ct/kWh), backfill window (days). Nothing else: issue times,
points and models come from the zone definition.

## Success criteria

- Error is strongly regime-dependent: 1.86–2.57 ct over the full backtest,
  but 2.8–3.8 ct in Aug–Oct 2026 with high gas prices. So the training report
  states both the full-period MAE and the trailing-3-month MAE per horizon,
  plus the skill against the naive "same as last week" baseline (1 − MAE /
  naive MAE), which depends much less on the regime. README and website show
  the trailing-3-month figures next to the full period. *(My call.)*
- Live check over the first 60 days, per horizon: MAE below the 4–5 ct
  benchmark, and skill against the naive baseline within 0.1 of the
  backtest's trailing-3-month skill.
- P90 exceedance live between 8 % and 15 %.
- Forecast run on a Raspberry Pi 4 under 5 s, extra memory under 100 MB.
- hassfest and HACS validation green; the HA test matrix green.
- SBP set to `sensor.spot_forecast_de_lu_price` builds a plan without errors
  (manual check).

## Left to the builder

- Tree export format details and whether inference is numpy or pure Python.
- Exact poll times around the day-ahead publication and ECMWF run times.
- Store layout, statistic metadata details, retention of the raw forecast
  store.
- Fallback source for known prices.
- How the Plotly cards read the attribute (`$fn` vs filters), as long as the
  look matches variant A.
- Test structure, beyond the two golden tests named above (features, model
  export).

## Calls I made (check these)

- Name and domain `spot_forecast` / "Spot Forecast".
- Anchor switch after the day-ahead publication, forecast up to today+7 in the
  afternoon.
- Backfill of 60 days by default, out of sample: shipped walk-forward
  predictions up to the training cutoff, recomputed on the device after it.
- Peak-risk binary sensor counts known prices and forecast P90 alike.
- Accuracy claims: full period and trailing 3 months side by side; live check
  on skill against the naive baseline.
- Main sensor merges known and forecast slots; no separate forecast-only
  sensor (the `source` key separates them).
- Peak threshold default = 97th percentile of the last training year.
- Day-level sensors included but disabled by default.
- Website in German and English only.
- Target scaled by the recent evening price level; no gas price feed in v1.
- P10 low-price value and a negative-price binary sensor, mirroring P90 and
  the peak-risk sensor.
- Model files in a rolling `models` release with a manifest; daily check, schema +
  SHA-256 + sample-prediction checks; downloaded models live in `.storage/`.
- Model older than 60 days → repairs issue.
