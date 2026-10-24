# How the model works

## The anchor day

The forecast is made relative to an **anchor day** X: prices are known through
the end of X, and the model predicts X+1 … X+6 (horizon *h* = 1 … 6).

* Before tomorrow's day-ahead result is published (around 12:45–13:15 CET),
  X is today, and the forecast covers tomorrow to today + 6.
* After it is published, X is tomorrow, and the forecast covers today + 2 to
  today + 7.

So the price sensor always shows the known prices first and the forecast
right after them. The anchor is in the sensor's `anchor_day` attribute.

## Data at run time

| What | Source | When |
|---|---|---|
| Day-ahead prices of the zone, the last 40 days | [Energy-Charts](https://www.energy-charts.info/) `/price?bzn=DE-LU` (or `NL`) | every run, one request |
| Weather forecast, hourly, 8 days, ECMWF IFS 0.25° | [Open-Meteo](https://open-meteo.com/), one multi-location request per set of points with the same variables (DE-LU: German points, neighbour points; NL: three) | every run |
| Installed wind and solar capacity (DE-LU) | the model file (last training value) | |
| Public holidays of the zone's country | computed in the integration | |

Every weather point has one set of variables, whatever zones use it. **DE-LU**
reads 10 points in Germany (wind at 100 m, solar radiation, temperature at 2 m,
cloud cover) and 12 points in the Netherlands, Belgium, France, Denmark,
Poland, the Alps, Norway, Sweden and Finland (wind, radiation, temperature,
precipitation). From these come a wind-power index, a solar estimate (clear
sky × cloud cover), neighbour aggregates and daily aggregates. Neighbour prices
are not used: in the backtest they made the forecast worse.

**NL** uses the zone-generic layout of the multi-zone study
(`research/zones/genfeat.py`), which differs from DE-LU's only in where the
points are and which aggregates they feed:

| Role | Points |
|---|---|
| Own | `nb_nl` (Noord-Holland coast), `nl_borssele` (offshore wind), `north_sea` (German Bight), `nl_flevoland`, `nl_groningen`, `nl_randstad`, `nl_brabant` |
| Wind index | `nb_nl`, `nl_borssele`, `north_sea`, `nl_flevoland`, `nl_groningen` |
| Solar estimate | `nl_flevoland`, `nl_groningen`, `nl_randstad`, `nl_brabant` |
| Neighbours | `ostfriesland`, `nrw`, `nb_be`, `nb_fr_north`, `nb_dk_west`, `schleswig`, `nb_no_south` |
| Neighbour wind | `ostfriesland`, `nb_be`, `nb_dk_west`, `schleswig` |
| Neighbour hydro (precipitation) | `nb_no_south` |

The five `nl_*` points have wind, temperature and cloud cover. The features are
each own point's variables, the wind mean and wind-power index, the temperature
and cloud mean, the solar estimate, the neighbours' wind-power index,
temperature mean and hydro precipitation, and their daily values, plus the
calendar (Dutch public holidays) and the same price features as DE-LU. There
is no installed-capacity feature. The model file's feature list is checked
against its zone's layout, so a model is never applied to another zone.

## Three tree models

Per zone, three gradient-boosted tree models, trained with scikit-learn's
`HistGradientBoostingRegressor`:

* the **median** model (absolute-error loss): the expected price;
* a **P90** model (quantile loss 0.9): the peak risk;
* a **P10** model (quantile loss 0.1): the low-price value.

The target is the price minus its mean over the last 7 days, divided by the
recent evening price level (the median of the 18–21 h prices over the last
14 days, at least 50 EUR/MWh). Evening hours are when gas plants usually set
the price, so this level follows the fuel cost without a gas price feed, and
it lets price spikes grow with it.

The price features are in the same units: the anchor day's price at the
target hour, the price of that hour a week before, and the anchor day's mean,
minimum and maximum with their 7- and 30-day windows are given as their
distance from the 7-day mean, divided by the evening level; the spreads
(standard deviations) are divided by it. So when prices rise, the trees do not
meet price levels they have never seen. In the research backtest this cut the
error by 0.15 ct in the year of the gas price rise and halved the bias, and it
changed nothing in the calm summer of 2025. The two price lags take the same
local clock hour, also across a daylight-saving switch: 18:00 a week before
stays 18:00, not 17:00 or 19:00.

The trees are exported to a compact JSON file per zone
(`custom_components/spot_forecast/models/de_lu.json.gz`, `nl.json.gz`): the feature list and
order, the tree arrays, training period, version, backtest metrics, default
calibration offsets, installed capacity and default peak threshold. On the
device a small tree walker over numpy arrays evaluates them; it reproduces
scikit-learn's predictions to 1e-6. Feature engineering is one module without
Home Assistant or pandas, shared by the integration and the training. A
forecast run needs nothing but numpy (no pandas, no scikit-learn). On a
Raspberry Pi 5 a full run, downloads included, takes about 1.2 seconds.

## Calibration

A raw P90 model is under-dispersed out of sample: in the backtest the actual
price exceeded it in 25 % of hours, not 10 %. So each horizon gets an offset:
the 90 % quantile of `actual − raw P90` over the last 56 target days, from the
integration's own stored forecasts. With fewer than 200 hours of history, the
offset shipped in the model file is used. The peak risk is reported as
`max(P90, median)`, so the band never dips below the line. The target is about
10–12 % exceedance.

The low-price value is calibrated the same way, mirrored: the 10 % quantile of
`actual − raw P10`, reported as `min(P10, median)`.

The [backfill](configuration.md#first-start-and-the-backfill) gives the
calibration its history on install. It must be out of sample: the shipped
model's own forecasts for days it was trained on would look better than
reality and would shrink the offsets towards zero.

## Model updates

A frozen model holds for six to nine months in a stable market. When the
market shifts, as with the gas price rise in 2026, it falls to the naive
baseline within about two quarters. So the models are retrained every month,
and they reach your install **separately from integration releases**:

* A GitHub Actions workflow trains every zone, one after the other, on all
  data through the end of the last month on the 3rd of each month.
* A **quality gate** compares each zone's model out of sample with the zone's
  published model on the last month. It is published only if its error is no
  more than 0.1 ct worse and it beats the naive baseline; otherwise the
  workflow opens an issue and publishes nothing for that zone. The other zones
  are published all the same.
* Published models go to the rolling
  [`models` release](https://github.com/cygnusb/ha-spot-forecast/releases/tag/models),
  with a manifest listing version, file, SHA-256, feature schema, training
  cutoff and headline metrics, and the model's backfill file.
* Five minutes after setup, and then once a day at a fixed time between 03:00
  and 05:59 that differs from install to install, the integration reads the
  manifest. It downloads a newer model only if the feature schema matches its own, the
  SHA-256 matches, and the file loads and predicts on a built-in sample. The
  file is data only; the integration never runs downloaded code.
* Downloaded models live in `<config>/.storage/spot_forecast/`, so a HACS
  update does not remove them. The integration uses the newest model by
  training data among the bundled one and the valid downloaded ones; a
  download counts only if it is newer than the bundled model.
  `sensor.spot_forecast_de_lu_model_version` shows which one runs (`source`:
  `bundled` or `downloaded`).
* An integration update that changes the features (a new feature schema)
  cannot use the models downloaded before: they are deleted, and the bundled
  model runs until the first model of the new schema is published. An older
  integration skips models of a newer schema and keeps its own.

Each integration release bundles the latest published model, so a fresh
install works without GitHub. Forecasts already written stay as they were;
a new model only affects new runs.

Turning the option *Update models automatically* off stops further downloads
only. A model downloaded before stays in use until an integration update
bundles a newer one. A repairs issue appears once the training data of the
model in use is more than 60 days old.

The training code is in
[`training/`](https://github.com/cygnusb/ha-spot-forecast/tree/main/training),
the feasibility study it grew out of in
[`research/`](https://github.com/cygnusb/ha-spot-forecast/tree/main/research).
