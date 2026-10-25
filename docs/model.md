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
| Day-ahead prices of the zone, the last 40 days | [Energy-Charts](https://www.energy-charts.info/) `/price?bzn=DE-LU` (or `NL`, `DK1`, `DK2`, `NO2`; DK1 and DK2 also read DE-LU's) | every run, one request (DK: two, one after the other) |
| Weather forecast, hourly, 8 days, ECMWF IFS 0.25° | [Open-Meteo](https://open-meteo.com/), one multi-location request per set of points with the same variables (DE-LU: German points, neighbour points; NL, NO2: three; DK1, DK2: four) | every run |
| NO2 only: precipitation and temperature of the hydro points, the last 92 days | Open-Meteo previous runs, the forecasts issued two days before each hour | every run, one request |
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

## DK1 and DK2: blended with DE-LU

Denmark's two zones are tightly coupled to Germany, and in the research
backtest neither an own model nor the DE-LU forecast used as is was best for
them: the 50/50 blend of both was (`research/zones/README.md`, section "DK1
and DK2"). So a DK forecast is the average of two models:

* the zone's **own model**: the zone-generic layout with the zone's points
  (below), plus inputs from DE-LU: the raw series of the ten German points,
  the German wind-power index, solar estimate and temperature mean with their
  daily values, and six DE-LU price features (the anchor day's DE-LU price at
  the target hour and a week before, DE-LU's daily mean, 7-day mean, maximum
  and evening level). The DE-LU prices are given relative to the DK zone's own
  7-day mean in its own evening-level units, like the zone's own price
  features;
* the **DE-LU model** itself, run on DE-LU's prices and points as for a DE-LU
  entry.

The median, P90 and P10 are each `0.5 × own + 0.5 × DE-LU`, hour by hour; an
hour is valid only when both sides are. The DK model file carries the complete
DE-LU model it blends with (a *composite* model), so a DK install always
blends with the DE-LU model it was trained and calibrated with, whatever model
a DE-LU entry next to it runs. The calibration offsets and the backtest
figures in the file are those of the blend. A DK model file without its DE-LU
model, with one of another zone or of another feature schema does not load.

| Role | DK1 (West Denmark) | DK2 (East Denmark) |
|---|---|---|
| Own | `nb_dk_west`, `dk1_horns_rev` (offshore), `dk1_north_jutland`, `dk1_aarhus`, `dk1_south_jutland` | `dk2_copenhagen`, `dk2_kriegers_flak` (offshore), `dk2_lolland`, `dk2_bornholm` |
| Wind index | all own points | all own points |
| Solar estimate | `dk1_north_jutland`, `dk1_aarhus`, `dk1_south_jutland` | `dk2_copenhagen`, `dk2_lolland` |
| Neighbours | `schleswig`, `north_sea`, `ostfriesland`, `nb_nl`, `nb_no_south`, `no2_sirdal`, `se3_gothenburg`, `dk2_copenhagen` | `baltic`, `nb_se_south`, `se4_malmo`, `dk1_aarhus`, `se3_gothenburg`, `nb_no_south`, `nb_se_north` |
| Neighbour wind | `schleswig`, `north_sea`, `ostfriesland`, `nb_nl`, `dk2_copenhagen` | `baltic`, `nb_se_south`, `se4_malmo`, `dk1_aarhus` |
| Neighbour hydro (precipitation) | `nb_no_south`, `no2_sirdal` | `nb_no_south`, `nb_se_north` |

The `dk*` points and `se4_malmo` have wind, temperature and cloud cover,
`no2_sirdal` and `se3_gothenburg` wind, temperature and precipitation. A DK run
fetches the weather of all its own and neighbour points and of every DE-LU
point (the DE-LU model needs them all). The calendar is the Danish one plus
Christmas Eve and New Year's Eve, which are days off in practice; Store
bededag counts up to 2023, when it was abolished.

## NO2: hydro features

Southern Norway's prices follow its hydro reservoirs more than the weather of
the next days, so NO2 has its own model with the zone-generic layout, its own
and neighbour points (below) and six **hydro features**, from the research
variant `hydrel` (`research/zones/README.md`):

| Feature | What |
|---|---|
| `hyd_precip_past14`, `_past30`, `_past90` | precipitation summed over the six hydro points in the 14, 30 and 90 days before the end of the anchor day |
| `hyd_temp_past14`, `_past30` | mean temperature of the six hydro points (snow or melt) in the 14 and 30 days before |
| `hyd_precip_fc6` | precipitation of the six hydro points in the forecast over all six target days |

The trailing values come from the archived ECMWF forecasts issued two days
before each hour (Open-Meteo previous runs, `_previous_day2`), the same series
the model was trained on: near-actual weather without a separate observation
source. The windows count real hours back from the end of the anchor day
(2160 for 90 days), whatever the clock changes in between. The hydro features
stay absolute values. NO2 has no day-of-year feature: next to the hydro
features it made the backtest worse (2.33 against 2.27 ct/kWh in the quarterly
research run, `hydro` against `hydnd`).

| Role | Points |
|---|---|
| Own | `nb_no_south` (Telemark), `no2_kristiansand`, `no2_stavanger`, `no2_sirdal` |
| Wind index | `no2_kristiansand`, `no2_stavanger` |
| Solar estimate | none |
| Own hydro (precipitation) | `nb_no_south`, `no2_sirdal`, `no2_stavanger` |
| Neighbours | `nb_dk_west`, `north_sea`, `schleswig`, `nb_nl`, `no5_bergen`, `no5_sogn`, `no1_oslo`, `no1_hallingdal` |
| Neighbour wind | `nb_dk_west`, `north_sea`, `schleswig`, `nb_nl` |
| Neighbour hydro (precipitation) | `no5_sogn`, `no1_hallingdal`, `no5_bergen` |

The six hydro points are the own and the neighbour hydro points. Each NO2 run
makes one more Open-Meteo request than the other zones: the two
`_previous_day2` columns of the hydro points for the last 92 days and the next
two (about 0.4 MB). If any hour of the 90 days is missing, the run fails like
any source error and is tried again; it never forecasts from sums that a hole
made smaller. The backfill fetches the same history from 91 days before its
first recomputed anchor, one request per month, and skips an anchor whose 90
days have a hole. The calendar is the Norwegian one plus Christmas Eve and New
Year's Eve. Prices and forecasts are in EUR, as Energy-Charts publishes them
for every zone; there is no conversion to NOK.

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
(`custom_components/spot_forecast/models/de_lu.json.gz`, `nl.json.gz`, `dk1.json.gz`,
`dk2.json.gz`): the feature list and
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
  data through the end of the last month on the 3rd of each month. DE-LU
  comes first: DK1 and DK2 embed the DE-LU model of the same run (also when
  DE-LU's own gate fails) and are skipped that month if DE-LU's training
  errors.
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
