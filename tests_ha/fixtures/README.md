# Recorded API payloads

Recorded with one live call each on 2026-10-22 at about 20:50 CEST (18:50 UTC), the
date the HA tests freeze to. Values are the real API responses; nothing was rounded.

| File | Request | Content |
|---|---|---|
| `energy_charts_price_tomorrow.json` | `GET https://api.energy-charts.info/price?bzn=DE-LU&start=2026-10-19&end=2026-10-23` | unchanged response: 480 quarter-hour slots, 2026-10-19 00:00 to 2026-10-23 23:45 Europe/Berlin (tomorrow's day-ahead was already published) |
| `energy_charts_price.json` | same | the response above with the 96 slots of 2026-10-23 (local) removed from `unix_seconds` and `price`: 384 slots, 2026-10-19 00:00 to 2026-10-22 23:45 Europe/Berlin, i.e. the state before tomorrow's auction is published |
| `open_meteo_de.json` | `GET https://api.open-meteo.com/v1/forecast` with the DE points (query below) | unchanged response: list of 10 location documents, 2026-10-22T00:00 to 2026-10-29T23:00 GMT |
| `open_meteo_nb.json` | same URL with the neighbour points | unchanged response: list of 12 location documents, same time range |
| `open_meteo_prev_de.json` | `GET https://previous-runs-api.open-meteo.com/v1/forecast` with the DE points | unchanged response: list of 10 location documents, 2026-10-01T00:00 to 2026-10-22T23:00 GMT, all 24 `{var}_previous_day{2..7}` columns |
| `open_meteo_prev_nb.json` | same URL with the neighbour points, recorded at about 21:50 CEST (19:50 UTC) | unchanged response: list of 12 location documents, same time range and column layout |

**Synthesized:** nothing. The Energy-Charts recording already contained 2026-10-23, so no
rows were copied forward, and the Open-Meteo forecasts were recorded on 2026-10-22, so they
already start at 2026-10-22 00:00 GMT and no timestamps were shifted. The only edit is the
removal of 2026-10-23 from `energy_charts_price.json`.

The HA tests synthesize more in code, never in these files: `conftest.Sources` serves the
prices with 40 days of history in front (the four recorded days tiled backwards, see
`_with_history`, since the integration needs 30 days) and can override single quarter hours
(`price_overrides`, e.g. one negative price). The backfill tests need prices back to
about 2026-08-25 (30 days of history before the first recomputed anchor), so the history
has 80 days (`HISTORY_BLOCKS = 20`, from 2026-07-31, all CEST).

The previous-runs recordings are served whole, whatever `start_date`/`end_date` a request
asks for. They cover the recompute span of the backfill tests (target days 2026-10-01 …
2026-10-22) and nothing was shifted or synthesized: target hours outside it (the first two
hours of local 2026-10-01, which are 2026-09-30 GMT, and the days after 2026-10-22) simply
have no weather, so those forecast slots are invalid and write no statistics.

The Energy-Charts API reads date-only `start`/`end` as whole market days in local time,
`end` inclusive.

Open-Meteo answers a multi-location request with a list in request order. The documents
carry grid-snapped coordinates (54.25/6.5 for `north_sea` at 54.2/6.6), so they are
matched to points by position, never by coordinates. Point order is that of
`core.zones.DE_LU.points` / `.neighbor_points`.

Query strings (`%2C` is a comma):

```
# open_meteo_de.json
latitude=54.2,54.7,54.5,53.3,53.2,51.9,51.4,50.4,48.7,48.4
longitude=6.6,13.8,9.2,7.6,13.8,11.6,7.3,9.3,9.2,11.8
hourly=wind_speed_100m,shortwave_radiation,temperature_2m,cloud_cover
models=ecmwf_ifs025  forecast_days=8  timezone=GMT

# open_meteo_nb.json
latitude=52.6,50.8,49.0,45.5,56.0,54.2,46.6,47.3,59.5,56.5,63.5,61.0
longitude=4.6,4.4,2.4,4.8,8.5,17.5,8.0,12.5,7.5,14.5,16.0,25.0
hourly=wind_speed_100m,shortwave_radiation,temperature_2m,precipitation
models=ecmwf_ifs025  forecast_days=8  timezone=GMT

# open_meteo_prev_de.json
latitude, longitude as for open_meteo_de.json
hourly={wind_speed_100m,shortwave_radiation,temperature_2m,cloud_cover}_previous_day{2..7}
       (variable-major: wind_speed_100m_previous_day2, ..._day3, ..., cloud_cover_previous_day7)
models=ecmwf_ifs025  start_date=2026-10-01  end_date=2026-10-22  timezone=GMT

# open_meteo_prev_nb.json
latitude, longitude as for open_meteo_nb.json
hourly={wind_speed_100m,shortwave_radiation,temperature_2m,precipitation}_previous_day{2..7}
       (variable-major, as above)
models=ecmwf_ifs025  start_date=2026-10-01  end_date=2026-10-22  timezone=GMT
```

## NL

Recorded with one live call each on 2026-10-23 at about 22:50 CEST (20:50 UTC), a day
after the DE-LU recordings, for the NL tests (`test_zones.py`), which freeze to the same
2026-10-22 10:00. NL has three Open-Meteo point sets (`core.zones.NL.point_sets()`):
`nb` (neighbour variables: `nb_nl`, `nb_be`, `nb_fr_north`, `nb_dk_west`, `nb_no_south`),
`nl` (`wind_speed_100m,temperature_2m,cloud_cover`: the five `nl_*` points) and `de`
(DE-LU variables: `north_sea`, `ostfriesland`, `nrw`, `schleswig`).

| File | Request | Content |
|---|---|---|
| `energy_charts_price_nl_tomorrow.json` | `GET https://api.energy-charts.info/price?bzn=NL&start=2026-10-19&end=2026-10-23` | unchanged response: 480 quarter-hour slots, 2026-10-19 00:00 to 2026-10-23 23:45 Europe/Amsterdam |
| `energy_charts_price_nl.json` | same | the response above without the 96 slots of 2026-10-23 (local): 384 slots, the state before tomorrow's auction |
| `open_meteo_nl_{nb,nl,de}.json` | `GET https://api.open-meteo.com/v1/forecast` with the point set's points and variables, `start_date=2026-10-22`, `end_date=2026-10-29` | unchanged responses: 5, 5 and 4 location documents, 2026-10-22T00:00 to 2026-10-29T23:00 GMT |
| `open_meteo_prev_nl_{nb,nl,de}.json` | `GET https://previous-runs-api.open-meteo.com/v1/forecast` with the point set, `start_date=2026-10-01`, `end_date=2026-10-22` | unchanged responses, 2026-10-01T00:00 to 2026-10-22T23:00 GMT, all `{var}_previous_day{2..7}` columns |

**Not as of 2026-10-22:** the forecasts were recorded a day later with an explicit
`start_date=2026-10-22` instead of `forecast_days=8`, so they cover the same hours as the
DE-LU ones; their 2026-10-22 values are what Open-Meteo had for that past day on the 23rd,
not the forecast of the 22nd. Nothing was shifted or synthesized; the only edit is the
removal of 2026-10-23 from `energy_charts_price_nl.json`. `conftest.Sources` serves the NL
prices by the request's `bzn`, with the same tiled history as DE-LU's.

## DK1 and DK2

Recorded with one live call each on 2026-10-24 at about 07:40 CEST (05:40 UTC), for the DK
tests (`test_dk.py`), which freeze to the same 2026-10-22 10:00. DK1 and DK2 have four
Open-Meteo point sets each (`core.zones.DK1.point_sets()`, `DK2.point_sets()`): one of
neighbour variables and one of DE-LU variables, which hold DE-LU points only, one of
`wind_speed_100m,temperature_2m,cloud_cover` (`s`: the `dk1_*` / `dk2_*` points, `se4_malmo`
and the other zone's point) and one of `wind_speed_100m,temperature_2m,precipitation` (`n`:
`no2_sirdal`, `se3_gothenburg`).

| File | Request | Content |
|---|---|---|
| `energy_charts_price_dk{1,2}_tomorrow.json` | `GET https://api.energy-charts.info/price?bzn=DK1&start=2026-10-19&end=2026-10-23` (`DK2` alike) | unchanged responses: 480 quarter-hour slots, 2026-10-19 00:00 to 2026-10-23 23:45 Europe/Copenhagen; `license_info` "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur \| SMARD.de" |
| `energy_charts_price_dk{1,2}.json` | same | the responses above without the 96 slots of 2026-10-23 (local): 384 slots, the state before tomorrow's auction |
| `open_meteo_dk{1,2}_{s,n}.json` | `GET https://api.open-meteo.com/v1/forecast` with the point set's points and variables, `start_date=2026-10-22`, `end_date=2026-10-29` | unchanged responses: DK1 5 and 2, DK2 6 and 1 location documents, 2026-10-22T00:00 to 2026-10-29T23:00 GMT |
| `open_meteo_prev_dk{1,2}_{s,n}.json` | `GET https://previous-runs-api.open-meteo.com/v1/forecast` with the point set, `start_date=2026-10-01`, `end_date=2026-10-22` | unchanged responses, 2026-10-01T00:00 to 2026-10-22T23:00 GMT, all `{var}_previous_day{2..7}` columns |

**Synthesized in code, not in these files:** the two point sets of DE-LU points are served
from the DE-LU recordings (`open_meteo_de.json`, `open_meteo_nb.json` and their `prev`
files), their location documents reordered to the set's point order
(`conftest.dk_weather`): the same points and variables, a different request order. As for
NL, the forecasts were recorded later with an explicit `start_date=2026-10-22`, so their
2026-10-22 values are what Open-Meteo had for that past day. The DE-LU prices a DK entry
reads are DE-LU's recording. The only edit to the files is the removal of 2026-10-23 from
`energy_charts_price_dk{1,2}.json`.

## Model

`model/de_lu.json.gz` and `model/de_lu.backfill.json.gz` are copies of the bundled model
`de_lu-2026-10` (training cutoff 2026-10-01, feature schema 2) and its backfill file (target
days 2026-04-04 … 2026-09-30), trained on 2026-10-23 with `python -m training.train --raw
research/data/raw --out dist --cutoff 2026-10-01 --walk-forward-from 2025-04-01`. `conftest` points `model_loader.MODELS_DIR` at them for every
test, so a refreshed model in `custom_components/spot_forecast/models/` does not move the
dates the tests depend on.

`model/nl.json.gz` and `model/nl.backfill.json.gz` are copies of the bundled model
`nl-2026-10` (training cutoff 2026-10-01, feature schema 2, the `generic` layout) and its
backfill file (target days 2026-04-04 … 2026-09-30, Europe/Amsterdam), trained on 2026-10-23
with `python -m training.train --zone NL --raw <raw> --out dist --cutoff 2026-10-01
--walk-forward-from 2025-04-01`, where `<raw>` held the NL prices and the 14 NL weather
points of `research/data/zones/` in the training layout.

`model/dk1.json.gz`, `model/dk2.json.gz` and their backfill files are copies of the bundled
composite models `dk1-2026-10` and `dk2-2026-10` (training cutoff 2026-10-01, feature schema
2, the `generic` layout with DE-LU inputs, blended 50/50 with the embedded `de_lu-2026-10`)
and their backfill files (target days 2026-04-04 … 2026-09-30, Europe/Copenhagen), trained on
2026-10-24 with `python -m training.train --zone DK1 --raw <raw> --out dist --cutoff
2026-10-01 --walk-forward-from 2025-04-01 --parent-model models/de_lu.json.gz
--parent-predictions <de_lu-2026-10.predictions.json.gz>` (DK2 alike), where `<raw>` held the
DE-LU, DK1 and DK2 prices, the installed capacity and the 33 weather points of the two zones
and of DE-LU, and the predictions file came from retraining `de_lu-2026-10` the same way as
above (its model file came out the same as the committed one, apart from `created`).
