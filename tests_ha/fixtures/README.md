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

## Model

`model/de_lu.json.gz` and `model/de_lu.backfill.json.gz` are copies of the bundled model
`de_lu-2026-10` (training cutoff 2026-10-01, feature schema 2) and its backfill file (target
days 2026-04-04 … 2026-09-30), trained on 2026-10-23 with `python -m training.train --raw
research/data/raw --out dist --cutoff 2026-10-01 --walk-forward-from 2025-04-01`. `conftest` points `model_loader.MODELS_DIR` at them for every
test, so a refreshed model in `custom_components/spot_forecast/models/` does not move the
dates the tests depend on.
