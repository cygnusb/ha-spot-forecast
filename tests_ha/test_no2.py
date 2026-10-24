"""NO2: the generic layout with the hydro features, which need the hydro points' history.

NO2 is served from its own price recording (``conftest.NO2_PRICES``), its weather by point set
(``conftest.dk_weather``) and its hydro history from the recordings ``conftest.NO2_HYDRO``
(the live ``past_days`` request) and ``NO2_HYDRO_RANGE`` (the backfill's date ranges). It
loads its bundled model from ``fixtures/model/no2.*``.
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta

from conftest import (
    FORECAST_URL,
    NO2_HYDRO,
    NO2_HYDRO_LATITUDES,
    NO2_HYDRO_RANGE,
    PREVIOUS_RUNS_URL,
    T10,
    SetUp,
    Sources,
    advance,
    stat_rows,
)
from freezegun.api import FrozenDateTimeFactory
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
import pytest

from custom_components.spot_forecast.const import CONF_BACKFILL_DAYS, CONF_SURCHARGE, DOMAIN
from custom_components.spot_forecast.core.features import feature_names
from custom_components.spot_forecast.core.zones import NO2

NO2_PRICE = "sensor.spot_forecast_no2_price"
NO2_TOTAL = "sensor.spot_forecast_no2_total_price"
NO2_MODEL = "sensor.spot_forecast_no2_model_version"
DE_PRICE = "sensor.spot_forecast_de_lu_price"
T14 = T10 + timedelta(hours=4)
TOMORROW = "energy_charts_price_no2_tomorrow.json"
HYDRO_COLUMNS = "precipitation_previous_day2,temperature_2m_previous_day2"


def _market_price(recorded, at: datetime) -> float:
    """The recorded NO2 quarter-hour price at ``at`` in EUR/kWh."""
    doc = recorded("energy_charts_price_no2.json")
    stamp = int(at.replace(minute=at.minute // 15 * 15).timestamp())
    return round(doc["price"][doc["unix_seconds"].index(stamp)] / 1000, 5)


def _requests(sources: Sources, url: str) -> list:
    return [u for _, u, _, _ in sources._mock.mock_calls if str(u).startswith(url)]


def _hydro_requests(sources: Sources) -> list:
    return [u for u in _requests(sources, PREVIOUS_RUNS_URL) if u.query["hourly"] == HYDRO_COLUMNS]


async def test_no2_entry_sets_up_and_forecasts(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up(zone="NO2")
    model = entry.runtime_data.model
    assert (model.zone, model.features, model.capacity) == ("no2", feature_names(NO2), {})
    assert model.parent is None and model.version == "no2-2026-10"

    registry = er.async_get(hass)
    for key, entity_id in (("price", NO2_PRICE), ("model_version", NO2_MODEL)):
        assert registry.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}") == (
            entity_id
        )
    state = hass.states.get(NO2_PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert float(state.state) == _market_price(recorded, T10)
    attrs = state.attributes
    assert attrs["anchor_day"] == "2026-10-22"
    assert attrs["attribution"].startswith("Prices: Bundesnetzagentur | SMARD.de")
    forecast = [e for e in attrs["prices"] if e["source"] == "forecast"]
    assert {e["h"] for e in forecast} == set(range(1, 7))
    assert forecast[0]["time"] == "2026-10-23T00:00:00+02:00"  # Europe/Oslo
    assert all(e["p10"] <= e["price"] <= e["p90"] for e in forecast)
    assert all(-0.5 < e["price"] < 1.0 for e in forecast)  # plausible EUR/kWh

    # NO2's three point sets, its own prices only, and one hydro history request
    latitudes = {u.query["latitude"]: u.query["hourly"] for u in _requests(sources, FORECAST_URL)}
    assert latitudes == {
        ",".join(str(lat) for lat, _ in points.values()): ",".join(variables)
        for points, variables in NO2.point_sets()
    }
    assert len(latitudes) == 3
    zones = [u.query["bzn"] for u in _requests(sources, "https://api.energy-charts.info")]
    assert zones == ["NO2"]
    (hydro,) = _requests(sources, PREVIOUS_RUNS_URL)
    assert dict(hydro.query) == {
        "latitude": NO2_HYDRO_LATITUDES,
        "longitude": "7.5,6.9,5.8,7.0,9.0,5.3",
        "hourly": HYDRO_COLUMNS,
        "models": "ecmwf_ifs025",
        "timezone": "GMT",
        "past_days": "92",
        "forecast_days": "2",
    }


async def test_fresh_no2_entry_charges_norwegian_vat(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_SURCHARGE: 0.15}, zone="NO2")
    assert entry.runtime_data.coordinator.vat == 25.0
    total = hass.states.get(NO2_TOTAL)
    assert float(total.state) == round(_market_price(recorded, T10) * 1.25 + 0.15, 5)


def _with_hole(recorded, point: str, var: str, hour: str) -> list:
    """The live hydro recording with the value of ``point``'s ``var`` at ``hour`` removed."""
    docs = copy.deepcopy(recorded(NO2_HYDRO))
    doc = docs[NO2.hydro_points.index(point)]
    doc["hourly"][f"{var}_previous_day2"][doc["hourly"]["time"].index(hour)] = None
    return docs


async def test_a_hole_in_the_hydro_history_fails_the_run_and_is_retried(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    recorded,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One missing hour, 38 days back: no forecast from lowered sums, a source error that
    names NO2 and the series, the last forecast stays, and the retry forecasts."""
    hass = hass_with_recorder
    entry = await set_up(zone="NO2")
    before = hass.states.get(NO2_PRICE).attributes

    hole = _with_hole(recorded, "no5_sogn", "precipitation", "2026-09-15T12:00")
    sources.serve(no2_prices=TOMORROW, hydro=hole)
    await advance(hass, freezer, T14)

    state = hass.states.get(NO2_PRICE)
    assert state.attributes["stale"] is True
    assert state.attributes["anchor_day"] == "2026-10-22"
    assert state.attributes["issued_at"] == before["issued_at"]
    assert date(2026, 10, 23) not in entry.runtime_data.coordinator.ledger.runs
    assert "NO2: the Open-Meteo hydro history misses hours" in caplog.text
    assert "no5_sogn__precipitation" in caplog.text
    market = [e for e in state.attributes["prices"] if e["source"] == "market"]
    assert market[-1]["time"] == "2026-10-23T23:45:00+02:00"  # the fresh prices are kept

    sources.serve(no2_prices=TOMORROW)
    await advance(hass, freezer, T14 + timedelta(minutes=20))  # the retry
    attrs = hass.states.get(NO2_PRICE).attributes
    assert attrs["stale"] is False and attrs["anchor_day"] == "2026-10-23"
    assert len(_hydro_requests(sources)) == 1


async def test_a_hole_outside_the_90_days_does_not_matter(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    # the window of anchor 2026-10-22 starts 2026-07-24 22:00 UTC
    sources.serve(hydro=_with_hole(recorded, "no5_sogn", "precipitation", "2026-07-24T21:00"))
    await set_up(zone="NO2")
    attrs = hass.states.get(NO2_PRICE).attributes
    assert attrs["stale"] is False and attrs["anchor_day"] == "2026-10-22"


async def test_no2_next_to_de_lu_only_no2_fetches_the_hydro_history(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    hass = hass_with_recorder
    de = await set_up()
    assert _requests(sources, PREVIOUS_RUNS_URL) == []
    no2 = await set_up(zone="NO2")
    assert len(_hydro_requests(sources)) == 1
    assert de.runtime_data.model.version == "de_lu-2026-10"
    assert no2.runtime_data.model.version == "no2-2026-10"

    sources.serve("energy_charts_price_tomorrow.json", no2_prices=TOMORROW)
    await advance(hass, freezer, T14)
    for entity_id in (DE_PRICE, NO2_PRICE):
        attrs = hass.states.get(entity_id).attributes
        assert attrs["stale"] is False and attrs["anchor_day"] == "2026-10-23", entity_id
    # both ran; only NO2's run asked for the hydro history, with NO2's hydro points
    zones = [u.query["bzn"] for u in _requests(sources, "https://api.energy-charts.info")]
    assert sorted(zones) == ["DE-LU", "NO2"]
    hydro = _hydro_requests(sources)
    assert len(hydro) == 1 and hydro[0].query["latitude"] == NO2_HYDRO_LATITUDES
    assert _requests(sources, PREVIOUS_RUNS_URL) == hydro


async def test_no2_backfill_fetches_the_hydro_history_from_91_days_before_its_first_anchor(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources
) -> None:
    """30 days at 2026-10-22: anchors 2026-09-25 … 2026-10-21 are recomputed, so the hydro
    history runs from 2026-06-26 to the day after the last one, one request per calendar
    month."""
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30}, zone="NO2")
    result = entry.runtime_data.coordinator.backfill_result
    assert result is not None and result.error is None
    assert result.imported > 0 and result.recomputed > 0
    assert hass.states.get("sensor.spot_forecast_no2_backfill_status").state == "done"
    ranges = [
        (u.query["start_date"], u.query["end_date"])
        for u in _hydro_requests(sources)
        if "start_date" in u.query
    ]
    assert ranges == [
        ("2026-06-26", "2026-06-30"),
        ("2026-07-01", "2026-07-31"),
        ("2026-08-01", "2026-08-31"),
        ("2026-09-01", "2026-09-30"),
        ("2026-10-01", "2026-10-22"),
    ]
    # the point sets' previous runs as for every zone: two calendar months, three sets
    others = [
        u for u in _requests(sources, PREVIOUS_RUNS_URL) if u.query["hourly"] != HYDRO_COLUMNS
    ]
    assert len(others) == 2 * 3
    runs = entry.runtime_data.coordinator.ledger.runs
    assert runs[date(2026, 9, 25)].backfill and runs[date(2026, 10, 21)].backfill
    assert await stat_rows(hass, "spot_forecast:no2_forecast_h1")


async def test_no2_backfill_skips_the_anchors_whose_hydro_history_has_a_hole(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    recorded,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing hour on 2026-07-10 lies in the 90 days of the anchors up to 2026-10-07 only."""
    caplog.set_level("DEBUG", logger="custom_components.spot_forecast.backfill")
    docs = copy.deepcopy(recorded(NO2_HYDRO_RANGE))
    hourly = docs[0]["hourly"]
    hourly["temperature_2m_previous_day2"][hourly["time"].index("2026-07-10T06:00")] = None
    sources.serve(hydro_range=docs)
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30}, zone="NO2")
    result = entry.runtime_data.coordinator.backfill_result
    assert result is not None and result.error is None and result.recomputed > 0
    runs = entry.runtime_data.coordinator.ledger.runs
    # anchor 2026-10-07: its 90 days start 2026-07-09 22:00 UTC; 2026-10-08: 2026-07-10 22:00
    assert date(2026, 10, 7) not in runs and runs[date(2026, 10, 8)].backfill
    # no recomputed target day from those anchors: before the cutoff only bundled ones stay
    for i in range(13):
        run = runs.get(date(2026, 9, 25) + timedelta(days=i))
        if run is not None:
            assert max(run.anchor + timedelta(days=s.h) for s in run.slots) < date(2026, 10, 1)
    assert "no hydro history for ['nb_no_south__temperature_2m']" in caplog.text
    # visible without debug logging: how many anchors, and the first and last of them
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "NO2 backfill skips 13 anchors (2026-09-25 … 2026-10-07): the Open-Meteo hydro history" in m
        for m in warnings
    ), warnings
    assert hass.states.get("sensor.spot_forecast_no2_backfill_status").state == "done"
