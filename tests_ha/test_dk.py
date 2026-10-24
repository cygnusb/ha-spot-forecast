"""DK1 and DK2: composite models that blend with the embedded DE-LU model and read DE-LU prices.

DK1 and DK2 are served from their own price recordings (``conftest.DK_PRICES``), DE-LU's
prices from its recording, and their weather by point set (``conftest.dk_weather``). They
load their bundled models from ``fixtures/model/dk1.*`` and ``dk2.*``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
import gzip
import hashlib
import json

from conftest import (
    FORECAST_URL,
    MODELS_BASE,
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

from custom_components.spot_forecast.const import (
    CONF_AUTO_UPDATE,
    CONF_BACKFILL_DAYS,
    CONF_SURCHARGE,
    CONF_VAT,
    DOMAIN,
    MANIFEST_URL,
)
from custom_components.spot_forecast.core.features import FEATURE_SCHEMA, feature_names
from custom_components.spot_forecast.core.model import ModelFile, dump_model
from custom_components.spot_forecast.core.zones import DK1
from custom_components.spot_forecast.diagnostics import async_get_config_entry_diagnostics
from custom_components.spot_forecast.model_loader import load_bundled

DK1_PRICE = "sensor.spot_forecast_dk1_price"
DK1_TOTAL = "sensor.spot_forecast_dk1_total_price"
DK1_MODEL = "sensor.spot_forecast_dk1_model_version"
DK2_PRICE = "sensor.spot_forecast_dk2_price"
DE_MODEL = "sensor.spot_forecast_de_lu_model_version"
T14 = T10 + timedelta(hours=4)
TOMORROW = {
    "DK1": "energy_charts_price_dk1_tomorrow.json",
    "DK2": "energy_charts_price_dk2_tomorrow.json",
}


def _market_price(recorded, at: datetime, name: str = "energy_charts_price_dk1.json") -> float:
    """The recorded quarter-hour price at ``at`` in EUR/kWh."""
    doc = recorded(name)
    stamp = int(at.replace(minute=at.minute // 15 * 15).timestamp())
    return round(doc["price"][doc["unix_seconds"].index(stamp)] / 1000, 5)


def _default(schema, key: str):
    for marker in schema.schema:
        if marker == key:
            return marker.default()
    raise AssertionError(f"{key} not in schema")


def _requests(sources: Sources, url: str) -> list:
    return [u for _, u, _, _ in sources._mock.mock_calls if str(u).startswith(url)]


async def test_dk1_entry_sets_up_and_forecasts(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up(zone="DK1")
    model = entry.runtime_data.model
    assert (model.zone, model.features, model.capacity) == ("dk1", feature_names(DK1), {})
    assert (model.parent.zone, model.parent.version, model.blend) == ("de_lu", "de_lu-2026-10", 0.5)

    registry = er.async_get(hass)
    for key, entity_id in (("price", DK1_PRICE), ("model_version", DK1_MODEL)):
        assert registry.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}") == (
            entity_id
        )
    state = hass.states.get(DK1_PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert float(state.state) == _market_price(recorded, T10)
    assert hass.states.get(DK1_MODEL).state == model.version == "dk1-2026-10"
    attrs = state.attributes
    assert attrs["anchor_day"] == "2026-10-22"
    assert attrs["attribution"].startswith("Prices: Bundesnetzagentur | SMARD.de")
    forecast = [e for e in attrs["prices"] if e["source"] == "forecast"]
    assert {e["h"] for e in forecast} == set(range(1, 7))
    assert forecast[0]["time"] == "2026-10-23T00:00:00+02:00"  # Europe/Copenhagen
    assert all(e["p10"] <= e["price"] <= e["p90"] for e in forecast)
    assert all(-0.5 < e["price"] < 1.0 for e in forecast)  # plausible EUR/kWh

    # DK1's four Open-Meteo point sets (the DE-LU points included), and two price requests
    latitudes = {u.query["latitude"]: u.query["hourly"] for u in _requests(sources, FORECAST_URL)}
    assert latitudes == {
        ",".join(str(lat) for lat, _ in points.values()): ",".join(variables)
        for points, variables in DK1.point_sets()
    }
    assert len(latitudes) == 4
    zones = [u.query["bzn"] for u in _requests(sources, "https://api.energy-charts.info")]
    assert zones == ["DK1", "DE-LU"]


async def test_fresh_dk1_entry_charges_danish_vat(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_SURCHARGE: 0.15}, zone="DK1")
    assert CONF_VAT not in entry.options
    assert entry.runtime_data.coordinator.vat == 25.0
    total = hass.states.get(DK1_TOTAL)
    assert float(total.state) == round(_market_price(recorded, T10) * 1.25 + 0.15, 5)
    assert total.attributes["vat"] == 25.0

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert _default(result["data_schema"], CONF_VAT) == 25.0


async def test_dk2_entry_sets_up_and_forecasts(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up(zone="DK2")
    model = entry.runtime_data.model
    assert (model.zone, model.version, model.parent.version) == (
        "dk2", "dk2-2026-10", "de_lu-2026-10",
    )  # fmt: skip
    state = hass.states.get(DK2_PRICE)
    assert float(state.state) == _market_price(recorded, T10, "energy_charts_price_dk2.json")
    forecast = [e for e in state.attributes["prices"] if e["source"] == "forecast"]
    assert {e["h"] for e in forecast} == set(range(1, 7))
    assert entry.runtime_data.coordinator.vat == 25.0
    assert len({u.query["latitude"] for u in _requests(sources, FORECAST_URL)}) == 4


async def test_failed_de_lu_price_request_keeps_the_last_forecast(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    recorded,
) -> None:
    """As any source error: the last forecast stays, marked stale; DK1's own fresh prices
    are kept."""
    hass = hass_with_recorder
    await set_up(zone="DK1")
    before = hass.states.get(DK1_PRICE).attributes

    sources.serve("energy_charts_price_tomorrow.json", dk_prices=TOMORROW,
                  zone_prices_status={"DE-LU": 500})  # fmt: skip
    await advance(hass, freezer, T14)

    state = hass.states.get(DK1_PRICE)
    assert state.attributes["stale"] is True
    assert state.attributes["anchor_day"] == "2026-10-22"
    assert state.attributes["issued_at"] == before["issued_at"]
    market = [e for e in state.attributes["prices"] if e["source"] == "market"]
    assert market[-1]["time"] == "2026-10-23T23:45:00+02:00"  # tomorrow's DK1 prices
    # the old forecast, after the hours that are known now
    forecast = [e for e in state.attributes["prices"] if e["source"] == "forecast"]
    known_until = datetime.fromisoformat("2026-10-24T00:00:00+02:00")
    assert forecast == [
        e
        for e in before["prices"]
        if e["source"] == "forecast" and datetime.fromisoformat(e["time"]) >= known_until
    ]

    sources.serve("energy_charts_price_tomorrow.json", dk_prices=TOMORROW)
    await advance(hass, freezer, T14 + timedelta(minutes=20))  # the retry
    attrs = hass.states.get(DK1_PRICE).attributes
    assert attrs["stale"] is False and attrs["anchor_day"] == "2026-10-23"


async def test_short_de_lu_prices_never_forecast_from_missing_de_lu_inputs(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DK1's prices of tomorrow are out, DE-LU's are not complete: no run for tomorrow,
    the last forecast stays."""
    hass = hass_with_recorder
    entry = await set_up(zone="DK1")
    before = hass.states.get(DK1_PRICE).attributes

    until = {"DE-LU": datetime.fromisoformat("2026-10-23T20:00:00+02:00")}
    sources.serve("energy_charts_price_tomorrow.json", dk_prices=TOMORROW, prices_until=until)
    await advance(hass, freezer, T14)

    state = hass.states.get(DK1_PRICE)
    assert state.attributes["stale"] is True
    assert state.attributes["issued_at"] == before["issued_at"]
    assert date(2026, 10, 23) not in entry.runtime_data.coordinator.ledger.runs
    assert "DE-LU prices of 2026-10-23 are not complete" in caplog.text


async def test_dk1_keeps_its_de_lu_parent_when_de_lu_updates(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """A DE-LU entry next to a DK1 entry: a new DE-LU model replaces DE-LU's model only."""
    hass = hass_with_recorder
    bundled, bundled_backfill = await hass.async_add_executor_job(load_bundled, "DE-LU")
    newer = replace(bundled, version="de_lu-2026-11", train_cutoff=date(2026, 10, 14))
    backfill = gzip.compress(gzip.decompress(bundled_backfill), mtime=14)
    sources.release = _published_de_lu(newer, backfill)
    de = await set_up({CONF_AUTO_UPDATE: True})
    dk = await set_up({CONF_AUTO_UPDATE: True}, zone="DK1")
    forecast = hass.states.get(DK1_PRICE).attributes["prices"]

    await advance(hass, freezer, T10 + timedelta(minutes=5))  # the checks after setup

    assert hass.states.get(DE_MODEL).state == "de_lu-2026-11"
    assert de.runtime_data.coordinator.model_source == "downloaded"
    dk_model = dk.runtime_data.coordinator.model
    assert (dk_model.version, dk_model.parent.version) == ("dk1-2026-10", "de_lu-2026-10")
    assert dk.runtime_data.coordinator.model_source == "bundled"
    assert hass.states.get(DK1_PRICE).attributes["prices"] == forecast


def _published_de_lu(model: ModelFile, backfill: bytes) -> dict[str, bytes]:
    """The release files of a manifest publishing ``model`` for DE-LU only."""
    data = dump_model(model)

    def asset(name: str, body: bytes) -> dict:
        digest = hashlib.sha256(body).hexdigest()
        return {"name": name, "url": MODELS_BASE + name, "sha256": digest, "size": len(body)}

    entry = {
        "version": model.version,
        "feature_schema": FEATURE_SCHEMA,
        "train_cutoff": model.train_cutoff.isoformat(),
        "created": model.created.isoformat(),
        "model": asset(f"{model.version}.json.gz", data),
        "backfill": asset(f"{model.version}.backfill.json.gz", backfill),
        "metrics": {"mae_h1": 17.0, "mae": 19.0, "skill": 0.4, "p90_exceedance": 0.1},
    }
    manifest = {"format": 1, "updated": "2026-11-03T02:10:00+00:00", "zones": {"de_lu": entry}}
    return {
        MANIFEST_URL: json.dumps(manifest).encode(),
        MODELS_BASE + f"{model.version}.json.gz": data,
        MODELS_BASE + f"{model.version}.backfill.json.gz": backfill,
    }


async def test_dk1_diagnostics_name_the_parent(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    entry = await set_up(zone="DK1")
    model = entry.runtime_data.model
    diag = json.loads(json.dumps(await async_get_config_entry_diagnostics(hass, entry)))
    assert diag["model"]["version"] == "dk1-2026-10"
    assert diag["model"]["parent"] == {
        "version": "de_lu-2026-10",
        "train_cutoff": "2026-10-01",
        "created": model.parent.created.isoformat(),
    }
    assert diag["model"]["blend"] == 0.5


async def test_dk1_backfill_recomputes_from_all_its_point_sets(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources
) -> None:
    """The bundled DK1 rows up to the cutoff, then recomputed from DK1's four point sets
    (the DE-LU model's points among them) and DK1's and DE-LU's prices."""
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30}, zone="DK1")
    result = entry.runtime_data.coordinator.backfill_result
    assert result is not None and result.error is None
    assert result.imported > 0 and result.recomputed > 0
    assert hass.states.get("sensor.spot_forecast_dk1_backfill_status").state == "done"
    calls = _requests(sources, PREVIOUS_RUNS_URL)
    assert len(calls) == 2 * 4  # two calendar months, four point sets each
    assert {url.query["latitude"] for url in calls} == {
        ",".join(str(lat) for lat, _ in points.values()) for points, _ in DK1.point_sets()
    }
    zones = [u.query["bzn"] for u in _requests(sources, "https://api.energy-charts.info")]
    assert zones.count("DE-LU") == 2  # the live run and the backfill
    assert await stat_rows(hass, "spot_forecast:dk1_forecast_h1")
