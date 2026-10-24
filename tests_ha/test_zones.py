"""NL, the second zone, and DE-LU and NL side by side, against a real Home Assistant.

NL is served from its own recordings (``conftest.NL_PRICES``, ``conftest.NL_WEATHER``) and
loads its bundled model from ``fixtures/model/nl.*``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path

from conftest import (
    FORECAST_URL,
    MODELS_BASE,
    NL_WEATHER,
    PREVIOUS_RUNS_URL,
    T10,
    SetUp,
    Sources,
    advance,
    stat_rows,
)
from freezegun.api import FrozenDateTimeFactory
from homeassistant import config_entries
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er

from custom_components.spot_forecast.const import (
    CONF_AUTO_UPDATE,
    CONF_BACKFILL_DAYS,
    CONF_SURCHARGE,
    CONF_VAT,
    CONF_ZONE,
    DOMAIN,
    MANIFEST_URL,
    MODEL_DIR,
)
from custom_components.spot_forecast.core.features import FEATURE_SCHEMA, feature_names
from custom_components.spot_forecast.core.model import ModelFile, dump_model
from custom_components.spot_forecast.core.zones import NL
from custom_components.spot_forecast.model_loader import load_bundled

NL_PRICE = "sensor.spot_forecast_nl_price"
NL_TOTAL = "sensor.spot_forecast_nl_total_price"
NL_MODEL = "sensor.spot_forecast_nl_model_version"
DE_PRICE = "sensor.spot_forecast_de_lu_price"
DE_MODEL = "sensor.spot_forecast_de_lu_model_version"
T14 = T10 + timedelta(hours=4)


def _default(schema, key: str):
    for marker in schema.schema:
        if marker == key:
            return marker.default()
    raise AssertionError(f"{key} not in schema")


def _market_price(recorded, at: datetime) -> float:
    """The recorded NL quarter-hour price at ``at`` in EUR/kWh."""
    doc = recorded("energy_charts_price_nl.json")
    stamp = int(at.replace(minute=at.minute // 15 * 15).timestamp())
    return round(doc["price"][doc["unix_seconds"].index(stamp)] / 1000, 5)


async def test_nl_entry_sets_up_and_forecasts(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up(zone="NL")
    model = entry.runtime_data.model
    assert (model.zone, model.features) == ("nl", feature_names(NL))
    assert model.capacity == {}

    registry = er.async_get(hass)
    for key, entity_id in (("price", NL_PRICE), ("model_version", NL_MODEL)):
        assert registry.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}") == (
            entity_id
        )
    state = hass.states.get(NL_PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert float(state.state) == _market_price(recorded, T10)
    assert hass.states.get(NL_MODEL).state == model.version
    attrs = state.attributes
    assert attrs["anchor_day"] == "2026-10-22"
    assert attrs["attribution"].startswith("Prices: Bundesnetzagentur | SMARD.de")
    forecast = [e for e in attrs["prices"] if e["source"] == "forecast"]
    assert {e["h"] for e in forecast} == set(range(1, 7))
    assert forecast[0]["time"] == "2026-10-23T00:00:00+02:00"  # Europe/Amsterdam
    assert all(e["p10"] <= e["price"] <= e["p90"] for e in forecast)
    assert all(-0.5 < e["price"] < 1.0 for e in forecast)  # plausible EUR/kWh

    # NL's three Open-Meteo point sets, each with its own variables; no DE-LU set
    latitudes = {
        url.query["latitude"]: url.query["hourly"]
        for _, url, _, _ in sources._mock.mock_calls
        if str(url).startswith(FORECAST_URL)
    }
    assert latitudes == {
        ",".join(str(lat) for lat, _ in points.values()): ",".join(variables)
        for points, variables in NL.point_sets()
    }
    assert len(NL_WEATHER) == 3
    assert {
        url.query["bzn"] for _, url, _, _ in sources._mock.mock_calls if "energy-charts" in str(url)
    } == {"NL"}


async def test_fresh_nl_entry_charges_dutch_vat(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    """No VAT option: the total price uses NL's 21 %, and the options form offers 21."""
    hass = hass_with_recorder
    entry = await set_up({CONF_SURCHARGE: 0.15}, zone="NL")
    assert CONF_VAT not in entry.options
    assert entry.runtime_data.coordinator.vat == 21.0
    total = hass.states.get(NL_TOTAL)
    assert float(total.state) == round(_market_price(recorded, T10) * 1.21 + 0.15, 5)
    assert total.attributes["vat"] == 21.0
    assert hass.states.get(NL_PRICE).attributes["vat"] == 21.0

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert _default(result["data_schema"], CONF_VAT) == 21.0


async def test_user_flow_offers_the_zones_with_their_names(
    hass_with_recorder: HomeAssistant, sources: Sources, freezer: FrozenDateTimeFactory
) -> None:
    hass = hass_with_recorder
    freezer.move_to(T10)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    for marker, selector in result["data_schema"].schema.items():
        if marker == CONF_ZONE:
            options = selector.config["options"]
    assert options == [
        {"value": "DE-LU", "label": "DE-LU \u2013 Germany and Luxembourg"},
        {"value": "NL", "label": "NL \u2013 Netherlands"},
    ]
    assert _default(result["data_schema"], CONF_ZONE) == "DE-LU"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_ZONE: "NL"})
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Spot Forecast NL"
    assert result["data"] == {CONF_ZONE: "NL"} and result["result"].unique_id == "NL"
    assert result["result"].runtime_data.model.zone == "nl"
    assert await hass.config_entries.async_unload(result["result"].entry_id)


async def test_de_lu_and_nl_side_by_side(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    recorded,
) -> None:
    """Two entries keep their own models, stores, sensors and statistics."""
    hass = hass_with_recorder
    de = await set_up()
    nl = await set_up(zone="NL")
    assert (de.runtime_data.model.zone, nl.runtime_data.model.zone) == ("de_lu", "nl")
    de_store = de.runtime_data.coordinator._store._store
    nl_store = nl.runtime_data.coordinator._store._store
    assert de_store.key == f"{DOMAIN}.{de.entry_id}" and nl_store.key == f"{DOMAIN}.{nl.entry_id}"
    assert float(hass.states.get(NL_PRICE).state) == _market_price(recorded, T10)
    assert hass.states.get(DE_PRICE).state != hass.states.get(NL_PRICE).state

    sources.serve(
        "energy_charts_price_tomorrow.json", nl_prices="energy_charts_price_nl_tomorrow.json"
    )
    await advance(hass, freezer, T14)
    nl_price = await stat_rows(hass, "spot_forecast:nl_price")
    de_price = await stat_rows(hass, "spot_forecast:de_lu_price")
    assert nl_price and de_price
    assert [r["mean"] for r in nl_price[-24:]] != [r["mean"] for r in de_price[-24:]]
    assert len(await stat_rows(hass, "spot_forecast:nl_forecast_h1")) == 24
    assert len(await stat_rows(hass, "spot_forecast:de_lu_forecast_h1")) == 24


def _published_nl(model: ModelFile, backfill: bytes) -> dict[str, bytes]:
    """The release files of a manifest publishing ``model`` for NL only."""
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
    manifest = {"format": 1, "updated": "2026-11-03T02:10:00+00:00", "zones": {"nl": entry}}
    return {
        MANIFEST_URL: json.dumps(manifest).encode(),
        MODELS_BASE + f"{model.version}.json.gz": data,
        MODELS_BASE + f"{model.version}.backfill.json.gz": backfill,
    }


async def test_an_nl_model_update_leaves_de_lu_alone(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    tmp_path: Path,
) -> None:
    hass = hass_with_recorder
    bundled, bundled_backfill = await hass.async_add_executor_job(load_bundled, "NL")
    newer = replace(bundled, version="nl-2026-11", train_cutoff=date(2026, 10, 14))
    backfill = gzip.compress(gzip.decompress(bundled_backfill), mtime=14)
    sources.release = _published_nl(newer, backfill)
    de = await set_up({CONF_AUTO_UPDATE: True})
    nl = await set_up({CONF_AUTO_UPDATE: True}, zone="NL")
    de_version = de.runtime_data.model.version

    await advance(hass, freezer, T10 + timedelta(minutes=5))  # the checks after setup

    assert hass.states.get(NL_MODEL).state == "nl-2026-11"
    assert nl.runtime_data.coordinator.model_source == "downloaded"
    assert hass.states.get(DE_MODEL).state == de_version
    assert de.runtime_data.coordinator.model_source == "bundled"
    assert sorted(p.name for p in (tmp_path / ".storage" / MODEL_DIR).iterdir()) == [
        "nl-2026-11.backfill.json.gz",
        "nl-2026-11.json.gz",
    ]


async def test_nl_backfill_recomputes_from_its_own_point_sets(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources
) -> None:
    """The bundled NL rows up to the cutoff, then recomputed from NL's three point sets."""
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30}, zone="NL")
    coordinator = entry.runtime_data.coordinator
    result = coordinator.backfill_result
    assert result is not None and result.imported > 0 and result.recomputed > 0
    assert hass.states.get("sensor.spot_forecast_nl_backfill_status").state == "done"
    calls = [
        url for _, url, _, _ in sources._mock.mock_calls if str(url).startswith(PREVIOUS_RUNS_URL)
    ]
    assert len(calls) == 2 * 3  # two calendar months, three point sets each
    assert {url.query["latitude"] for url in calls} == {
        ",".join(str(lat) for lat, _ in points.values()) for points, _ in NL.point_sets()
    }
    assert await stat_rows(hass, "spot_forecast:nl_forecast_h1")
