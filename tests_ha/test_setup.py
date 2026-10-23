"""Coordinator, Store, statistics and entities against a real Home Assistant.

Time is frozen to 2026-10-22 10:00 Europe/Berlin, before the day-ahead auction; the
sources serve the recorded payloads (see ``conftest.Sources``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
import logging
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

from conftest import T10, SetUp, Sources, advance, stat_rows
from freezegun.api import FrozenDateTimeFactory
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import (
    get_metadata,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.spot_forecast import coordinator as coordinator_module
from custom_components.spot_forecast import sensor, store
from custom_components.spot_forecast.const import (
    CONF_PEAK_THRESHOLD,
    CONF_SURCHARGE,
    CONF_VAT,
    CONF_ZONE,
    DOMAIN,
)
from custom_components.spot_forecast.core.ingest import parse_energy_charts

BERLIN = ZoneInfo("Europe/Berlin")
T14 = datetime(2026, 10, 22, 14, tzinfo=BERLIN)

PRICE = "sensor.spot_forecast_de_lu_price"
PEAK = "binary_sensor.spot_forecast_de_lu_peak_risk"
NEGATIVE = "binary_sensor.spot_forecast_de_lu_negative_price"
LAST_RUN = "sensor.spot_forecast_de_lu_last_run"
TOTAL = "sensor.spot_forecast_de_lu_total_price"
STAT_PRICE = "spot_forecast:de_lu_price"
STAT_H1 = "spot_forecast:de_lu_forecast_h1"


def _market_price(recorded, at: datetime, name: str = "energy_charts_price.json") -> float:
    """EUR/kWh of the recorded quarter hour running at ``at``."""
    slot = next(s for s in parse_energy_charts(recorded(name)) if s.start <= at < s.end)
    return round(slot.price / 1000, 5)


async def test_setup_creates_entities_with_expected_ids(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    entry = await set_up()

    registry = er.async_get(hass)
    expected = {
        ("sensor", "price"): PRICE,
        ("sensor", "last_run"): "sensor.spot_forecast_de_lu_last_run",
        ("sensor", "model_version"): "sensor.spot_forecast_de_lu_model_version",
        ("sensor", "forecast_error"): "sensor.spot_forecast_de_lu_forecast_error",
        ("sensor", "backfill"): "sensor.spot_forecast_de_lu_backfill_status",
        ("binary_sensor", "peak_risk"): PEAK,
        ("binary_sensor", "negative_price"): NEGATIVE,
        **{("sensor", f"day_{i}"): f"sensor.spot_forecast_de_lu_day_{i}" for i in range(7)},
    }
    for (platform, key), entity_id in expected.items():
        assert registry.async_get_entity_id(platform, DOMAIN, f"{entry.entry_id}_{key}") == (
            entity_id
        )
    for entity_id in (PRICE, PEAK, NEGATIVE, "sensor.spot_forecast_de_lu_last_run"):
        assert hass.states.get(entity_id).state != STATE_UNAVAILABLE
    assert hass.states.get("sensor.spot_forecast_de_lu_model_version").state == (
        entry.runtime_data.model.version
    )
    assert registry.async_get("sensor.spot_forecast_de_lu_last_run").entity_category == (
        er.EntityCategory.DIAGNOSTIC
    )


async def test_entity_ids_stay_english_on_a_german_instance(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    """The cards and Smart Battery Pilot use these ids; the language must not change them."""
    hass = hass_with_recorder
    hass.config.language = "de"
    await set_up()

    for entity_id in (
        PRICE,
        PEAK,
        NEGATIVE,
        "sensor.spot_forecast_de_lu_backfill_status",
        "sensor.spot_forecast_de_lu_forecast_error",
    ):
        assert hass.states.get(entity_id) is not None, entity_id
    # The names do follow the language.
    assert hass.states.get(PRICE).attributes["friendly_name"] == "Spot Forecast DE-LU Preis"


async def test_price_attribute_has_market_then_forecast_and_sbp_keys(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    hass = hass_with_recorder
    entry = await set_up()

    state = hass.states.get(PRICE)
    assert float(state.state) == _market_price(recorded, T10)
    attrs = state.attributes
    assert attrs["unit_of_measurement"] == "EUR/kWh"
    assert attrs["unit"] == "EUR/kWh"
    assert attrs["state_class"] == "measurement"
    assert attrs["anchor_day"] == "2026-10-22"
    assert attrs["model_version"] == entry.runtime_data.model.version
    assert attrs["stale"] is False
    assert datetime.fromisoformat(attrs["issued_at"]) == T10
    assert attrs["attribution"].startswith("Prices: Bundesnetzagentur | SMARD.de")

    prices = attrs["prices"]
    times = [datetime.fromisoformat(e["time"]) for e in prices]
    assert times[0] == datetime(2026, 10, 22, tzinfo=BERLIN)
    assert prices[0]["time"] == "2026-10-22T00:00:00+02:00"
    assert times == sorted(times) and len(set(times)) == len(times)
    sources = [e["source"] for e in prices]
    n_market = sources.count("market")
    assert n_market == 96  # today's quarter hours
    assert sources == ["market"] * n_market + ["forecast"] * (len(prices) - n_market)
    first = prices[n_market]
    assert first["time"] == "2026-10-23T00:00:00+02:00"
    assert set(first) == {"time", "price", "source", "p90", "p10", "h"}
    assert first["h"] == 1
    assert first["p10"] <= first["price"] <= first["p90"]
    assert prices[-1]["h"] == 6
    assert set(prices[0]) == {"time", "price", "source"}


async def test_total_price_sensor_adds_vat_and_the_surcharge(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    hass = hass_with_recorder
    await set_up({CONF_SURCHARGE: 0.187})

    price, total = hass.states.get(PRICE), hass.states.get(TOTAL)
    # no VAT option set: 19 % on the market price, then the gross surcharge
    assert float(total.state) == round(_market_price(recorded, T10) * 1.19 + 0.187, 5)
    assert total.attributes["unit_of_measurement"] == "EUR/kWh"
    assert total.attributes["surcharge"] == 0.187
    assert price.attributes["surcharge"] == 0.187  # the cards read it here
    market, shifted = price.attributes["prices"], total.attributes["prices"]
    assert [e["time"] for e in shifted] == [e["time"] for e in market]
    assert [e["price"] for e in shifted] == [round(e["price"] * 1.19 + 0.187, 5) for e in market]
    first_fc = next(e for e in shifted if e["source"] == "forecast")
    assert set(first_fc) == {"time", "price", "source", "p90", "p10", "h"}
    # everything else stays on the market price
    assert float(price.state) == _market_price(recorded, T10)


async def test_total_price_sensor_only_with_a_surcharge(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    """Without a surcharge there is no sensor; clearing the option removes it again."""
    hass = hass_with_recorder
    entry = await set_up()
    assert hass.states.get(TOTAL) is None
    assert hass.states.get(PRICE).attributes["surcharge"] == 0.0

    hass.config_entries.async_update_entry(entry, options={**entry.options, CONF_SURCHARGE: 0.2})
    await hass.async_block_till_done()  # the update listener reloads the entry
    assert float(hass.states.get(TOTAL).state) > 0.2

    options = {k: v for k, v in entry.options.items() if k != CONF_SURCHARGE}
    hass.config_entries.async_update_entry(entry, options=options)
    await hass.async_block_till_done()
    assert hass.states.get(TOTAL) is None
    assert er.async_get(hass).async_get(TOTAL) is None


async def test_display_options_apply_without_a_reload(
    hass_with_recorder: HomeAssistant, set_up: SetUp, aioclient_mock: AiohttpClientMocker
) -> None:
    """Surcharge, VAT and peak threshold: the same coordinator, no fetch, entities updated."""
    hass = hass_with_recorder
    entry = await set_up({CONF_SURCHARGE: 0.1, CONF_PEAK_THRESHOLD: 1000.0})
    coordinator = entry.runtime_data.coordinator
    assert hass.states.get(PEAK).state == STATE_OFF
    calls = aioclient_mock.call_count

    options = {**entry.options, CONF_SURCHARGE: 0.2, CONF_VAT: 7.0, CONF_PEAK_THRESHOLD: 0.0}
    with patch.object(
        hass.config_entries, "async_reload", wraps=hass.config_entries.async_reload
    ) as reload:
        hass.config_entries.async_update_entry(entry, options=options)
        await hass.async_block_till_done()

    reload.assert_not_called()
    assert entry.runtime_data.coordinator is coordinator
    assert aioclient_mock.call_count == calls
    assert coordinator.vat == 7.0
    assert hass.states.get(PRICE).attributes["surcharge"] == 0.2
    assert hass.states.get(TOTAL).attributes["surcharge"] == 0.2
    peak = hass.states.get(PEAK)
    assert peak.state == STATE_ON
    assert peak.attributes["threshold"] == 0.0


async def test_other_options_reload_the_entry(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_SURCHARGE: 0.1})
    coordinator = entry.runtime_data.coordinator

    options = {**entry.options, CONF_SURCHARGE: 0.2, "something_else": 1}
    hass.config_entries.async_update_entry(entry, options=options)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.coordinator is not coordinator
    assert hass.states.get(TOTAL).attributes["surcharge"] == 0.2


async def test_anchor_switches_after_publication(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    hass = hass_with_recorder
    await set_up()
    assert hass.states.get(PRICE).attributes["anchor_day"] == "2026-10-22"

    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)

    attrs = hass.states.get(PRICE).attributes
    assert attrs["anchor_day"] == "2026-10-23"
    assert datetime.fromisoformat(attrs["issued_at"]) == T14
    prices = attrs["prices"]
    market = [e for e in prices if e["source"] == "market"]
    forecast = [e for e in prices if e["source"] == "forecast"]
    assert len(market) == 2 * 96
    assert market[-1]["time"] == "2026-10-23T23:45:00+02:00"
    assert forecast[0]["time"] == "2026-10-24T00:00:00+02:00"
    assert forecast[0]["h"] == 1


async def test_peak_and_negative_binary_sensors(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, recorded
) -> None:
    hass = hass_with_recorder
    negative_at = datetime(2026, 10, 22, 13, 15, tzinfo=BERLIN)
    sources.serve(price_overrides={negative_at: -5.0})
    # threshold at the dearest known quarter hour still ahead today -> certain peak
    ahead = [
        s
        for s in parse_energy_charts(recorded("energy_charts_price.json"))
        if s.end > T10 and s.start < datetime(2026, 10, 23, tzinfo=BERLIN)
    ]
    dearest = max(ahead, key=lambda s: s.price)
    threshold_ct = dearest.price / 10
    await set_up({CONF_PEAK_THRESHOLD: threshold_ct})

    peak = hass.states.get(PEAK)
    assert peak.state == STATE_ON
    assert peak.attributes["certain"] is True
    assert peak.attributes["threshold"] == pytest.approx(threshold_ct / 100)
    flagged = peak.attributes["slots"]
    assert dearest.start.astimezone(BERLIN).isoformat() in [e["time"] for e in flagged]
    assert all(
        (e["price"] if e["source"] == "market" else e["p90"]) >= threshold_ct / 100 for e in flagged
    )

    negative = hass.states.get(NEGATIVE)
    assert negative.state == STATE_ON
    assert negative.attributes["certain"] is True
    assert negative.attributes["threshold"] == 0.0
    assert negative.attributes["slots"][0] == {
        "time": "2026-10-22T13:15:00+02:00",
        "price": -0.005,
        "source": "market",
    }


async def test_peak_risk_off_above_every_price(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    await set_up({CONF_PEAK_THRESHOLD: 1000.0})

    peak = hass.states.get(PEAK)
    assert peak.state == STATE_OFF
    assert peak.attributes["slots"] == []
    assert peak.attributes["certain"] is False
    assert peak.attributes["threshold"] == 10.0


async def test_flags_unknown_without_prices_in_the_window(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Nothing known for the next 48 h is neither "risk" nor "no risk"."""
    hass = hass_with_recorder
    await set_up()
    sources.fail_all()

    # 2026-10-23 12:00: no market prices for today, the forecast still shown
    await advance(hass, freezer, T10 + timedelta(hours=26))
    for entity_id in (PEAK, NEGATIVE):
        assert hass.states.get(entity_id).state in (STATE_ON, STATE_OFF), entity_id

    # 36 h after the last success the forecast is dropped too: an empty window
    await advance(hass, freezer, T10 + timedelta(hours=36, minutes=15))
    assert hass.states.get(PRICE).attributes["prices"] == []
    for entity_id in (PEAK, NEGATIVE):
        state = hass.states.get(entity_id)
        assert state.state == STATE_UNKNOWN, entity_id
        assert state.attributes["slots"] == []
        assert state.attributes["certain"] is False


async def test_day_sensors_disabled_by_default(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    await set_up()

    registry = er.async_get(hass)
    for i in range(7):
        entity_id = f"sensor.spot_forecast_de_lu_day_{i}"
        assert registry.async_get(entity_id).disabled_by is er.RegistryEntryDisabler.INTEGRATION
        assert hass.states.get(entity_id) is None


async def test_statistics_written_for_completed_day(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    hass = hass_with_recorder
    await set_up()
    price_rows = await stat_rows(hass, STAT_PRICE)
    assert price_rows  # every complete day of the 40-day history
    assert datetime.fromtimestamp(price_rows[-1]["start"], UTC) == datetime(
        2026, 10, 22, 21, tzinfo=UTC
    )
    assert await stat_rows(hass, STAT_H1) == []  # no target day complete yet

    # 14:00: 2026-10-23 is published, so the 10:00 run's D+1 day is complete
    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)

    h1 = await stat_rows(hass, STAT_H1)
    assert len(h1) == 24
    assert datetime.fromtimestamp(h1[0]["start"], UTC) == datetime(2026, 10, 22, 22, tzinfo=UTC)
    assert all(row["mean"] is not None for row in h1)
    price_rows = await stat_rows(hass, STAT_PRICE)
    assert datetime.fromtimestamp(price_rows[-1]["start"], UTC) == datetime(
        2026, 10, 23, 21, tzinfo=UTC
    )
    for series in ("peak_risk_h1", "low_h1"):
        assert len(await stat_rows(hass, f"spot_forecast:de_lu_{series}")) == 24
    assert await stat_rows(hass, "spot_forecast:de_lu_forecast_h2") == []

    metadata = await get_instance(hass).async_add_executor_job(
        lambda: get_metadata(hass, statistic_ids={STAT_PRICE, STAT_H1})
    )
    meta = {statistic_id: m for statistic_id, (_, m) in metadata.items()}
    assert meta[STAT_PRICE]["name"] == "Spot Forecast DE-LU price"
    assert meta[STAT_H1]["name"] == "Spot Forecast DE-LU forecast D+1"
    assert meta[STAT_H1]["source"] == DOMAIN
    assert meta[STAT_H1]["unit_of_measurement"] == "EUR/kWh"
    assert meta[STAT_H1]["has_sum"] is False


async def test_weather_outage_keeps_last_forecast(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    aioclient_mock: AiohttpClientMocker,
    recorded,
) -> None:
    hass = hass_with_recorder
    await set_up()
    before = hass.states.get(PRICE).attributes
    last_run = hass.states.get(LAST_RUN).state

    sources.serve(weather_status=500)
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # weather run 08:15 UTC

    assert any("open-meteo" in str(url) for _, url, _, _ in aioclient_mock.mock_calls)
    state = hass.states.get(PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert state.attributes["stale"] is True  # the old forecast is shown, marked stale
    assert state.attributes["issued_at"] == before["issued_at"]
    assert state.attributes["prices"] == before["prices"]
    assert hass.states.get(PEAK).state != STATE_UNAVAILABLE
    assert hass.states.get(LAST_RUN).state == last_run

    # 36 h after the last success: forecast dropped, known prices kept
    sources.serve("energy_charts_price_tomorrow.json", weather_status=500)
    later = T10 + timedelta(hours=36, minutes=15)
    await advance(hass, freezer, later)

    assert aioclient_mock.call_count > 0  # a run was attempted, and failed again
    state = hass.states.get(PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert state.attributes["stale"] is True
    assert float(state.state) == _market_price(recorded, later, "energy_charts_price_tomorrow.json")
    prices = state.attributes["prices"]
    assert prices and {e["source"] for e in prices} == {"market"}
    assert hass.states.get(PEAK).state in (STATE_ON, STATE_OFF)  # from the market prices


async def test_successful_run_clears_stale(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    hass = hass_with_recorder
    await set_up()
    sources.serve(weather_status=500)
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # weather run 08:15 UTC
    assert hass.states.get(PRICE).attributes["stale"] is True

    sources.serve()
    await advance(hass, freezer, datetime(2026, 10, 22, 13, tzinfo=BERLIN))  # first poll
    attrs = hass.states.get(PRICE).attributes
    assert attrs["stale"] is False
    assert datetime.fromisoformat(attrs["issued_at"]) == datetime(2026, 10, 22, 13, tzinfo=BERLIN)


async def test_failed_run_is_retried_20_min_later(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """A source outage at a scheduled run: tried again 20 min later, not at 13:00."""
    hass = hass_with_recorder
    await set_up()
    sources.serve(weather_status=500)
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # weather run 08:15 UTC
    assert hass.states.get(PRICE).attributes["stale"] is True

    sources.serve()  # clears the call log
    await advance(hass, freezer, T10 + timedelta(minutes=34))
    assert aioclient_mock.call_count == 0
    await advance(hass, freezer, T10 + timedelta(minutes=35))
    assert aioclient_mock.call_count > 0
    attrs = hass.states.get(PRICE).attributes
    assert attrs["stale"] is False
    assert datetime.fromisoformat(attrs["issued_at"]) == T10 + timedelta(minutes=35)

    # after the success the regular slot again: the first poll at 13:00
    sources.serve()
    await advance(hass, freezer, T10 + timedelta(minutes=55))
    await advance(hass, freezer, datetime(2026, 10, 22, 12, 59, tzinfo=BERLIN))
    assert aioclient_mock.call_count == 0
    await advance(hass, freezer, datetime(2026, 10, 22, 13, tzinfo=BERLIN))
    assert aioclient_mock.call_count > 0


async def test_stale_after_7h_without_success(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    freezer: FrozenDateTimeFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No run due (moved away): the quarter-hour tick alone turns the old forecast stale."""
    hass = hass_with_recorder
    monkeypatch.setattr(
        coordinator_module, "next_run", lambda now, *_, **__: now + timedelta(days=3)
    )
    await set_up()
    await advance(hass, freezer, T10 + timedelta(hours=7))
    assert hass.states.get(PRICE).attributes["stale"] is False
    await advance(hass, freezer, T10 + timedelta(hours=7, minutes=15))
    attrs = hass.states.get(PRICE).attributes
    assert attrs["stale"] is True
    assert {e["source"] for e in attrs["prices"]} == {"market", "forecast"}


async def test_restart_keeps_forecast_and_does_not_duplicate_statistics(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    hass = hass_with_recorder
    entry = await set_up()
    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)
    before = hass.states.get(PRICE).attributes
    h1_before = len(await stat_rows(hass, STAT_H1))
    assert h1_before > 0

    sources.fail_all()
    with patch.object(
        coordinator_module.long_term_statistics,
        "async_write",
        wraps=coordinator_module.long_term_statistics.async_write,
    ) as write:
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

    # the restored ledger knows what was finalized: nothing is written again
    assert write.call_args_list and all(c.args[1] == [] for c in write.call_args_list)

    assert entry.state is ConfigEntryState.LOADED
    assert aioclient_mock.call_count > 0  # the first refresh tried and failed
    state = hass.states.get(PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert state.attributes["prices"] == before["prices"]
    assert state.attributes["issued_at"] == before["issued_at"]
    assert state.attributes["stale"] is True  # the first refresh after the restart failed
    assert len(await stat_rows(hass, STAT_H1)) == h1_before


async def test_statistics_lost_by_the_recorder_are_written_again(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marked written in the Store but never committed (a crash, a restored database)."""
    hass = hass_with_recorder
    lts = coordinator_module.long_term_statistics
    write = lts.async_write
    monkeypatch.setattr(lts, "async_write", lambda hass, imports: None)  # all lost
    entry = await set_up()
    assert entry.runtime_data.coordinator.ledger.finalized
    assert await stat_rows(hass, STAT_PRICE) == []

    monkeypatch.setattr(lts, "async_write", write)
    sources.fail_all()  # the days come from the stored ledger, not from a run
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    rows = await stat_rows(hass, STAT_PRICE)
    assert rows
    assert datetime.fromtimestamp(rows[-1]["start"], UTC) == datetime(2026, 10, 22, 21, tzinfo=UTC)


async def test_statistics_the_recorder_lacks_after_its_last_row_are_written_again(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = hass_with_recorder
    lts = coordinator_module.long_term_statistics
    write = lts.async_write
    entry = await set_up()  # price rows through 2026-10-22
    assert len(await stat_rows(hass, STAT_PRICE)) > 0
    monkeypatch.setattr(lts, "async_write", lambda hass, imports: None)
    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)  # 2026-10-23 and its D+1 rows, lost

    sources.fail_all()
    with patch.object(lts, "async_write", wraps=write) as written:
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

    starts = [row["start"] for c in written.call_args_list for _, rows in c.args[1] for row in rows]
    assert starts  # only the days after the recorder's last row
    assert min(starts) == datetime(2026, 10, 22, 22, tzinfo=UTC)
    price = await stat_rows(hass, STAT_PRICE)
    assert datetime.fromtimestamp(price[-1]["start"], UTC) == datetime(2026, 10, 23, 21, tzinfo=UTC)
    assert len(await stat_rows(hass, STAT_H1)) == 24


async def test_setup_retries_without_data(
    hass_with_recorder: HomeAssistant, sources: Sources, freezer: FrozenDateTimeFactory
) -> None:
    hass = hass_with_recorder
    freezer.move_to(T10)
    sources.fail_all()
    entry = MockConfigEntry(domain=DOMAIN, unique_id="DE-LU", data={CONF_ZONE: "DE-LU"})
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_RETRY
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_state_advances_each_quarter_hour_without_fetching(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    freezer: FrozenDateTimeFactory,
    aioclient_mock: AiohttpClientMocker,
    monkeypatch: pytest.MonkeyPatch,
    recorded,
) -> None:
    """10:15 is also a weather run (08:15 UTC); the run is moved away so the tick is alone."""
    hass = hass_with_recorder
    monkeypatch.setattr(
        coordinator_module, "next_run", lambda now, *_, **__: now + timedelta(days=1)
    )
    await set_up()
    assert float(hass.states.get(PRICE).state) == _market_price(recorded, T10)
    calls = aioclient_mock.call_count

    await advance(hass, freezer, T10 + timedelta(minutes=15))

    assert float(hass.states.get(PRICE).state) == _market_price(
        recorded, T10 + timedelta(minutes=15)
    )
    assert _market_price(recorded, T10 + timedelta(minutes=15)) != _market_price(recorded, T10)
    assert aioclient_mock.call_count == calls


@pytest.mark.parametrize(
    "stored",
    [
        {"ledger": {"runs": "garbage"}, "market": [], "last_success": None},
        {"market": [["not a time", "x", 1.0]], "ledger": {}, "last_success": None},
        ["an", "old", "layout"],
    ],
)
async def test_unreadable_store_starts_empty(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    hass_storage: dict[str, Any],
    stored: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hass = hass_with_recorder
    entry_id = "unreadable"
    hass_storage[f"{DOMAIN}.{entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{entry_id}",
        "data": stored,
    }

    entry = await set_up(entry_id=entry_id)

    assert entry.state is ConfigEntryState.LOADED
    assert "cannot be read, starting empty" in caplog.text
    assert hass.states.get(PRICE).attributes["anchor_day"] == "2026-10-22"


async def test_forecast_bug_keeps_price_sensor_with_fresh_market(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    recorded,
) -> None:
    hass = hass_with_recorder

    def broken(*args: Any) -> None:
        raise IndexError("bug in core")

    monkeypatch.setattr(coordinator_module, "run_forecast", broken)
    entry = await set_up()

    assert entry.state is ConfigEntryState.LOADED
    state = hass.states.get(PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert float(state.state) == _market_price(recorded, T10)
    assert {e["source"] for e in state.attributes["prices"]} == {"market"}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and errors[0].exc_info[0] is IndexError


async def test_refreshes_never_interleave(
    hass_with_recorder: HomeAssistant, set_up: SetUp, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = await set_up()
    coordinator = entry.runtime_data.coordinator
    active, peak = 0, 0
    fetch = coordinator_module.fetch_prices

    async def counting_fetch(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        try:
            return await fetch(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(coordinator_module, "fetch_prices", counting_fetch)
    # Straight at the update method: newer HA locks async_refresh itself, 2024.11 does not.
    await asyncio.gather(coordinator._async_update_data(), coordinator._async_update_data())

    assert peak == 1


async def test_outage_flag_is_stored(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    hass_storage: dict[str, Any],
) -> None:
    """A restart during an outage starts stale, before its own first run fails too."""
    hass = hass_with_recorder
    entry = await set_up(entry_id="outage")
    sources.serve(weather_status=500)
    await advance(hass, freezer, T10 + timedelta(minutes=15))
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert hass_storage[f"{DOMAIN}.outage"]["data"]["last_attempt_failed"] is True

    sources.serve()
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert hass_storage[f"{DOMAIN}.outage"]["data"]["last_attempt_failed"] is False


async def test_removing_the_entry_deletes_its_store(
    hass_with_recorder: HomeAssistant, set_up: SetUp, hass_storage: dict[str, Any]
) -> None:
    hass = hass_with_recorder
    entry = await set_up(entry_id="gone")
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert f"{DOMAIN}.gone" in hass_storage
    rows = len(await stat_rows(hass, STAT_PRICE))

    assert await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert f"{DOMAIN}.gone" not in hass_storage
    assert len(await stat_rows(hass, STAT_PRICE)) == rows > 0  # statistics are kept


def _hold(monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, ...]:
    """Make the next price fetch wait: returns (fetch entered, release it, it was cancelled)."""
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    fetch = coordinator_module.fetch_prices

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return await fetch(*args, **kwargs)

    monkeypatch.setattr(coordinator_module, "fetch_prices", held)
    return entered, release, cancelled


async def _remove_while_held(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Remove ``entry`` while a run waits in a held fetch, past every unload timeout."""
    coordinator = entry.runtime_data.coordinator
    remove = hass.async_create_task(hass.config_entries.async_remove(entry.entry_id))
    while coordinator._started:  # until the unload waits for the run lock
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    # the loop clock is frozen too: let the unload's waits run out (HA's own is 10 s)
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=coordinator_module.UNLOAD_WAIT + 11)
    )
    assert await remove


async def _saves_flushed(hass: HomeAssistant) -> None:
    """Let delayed Store saves fall due and be written."""
    await hass.async_block_till_done(wait_background_tasks=True)
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=store.SAVE_DELAY + 1))
    await hass.async_block_till_done(wait_background_tasks=True)


async def test_removing_the_entry_cancels_a_stalled_scheduled_run(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    freezer: FrozenDateTimeFactory,
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run stuck on a slow API must not write the Store file of a removed entry again."""
    hass = hass_with_recorder
    entry = await set_up(entry_id="stalled")
    entered, release, cancelled = _hold(monkeypatch)
    freezer.move_to(T10 + timedelta(minutes=15))  # weather run 08:15 UTC
    async_fire_time_changed(hass, dt_util.utcnow())
    await entered.wait()

    await _remove_while_held(hass, entry)
    assert cancelled.is_set()  # unload does not wait for the slow API
    release.set()
    await _saves_flushed(hass)

    assert f"{DOMAIN}.stalled" not in hass_storage


async def test_a_refresh_ending_after_removal_saves_nothing(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh not tied to the entry (e.g. ``homeassistant.update_entity``) outlives unload."""
    hass = hass_with_recorder
    entry = await set_up(entry_id="late")
    coordinator = entry.runtime_data.coordinator
    entered, release, _ = _hold(monkeypatch)
    refresh = hass.async_create_background_task(coordinator.async_refresh(), "refresh")
    await entered.wait()

    await _remove_while_held(hass, entry)
    release.set()
    await refresh
    await _saves_flushed(hass)

    assert f"{DOMAIN}.late" not in hass_storage


@pytest.mark.parametrize("step", ["_take_prices", "_settle"])
async def test_a_bug_in_a_run_keeps_the_last_data(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    freezer: FrozenDateTimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    step: str,
) -> None:
    hass = hass_with_recorder
    entry = await set_up()
    coordinator = entry.runtime_data.coordinator
    before = hass.states.get(PRICE).attributes

    def broken(*args: Any) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(coordinator, step, broken)
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # weather run 08:15 UTC

    assert coordinator.last_update_success
    state = hass.states.get(PRICE)
    assert state.state != STATE_UNAVAILABLE
    assert state.attributes["stale"] is True
    assert state.attributes["issued_at"] == before["issued_at"]
    assert hass.states.get(PEAK).state in (STATE_ON, STATE_OFF)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and errors[0].exc_info[0] is RuntimeError


async def test_midnight_shifts_days_then_runs_at_0005(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = hass_with_recorder
    monkeypatch.setattr(sensor.DaySensor, "_attr_entity_registry_enabled_default", True)
    await set_up()
    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)
    await advance(hass, freezer, datetime(2026, 10, 22, 23, 50, tzinfo=BERLIN))

    def days() -> list[str]:
        return [hass.states.get(f"sensor.spot_forecast_de_lu_day_{i}").state for i in range(7)]

    before = days()
    attrs = hass.states.get(PRICE).attributes
    assert attrs["prices"][0]["time"] == "2026-10-22T00:00:00+02:00"
    issued = attrs["issued_at"]

    await advance(hass, freezer, datetime(2026, 10, 23, 0, 0, tzinfo=BERLIN))  # the tick
    attrs = hass.states.get(PRICE).attributes
    assert attrs["prices"][0]["time"] == "2026-10-23T00:00:00+02:00"
    assert attrs["issued_at"] == issued  # no run yet
    assert days()[:6] == before[1:]

    await advance(hass, freezer, datetime(2026, 10, 23, 0, 5, tzinfo=BERLIN))  # rollover run
    attrs = hass.states.get(PRICE).attributes
    assert datetime.fromisoformat(attrs["issued_at"]) == datetime(2026, 10, 23, 0, 5, tzinfo=BERLIN)
    assert attrs["anchor_day"] == "2026-10-23"
    assert attrs["prices"][0]["time"] == "2026-10-23T00:00:00+02:00"
    assert attrs["stale"] is False


async def test_runs_never_change_a_published_ledger(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """The Store serialises a saved ledger in a worker thread: a run must swap, not change it."""
    hass = hass_with_recorder
    entry = await set_up()
    coordinator = entry.runtime_data.coordinator
    published = coordinator.ledger
    saved = published.to_json()

    sources.serve("energy_charts_price_tomorrow.json")
    await advance(hass, freezer, T14)  # a new run, new actuals, new statistics

    assert coordinator.ledger is not published
    assert coordinator.ledger.to_json() != saved
    assert published.to_json() == saved


async def test_fall_back_day_end_to_end(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """2026-10-25 has 25 hours: in the forecast, in the statistics and on the clock."""
    hass = hass_with_recorder
    fall_back = date(2026, 10, 25)
    first_two = datetime(2026, 10, 25, 0, tzinfo=UTC)  # 02:00 CEST
    second_two = datetime(2026, 10, 25, 1, tzinfo=UTC)  # 02:00 CET
    freezer.move_to(datetime(2026, 10, 24, 10, tzinfo=BERLIN))
    sources.serve("energy_charts_price_tomorrow.json", through=date(2026, 10, 24))
    await set_up()

    attrs = hass.states.get(PRICE).attributes
    assert attrs["anchor_day"] == "2026-10-24"
    day = [e for e in attrs["prices"] if e["time"].startswith("2026-10-25")]
    assert len(day) == 25
    assert {e["source"] for e in day} == {"forecast"} and {e["h"] for e in day} == {1}
    times = [e["time"] for e in day]
    assert times[2:4] == ["2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"]

    # 14:00: the fall-back day's 100 quarter hours are published
    sources.serve(
        "energy_charts_price_tomorrow.json",
        through=fall_back,
        price_overrides={first_two: 11.0, second_two: 22.0},
    )
    await advance(hass, freezer, datetime(2026, 10, 24, 14, tzinfo=BERLIN))
    attrs = hass.states.get(PRICE).attributes
    assert attrs["anchor_day"] == "2026-10-25"
    market = [e for e in attrs["prices"] if e["time"].startswith("2026-10-25")]
    assert len(market) == 100 and {e["source"] for e in market} == {"market"}

    def of_day(rows: list[dict]) -> list[datetime]:
        starts = [datetime.fromtimestamp(r["start"], UTC) for r in rows]
        return [t for t in starts if t.astimezone(BERLIN).date() == fall_back]

    for statistic_id in (STAT_PRICE, STAT_H1):
        hours = of_day(await stat_rows(hass, statistic_id))
        assert len(hours) == len(set(hours)) == 25, statistic_id

    # the clock across both 02:00: the price follows the quarter hours
    shown = {}
    for at in (
        first_two - timedelta(minutes=15),
        first_two,
        first_two + timedelta(minutes=45),
        second_two,
        second_two + timedelta(minutes=15),
    ):
        await advance(hass, freezer, at)
        state = hass.states.get(PRICE)
        assert state.state != STATE_UNAVAILABLE
        shown[at] = float(state.state)
    assert shown[first_two] == 0.011
    assert shown[second_two] == 0.022
    assert hass.states.get(PEAK).state in (STATE_ON, STATE_OFF)
