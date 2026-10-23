"""Coordinator, Store, statistics and entities against a real Home Assistant.

Time is frozen to 2026-10-22 10:00 Europe/Berlin, before the day-ahead auction; the
sources serve the recorded payloads (see ``conftest.Sources``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.spot_forecast import coordinator as coordinator_module
from custom_components.spot_forecast import sensor
from custom_components.spot_forecast.const import (
    CONF_PEAK_THRESHOLD,
    CONF_SURCHARGE,
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


async def test_total_price_sensor_adds_the_surcharge(
    hass_with_recorder: HomeAssistant, set_up: SetUp, recorded
) -> None:
    hass = hass_with_recorder
    await set_up({CONF_SURCHARGE: 0.187})

    price, total = hass.states.get(PRICE), hass.states.get(TOTAL)
    assert float(total.state) == round(_market_price(recorded, T10) + 0.187, 5)
    assert total.attributes["unit_of_measurement"] == "EUR/kWh"
    assert total.attributes["surcharge"] == 0.187
    assert price.attributes["surcharge"] == 0.187  # the cards read it here
    market, shifted = price.attributes["prices"], total.attributes["prices"]
    assert [e["time"] for e in shifted] == [e["time"] for e in market]
    assert [e["price"] for e in shifted] == [round(e["price"] + 0.187, 5) for e in market]
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
    assert hass.states.get(PEAK).state != STATE_UNAVAILABLE


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
