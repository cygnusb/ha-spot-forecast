"""Backfill on install: bundled out-of-sample runs, then on-device recompute after the cutoff.

Time is frozen to 2026-10-22 10:00 Europe/Berlin. The bundled model's training cutoff is
2026-10-01 and its backfill file covers the target days 2026-04-04 … 2026-09-30, so with a
30-day window 2026-09-25 comes from the file and 2026-10-05 is recomputed from the
recorded previous runs (see ``conftest.Sources``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
import gzip
import json
import logging
from typing import Any
from zoneinfo import ZoneInfo

from conftest import PREVIOUS_RUNS_URL, T10, SetUp, Sources, advance, stat_rows
from freezegun.api import FrozenDateTimeFactory
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
import numpy as np
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.spot_forecast import backfill as backfill_module
from custom_components.spot_forecast.const import CONF_BACKFILL_DAYS, DOMAIN
from custom_components.spot_forecast.coordinator import LEDGER_KEEP_DAYS
from custom_components.spot_forecast.core.ledger import Ledger
from custom_components.spot_forecast.model_loader import MODELS_DIR

BERLIN = ZoneInfo("Europe/Berlin")
TODAY = date(2026, 10, 22)
BACKFILL = "sensor.spot_forecast_de_lu_backfill_status"


def _hours(day: date) -> list[datetime]:
    start = datetime.combine(day, time(), BERLIN).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time(), BERLIN).astimezone(UTC)
    return [start + timedelta(hours=i) for i in range((end - start) // timedelta(hours=1))]


def _rows_of(rows: list[dict], day: date) -> dict[datetime, float]:
    """Statistic means of the local day ``day``, by UTC hour start."""
    hours = set(_hours(day))
    by_start = {datetime.fromtimestamp(r["start"], UTC): r["mean"] for r in rows}
    return {t: v for t, v in by_start.items() if t in hours}


def _previous_runs_calls(aioclient_mock: AiohttpClientMocker) -> int:
    return sum(str(url).startswith(PREVIOUS_RUNS_URL) for _, url, _, _ in aioclient_mock.mock_calls)


def _bundled_medians(h: int, day: date) -> dict[datetime, float]:
    """EUR/kWh medians of the bundled backfill file for target day ``day`` at horizon ``h``."""
    doc = json.loads(gzip.decompress((MODELS_DIR / "de_lu.backfill.json.gz").read_bytes()))
    hours = set(_hours(day))
    return {
        t: round(median / 1000, 5)
        for t_iso, hh, median, _, _ in doc["rows"]
        if hh == h and (t := datetime.fromisoformat(t_iso).astimezone(UTC)) in hours
    }


async def test_backfill_imports_bundled_rows_and_recomputes_after_cutoff(
    hass_with_recorder: HomeAssistant, set_up: SetUp, aioclient_mock: AiohttpClientMocker
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30})
    coordinator = entry.runtime_data.coordinator

    result = coordinator.backfill_result
    assert result is not None
    assert result.imported > 0 and result.recomputed > 0
    assert coordinator.backfill_done_days == 30
    # 2026-09-30 (GMT; local 2026-10-01 starts 22:00 UTC) through 2026-10-27: two calendar
    # months, two point sets each
    assert _previous_runs_calls(aioclient_mock) == 4

    h1 = await stat_rows(hass, "spot_forecast:de_lu_forecast_h1")
    # before the cutoff: the bundled walk-forward rows, unchanged
    bundled = _rows_of(h1, date(2026, 9, 25))
    assert len(bundled) == 24
    assert bundled == pytest.approx(_bundled_medians(1, date(2026, 9, 25)))
    # after the cutoff: recomputed on the device
    recomputed = _rows_of(h1, date(2026, 10, 5))
    assert len(recomputed) == 24
    run = coordinator.ledger.runs[date(2026, 10, 4)]
    assert run.backfill
    assert recomputed == pytest.approx(
        {s.time: round(s.median / 1000, 5) for s in run.slots if s.h == 1}
    )
    # the first target days after the cutoff get every horizon, from anchors before it
    for h in range(1, 7):
        rows = await stat_rows(hass, f"spot_forecast:de_lu_forecast_h{h}")
        assert len(_rows_of(rows, date(2026, 10, 3))) == 24, h
    # nothing before the window
    assert _rows_of(h1, date(2026, 9, 21)) == {}
    assert result.finalized_days == len(
        {d for d, s in coordinator.ledger.finalized if s != "price" and d >= TODAY - timedelta(30)}
    )


async def test_backfill_runs_once_and_again_when_window_grows(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 10})
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 10
    # targets from 2026-10-12: anchors 2026-10-06 … 2026-10-21, nothing from the file
    assert coordinator.backfill_result.recomputed == 16
    assert coordinator.backfill_result.imported == 0

    sources.serve()  # clears the call log
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 10  # read back from the Store
    assert coordinator.backfill_result is None
    assert _previous_runs_calls(aioclient_mock) == 0

    sources.serve()
    hass.config_entries.async_update_entry(entry, options={CONF_BACKFILL_DAYS: 20})
    await hass.async_block_till_done(wait_background_tasks=True)  # the update reloads
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 20
    # targets from 2026-10-02: anchors 2026-09-26 … 2026-10-10 are new or incomplete;
    # 2026-10-11 … 2026-10-21 already hold every target day from 2026-10-12 on
    assert coordinator.backfill_result.recomputed == 15
    assert _previous_runs_calls(aioclient_mock) == 2


async def test_backfill_source_error_keeps_imports_and_retries_on_next_start(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources
) -> None:
    hass = hass_with_recorder
    sources.serve(previous_runs_status=500)
    entry = await set_up({CONF_BACKFILL_DAYS: 30})
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 0
    assert coordinator.backfill_result.imported > 0
    assert coordinator.backfill_result.recomputed == 0
    state = hass.states.get(BACKFILL)
    assert state.state == "failed"
    assert "previous-runs-api.open-meteo.com" in state.attributes["last_error"]
    assert state.attributes["days"] == 0
    h1 = await stat_rows(hass, "spot_forecast:de_lu_forecast_h1")
    assert len(_rows_of(h1, date(2026, 9, 25))) == 24  # the bundled import stays

    sources.serve()
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 30
    assert coordinator.backfill_result.recomputed > 0
    h1 = await stat_rows(hass, "spot_forecast:de_lu_forecast_h1")
    assert len(_rows_of(h1, date(2026, 10, 5))) == 24
    assert hass.states.get(BACKFILL).state == "done"


async def test_backfill_waits_for_a_successful_run(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """A start on stored data with the sources down defers the backfill to the next run."""
    hass = hass_with_recorder
    entry = await set_up()
    sources.fail_all()
    hass.config_entries.async_update_entry(entry, options={CONF_BACKFILL_DAYS: 10})
    await hass.async_block_till_done(wait_background_tasks=True)  # the update reloads
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_result is None
    assert hass.states.get(BACKFILL).state == "pending"

    sources.serve()
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # weather run 08:15 UTC
    assert coordinator.backfill_result is not None
    assert coordinator.backfill_done_days == 10
    assert hass.states.get(BACKFILL).state == "done"


async def test_store_without_backfill_record_backfills(
    hass_with_recorder: HomeAssistant, set_up: SetUp, hass_storage: dict
) -> None:
    """A Store written before the backfill existed reads as nothing backfilled yet."""
    hass_storage[f"{DOMAIN}.old"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.old",
        "data": {"ledger": Ledger().to_json(), "market": [], "last_success": None},
    }
    entry = await set_up({CONF_BACKFILL_DAYS: 10}, entry_id="old")
    assert entry.runtime_data.coordinator.backfill_done_days == 10


def _p90_offset(ledger: Ledger, h: int, anchor: date) -> tuple[float, int]:
    """0.9-quantile of actual - raw P90 at horizon ``h`` over target days before ``anchor``.

    Written out here instead of calling ``Ledger.offsets``, which the code under test uses.
    """
    lo = anchor - timedelta(days=56)
    residuals = [
        ledger.actual[s.time] - s.p90_raw
        for run in ledger.runs.values()
        for s in run.slots
        if s.h == h
        and s.valid
        and s.time in ledger.actual
        and lo <= run.anchor + timedelta(days=h) < anchor
    ]
    return float(np.quantile(residuals, 0.9)) if residuals else float("nan"), len(residuals)


async def test_backfilled_calibration_uses_only_earlier_rows(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 30})
    coordinator = entry.runtime_data.coordinator
    ledger = coordinator.ledger
    peak_h1 = await stat_rows(hass, "spot_forecast:de_lu_peak_risk_h1")

    # 2026-10-20 (anchor 10-19): enough earlier residuals for a data-driven offset, and
    # one that differs from the offset all rows up to today would give
    anchor = date(2026, 10, 19)
    offset, n = _p90_offset(ledger, 1, anchor)
    assert n >= 200
    assert offset != pytest.approx(_p90_offset(ledger, 1, TODAY)[0], abs=1e-3)
    run = ledger.runs[anchor]
    expected = {
        s.time: round(max(s.p90_raw + offset, s.median) / 1000, 5) for s in run.slots if s.h == 1
    }
    assert _rows_of(peak_h1, anchor + timedelta(days=1)) == pytest.approx(expected)

    # the window's first target day has no earlier rows: the model's default offset
    first = TODAY - timedelta(days=30)
    assert _p90_offset(ledger, 1, first - timedelta(days=1))[1] == 0
    default = coordinator.model.p90_offsets[1]
    run = ledger.runs[first - timedelta(days=1)]
    expected = {
        s.time: round(max(s.p90_raw + default, s.median) / 1000, 5) for s in run.slots if s.h == 1
    }
    assert _rows_of(peak_h1, first) == pytest.approx(expected)


async def test_bundled_rows_use_every_earlier_row_of_the_window(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    """60-day window: a bundled day inside the 40-day price history the first run fetched
    is calibrated with the window's older rows too, as if every actual came first."""
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 60})
    coordinator = entry.runtime_data.coordinator
    ledger = coordinator.ledger
    day = date(2026, 9, 15)  # bundled; the first run's prices start 2026-09-12
    offset, n = _p90_offset(ledger, 1, day - timedelta(days=1))
    assert n >= 200
    assert offset != pytest.approx(coordinator.model.p90_offsets[1], abs=1e-3)

    pure = Ledger(
        runs={a: r for a, r in ledger.runs.items() if r.backfill}, actual=dict(ledger.actual)
    )
    rows = pure.finalize(
        "Europe/Berlin",
        today=TODAY,
        zone_id="de_lu",
        default_p90=coordinator.model.p90_offsets,
        default_p10=coordinator.model.p10_offsets,
    )
    hours = set(_hours(day))
    expected = {
        r.start: r.value
        for r in rows
        if r.statistic_id == "spot_forecast:de_lu_peak_risk_h1" and r.start in hours
    }
    assert len(expected) == 24
    peak_h1 = await stat_rows(hass, "spot_forecast:de_lu_peak_risk_h1")
    assert _rows_of(peak_h1, day) == pytest.approx(expected)


async def test_backfill_status_sensor(hass_with_recorder: HomeAssistant, set_up: SetUp) -> None:
    hass = hass_with_recorder
    entry = await set_up()
    assert hass.states.get(BACKFILL).state == "off"
    assert hass.states.get(BACKFILL).attributes["options"] == [
        "off",
        "pending",
        "running",
        "done",
        "failed",
    ]

    hass.config_entries.async_update_entry(entry, options={CONF_BACKFILL_DAYS: 10})
    await hass.async_block_till_done(wait_background_tasks=True)  # the update reloads
    state = hass.states.get(BACKFILL)
    assert state.state == "done"
    result = entry.runtime_data.coordinator.backfill_result
    assert state.attributes["days"] == 10
    assert state.attributes["imported"] == 0
    assert state.attributes["recomputed"] == 16
    assert state.attributes["finalized_days"] == result.finalized_days > 0
    assert state.attributes["last_error"] is None


async def test_backfill_skips_an_anchor_whose_forecast_fails(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hass = hass_with_recorder
    forecast = backfill_module.run_forecast

    def broken_once(model: Any, anchor: date, *args: Any) -> Any:
        if anchor == date(2026, 10, 10):
            raise IndexError("bug in core")
        return forecast(model, anchor, *args)

    monkeypatch.setattr(backfill_module, "run_forecast", broken_once)
    entry = await set_up({CONF_BACKFILL_DAYS: 10})

    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_result.recomputed == 15  # 16 anchors, one skipped
    assert date(2026, 10, 10) not in coordinator.ledger.runs
    assert hass.states.get(BACKFILL).state == "done"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and errors[0].exc_info[0] is IndexError


async def test_backfill_bug_sets_failed_state(
    hass_with_recorder: HomeAssistant, set_up: SetUp, monkeypatch: pytest.MonkeyPatch
) -> None:
    hass = hass_with_recorder

    def unreadable(*args: Any) -> Any:
        raise ValueError("unsupported backfill format 2")

    monkeypatch.setattr(backfill_module, "runs_from_backfill_file", unreadable)
    entry = await set_up({CONF_BACKFILL_DAYS: 10})

    assert entry.state is ConfigEntryState.LOADED
    state = hass.states.get(BACKFILL)
    assert state.state == "failed"
    assert state.attributes["last_error"] == "ValueError: unsupported backfill format 2"
    assert entry.runtime_data.coordinator.backfill_done_days == 0


async def test_unload_stops_a_running_backfill(
    hass_with_recorder: HomeAssistant, set_up: SetUp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unload cancels the backfill (it never resumes); the next start does it from scratch."""
    hass = hass_with_recorder
    release = asyncio.Event()
    fetch = backfill_module.fetch_previous_runs

    async def held(*args: Any, **kwargs: Any) -> Any:
        await release.wait()
        return await fetch(*args, **kwargs)

    monkeypatch.setattr(backfill_module, "fetch_previous_runs", held)
    entry = await set_up({CONF_BACKFILL_DAYS: 30}, wait=False)
    assert hass.states.get(BACKFILL).state == "running"

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)  # hangs if it survived
    assert not release.is_set()

    monkeypatch.setattr(backfill_module, "fetch_previous_runs", fetch)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 30
    assert coordinator.backfill_result.recomputed > 0
    assert hass.states.get(BACKFILL).state == "done"


async def test_failed_backfill_is_retried_a_day_later(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """After a successful run, at most once per 24 h; no restart needed."""
    hass = hass_with_recorder
    sources.serve(previous_runs_status=500)
    entry = await set_up({CONF_BACKFILL_DAYS: 30})
    coordinator = entry.runtime_data.coordinator
    assert hass.states.get(BACKFILL).state == "failed"

    sources.serve()
    await advance(hass, freezer, T10 + timedelta(minutes=15))  # a successful run
    assert hass.states.get(BACKFILL).state == "failed"
    assert _previous_runs_calls(aioclient_mock) == 0

    sources.serve("energy_charts_price_tomorrow.json")  # 2026-10-23 complete: its run works
    await advance(hass, freezer, T10 + timedelta(days=1, minutes=15))
    assert _previous_runs_calls(aioclient_mock) > 0
    assert hass.states.get(BACKFILL).state == "done"
    assert coordinator.backfill_done_days == 30
    assert coordinator.backfill_error is None


async def test_afternoon_install_recomputes_anchor_today(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    """At 14:30 the live run is anchored on tomorrow; today's anchor comes from the backfill."""
    freezer.move_to(datetime(2026, 10, 22, 14, 30, tzinfo=BERLIN))
    sources.serve("energy_charts_price_tomorrow.json")
    entry = await set_up({CONF_BACKFILL_DAYS: 10})
    coordinator = entry.runtime_data.coordinator

    assert not coordinator.ledger.runs[TODAY + timedelta(days=1)].backfill  # the live run
    run = coordinator.ledger.runs[TODAY]
    assert run.backfill
    assert {s.h for s in run.slots} == set(range(1, 7))
    assert run.issued_at == datetime(2026, 10, 22, 6, tzinfo=BERLIN)
    assert coordinator.backfill_result.recomputed == 17  # 2026-10-06 … 2026-10-22


async def test_wide_backfill_is_pruned_and_not_redone(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """180 days: old days get their rows, then the ledger keeps 70 days; a restart with the
    same option does not backfill again, though the pruned runs are gone."""
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 180})
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 180
    keep_from = TODAY - timedelta(days=LEDGER_KEEP_DAYS)
    assert min(coordinator.ledger.runs) >= keep_from
    assert min(d for d, _ in coordinator.ledger.finalized) >= keep_from
    old = date(2026, 8, 5)  # bundled, inside the served prices, older than 70 days
    assert old < keep_from
    h1 = await stat_rows(hass, "spot_forecast:de_lu_forecast_h1")
    assert _rows_of(h1, old) == pytest.approx(_bundled_medians(1, old))
    runs = dict(coordinator.ledger.runs)

    sources.serve()  # clears the call log
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data.coordinator
    assert coordinator.backfill_done_days == 180
    assert coordinator.backfill_result is None
    assert _previous_runs_calls(aioclient_mock) == 0
    assert {a for a, r in coordinator.ledger.runs.items() if r.backfill} == {
        a for a, r in runs.items() if r.backfill
    }
