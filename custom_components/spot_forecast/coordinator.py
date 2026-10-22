"""Scheduled forecast runs, the quarter-hour clock tick and the data the entities show."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
import logging
from zoneinfo import ZoneInfo

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CALLBACK_TYPE, Event, HassJob, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_point_in_utc_time, async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from . import long_term_statistics
from .backfill import BackfillResult, run_backfill
from .const import (
    CONF_BACKFILL_DAYS,
    CONF_PEAK_THRESHOLD,
    CONF_ZONE,
    DEFAULT_BACKFILL_DAYS,
    DOMAIN,
    MAX_BACKFILL_DAYS,
    PRICE_HISTORY_DAYS,
)
from .core.features import HORIZONS, Hourly, InsufficientHistory
from .core.forecast import Slot, run_forecast
from .core.ingest import MarketSlot, hourly_prices
from .core.ledger import PRICE, Ledger, Run, StatRow
from .core.model import ModelFile
from .core.output import prices_attribute
from .core.schedule import choose_anchor, is_stale, next_run
from .core.zones import ZONES, Zone
from .issues import async_check_model_age, async_delete_model_issue
from .sources.energy_charts import fetch_prices
from .sources.http import SourceError
from .sources.open_meteo import fetch_forecast
from .store import ForecastStore

_LOGGER = logging.getLogger(__name__)

MARKET_KEEP_DAYS = 3  # market slots kept for display and in the Store, before today
# ledger days kept: the widest backfill window, its anchors' horizons, and a margin
LEDGER_KEEP_DAYS = MAX_BACKFILL_DAYS + max(HORIZONS) + 14

BACKFILL_OFF = "off"
BACKFILL_PENDING = "pending"
BACKFILL_RUNNING = "running"
BACKFILL_DONE = "done"
BACKFILL_FAILED = "failed"
BACKFILL_STATES = (
    BACKFILL_OFF,
    BACKFILL_PENDING,
    BACKFILL_RUNNING,
    BACKFILL_DONE,
    BACKFILL_FAILED,
)


@dataclass(frozen=True)
class SpotData:
    """What the entities show; prices inside are EUR/MWh except ``entries`` (EUR/kWh)."""

    market: list[MarketSlot]
    forecast: list[Slot]  # the latest run's slots; empty while stale
    p90_off: dict[int, float]
    p10_off: dict[int, float]
    anchor: date
    issued_at: datetime | None
    last_success: datetime | None
    stale: bool
    model: ModelFile
    entries: list[dict]  # the ``prices`` attribute, from today 00:00 (market local time)
    accuracy: dict


class SpotForecastCoordinator(DataUpdateCoordinator[SpotData]):
    """Runs the forecast at the times ``core.schedule.next_run`` names, not on an interval."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, model: ModelFile) -> None:
        self.zone: Zone = ZONES[entry.data[CONF_ZONE]]
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {self.zone.code}",
            update_interval=None,
        )
        self.model = model
        self._store = ForecastStore(hass, entry.entry_id)
        self._ledger = Ledger()
        self._market: list[MarketSlot] = []
        self._last_success: datetime | None = None
        self._p90_off = dict(model.p90_offsets)
        self._p10_off = dict(model.p10_offsets)
        self._accuracy: dict = {}
        self._unsub_run: CALLBACK_TYPE | None = None
        self._unsub_tick: CALLBACK_TYPE | None = None
        self._unsub_stop: CALLBACK_TYPE | None = None
        self._run_lock = asyncio.Lock()  # runs, refreshes and backfill never interleave
        self._started = False  # between async_start and unload / HA stop
        self._ran = False  # a run succeeded since this setup
        self._backfill_done_days = 0  # the widest backfill window completed so far
        self._backfill_started = False  # one backfill attempt per setup
        self._backfill_running = False
        self.backfill_result: BackfillResult | None = None  # of this setup's backfill
        self.backfill_error: str | None = None  # why this setup's backfill failed

    @property
    def peak_threshold(self) -> float:
        """Peak threshold in ct/kWh: the option, else the model's (EUR/MWh) rounded to 0.1."""
        option = self.config_entry.options.get(CONF_PEAK_THRESHOLD)
        return float(option) if option is not None else round(self.model.peak_threshold / 10, 1)

    @property
    def ledger(self) -> Ledger:
        """The forecast ledger, to read; change it only through ``async_add_history``."""
        return self._ledger

    @property
    def backfill_done_days(self) -> int:
        """The widest backfill window (days) completed, 0 if none."""
        return self._backfill_done_days

    @property
    def _backfill_days(self) -> int:
        """The backfill window the options ask for; 0 turns the backfill off."""
        return int(self.config_entry.options.get(CONF_BACKFILL_DAYS, DEFAULT_BACKFILL_DAYS))

    @property
    def backfill_state(self) -> str:
        """One of ``BACKFILL_STATES``."""
        days = self._backfill_days
        if days <= 0:
            return BACKFILL_OFF
        if self._backfill_running:
            return BACKFILL_RUNNING
        if self._backfill_done_days >= days:
            return BACKFILL_DONE
        if self.backfill_error is not None:
            return BACKFILL_FAILED
        return BACKFILL_PENDING

    async def _async_setup(self) -> None:
        """Rebuild the last state from the Store before the first run."""
        entry_id = self.config_entry.entry_id
        self.config_entry.async_on_unload(lambda: async_delete_model_issue(self.hass, entry_id))
        (
            self._ledger,
            self._market,
            self._last_success,
            self._backfill_done_days,
        ) = await self._store.async_load()
        self._calibrate(dt_util.utcnow())

    async def _async_update_data(self) -> SpotData:
        async with self._run_lock:
            return await self._async_update_locked()

    async def _async_update_locked(self) -> SpotData:
        now = dt_util.utcnow()
        async_check_model_age(self.hass, self.config_entry.entry_id, self.model, self._today(now))
        try:
            await self._async_run(now)
        except (SourceError, InsufficientHistory) as err:
            if not self._market and self._latest_run() is None:
                raise UpdateFailed(f"no forecast yet and nothing stored: {err}") from err
            _LOGGER.warning("Forecast run failed, keeping the last forecast: %s", err)
        self._write_statistics()
        self._ledger.prune(LEDGER_KEEP_DAYS)
        self._calibrate(now)
        self._save()
        return self._build(now)

    def _write_statistics(self) -> None:
        """Queue the rows of every newly complete day; ``finalize`` marks them written.

        No ``await`` between finalize, the write and the save, so a cancelled task can never
        leave days marked written that were not.
        """
        long_term_statistics.async_write(self.hass, self._finalize(self._ledger)())

    def _finalize(self, ledger: Ledger) -> partial[list[StatRow]]:
        """``ledger.finalize`` with this zone's ids and the model's default offsets."""
        return partial(
            ledger.finalize,
            self.zone.tz,
            zone_id=self.zone.slug,
            default_p90=self.model.p90_offsets,
            default_p10=self.model.p10_offsets,
        )

    def _save(self) -> None:
        self._store.async_save(
            self._ledger, self._market, self._last_success, self._backfill_done_days
        )

    async def async_add_history(
        self, runs: Sequence[Run], actual: Hourly | None = None
    ) -> tuple[int, set[date]]:
        """Add backfilled runs and past prices (EUR/MWh) and write the statistics they complete.

        Holds the run lock, so it never interleaves with a run. The rows are built in the
        executor from a copy of the ledger, whose ``finalized`` is taken over only once the
        rows are back: cancelled in between, nothing counts as written. Returns how many
        of ``runs`` were stored and the target days that got forecast statistics.
        """
        async with self._run_lock:
            if actual is not None:
                self._ledger.record_actual(actual)
            stored = 0
            for run in runs:
                self._ledger.add_run(run)
                stored += self._ledger.runs.get(run.anchor) is run
            before = set(self._ledger.finalized)
            snapshot = Ledger(
                dict(self._ledger.runs), set(self._ledger.finalized), dict(self._ledger.actual)
            )
            rows = await self.hass.async_add_executor_job(self._finalize(snapshot))
            self._ledger.finalized = snapshot.finalized
            long_term_statistics.async_write(self.hass, rows)
            now = dt_util.utcnow()
            self._calibrate(now)
            self._save()
            if self.data is not None:
                self.data = self._build(now)
                self.async_update_listeners()
            written = {day for day, series in self._ledger.finalized - before if series != PRICE}
            return stored, written

    @callback
    def mark_backfill_done(self, days: int) -> None:
        """Record that the backfill of a ``days`` window completed; it reruns only for wider."""
        self._backfill_done_days = max(self._backfill_done_days, days)
        self._save()

    async def _async_run(self, now: datetime) -> None:
        """Fetch prices and weather, forecast and record the run; raises on missing data.

        Fresh prices are kept even when the weather or the forecast fails afterwards; an
        unexpected error in the forecast itself is logged, not raised.
        """
        today = self._today(now)
        session = async_get_clientsession(self.hass)
        market = await fetch_prices(
            session,
            self.zone.code,
            today - timedelta(days=PRICE_HISTORY_DAYS),
            today + timedelta(days=1),
        )
        prices = hourly_prices(market)
        self._ledger.record_actual(prices)
        keep_from = self._midnight(today - timedelta(days=MARKET_KEEP_DAYS))
        self._market = [m for m in market if m.start >= keep_from]

        anchor = choose_anchor(now, market, self.zone.tz)
        weather = await fetch_forecast(session, self.zone)
        try:
            slots = await self.hass.async_add_executor_job(
                run_forecast,
                self.model,
                anchor,
                prices,
                dict.fromkeys(HORIZONS, weather),
                self.zone,
            )
        except InsufficientHistory:
            raise
        except Exception:  # a bug in core: keep the last forecast, still show the prices
            _LOGGER.exception("Forecast for %s failed, keeping the last forecast", anchor)
            return
        self._ledger.add_run(Run(anchor, now, self.model.version, tuple(slots), False))
        self._last_success = now
        self._ran = True

    def _calibrate(self, now: datetime) -> None:
        """Offsets for the latest run's anchor and the 30-day accuracy, from the ledger."""
        run = self._latest_run()
        anchor = run.anchor if run is not None else self._today(now)
        self._p90_off, self._p10_off = self._ledger.offsets(
            anchor, self.model.p90_offsets, self.model.p10_offsets
        )
        self._accuracy = self._ledger.accuracy(now, self.peak_threshold * 10, tz=self.zone.tz)

    def _build(self, now: datetime) -> SpotData:
        """``SpotData`` for ``now`` from what is held in memory; no network."""
        run = self._latest_run()
        stale = is_stale(self._last_success, now)
        forecast = [] if stale or run is None else list(run.slots)
        if run is None:
            anchor, issued_at = choose_anchor(now, self._market, self.zone.tz), None
        else:
            anchor, issued_at = run.anchor, run.issued_at
        return SpotData(
            market=self._market,
            forecast=forecast,
            p90_off=self._p90_off,
            p10_off=self._p10_off,
            anchor=anchor,
            issued_at=issued_at,
            last_success=self._last_success,
            stale=stale,
            model=self.model,
            entries=prices_attribute(
                self._market,
                forecast,
                self._p90_off,
                self._p10_off,
                self.zone.tz,
                self._midnight(self._today(now)),
            ),
            accuracy=self._accuracy,
        )

    def _latest_run(self) -> Run | None:
        """The live run with the latest anchor (backfilled runs are history, not shown)."""
        live = [r for r in self._ledger.runs.values() if not r.backfill]
        return max(live, key=lambda r: r.anchor, default=None)

    def _today(self, now: datetime) -> date:
        return now.astimezone(ZoneInfo(self.zone.tz)).date()

    def _midnight(self, day: date) -> datetime:
        return datetime.combine(day, time(), ZoneInfo(self.zone.tz)).astimezone(dt_util.UTC)

    # --- scheduling ---------------------------------------------------------

    @callback
    def async_start(self) -> None:
        """Start the run schedule and the quarter-hour tick; both stop on unload and HA stop."""
        self._started = True
        self._unsub_tick = async_track_time_change(
            self.hass, self._async_tick, minute=(0, 15, 30, 45), second=0
        )
        self._schedule_next_run(dt_util.utcnow())
        self._unsub_stop = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_on_hass_stop
        )
        self.config_entry.async_on_unload(self._async_on_unload)
        self._maybe_start_backfill()

    @callback
    def _maybe_start_backfill(self) -> None:
        """Start the backfill in the background after this setup's first successful run.

        Once per setup, and only while the option's window is wider than the one done.
        """
        days = self._backfill_days
        if (
            self._backfill_started
            or not self._started
            or not self._ran
            or days <= self._backfill_done_days
        ):
            return
        self._backfill_started = True
        self.config_entry.async_create_background_task(
            self.hass, self._async_backfill(days), f"{self.name} backfill"
        )

    async def _async_backfill(self, days: int) -> None:
        self._backfill_running = True
        self.async_update_listeners()
        try:
            self.backfill_result = await run_backfill(self.hass, self, days)
            self.backfill_error = self.backfill_result.error
        except Exception as err:  # a bug: report it on the sensor, retry on the next start
            _LOGGER.exception("Backfill failed")
            self.backfill_error = f"{type(err).__name__}: {err}"
        finally:
            self._backfill_running = False
        self.async_update_listeners()

    def _schedule_next_run(self, now: datetime) -> None:
        if not self._started:  # a run that was still going during unload
            return
        if self._unsub_run is not None:
            self._unsub_run()
        run = self._latest_run()
        anchor_is_tomorrow = run is not None and run.anchor > self._today(now)
        when = next_run(now, anchor_is_tomorrow, self.zone.tz)
        self._unsub_run = async_track_point_in_utc_time(
            self.hass,
            HassJob(self._on_run_due, f"{self.name} run", cancel_on_shutdown=True),
            when,
        )

    @callback
    def _on_run_due(self, _now: datetime) -> None:
        self._unsub_run = None
        self.config_entry.async_create_task(
            self.hass, self._async_scheduled_run(), f"{self.name} scheduled run"
        )

    async def _async_scheduled_run(self) -> None:
        await self.async_refresh()
        self._schedule_next_run(dt_util.utcnow())
        self._maybe_start_backfill()

    @callback
    def _async_tick(self, _now: datetime) -> None:
        """Re-derive the shown data for the new quarter hour from what is held; no fetch."""
        if self.data is None:
            return
        self.data = self._build(dt_util.utcnow())
        self.async_update_listeners()

    @callback
    def _cancel_timers(self) -> None:
        self._started = False
        for unsub in (self._unsub_run, self._unsub_tick):
            if unsub is not None:
                unsub()
        self._unsub_run = self._unsub_tick = None

    @callback
    def _async_on_hass_stop(self, _event: Event) -> None:
        self._unsub_stop = None
        self._cancel_timers()

    async def _async_on_unload(self) -> None:
        self._cancel_timers()
        if self._unsub_stop is not None:
            self._unsub_stop()
            self._unsub_stop = None
        await self._store.async_flush()
