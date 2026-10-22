"""Scheduled forecast runs, the quarter-hour clock tick and the data the entities show."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
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
from .const import CONF_PEAK_THRESHOLD, CONF_ZONE, DOMAIN, PRICE_HISTORY_DAYS
from .core.features import HORIZONS, InsufficientHistory
from .core.forecast import Slot, run_forecast
from .core.ingest import MarketSlot, hourly_prices
from .core.ledger import Ledger, Run
from .core.model import ModelFile
from .core.output import prices_attribute
from .core.schedule import choose_anchor, is_stale, next_run
from .core.zones import ZONES, Zone
from .sources.energy_charts import fetch_prices
from .sources.http import SourceError
from .sources.open_meteo import fetch_forecast
from .store import ForecastStore

_LOGGER = logging.getLogger(__name__)

MARKET_KEEP_DAYS = 3  # market slots kept for display and in the Store, before today


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
        self._started = False  # between async_start and unload / HA stop

    @property
    def peak_threshold(self) -> float:
        """Peak threshold in ct/kWh: the option, else the model's (EUR/MWh) rounded to 0.1."""
        option = self.config_entry.options.get(CONF_PEAK_THRESHOLD)
        return float(option) if option is not None else round(self.model.peak_threshold / 10, 1)

    async def _async_setup(self) -> None:
        """Rebuild the last state from the Store before the first run."""
        self._ledger, self._market, self._last_success = await self._store.async_load()
        self._calibrate(dt_util.utcnow())

    async def _async_update_data(self) -> SpotData:
        now = dt_util.utcnow()
        try:
            await self._async_run(now)
        except (SourceError, InsufficientHistory) as err:
            if not self._market and self._latest_run() is None:
                raise UpdateFailed(f"no forecast yet and nothing stored: {err}") from err
            _LOGGER.warning("Forecast run failed, keeping the last forecast: %s", err)
        long_term_statistics.async_write(
            self.hass,
            self._ledger.finalize(
                self.zone.tz,
                zone_id=self.zone.slug,
                default_p90=self.model.p90_offsets,
                default_p10=self.model.p10_offsets,
            ),
        )
        self._ledger.prune()
        self._calibrate(now)
        self._store.async_save(self._ledger, self._market, self._last_success)
        return self._build(now)

    async def _async_run(self, now: datetime) -> None:
        """Fetch prices and weather, forecast and record the run; raises on missing data.

        Fresh prices are kept even when the weather fails afterwards.
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
        slots = await self.hass.async_add_executor_job(
            run_forecast, self.model, anchor, prices, dict.fromkeys(HORIZONS, weather), self.zone
        )
        self._ledger.add_run(Run(anchor, now, self.model.version, tuple(slots), False))
        self._last_success = now

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
