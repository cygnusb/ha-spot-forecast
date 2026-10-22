"""Past forecasts on install, so the look-back statistics and the calibration have history.

Out of sample only: target days before the model's training cutoff come from the bundled
walk-forward backfill file; target days from the cutoff on are recomputed on the device
from archived weather forecasts (Open-Meteo previous runs) and Energy-Charts prices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
import logging
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import PRICE_HISTORY_DAYS
from .core.features import HORIZONS, Hourly, InsufficientHistory
from .core.forecast import run_forecast
from .core.ingest import hourly_prices
from .core.ledger import ISSUE_HOUR, Run, runs_from_backfill_file
from .core.model import ModelFile
from .core.zones import Zone
from .sources.energy_charts import fetch_prices
from .sources.http import SourceError
from .sources.open_meteo import fetch_previous_runs

if TYPE_CHECKING:
    from .coordinator import SpotForecastCoordinator

_LOGGER = logging.getLogger(__name__)

_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_MAX_H = max(HORIZONS)


@dataclass(frozen=True)
class BackfillResult:
    """What one backfill pass added."""

    imported: int  # runs from the bundled file that were stored
    recomputed: int  # runs recomputed on the device that were stored
    finalized_days: int  # target days whose forecast statistics this pass wrote
    error: str | None = None  # why the pass stopped early; None when it completed


async def run_backfill(
    hass: HomeAssistant, coordinator: SpotForecastCoordinator, days: int
) -> BackfillResult:
    """Backfill forecasts for the target days in [today - ``days``, today).

    Bundled runs cover the target days before the training cutoff; the rest is recomputed,
    from anchors up to six days before the first such target day through yesterday, and
    keeps target days from today on, which complete later. The window's prices are recorded
    together with the bundled runs, so every row is calibrated with all earlier rows of the
    window. Anchors whose prices are too short, or whose forecast fails, are skipped. A
    source error ends the pass with what was added so far (``error`` set); the window is
    then not marked done, so the next start tries again.
    """
    model, zone = coordinator.model, coordinator.zone
    tz = ZoneInfo(zone.tz)
    today = dt_util.utcnow().astimezone(tz).date()
    first = today - timedelta(days=days)
    start = max(model.train_cutoff, first)  # first recomputed target day
    session = async_get_clientsession(hass)

    bundled = await hass.async_add_executor_job(
        runs_from_backfill_file,
        coordinator.config_entry.runtime_data.backfill,
        model.version,
        zone.tz,
    )
    try:
        market = await fetch_prices(
            session,
            zone.code,
            min(first, start - (_MAX_H + PRICE_HISTORY_DAYS) * _DAY),
            today,
        )
    except SourceError as err:
        return _stopped(err, BackfillResult(0, 0, 0))
    prices = hourly_prices(market)
    in_window = [r for r in (_targets_in(run, first, model.train_cutoff) for run in bundled) if r]
    imported, finalized = await coordinator.async_add_history(
        _merged(coordinator.ledger.runs, in_window), actual=_since(prices, _midnight(first, tz))
    )

    candidates = [start + (i - _MAX_H) * _DAY for i in range((today - start).days + _MAX_H)]
    anchors = [a for a in candidates if _needs_recompute(coordinator.ledger.runs.get(a), start)]
    recomputed = 0
    if anchors:
        try:
            weather = await fetch_previous_runs(
                session, zone, max(start - _DAY, anchors[0]), anchors[-1] + _MAX_H * _DAY
            )
        except SourceError as err:
            return _stopped(err, BackfillResult(imported, 0, len(finalized)))
        runs = []
        for anchor in anchors:  # one job per anchor: a cancelled backfill stops promptly
            run = await hass.async_add_executor_job(
                _recompute, model, zone, anchor, prices, weather, start
            )
            if run is not None:
                runs.append(run)
        recomputed, days_written = await coordinator.async_add_history(
            _merged(coordinator.ledger.runs, runs)
        )
        finalized |= days_written

    coordinator.mark_backfill_done(days)
    result = BackfillResult(imported, recomputed, len(finalized))
    _LOGGER.debug("Backfill of %d days done: %s", days, result)
    return result


def _stopped(err: SourceError, result: BackfillResult) -> BackfillResult:
    _LOGGER.warning("Backfill stopped early, it is retried on the next start: %s", err)
    return replace(result, error=str(err))


def _recompute(
    model: ModelFile,
    zone: Zone,
    anchor: date,
    prices: Hourly,
    weather: Mapping[int, Mapping[str, Hourly]],
    first_target: date,
) -> Run | None:
    """The backfill run of ``anchor`` with the target days from ``first_target`` on (blocking).

    None when the anchor cannot be forecast: too short prices, or a bug in the forecast,
    which is logged and skips only this anchor (as a live run keeps the last forecast).
    """
    tz = ZoneInfo(zone.tz)
    known = _until(prices, _midnight(anchor + _DAY, tz))  # what was published by then
    try:
        slots = run_forecast(model, anchor, known, weather, zone)
    except InsufficientHistory as err:  # IncompleteAnchorDay included
        _LOGGER.debug("Backfill skips anchor %s: %s", anchor, err)
        return None
    except Exception:  # a bug in core: skip this anchor, keep the rest
        _LOGGER.exception("Backfill forecast for %s failed, skipping it", anchor)
        return None
    issued = datetime.combine(anchor, time(ISSUE_HOUR), tz).astimezone(UTC)
    kept = tuple(s for s in slots if anchor + timedelta(days=s.h) >= first_target)
    return Run(anchor, issued, model.version, kept, True)


def _targets_in(run: Run, first: date, end: date) -> Run | None:
    """``run`` with the slots whose target day is in [first, end); None if none is."""
    slots = tuple(s for s in run.slots if first <= run.anchor + timedelta(days=s.h) < end)
    return replace(run, slots=slots) if slots else None


def _needs_recompute(stored: Run | None, first_target: date) -> bool:
    """Whether the anchor's stored run lacks a target day from ``first_target`` on.

    A live run is never recomputed: what the user saw stays.
    """
    if stored is None:
        return True
    if not stored.backfill:
        return False
    have = {stored.anchor + timedelta(days=s.h) for s in stored.slots}
    need = {stored.anchor + timedelta(days=h) for h in HORIZONS}
    return not {d for d in need if d >= first_target} <= have


def _merged(stored: Mapping[date, Run], runs: Sequence[Run]) -> list[Run]:
    """``runs`` completed with the slots a stored backfill run of the same anchor holds.

    The bundled part and the recompute can each hold some target days of one anchor (the
    last anchors before the cutoff), and a wider window adds target days to stored runs.
    """
    out = []
    for run in runs:
        old = stored.get(run.anchor)
        if old is not None and old.backfill:
            slots = {(s.time, s.h): s for s in old.slots} | {(s.time, s.h): s for s in run.slots}
            run = replace(run, slots=tuple(sorted(slots.values(), key=lambda s: (s.h, s.time))))
        out.append(run)
    return out


def _midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time(), tz).astimezone(UTC)


def _until(s: Hourly, end: datetime) -> Hourly:
    """``s`` cut to the hours before ``end``."""
    return Hourly(s.start, s.values[: max((end - s.start) // _HOUR, 0)])


def _since(s: Hourly, start: datetime) -> Hourly:
    """``s`` cut to the hours from ``start`` on."""
    i = min(max((start - s.start) // _HOUR, 0), len(s.values))
    return Hourly(s.start + i * _HOUR, s.values[i:])
