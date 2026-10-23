"""Which day's prices are the latest known, and when to run next."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .ingest import MarketSlot

# Provisional: UTC times shortly after the weather model runs become available.
WEATHER_RUNS_UTC = (time(2, 15), time(8, 15), time(14, 15), time(20, 15))

POLL_START = time(13, 0)  # local; day-ahead prices appear around 13:00
POLL_END = time(16, 0)  # local, exclusive
POLL_STEP = timedelta(minutes=15)
ROLLOVER = time(0, 5)  # local
# more than the longest regular gap between runs (6 h between weather refreshes)
STALE_AFTER = timedelta(hours=7)
DROP_AFTER = timedelta(hours=36)  # the forecast is no longer shown


def _local_midnight(day: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time(0), zone).astimezone(UTC)


def choose_anchor(now: datetime, market: Sequence[MarketSlot], tz: str) -> date:
    """Tomorrow (local) if the market slots cover all of it without gaps, else today."""
    zone = ZoneInfo(tz)
    today = now.astimezone(zone).date()
    tomorrow = today + timedelta(days=1)
    cursor = _local_midnight(tomorrow, zone)
    end = _local_midnight(tomorrow + timedelta(days=1), zone)
    for slot in sorted(market, key=lambda s: s.start):
        if slot.end <= cursor:
            continue
        if slot.start > cursor:
            break
        cursor = slot.end
        if cursor >= end:
            return tomorrow
    return today


def next_run(now: datetime, anchor_is_tomorrow: bool, tz: str) -> datetime:
    """Earliest of day-ahead poll, weather refresh and midnight rollover after ``now`` (UTC)."""
    zone = ZoneInfo(tz)
    now = now.astimezone(UTC)
    local = now.astimezone(zone)
    candidates: list[datetime] = []

    if not anchor_is_tomorrow:
        if local.time() < POLL_START:
            candidates.append(datetime.combine(local.date(), POLL_START, zone))
        elif local.time() < POLL_END:
            step = POLL_STEP.total_seconds()
            tick = now.replace(second=0, microsecond=0)
            tick = datetime.fromtimestamp((tick.timestamp() // step + 1) * step, UTC)
            if tick.astimezone(zone).time() < POLL_END:
                candidates.append(tick)

    for day in (now.date(), now.date() + timedelta(days=1)):
        candidates.extend(datetime.combine(day, t, UTC) for t in WEATHER_RUNS_UTC)

    for day in (local.date(), local.date() + timedelta(days=1)):
        candidates.append(datetime.combine(day, ROLLOVER, zone))

    return min(c.astimezone(UTC) for c in candidates if c.astimezone(UTC) > now)


def is_stale(last_success: datetime | None, now: datetime, last_attempt_failed: bool) -> bool:
    """True when the last run attempt failed, without a success, or when it is over 7 h old."""
    return last_attempt_failed or last_success is None or now - last_success > STALE_AFTER


def is_expired(last_success: datetime | None, now: datetime) -> bool:
    """True without a success or when the last one is more than 36 h old: drop the forecast."""
    return last_success is None or now - last_success > DROP_AFTER
