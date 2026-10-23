from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from core.ingest import MarketSlot
from core.schedule import WEATHER_RUNS_UTC, choose_anchor, is_expired, is_stale, next_run

TZ = "Europe/Berlin"
BERLIN = ZoneInfo(TZ)


def _local(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=BERLIN)


def _slots(day: date, step_min: int, skip=()):
    """Slots covering the local day ``day``; ``skip`` lists slot indexes to omit."""
    start = datetime.combine(day, time(0), BERLIN).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time(0), BERLIN).astimezone(UTC)
    step = timedelta(minutes=step_min)
    out, i = [], 0
    while start < end:
        if i not in skip:
            out.append(MarketSlot(start, start + step, 50.0))
        start += step
        i += 1
    return out


def test_anchor_tomorrow_when_published():
    now = _local(2026, 10, 22, 14, 5)
    for step in (15, 60):
        market = _slots(date(2026, 10, 21), step) + _slots(date(2026, 10, 22), step)
        assert choose_anchor(now, market + _slots(date(2026, 10, 23), step), TZ) == date(
            2026, 10, 23
        )


def test_anchor_stays_today_without_tomorrow():
    now = _local(2026, 10, 22, 14, 5)
    today = _slots(date(2026, 10, 22), 15)
    assert choose_anchor(now, today, TZ) == date(2026, 10, 22)
    assert choose_anchor(now, [], TZ) == date(2026, 10, 22)
    # partial tomorrow, and tomorrow with a gap in the middle
    assert choose_anchor(now, today + _slots(date(2026, 10, 23), 15)[:50], TZ) == date(2026, 10, 22)
    gap = today + _slots(date(2026, 10, 23), 15, skip={40})
    assert choose_anchor(now, gap, TZ) == date(2026, 10, 22)


def test_anchor_dst_fall_back_tomorrow_has_100_quarters():
    now = _local(2026, 10, 24, 14, 0)
    tomorrow = _slots(date(2026, 10, 25), 15)
    assert len(tomorrow) == 100
    assert choose_anchor(now, tomorrow, TZ) == date(2026, 10, 25)
    assert choose_anchor(now, tomorrow[:-1], TZ) == date(2026, 10, 24)
    # spring forward: 92 quarters, 23 hours
    assert len(_slots(date(2026, 3, 29), 15)) == 92
    assert choose_anchor(_local(2026, 3, 28, 14), _slots(date(2026, 3, 29), 60), TZ) == date(
        2026, 3, 29
    )


def test_next_run_polls_every_15_min_between_13_and_16():
    assert next_run(_local(2026, 10, 22, 11, 0), False, TZ) == _local(
        2026, 10, 22, 13, 0
    ).astimezone(UTC)
    assert next_run(_local(2026, 10, 22, 13, 0), False, TZ) == _local(
        2026, 10, 22, 13, 15
    ).astimezone(UTC)
    assert next_run(_local(2026, 10, 22, 13, 7), False, TZ) == _local(
        2026, 10, 22, 13, 15
    ).astimezone(UTC)
    assert next_run(_local(2026, 10, 22, 15, 45), False, TZ) == datetime(
        2026, 10, 22, 14, 15, tzinfo=UTC
    )
    # anchor already tomorrow: no polling, next is the weather run
    assert next_run(_local(2026, 10, 22, 13, 7), True, TZ) == datetime(
        2026, 10, 22, 14, 15, tzinfo=UTC
    )


def test_next_run_polling_window_across_dst_dates():
    for y, m, d in ((2026, 10, 25), (2026, 3, 29)):
        now = _local(y, m, d, 12, 0)
        t = next_run(now, False, TZ)
        assert t == _local(y, m, d, 13, 0).astimezone(UTC)
        seen = []
        while t.astimezone(BERLIN).time() < time(16, 0):
            seen.append(t)
            t = next_run(t, False, TZ)
        # 13:00 .. 15:45 local, every 15 minutes
        assert len(seen) == 12
        assert all(b - a == timedelta(minutes=15) for a, b in pairwise(seen))
        assert seen[-1] == _local(y, m, d, 15, 45).astimezone(UTC)


def test_next_run_after_16_without_tomorrow_waits_for_weather_or_midnight():
    # 16:00 local (14:00 UTC): next weather 14:15 UTC
    assert next_run(_local(2026, 10, 22, 16, 0), False, TZ) == datetime(
        2026, 10, 22, 14, 15, tzinfo=UTC
    )
    # 22:30 local (20:30 UTC): next is midnight rollover 00:05 local, before 02:15 UTC
    assert next_run(_local(2026, 10, 22, 22, 30), False, TZ) == _local(
        2026, 10, 23, 0, 5
    ).astimezone(UTC)
    # 15:50 local: window's last mark has passed, so weather
    assert next_run(_local(2026, 10, 22, 15, 50), False, TZ) == datetime(
        2026, 10, 22, 14, 15, tzinfo=UTC
    )


def test_next_run_hits_weather_times():
    expected = [time(*t) for t in ((2, 15), (8, 15), (14, 15), (20, 15))]
    assert list(WEATHER_RUNS_UTC) == expected
    now = datetime(2026, 7, 1, 2, 0, tzinfo=UTC)  # 04:00 local, anchor tomorrow
    got = []
    for _ in range(4):
        now = next_run(now, True, TZ)
        got.append(now)
    assert got[0] == datetime(2026, 7, 1, 2, 15, tzinfo=UTC)
    assert got[1] == datetime(2026, 7, 1, 8, 15, tzinfo=UTC)
    assert got[2] == datetime(2026, 7, 1, 14, 15, tzinfo=UTC)
    assert got[3] == datetime(2026, 7, 1, 20, 15, tzinfo=UTC)
    # strictly after, UTC-aware, even when exactly on a run time
    t = datetime(2026, 7, 1, 8, 15, tzinfo=UTC)
    assert next_run(t, True, TZ) > t
    assert next_run(t, True, TZ).utcoffset() == timedelta(0)
    # midnight rollover wins late in the day (summer: 22:05 UTC = 00:05 local)
    assert next_run(datetime(2026, 7, 1, 20, 15, tzinfo=UTC), True, TZ) == datetime(
        2026, 7, 1, 22, 5, tzinfo=UTC
    )


def test_stale_after_failed_attempt_or_7h():
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    assert is_stale(None, now, False)
    assert not is_stale(now - timedelta(hours=1), now, False)
    assert is_stale(now - timedelta(hours=1), now, True)
    assert not is_stale(now - timedelta(hours=7), now, False)
    assert is_stale(now - timedelta(hours=7, seconds=1), now, False)


def test_regular_runs_never_go_stale():
    """Every gap between scheduled runs stays under the stale limit."""
    t = datetime(2026, 10, 20, 0, tzinfo=UTC)
    while t < datetime(2026, 10, 27, tzinfo=UTC):
        nxt = next_run(t, True, TZ)
        assert not is_stale(t, nxt, False), (t, nxt)
        t = nxt


def test_expired_after_36h():
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    assert is_expired(None, now)
    assert not is_expired(now - timedelta(hours=36), now)
    assert is_expired(now - timedelta(hours=36, seconds=1), now)
    assert not is_expired(now - timedelta(hours=1), now)
