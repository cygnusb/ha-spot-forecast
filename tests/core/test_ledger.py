from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import gzip
import json
import math
from zoneinfo import ZoneInfo

import numpy as np

from core.features import Hourly
from core.forecast import Slot
from core.ledger import Ledger, Run, StatRow, runs_from_backfill_file

TZ = "Europe/Berlin"
BERLIN = ZoneInfo(TZ)
ZERO = dict.fromkeys(range(1, 7), 0.0)


def fin(led: Ledger, p90=None, p10=None) -> list[StatRow]:
    return led.finalize(TZ, zone_id="de_lu", default_p90=p90 or ZERO, default_p10=p10 or ZERO)


def day_hours(d: date) -> list[datetime]:
    start = datetime.combine(d, time(), BERLIN).astimezone(UTC)
    end = datetime.combine(d + timedelta(days=1), time(), BERLIN).astimezone(UTC)
    return [start + timedelta(hours=i) for i in range((end - start) // timedelta(hours=1))]


def make_run(anchor: date, hs=range(1, 7), median=50.0, p90=60.0, p10=40.0, issued=None) -> Run:
    slots = tuple(
        Slot(t, h, median, p90, p10, True)
        for h in hs
        for t in day_hours(anchor + timedelta(days=h))
    )
    issued = issued or datetime.combine(anchor, time(6), BERLIN).astimezone(UTC)
    return Run(anchor, issued, "v1", slots, False)


def hourly_for(days: list[date], value) -> Hourly:
    """Contiguous Hourly covering the given consecutive local days."""
    times = [t for d in days for t in day_hours(d)]
    vals = np.array([value(t) if callable(value) else value for t in times], dtype=float)
    return Hourly(times[0], vals)


def consecutive(first: date, n: int) -> list[date]:
    return [first + timedelta(days=i) for i in range(n)]


def test_last_run_per_anchor_wins():
    led = Ledger()
    a = date(2026, 3, 10)
    early = make_run(a, median=1.0, issued=datetime(2026, 3, 10, 5, tzinfo=UTC))
    late = make_run(a, median=2.0, issued=datetime(2026, 3, 10, 9, tzinfo=UTC))
    led.add_run(late)
    led.add_run(early)
    assert led.runs[a].slots[0].median == 2.0
    later = make_run(a, median=3.0, issued=datetime(2026, 3, 10, 10, tzinfo=UTC))
    led.add_run(later)
    assert led.runs[a].slots[0].median == 3.0


def test_backfill_never_replaces_live_run():
    led = Ledger()
    a = date(2026, 3, 10)
    live = make_run(a, median=1.0, issued=datetime(2026, 3, 10, 4, tzinfo=UTC))
    led.add_run(live)
    later_backfill = replace(make_run(a, median=2.0), backfill=True)  # issued 06:00 local
    assert later_backfill.issued_at > live.issued_at
    led.add_run(later_backfill)
    assert led.runs[a] is live  # what the user saw wins

    b = date(2026, 3, 11)
    backfilled = replace(make_run(b, median=3.0), backfill=True)
    led.add_run(backfilled)
    earlier_live = make_run(b, median=4.0, issued=backfilled.issued_at - timedelta(hours=2))
    led.add_run(earlier_live)
    assert led.runs[b] is earlier_live


def test_backfill_run_replaces_backfill_run_of_same_issue_time():
    """A re-import with a wider window completes the stored run of an anchor."""
    led = Ledger()
    a = date(2026, 3, 10)
    led.add_run(replace(make_run(a, hs=[5, 6]), backfill=True))
    wider = replace(make_run(a), backfill=True)
    led.add_run(wider)
    assert led.runs[a] is wider


def _ledger_with_residuals(anchor: date, before: float, after: float) -> Ledger:
    """One run per day over 80 days; actual = p90_raw + residual, residual by target day."""
    led = Ledger()
    first = anchor - timedelta(days=60)
    for i in range(80):
        led.add_run(make_run(first + timedelta(days=i)))
    for d in consecutive(first, 90):
        res = before if d < anchor else after
        led.record_actual(hourly_for([d], 60.0 + res))
    return led


def test_offsets_ignore_targets_on_or_after_anchor():
    a = date(2026, 5, 1)
    led = _ledger_with_residuals(a, before=1.0, after=100.0)
    p90, p10 = led.offsets(a, ZERO, ZERO)
    assert all(abs(v - 1.0) < 1e-9 for v in p90.values())
    # p10_raw is 40 -> residual before = 21
    assert all(abs(v - 21.0) < 1e-9 for v in p10.values())


def test_offsets_use_only_trailing_56_target_days():
    a = date(2026, 5, 1)
    led = Ledger()
    for i in range(70):
        led.add_run(make_run(a - timedelta(days=75) + timedelta(days=i)))
    # target days older than 56 days carry +50, the window +2
    for d in consecutive(a - timedelta(days=80), 90):
        led.record_actual(hourly_for([d], 60.0 + (2.0 if d >= a - timedelta(days=56) else 50.0)))
    p90, _ = led.offsets(a, ZERO, ZERO)
    assert all(abs(v - 2.0) < 1e-9 for v in p90.values())


def test_offsets_default_below_200():
    a = date(2026, 5, 1)
    led = Ledger()
    for i in range(3):
        led.add_run(make_run(a - timedelta(days=10) + timedelta(days=i)))
    for d in consecutive(a - timedelta(days=10), 20):
        led.record_actual(hourly_for([d], 99.0))
    d90 = {h: 7.0 + h for h in range(1, 7)}
    d10 = {h: -3.0 - h for h in range(1, 7)}
    assert led.offsets(a, d90, d10) == (d90, d10)


def _complete_ledger(day: date) -> Ledger:
    led = Ledger()
    for h in range(1, 7):
        led.add_run(make_run(day - timedelta(days=h), hs=[h]))
    led.record_actual(hourly_for([day], lambda t: 100.0 + t.hour))
    return led


def test_finalize_emits_price_and_19_series_rows():
    day = date(2026, 6, 10)
    led = _complete_ledger(day)
    rows = fin(led)
    by_id: dict[str, list[StatRow]] = {}
    for r in rows:
        by_id.setdefault(r.statistic_id, []).append(r)
    assert len(by_id["spot_forecast:de_lu_price"]) == 24
    assert len(rows) == 24 + 24 * 6 * 3
    assert len(by_id) == 1 + 18
    price = by_id["spot_forecast:de_lu_price"][0]
    assert price.start == day_hours(day)[0]
    assert price.value == round((100.0 + price.start.hour) / 1000, 5)
    fc = by_id["spot_forecast:de_lu_forecast_h3"]
    assert {r.value for r in fc} == {0.05}
    assert {r.value for r in by_id["spot_forecast:de_lu_peak_risk_h3"]} == {0.06}
    assert {r.value for r in by_id["spot_forecast:de_lu_low_h3"]} == {0.04}
    assert all(r.start.tzinfo is not None and r.start.utcoffset() == timedelta(0) for r in rows)


def test_finalize_uses_offsets_of_run_anchor():
    day = date(2026, 6, 10)
    led = _complete_ledger(day)
    d90 = dict.fromkeys(range(1, 7), 20.0)
    d10 = dict.fromkeys(range(1, 7), -30.0)
    rows = fin(led, d90, d10)
    vals = {r.statistic_id: r.value for r in rows}
    assert vals["spot_forecast:de_lu_peak_risk_h1"] == 0.08
    assert vals["spot_forecast:de_lu_low_h1"] == 0.01


def test_finalize_skips_invalid_slots():
    day = date(2026, 6, 10)
    led = _complete_ledger(day)
    run = led.runs[day - timedelta(days=2)]
    first = run.slots[0]
    bad = Slot(first.time, first.h, math.nan, math.nan, math.nan, False)
    led.runs[run.anchor] = Run(
        run.anchor, run.issued_at, run.model_version, (bad, *run.slots[1:]), run.backfill
    )
    rows = fin(led)
    assert len(rows) == 24 + 24 * 6 * 3 - 3
    assert not [r for r in rows if r.start == first.time and r.statistic_id.endswith("h2")]


def test_finalize_is_idempotent():
    led = _complete_ledger(date(2026, 6, 10))
    assert fin(led)
    assert fin(led) == []
    assert (date(2026, 6, 10), "price") in led.finalized


def test_finalize_waits_for_complete_day():
    day = date(2026, 6, 10)
    led = _complete_ledger(day)
    ts = day_hours(day)
    partial = Ledger(runs=led.runs)
    partial.record_actual(Hourly(ts[0], np.arange(23, dtype=float)))
    assert fin(partial) == []
    partial.record_actual(Hourly(ts[23], np.array([5.0])))
    assert len(fin(partial)) == 24 + 24 * 18


def test_finalize_dst_days():
    for day, n in ((date(2026, 3, 29), 23), (date(2026, 10, 25), 25)):
        led = Ledger()
        for h in range(1, 7):
            led.add_run(make_run(day - timedelta(days=h), hs=[h]))
        led.record_actual(hourly_for([day], 10.0))
        rows = fin(led)
        assert len(rows) == n + n * 18
        assert len({(r.statistic_id, r.start) for r in rows}) == len(rows)


def test_peak_risk_row_never_below_forecast_row():
    day = date(2026, 6, 10)
    led = _complete_ledger(day)
    rows = fin(led, p90=dict.fromkeys(range(1, 7), -500.0))
    fc = {(r.start, r.statistic_id[-2:]): r.value for r in rows if "forecast_h" in r.statistic_id}
    pk = {(r.start, r.statistic_id[-2:]): r.value for r in rows if "peak_risk_h" in r.statistic_id}
    assert fc and all(pk[k] >= v for k, v in fc.items())
    led2 = _complete_ledger(day)
    rows2 = fin(led2, p10=dict.fromkeys(range(1, 7), 500.0))
    lo2 = [r.value for r in rows2 if "_low_h" in r.statistic_id]
    assert lo2 and all(v <= 0.05 for v in lo2)


def test_accuracy_mae_per_horizon():
    now = datetime(2026, 6, 20, 12, tzinfo=UTC)
    led = Ledger()
    d = date(2026, 6, 15)
    # h=1 run issued for target 6/16 (median 50), h=2 for 6/16 from anchor 6/14 (median 70)
    led.add_run(make_run(d, hs=[1], median=50.0))
    led.add_run(make_run(d - timedelta(days=1), hs=[2], median=70.0))
    led.record_actual(hourly_for([d + timedelta(days=1)], lambda t: 100.0 if t.hour < 12 else 40.0))
    acc = led.accuracy(now, peak_threshold=90.0)
    # h1 errors 50 or 10 -> mean 30; h2 errors 30 or 30 -> 30 ; 24 hours each
    n_h = 24
    assert acc["n"] == 2 * n_h
    assert math.isclose(acc["mae_h"][1], 0.030)
    assert math.isclose(acc["mae_h"][2], 0.030)
    assert math.isclose(acc["mae"], 0.030)
    # peak hours (actual 100 >= 90): errors h1 50, h2 30
    assert math.isclose(acc["peak_mae"], 0.040)


def test_accuracy_window_excludes_today_and_old_days():
    now = datetime(2026, 6, 20, 12, tzinfo=UTC)
    led = Ledger()
    for target in (date(2026, 5, 1), date(2026, 6, 20)):
        led.add_run(make_run(target - timedelta(days=1), hs=[1]))
        led.record_actual(hourly_for([target], 10.0))
    acc = led.accuracy(now, peak_threshold=1e9, days=30)
    assert acc["n"] == 0 and acc["mae"] is None and acc["peak_mae"] is None


def test_json_roundtrip():
    led = _complete_ledger(date(2026, 6, 10))
    nan_slot = Slot(day_hours(date(2026, 6, 20))[0], 1, math.nan, math.nan, math.nan, False)
    led.add_run(
        Run(date(2026, 6, 19), datetime(2026, 6, 19, 4, tzinfo=UTC), "v", (nan_slot,), True)
    )
    fin(led)
    d = led.to_json()
    assert json.loads(json.dumps(d)) == d
    back = Ledger.from_json(json.loads(json.dumps(d)))
    assert back.to_json() == d
    assert back.finalized == led.finalized and back.actual == led.actual
    r = back.runs[date(2026, 6, 19)]
    assert r.backfill and r.issued_at == datetime(2026, 6, 19, 4, tzinfo=UTC)
    assert math.isnan(r.slots[0].median) and r.slots[0].valid is False
    assert back.runs[date(2026, 6, 9)].slots == led.runs[date(2026, 6, 9)].slots


def test_prune_drops_old_entries():
    led = Ledger()
    old, new = date(2026, 1, 10), date(2026, 6, 10)
    for a in (old, new):
        led.add_run(make_run(a - timedelta(days=1), hs=[1]))
        led.record_actual(hourly_for([a], 10.0))
    fin(led)
    assert {d for d, _ in led.finalized} == {old, new}
    led.prune(keep_days=30)
    assert {d for d, _ in led.finalized} == {new}
    assert set(led.runs) == {new - timedelta(days=1)}
    assert all(t >= datetime(2026, 5, 1, tzinfo=UTC) for t in led.actual)
    assert fin(led) == []


def _backfill_bytes(rows, version="v1") -> bytes:
    doc = {"format": 1, "model_version": version, "rows": rows}
    return gzip.compress(json.dumps(doc).encode(), mtime=0)


def test_backfill_file_groups_by_anchor():
    rows = []
    for anchor in (date(2026, 2, 1), date(2026, 2, 2)):
        for h in (1, 2):
            for t in day_hours(anchor + timedelta(days=h)):
                rows.append([t.isoformat(), h, 50.0, 60.0, 40.0])
    runs = runs_from_backfill_file(_backfill_bytes(rows), "v9")
    assert [r.anchor for r in runs] == [date(2026, 2, 1), date(2026, 2, 2)]
    r = runs[0]
    assert r.backfill and r.model_version == "v9"
    assert r.issued_at == datetime(2026, 2, 1, 6, tzinfo=BERLIN).astimezone(UTC)
    assert r.issued_at.utcoffset() == timedelta(0)
    assert {s.h for s in r.slots} == {1, 2} and len(r.slots) == 48
    assert all(s.valid for s in r.slots)
    assert (r.slots[0].median, r.slots[0].p90_raw, r.slots[0].p10_raw) == (50.0, 60.0, 40.0)
    assert r.slots[0].time.tzinfo is not None


def test_finalize_emits_forecast_rows_for_runs_added_after_price_rows():
    day = date(2026, 6, 10)
    led = Ledger()
    led.record_actual(hourly_for([day], 80.0))
    first = fin(led)
    assert len(first) == 24 and {r.statistic_id for r in first} == {"spot_forecast:de_lu_price"}
    for h in range(1, 7):
        led.add_run(make_run(day - timedelta(days=h), hs=[h]))
    second = fin(led)
    assert len(second) == 24 * 18
    assert not [r for r in second if r.statistic_id.endswith("_price")]
    assert fin(led) == []
    again = Ledger.from_json(json.loads(json.dumps(led.to_json())))
    assert fin(again) == []


def test_finalize_waits_for_a_run_that_holds_the_horizon():
    """A backfill run holding only some horizons leaves the others open for a later merge."""
    day = date(2026, 6, 10)
    anchor = day - timedelta(days=4)
    led = Ledger()
    led.record_actual(hourly_for([day], 80.0))
    led.add_run(replace(make_run(anchor, hs=[1, 2]), backfill=True))
    assert not [r for r in fin(led) if "_h4" in r.statistic_id]
    led.add_run(replace(make_run(anchor), backfill=True))
    assert len([r for r in fin(led) if r.statistic_id.endswith("forecast_h4")]) == 24
