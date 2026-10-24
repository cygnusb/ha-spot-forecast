"""Build the training dataset once (current core.features), plus the candidate columns.

Writes research/data/holidays_dst/cache.npz with the baseline Dataset arrays and:
  hol_share, hol_bridge, hol_eve, lu_holiday   (per row, from target_date)
  p_lastday_dst, p_lastweek_dst                (same local wall-clock hour)
Run from the repo root:  .venv/bin/python research/holidays_dst/build.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import training  # noqa: E402,F401  (puts core on sys.path)
import holidays  # noqa: E402
import numpy as np  # noqa: E402

from core.features import FEATURE_NAMES  # noqa: E402
from training.dataset import load_raw  # noqa: E402
from training.train import build_training_data, check_data_reaches  # noqa: E402

OUT = REPO / "research" / "data" / "holidays_dst"
TZ = ZoneInfo("Europe/Berlin")
CUTOFF = date(2026, 10, 22)

# Population by state (millions, Destatis 2023), as research/backtest.py STATE_POP.
STATE_POP = {"BW": 11.3, "BY": 13.4, "BE": 3.9, "BB": 2.6, "HB": 0.7, "HH": 1.9, "HE": 6.4, "MV": 1.6,
             "NI": 8.1, "NW": 18.1, "RP": 4.2, "SL": 1.0, "SN": 4.0, "ST": 2.2, "SH": 3.0, "TH": 2.1}
# Mariä Himmelfahrt in BY is a holiday only in predominantly Catholic municipalities
# (1704 of 2056, incl. Munich, not Nuremberg); weight it with ~0.7 of BY's population.
BY_ASSUMPTION_WEIGHT = 0.7


def holiday_table(years: range) -> dict[date, tuple[float, float, float, float]]:
    total = sum(STATE_POP.values())
    days = [date(years.start, 1, 1) + timedelta(days=i)
            for i in range((date(years.stop, 1, 1) - date(years.start, 1, 1)).days)]
    share = dict.fromkeys(days, 0.0)
    for st, pop in STATE_POP.items():
        hs = holidays.Germany(subdiv=st, years=years)
        for d in days:
            if d in hs:
                share[d] += pop / total
    by_cath = holidays.Germany(subdiv="BY", years=years, categories=("catholic",))
    for d in days:
        if d in by_cath and by_cath.get(d) == "Mariä Himmelfahrt":
            share[d] += BY_ASSUMPTION_WEIGHT * STATE_POP["BY"] / total
    lu = holidays.Luxembourg(years=years)

    out = {}
    for d in days:
        s = round(min(share[d], 1.0), 4)
        bridge = 0.0
        if d.weekday() < 5 and s < 0.5:
            # working day between a holiday (share >= 0.5) and a weekend day
            prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
            if d.weekday() == 4 and share.get(prev, 0) >= 0.5:  # Friday after Thu holiday
                bridge = share[prev]
            if d.weekday() == 0 and share.get(nxt, 0) >= 0.5:  # Monday before Tue holiday
                bridge = max(bridge, share[nxt])
        eve = 1.0 if (d.month == 12 and d.day in (24, 31)) else 0.0
        out[d] = (s, round(bridge, 4), eve, float(d in lu))
    return out


def dst_lags(ds, prices) -> tuple[np.ndarray, np.ndarray]:
    """p_lastday / p_lastweek at the same local wall-clock hour.

    The local hour of each target row is taken from its UTC time. The lag hour is the same
    local wall-clock time on (target - h) / (target - 7) days. Ambiguous wall-clock times
    (02:00 on a fall-back lag day) take the first occurrence (fold=0); nonexistent ones
    (02:00 on a spring-forward lag day) take the hour before (01:00 local), the closest
    existing hour. The second 02:00 on a fall-back target day maps to the lag day's 02:00.
    Hours after the anchor day read NaN, as in core.features.
    """
    n = len(ds.times)
    lastday, lastweek = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        t_loc = ds.times[i].astimezone(TZ)
        tgt = ds.target_date[i]
        known_until = datetime.combine(ds.anchor[i] + timedelta(days=1), time(), TZ).astimezone(UTC)
        for arr, back in ((lastday, int(ds.h[i])), (lastweek, 7)):
            d = tgt - timedelta(days=back)
            cand = datetime(d.year, d.month, d.day, t_loc.hour, tzinfo=TZ, fold=0)
            u = cand.astimezone(UTC)
            if u.astimezone(TZ).replace(tzinfo=None) != cand.replace(tzinfo=None):  # nonexistent
                u = datetime(d.year, d.month, d.day, t_loc.hour - 1, tzinfo=TZ).astimezone(UTC)
            if u < known_until:
                arr[i] = prices.at(u)
    return lastday, lastweek


def main() -> None:
    t0 = _time.time()
    raw = load_raw(REPO / "research" / "data" / "raw")
    check_data_reaches(raw, CUTOFF)
    ds = build_training_data(raw, CUTOFF)
    print(f"dataset rows={len(ds.y_price)} in {_time.time() - t0:.0f}s")
    hol = holiday_table(range(2023, 2028))
    H = np.array([hol[d] for d in ds.target_date])
    t1 = _time.time()
    pld, plw = dst_lags(ds, raw.prices)
    print(f"dst lags in {_time.time() - t1:.0f}s")
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    old_ld, old_lw = ds.X[:, col["p_lastday"]], ds.X[:, col["p_lastweek"]]
    for name, a, b in (("lastday", old_ld, pld), ("lastweek", old_lw, plw)):
        diff = ~np.isclose(a, b, equal_nan=True)
        print(f"{name}: rows differing {diff.sum()}  target days {len(set(ds.target_date[diff]))}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "cache.npz",
        X=ds.X, y_price=ds.y_price, h=ds.h, mean7=ds.mean7, scale=ds.scale, p_lastweek=ds.p_lastweek,
        times=np.array([t.timestamp() for t in ds.times]),
        anchor=np.array([d.toordinal() for d in ds.anchor]),
        target_date=np.array([d.toordinal() for d in ds.target_date]),
        hol=H, p_lastday_dst=pld, p_lastweek_dst=plw,
        feature_names=np.array(FEATURE_NAMES),
    )
    # holiday table check
    for d, v in sorted(hol.items()):
        if d.year in (2025, 2026) and any(v) and v[0] < 1:
            print(d, d.strftime("%a"), v)


if __name__ == "__main__":
    main()
