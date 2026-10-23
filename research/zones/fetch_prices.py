"""Fetch Energy-Charts day-ahead prices for zones not in the research cache (cached, slow)."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data" / "zones"
START = dt.date(2024, 1, 21)  # research START (2024-03-01) - 40 days
END = dt.date(2026, 10, 21)
ZONES = sys.argv[1:] or ["SE3", "FI", "NO1", "SE1", "SE2", "NO3", "NO4", "NO5"]


def month_chunks(start, end, months=6):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur
        for _ in range(months):
            nxt = (nxt.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def get(params):
    for attempt in range(6):
        r = requests.get("https://api.energy-charts.info/price", params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        wait = 120 if r.status_code == 429 else 10 * (attempt + 1)
        print(f"  HTTP {r.status_code} - retry in {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


for zone in ZONES:
    out = DATA / "prices" / zone
    out.mkdir(parents=True, exist_ok=True)
    for a, b in month_chunks(START, END):
        path = out / f"{a:%Y-%m}_6m.json"
        if path.exists():
            continue
        print(f"prices {zone} {a}..{b}", flush=True)
        t0 = time.time()
        d = get({"bzn": zone, "start": f"{a}T00:00Z", "end": f"{b + dt.timedelta(days=1)}T00:00Z"})
        path.write_text(json.dumps(d))
        print(f"  {len(d['unix_seconds'])} slots, {time.time() - t0:.1f}s", flush=True)
        time.sleep(20)
print("done")
