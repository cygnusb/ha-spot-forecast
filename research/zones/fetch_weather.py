"""Fetch Open-Meteo previous-runs (ECMWF IFS 0.25) for the new points, multi-location, cached.

One file per point and month (research layout). Aborts on the first HTTP 429.
Usage: fetch_weather.py <max_tier> [sleep_s]
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time

import requests

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from zonecfg import DATA, NEW_POINTS

START = dt.date(2024, 3, 1)
END = dt.date(2026, 10, 21)
OFFSETS = range(2, 8)
BATCH = 8
MAX_TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SLEEP = float(sys.argv[2]) if len(sys.argv) > 2 else 38


def month_chunks(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


out = DATA / "weather"
out.mkdir(parents=True, exist_ok=True)
groups: dict[tuple, list[str]] = {}
for name, (_, vars_, tier) in NEW_POINTS.items():
    if tier <= MAX_TIER:
        groups.setdefault((tier, vars_), []).append(name)

weight_total = 0.0
for (tier, vars_), names in sorted(groups.items(), key=lambda kv: kv[0][0]):
    hourly = [f"{v}_previous_day{o}" for v in vars_ for o in OFFSETS]
    for i in range(0, len(names), BATCH):
        batch = names[i : i + BATCH]
        for a, b in month_chunks(START, END):
            todo = [n for n in batch if not (out / f"{n}_{a:%Y-%m}.json").exists()]
            if not todo:
                continue
            params = {
                "latitude": ",".join(str(NEW_POINTS[n][0][0]) for n in todo),
                "longitude": ",".join(str(NEW_POINTS[n][0][1]) for n in todo),
                "hourly": ",".join(hourly),
                "models": "ecmwf_ifs025",
                "start_date": str(a),
                "end_date": str(b),
                "timezone": "GMT",
            }
            w = len(todo) * max(len(hourly) / 10, 1) * max(((b - a).days + 1) / 14, 1)
            t0 = time.time()
            for attempt in range(3):
                try:
                    r = requests.get(
                        "https://previous-runs-api.open-meteo.com/v1/forecast", params=params, timeout=180
                    )
                    break
                except (requests.Timeout, requests.ConnectionError) as err:
                    print(f"  {type(err).__name__}, retry in 60s", flush=True)
                    time.sleep(60)
            else:
                sys.exit("network errors, abort")
            if r.status_code == 429:
                print(f"HTTP 429 at tier {tier} {todo} {a}: {r.text[:300]}", flush=True)
                sys.exit(2)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}: {r.text[:300]}", flush=True)
                sys.exit(3)
            docs = r.json()
            if isinstance(docs, dict):
                docs = [docs]
            for n, d in zip(todo, docs, strict=True):
                (out / f"{n}_{a:%Y-%m}.json").write_text(json.dumps(d))
            weight_total += w
            print(f"tier{tier} {a:%Y-%m} {len(todo)} pts w={w:.1f} total={weight_total:.0f} "
                  f"{time.time() - t0:.1f}s", flush=True)
            time.sleep(SLEEP * min(w / 32, 1))
print("done", weight_total)
