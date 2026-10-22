"""Download the raw DE-LU training inputs into a cache directory (no API key needed).

Sources: Energy-Charts day-ahead prices DE-LU (data from SMARD, CC BY 4.0), Energy-Charts
installed capacity, Open-Meteo Previous Runs API (archived ECMWF IFS 0.25 forecasts).
Same layout as research/data/raw: prices/DE-LU/*.json, weather/<point>_<YYYY-MM>.json,
installed_power.json. Finished chunks are cached; chunks near the end are refetched.

Usage: python -m training.fetch --raw training/data/raw [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import datetime as dt
import json
from pathlib import Path
import sys
import time

import requests

from core.zones import DE_LU

from .constants import OFFSETS, WEATHER_MODEL

START = dt.date(2024, 3, 1)


def _get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        wait = 60 if r.status_code == 429 else 5 * (attempt + 1)
        print(f"  HTTP {r.status_code} {r.text[:200]} - retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    r.raise_for_status()
    raise RuntimeError("unreachable")


def month_chunks(
    start: dt.date, end: dt.date, months: int = 1
) -> Iterator[tuple[dt.date, dt.date]]:
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur
        for _ in range(months):
            nxt = (nxt.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _is_complete(path: Path, chunk_end: dt.date, end: dt.date) -> bool:
    # Chunks touching the last few days are refetched so late data gets filled in.
    return path.exists() and chunk_end < end - dt.timedelta(days=3)


def fetch_prices(raw: Path, end: dt.date, start: dt.date = START) -> None:
    out = raw / "prices" / DE_LU.code
    out.mkdir(parents=True, exist_ok=True)
    # Energy-Charts rate-limits aggressively: few large requests with pauses.
    for a, b in month_chunks(start - dt.timedelta(days=40), end, months=6):
        path = out / f"{a:%Y-%m}_6m.json"
        if _is_complete(path, b, end):
            continue
        print(f"prices {DE_LU.code} {a}..{b}")
        data = _get(
            "https://api.energy-charts.info/price",
            {
                "bzn": DE_LU.code,
                "start": f"{a}T00:00Z",
                "end": f"{b + dt.timedelta(days=1)}T00:00Z",
            },
        )
        path.write_text(json.dumps(data))
        time.sleep(15)


def fetch_installed(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    data = _get(
        "https://api.energy-charts.info/installed_power",
        {"country": "de", "time_step": "monthly", "installation_decommission": "false"},
    )
    (raw / "installed_power.json").write_text(json.dumps(data))


def fetch_weather(
    raw: Path,
    end: dt.date,
    points: dict[str, tuple[float, float]],
    variables: tuple[str, ...],
    start: dt.date = START,
) -> None:
    out = raw / "weather"
    out.mkdir(parents=True, exist_ok=True)
    hourly = [f"{v}_previous_day{o}" for v in variables for o in OFFSETS]
    for name, (lat, lon) in points.items():
        for a, b in month_chunks(start, end):
            path = out / f"{name}_{a:%Y-%m}.json"
            if _is_complete(path, b, end):
                continue
            print(f"weather {name} {a}..{b}")
            data = _get(
                "https://previous-runs-api.open-meteo.com/v1/forecast",
                {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": ",".join(hourly),
                    "models": WEATHER_MODEL,
                    "start_date": str(a),
                    "end_date": str(b),
                    "timezone": "GMT",
                },
            )
            path.write_text(json.dumps(data))
            time.sleep(3)


def fetch_all(raw: Path, end: dt.date) -> None:
    fetch_installed(raw)
    fetch_prices(raw, end)
    fetch_weather(raw, end, DE_LU.points, DE_LU.weather_vars)
    fetch_weather(raw, end, DE_LU.neighbor_points, DE_LU.neighbor_vars)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, required=True, help="cache directory")
    p.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=dt.date.today() - dt.timedelta(days=1),
        help="last day to fetch (default: yesterday)",
    )
    args = p.parse_args(argv)
    fetch_all(args.raw, args.end)


if __name__ == "__main__":
    main()
