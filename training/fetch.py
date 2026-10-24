"""Download the raw training inputs of the zones into a cache directory (no API key needed).

Sources: Energy-Charts day-ahead prices per zone (data from SMARD, CC BY 4.0), Energy-Charts
installed capacity (Germany, for DE-LU), Open-Meteo Previous Runs API (archived ECMWF IFS
0.25 forecasts). Same layout as research/data/raw, shared by the zones:
prices/<zone code>/*.json, weather/<point>_<YYYY-MM>.json (one file set per point, whatever
zones use it), installed_power.json. Chunks whose cached data reaches the chunk end are
reused; chunks near the end are refetched.

Usage: python -m training.fetch --raw training/data/raw [--end YYYY-MM-DD] [--zones DE-LU,NL]
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

from core.features import needs_capacity
from core.zones import ZONES, Point, Zone

from .constants import OFFSETS, WEATHER_MODEL

START = dt.date(2024, 3, 1)


def _get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
        except (requests.Timeout, requests.ConnectionError) as err:
            if attempt == retries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  {type(err).__name__} - retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
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


# Weather hours are labelled at their start, so a full chunk ends at 23:00 UTC of its last day.
_END_TOLERANCE = dt.timedelta(hours=1)


def _last_timestamp(path: Path) -> dt.datetime | None:
    """Last UTC time with data in a cached Energy-Charts or Open-Meteo file, else None."""
    try:
        doc = json.loads(path.read_text())
        if "unix_seconds" in doc:
            times = [
                t for t, v in zip(doc["unix_seconds"], doc["price"], strict=True) if v is not None
            ]
            return dt.datetime.fromtimestamp(max(times), dt.UTC) if times else None
        hourly = doc["hourly"]
        columns = [v for k, v in hourly.items() if k != "time"]
        times = [
            t
            for t, *row in zip(hourly["time"], *columns, strict=True)
            if any(x is not None for x in row)
        ]
        return dt.datetime.fromisoformat(max(times)).replace(tzinfo=dt.UTC) if times else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _is_complete(path: Path, chunk_end: dt.date, end: dt.date) -> bool:
    """A cached chunk is reused only if its data reaches the chunk end.

    A chunk fetched while still open stops early and must be fetched again once it has
    closed. Chunks touching the last few days are refetched so late data gets filled in.
    """
    if chunk_end >= end - dt.timedelta(days=3):
        return False
    last = _last_timestamp(path)
    chunk_stop = dt.datetime.combine(chunk_end + dt.timedelta(days=1), dt.time(), dt.UTC)
    return last is not None and last >= chunk_stop - _END_TOLERANCE


def fetch_prices(raw: Path, end: dt.date, zone: Zone, start: dt.date = START) -> None:
    out = raw / "prices" / zone.code
    out.mkdir(parents=True, exist_ok=True)
    # Energy-Charts rate-limits aggressively: few large requests with pauses.
    for a, b in month_chunks(start - dt.timedelta(days=40), end, months=6):
        path = out / f"{a:%Y-%m}_6m.json"
        if _is_complete(path, b, end):
            continue
        print(f"prices {zone.code} {a}..{b}")
        data = _get(
            "https://api.energy-charts.info/price",
            {
                "bzn": zone.code,
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


def weather_sets(zones: list[Zone]) -> list[tuple[dict[str, Point], tuple[str, ...]]]:
    """The zones' point sets with every point once (a point has one variable set)."""
    seen: set[str] = set()
    out = []
    for zone in zones:
        for points, variables in zone.point_sets():
            new = {name: coords for name, coords in points.items() if name not in seen}
            seen |= set(new)
            if new:
                out.append((new, variables))
    return out


def price_zones(zones: list[Zone]) -> list[Zone]:
    """The zones whose prices the zones need: each zone and its price inputs, once each."""
    codes = dict.fromkeys(c for z in zones for c in (z.code, *z.price_inputs))
    return [ZONES[c] for c in codes]


def fetch_all(raw: Path, end: dt.date, zones: list[Zone]) -> None:
    if any(needs_capacity(z) for z in zones):
        fetch_installed(raw)
    # serial, with the pauses in fetch_prices: Energy-Charts rate limits
    for zone in price_zones(zones):
        fetch_prices(raw, end, zone)
    for points, variables in weather_sets(zones):
        fetch_weather(raw, end, points, variables)


def _zones(value: str) -> list[Zone]:
    """argparse type for a comma-separated list of zone codes."""
    codes = [c.strip() for c in value.split(",") if c.strip()]
    unknown = [c for c in codes if c not in ZONES]
    if unknown or not codes:
        raise argparse.ArgumentTypeError(f"unknown zones {unknown}; known: {', '.join(ZONES)}")
    return [ZONES[c] for c in dict.fromkeys(codes)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, required=True, help="cache directory")
    p.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=dt.date.today() - dt.timedelta(days=1),
        help="last day to fetch (default: yesterday)",
    )
    p.add_argument(
        "--zones",
        type=_zones,
        default=list(ZONES.values()),
        help=f"comma-separated zone codes (default: all, {','.join(ZONES)})",
    )
    args = p.parse_args(argv)
    fetch_all(args.raw, args.end, args.zones)


if __name__ == "__main__":
    main()
