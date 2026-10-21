"""Download raw input data for the DE-LU backtest into data/raw/ (cached).

Sources (no API key):
  - Energy-Charts (Fraunhofer ISE), day-ahead prices DE-LU, data from SMARD (CC BY 4.0)
  - Energy-Charts installed capacity (monthly)
  - Open-Meteo Previous Runs API: archived ECMWF IFS 0.25 forecasts at fixed lead times
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

RAW = Path(__file__).parent / "data" / "raw"

START = dt.date(2024, 3, 1)
END = dt.date.today() - dt.timedelta(days=1)

# Points chosen to cover wind (north/offshore), solar (south), load (population centres).
POINTS = {
    "north_sea": (54.2, 6.6),
    "baltic": (54.7, 13.8),
    "schleswig": (54.5, 9.2),
    "ostfriesland": (53.3, 7.6),
    "uckermark": (53.2, 13.8),
    "sachsen_anhalt": (51.9, 11.6),
    "nrw": (51.4, 7.3),
    "hessen": (50.4, 9.3),
    "bawue": (48.7, 9.2),
    "bayern": (48.4, 11.8),
}
WEATHER_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]

# Coupled neighbours: wind (NL/DK/PL/SE), heating load and nuclear (FR), hydro (Alps, Nordics).
NEIGHBOR_POINTS = {
    "nb_nl": (52.6, 4.6),
    "nb_be": (50.8, 4.4),
    "nb_fr_north": (49.0, 2.4),
    "nb_fr_south": (45.5, 4.8),
    "nb_dk_west": (56.0, 8.5),
    "nb_pl_north": (54.2, 17.5),
    "nb_alps_ch": (46.6, 8.0),
    "nb_alps_at": (47.3, 12.5),
    "nb_no_south": (59.5, 7.5),
    "nb_se_south": (56.5, 14.5),
    "nb_se_north": (63.5, 16.0),
    "nb_fi": (61.0, 25.0),
}
NEIGHBOR_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation"]
NEIGHBOR_ZONES = ["FR", "NL", "BE", "AT", "CH", "PL", "CZ", "DK1", "DK2", "NO2", "SE4"]
# Lead offsets in days. Target day D+h is forecast with offset h+1 so that every
# value comes from a model run that was available at the 06:00 issue time on day D.
OFFSETS = range(2, 8)
WEATHER_MODEL = "ecmwf_ifs025"


def _get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        wait = 60 if r.status_code == 429 else 5 * (attempt + 1)
        print(f"  HTTP {r.status_code} {r.text[:200]} – retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    r.raise_for_status()
    raise RuntimeError("unreachable")


def month_chunks(start: dt.date, end: dt.date, months: int = 1):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur
        for _ in range(months):
            nxt = (nxt.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _is_complete(path: Path, chunk_end: dt.date) -> bool:
    # Chunks touching the last few days are refetched so late data gets filled in.
    return path.exists() and chunk_end < END - dt.timedelta(days=3)


def fetch_prices(zone: str = "DE-LU") -> None:
    out = RAW / "prices" / zone
    out.mkdir(parents=True, exist_ok=True)
    # Energy-Charts rate-limits aggressively: few large requests with pauses.
    for a, b in month_chunks(START - dt.timedelta(days=40), END, months=6):
        path = out / f"{a:%Y-%m}_6m.json"
        if _is_complete(path, b):
            continue
        print(f"prices {zone} {a}..{b}")
        data = _get(
            "https://api.energy-charts.info/price",
            {"bzn": zone, "start": f"{a}T00:00Z", "end": f"{b + dt.timedelta(days=1)}T00:00Z"},
        )
        path.write_text(json.dumps(data))
        time.sleep(15)


def fetch_installed() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    data = _get(
        "https://api.energy-charts.info/installed_power",
        {"country": "de", "time_step": "monthly", "installation_decommission": "false"},
    )
    (RAW / "installed_power.json").write_text(json.dumps(data))


def fetch_weather(points: dict = POINTS, variables: list[str] = WEATHER_VARS) -> None:
    out = RAW / "weather"
    out.mkdir(parents=True, exist_ok=True)
    hourly = [f"{v}_previous_day{o}" for v in variables for o in OFFSETS]
    for name, (lat, lon) in points.items():
        for a, b in month_chunks(START, END):
            path = out / f"{name}_{a:%Y-%m}.json"
            if _is_complete(path, b):
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


if __name__ == "__main__":
    fetch_installed()
    fetch_prices()
    fetch_weather()
    fetch_weather(NEIGHBOR_POINTS, NEIGHBOR_VARS)
    for zone in NEIGHBOR_ZONES:
        fetch_prices(zone)
