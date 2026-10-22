"""Weather from Open-Meteo: the live ECMWF IFS forecast and archived previous runs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import aiohttp

from ..core.features import HORIZONS, Hourly
from ..core.ingest import merge_hourly, parse_open_meteo, weather_by_horizon_from_previous_runs
from ..core.zones import Point, Zone
from . import http
from .http import SourceError, get_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "ecmwf_ifs025"
FORECAST_DAYS = 8
REQUEST_PAUSE = 3.0  # seconds between previous-runs requests


def _point_sets(zone: Zone) -> list[tuple[dict[str, Point], tuple[str, ...]]]:
    """The zone's own points with ``weather_vars``, then its neighbours with ``neighbor_vars``."""
    return [(zone.points, zone.weather_vars), (zone.neighbor_points, zone.neighbor_vars)]


def _locations(points: Mapping[str, Point]) -> dict[str, str]:
    """Multi-location query: the API answers with one document per point, in this order."""
    return {
        "latitude": ",".join(str(lat) for lat, _ in points.values()),
        "longitude": ",".join(str(lon) for _, lon in points.values()),
    }


def _parse[T](parse: Callable[..., T], payload: Any, *args: Any) -> T:
    """``parse(docs, *args)``; a single location comes back as a bare document."""
    docs = payload if isinstance(payload, list) else [payload]
    try:
        return parse(docs, *args)
    except (KeyError, TypeError, ValueError) as e:
        raise SourceError(f"unexpected Open-Meteo payload: {type(e).__name__}: {e}") from e


def _month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """``[start, end]`` split at month boundaries, both bounds inclusive."""
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(next_month - timedelta(days=1), end)
        cur = next_month


async def fetch_forecast(session: aiohttp.ClientSession, zone: Zone) -> dict[str, Hourly]:
    """``"{point}__{var}"`` hourly forecast series from today 00:00 GMT for ``FORECAST_DAYS``.

    Two requests: the zone's points and its neighbour points, each with its variable set.
    """
    out: dict[str, Hourly] = {}
    for points, variables in _point_sets(zone):
        payload = await get_json(
            session,
            FORECAST_URL,
            {
                **_locations(points),
                "hourly": ",".join(variables),
                "models": WEATHER_MODEL,
                "forecast_days": FORECAST_DAYS,
                "timezone": "GMT",
            },
        )
        out |= _parse(parse_open_meteo, payload, list(points), variables)
    return out


async def fetch_previous_runs(
    session: aiohttp.ClientSession, zone: Zone, start: date, end: date
) -> dict[int, dict[str, Hourly]]:
    """Archived forecasts per horizon h for the target days ``start`` through ``end``.

    ``out[h]["{point}__{var}"]`` comes from the run h + 1 days before each target day
    (see ``weather_by_horizon_from_previous_runs``). One request per calendar month and
    point set, ``REQUEST_PAUSE`` apart.
    """
    out: dict[int, dict[str, Hourly]] = {h: {} for h in HORIZONS}
    first = True
    for chunk_start, chunk_end in _month_chunks(start, end):
        for points, variables in _point_sets(zone):
            if not first:
                await http.pause(REQUEST_PAUSE)
            first = False
            payload = await get_json(
                session,
                PREVIOUS_RUNS_URL,
                {
                    **_locations(points),
                    "hourly": _previous_columns(variables),
                    "models": WEATHER_MODEL,
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "timezone": "GMT",
                },
            )
            chunk = _parse(weather_by_horizon_from_previous_runs, payload, list(points), variables)
            for h, series in chunk.items():
                for key, s in series.items():
                    out[h][key] = merge_hourly(out[h][key], s) if key in out[h] else s
    return out


def _previous_columns(variables: Sequence[str]) -> str:
    """``{var}_previous_day{h + 1}`` for every variable and horizon, variable-major."""
    return ",".join(f"{v}_previous_day{h + 1}" for v in variables for h in HORIZONS)
