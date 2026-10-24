"""Weather from Open-Meteo: the live ECMWF IFS forecast and archived previous runs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import aiohttp

from ..const import MAX_WEATHER_BYTES
from ..core.features import HORIZONS, Hourly
from ..core.ingest import merge_hourly, parse_open_meteo, weather_by_horizon_from_previous_runs
from ..core.zones import Point, Zone
from . import http
from .http import RunBlocking, SourceError, decode_json, get_body

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "ecmwf_ifs025"
FORECAST_DAYS = 8
REQUEST_PAUSE = 3.0  # seconds between previous-runs requests


def _locations(points: Mapping[str, Point]) -> dict[str, str]:
    """Multi-location query: the API answers with one document per point, in this order."""
    return {
        "latitude": ",".join(str(lat) for lat, _ in points.values()),
        "longitude": ",".join(str(lon) for _, lon in points.values()),
    }


def _parse[T](parse: Callable[..., T], body: bytes, url: str, *args: Any) -> T:
    """``parse(docs, *args)`` of a JSON body (blocking); one location is a bare document."""
    payload = decode_json(body, url)
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


async def fetch_forecast(
    session: aiohttp.ClientSession, zone: Zone, *, run_blocking: RunBlocking
) -> dict[str, Hourly]:
    """``"{point}__{var}"`` hourly forecast series from today 00:00 GMT for ``FORECAST_DAYS``.

    One request per point set of the zone (``Zone.point_sets``: points of one variable
    set; for DE-LU its own points, then its neighbours). Bodies are decoded and parsed via
    ``run_blocking``.
    """
    out: dict[str, Hourly] = {}
    for points, variables in zone.point_sets():
        body = await get_body(
            session,
            FORECAST_URL,
            {
                **_locations(points),
                "hourly": ",".join(variables),
                "models": WEATHER_MODEL,
                "forecast_days": FORECAST_DAYS,
                "timezone": "GMT",
            },
            MAX_WEATHER_BYTES,
        )
        out |= await run_blocking(
            _parse, parse_open_meteo, body, FORECAST_URL, list(points), variables
        )
    return out


async def fetch_previous_runs(
    session: aiohttp.ClientSession,
    zone: Zone,
    start: date,
    end: date,
    *,
    run_blocking: RunBlocking,
) -> dict[int, dict[str, Hourly]]:
    """Archived forecasts per horizon h for the target days ``start`` through ``end``.

    ``out[h]["{point}__{var}"]`` comes from the run h + 1 days before each target day
    (see ``weather_by_horizon_from_previous_runs``). One request per calendar month and
    point set, ``REQUEST_PAUSE`` apart; each body is decoded, parsed and merged via
    ``run_blocking``.
    """
    out: dict[int, dict[str, Hourly]] = {h: {} for h in HORIZONS}
    first = True
    for chunk_start, chunk_end in _month_chunks(start, end):
        for points, variables in zone.point_sets():
            if not first:
                await http.pause(REQUEST_PAUSE)
            first = False
            body = await get_body(
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
                MAX_WEATHER_BYTES,
            )
            await run_blocking(_add_chunk, out, body, list(points), variables)
    return out


def _add_chunk(
    out: dict[int, dict[str, Hourly]], body: bytes, names: list[str], variables: Sequence[str]
) -> None:
    """Merge one previous-runs body into ``out`` (blocking; ``out`` belongs to one fetch)."""
    chunk = _parse(weather_by_horizon_from_previous_runs, body, PREVIOUS_RUNS_URL, names, variables)
    for h, series in chunk.items():
        for key, s in series.items():
            out[h][key] = merge_hourly(out[h][key], s) if key in out[h] else s


def _previous_columns(variables: Sequence[str]) -> str:
    """``{var}_previous_day{h + 1}`` for every variable and horizon, variable-major."""
    return ",".join(f"{v}_previous_day{h + 1}" for v in variables for h in HORIZONS)
