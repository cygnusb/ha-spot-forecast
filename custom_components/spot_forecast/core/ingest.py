"""Raw API payloads (Energy-Charts prices, Open-Meteo weather) to ``Hourly`` series.

The same functions read the live API responses and the research raw files, which have
the API's shape, so training and the integration see identical series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from .features import Hourly

_HOUR = timedelta(hours=1)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Open-Meteo previous-runs radiation at lead offsets >= 6 days is wrong before this date
# (shifted/scaled by 0.5-1.6x; ECMWF switches to coarser output steps at 144 h). Masked.
RADIATION_BROKEN: dict[int, date] = {6: date(2025, 7, 1), 7: date(2025, 7, 1)}
_RADIATION = "shortwave_radiation"


@dataclass(frozen=True)
class MarketSlot:
    """One day-ahead market period (15 or 60 minutes); price in EUR/MWh."""

    start: datetime  # UTC
    end: datetime  # UTC
    price: float


def parse_energy_charts(doc: Mapping) -> list[MarketSlot]:
    """Slots of an Energy-Charts ``price`` document; slots without a price are skipped.

    Each slot ends where the next one starts; the last keeps the length of the one before.
    """
    stamps = [datetime.fromtimestamp(s, UTC) for s in doc["unix_seconds"]]
    ends = stamps[1:]
    if stamps:
        ends.append(stamps[-1] + (stamps[-1] - stamps[-2] if len(stamps) > 1 else _HOUR))
    return [
        MarketSlot(start, end, float(price))
        for start, end, price in zip(stamps, ends, doc["price"], strict=True)
        if price is not None
    ]


def _hourly_from(items: Sequence[tuple[datetime, float]]) -> Hourly:
    """Series from (hour start, value) pairs, spanning first to last hour, NaN in gaps."""
    if not items:
        return Hourly(_EPOCH, np.array([]))
    start = min(t for t, _ in items)
    values = np.full((max(t for t, _ in items) - start) // _HOUR + 1, np.nan)
    for t, v in items:
        values[(t - start) // _HOUR] = v
    return Hourly(start, values)


def hourly_prices(slots: Sequence[MarketSlot]) -> Hourly:
    """Mean price per UTC hour (15-minute products to hourly); hours without a slot are NaN."""
    by_hour: dict[datetime, list[float]] = {}
    for s in slots:
        hour = s.start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        by_hour.setdefault(hour, []).append(s.price)
    return _hourly_from([(h, float(np.mean(p))) for h, p in by_hour.items()])


def merge_hourly(a: Hourly, b: Hourly) -> Hourly:
    """Union of two series; ``b`` wins where both have a value (its NaN gaps keep ``a``)."""
    if not len(a.values):
        return b
    if not len(b.values):
        return a
    start = min(a.start, b.start)
    end = max(a.start + len(a.values) * _HOUR, b.start + len(b.values) * _HOUR)
    out = np.full((end - start) // _HOUR, np.nan)
    for s in (a, b):
        i = (s.start - start) // _HOUR
        seg = out[i : i + len(s.values)]
        seg[:] = np.where(np.isnan(s.values), seg, s.values)
    return Hourly(start, out)


def parse_open_meteo(
    docs: Sequence[Mapping],
    names: Sequence[str],
    variables: Sequence[str],
    suffix: str = "",
) -> dict[str, Hourly]:
    """``"{name}__{var}"`` series from multi-location Open-Meteo documents.

    ``docs[i]`` belongs to ``names[i]``; its times are GMT without offset, and each
    variable is read from the ``"{var}{suffix}"`` column.
    """
    if len(docs) != len(names):
        raise ValueError(f"{len(docs)} weather documents for {len(names)} points")
    out = {}
    for name, doc in zip(names, docs, strict=True):
        hourly = doc["hourly"]
        times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in hourly["time"]]
        for var in variables:
            vals = [np.nan if v is None else float(v) for v in hourly[f"{var}{suffix}"]]
            out[f"{name}__{var}"] = _hourly_from(list(zip(times, vals, strict=True)))
    return out


def weather_by_horizon_from_previous_runs(
    docs: Sequence[Mapping], names: Sequence[str], variables: Sequence[str]
) -> dict[int, dict[str, Hourly]]:
    """Per horizon h = 1…6 the series of day anchor+h, from the ``_previous_day{h+1}`` columns.

    Radiation at the offsets in ``RADIATION_BROKEN`` is masked before its cutoff date.
    """
    out = {}
    for h in range(1, 7):
        series = parse_open_meteo(docs, names, variables, f"_previous_day{h + 1}")
        cutoff = RADIATION_BROKEN.get(h + 1)
        if cutoff is not None:
            limit = datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=UTC)
            for key, s in series.items():
                if key.endswith(f"__{_RADIATION}"):
                    n = min(max((limit - s.start) // _HOUR, 0), len(s.values))
                    s.values[:n] = np.nan
        out[h] = series
    return out
