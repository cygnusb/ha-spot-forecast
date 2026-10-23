"""Raw-data loading and the training dataset, built on the shared ``core.features``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import reduce
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.features import (
    HORIZONS,
    Hourly,
    InsufficientHistory,
    build_features,
    daily_stats,
    decode,
    feature_names,
    uses_capacity,
)
from core.ingest import (
    hourly_prices,
    merge_hourly,
    parse_energy_charts,
    weather_by_horizon_from_previous_runs,
)
from core.zones import DE_LU, Zone


@dataclass(frozen=True, eq=False)
class RawData:
    prices: Hourly  # EUR/MWh, hourly means
    weather: dict[int, dict[str, Hourly]]  # offset -> "{point}__{var}" -> series
    capacity: dict[str, dict[str, float]]  # "YYYY-MM" -> cap_* (forward-filled); {} if unused
    zone: Zone = DE_LU


@dataclass(frozen=True, eq=False)
class Dataset:
    """One row per (anchor, target hour); all arrays have ``n`` rows."""

    X: np.ndarray  # n x len(feature_names(zone))
    y_price: np.ndarray  # actual price, EUR/MWh
    times: np.ndarray  # target hour start, UTC datetime (object array)
    h: np.ndarray
    anchor: np.ndarray  # date (object array)
    target_date: np.ndarray  # local date (object array)
    mean7: np.ndarray
    scale: np.ndarray
    p_lastweek: np.ndarray  # naive baseline, EUR/MWh (the feature column in X is relative)

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})


def _merge_files(parts: list[Hourly]) -> Hourly:
    """Merge series of files in name order; as in research, the earlier file wins on overlap."""
    return reduce(
        merge_hourly, reversed(parts), Hourly(datetime(1970, 1, 1, tzinfo=UTC), np.array([]))
    )


def _json_files(directory: Path, pattern: str) -> list[Path]:
    """Data files, sorted, without hidden files (``._*`` AppleDouble files from macOS tar)."""
    return sorted(f for f in directory.glob(pattern) if not f.name.startswith("."))


def _load_prices(raw: Path, zone: Zone) -> Hourly:
    slots = {}
    for f in _json_files(raw / "prices" / zone.code, "*.json"):
        for s in parse_energy_charts(json.loads(f.read_text())):
            slots.setdefault(s.start, s)
    return hourly_prices(sorted(slots.values(), key=lambda s: s.start))


def _load_weather(raw: Path, zone: Zone) -> dict[int, dict[str, Hourly]]:
    """The series of the zone's own and neighbour points (the files of other points stay)."""
    point_vars = {name: variables for points, variables in zone.point_sets() for name in points}
    out: dict[int, dict[str, Hourly]] = {h + 1: {} for h in HORIZONS}
    for name, variables in point_vars.items():
        files = [
            weather_by_horizon_from_previous_runs([json.loads(f.read_text())], [name], variables)
            for f in _json_files(raw / "weather", f"{name}_*.json")
        ]
        for h in HORIZONS:
            for key in files[0][h]:
                out[h + 1][key] = _merge_files([f[h][key] for f in files])
    return out


def _load_capacity(raw: Path) -> dict[str, dict[str, float]]:
    d = json.loads((raw / "installed_power.json").read_text())
    idx = [f"{t[3:]}-{t[:2]}" for t in d["time"]]  # "MM.YYYY" -> "YYYY-MM"
    cols = {p["name"]: p["data"] for p in d["production_types"]}
    df = pd.DataFrame(cols, index=idx)
    caps = pd.DataFrame(
        {
            "cap_solar": df.get("Solar AC", df.get("Solar DC")),
            "cap_wind_on": df["Wind onshore"],
            "cap_wind_off": df["Wind offshore"],
        }
    ).ffill()
    return {month: {k: float(v) for k, v in row.items()} for month, row in caps.iterrows()}


def load_raw(raw: Path, zone: Zone = DE_LU) -> RawData:
    """The zone's prices and weather from the shared raw directory (``training.fetch``),
    and the installed capacity if its features use it."""
    capacity = _load_capacity(raw) if uses_capacity(zone) else {}
    return RawData(_load_prices(raw, zone), _load_weather(raw, zone), capacity, zone)


def build_dataset(raw: RawData, first_anchor: date, last_anchor: date) -> Dataset:
    """Feature rows for every anchor in the range, as research ``build_samples`` keeps them.

    Anchors without 30 days of history are skipped; rows without an actual price,
    ``wind_mean`` or ``p_lastweek`` are dropped (research ``dropna``).
    """
    zone = raw.zone
    col = {name: i for i, name in enumerate(feature_names(zone))}
    daily = daily_stats(raw.prices, zone)  # once; build_features ignores prices after the anchor
    weather = {h: raw.weather[h + 1] for h in HORIZONS}
    cols: dict[str, list[np.ndarray]] = {k: [] for k in Dataset.__dataclass_fields__}
    for anchor in (
        first_anchor + timedelta(days=i) for i in range((last_anchor - first_anchor).days + 1)
    ):
        capacity = raw.capacity.get(f"{anchor:%Y-%m}", {})
        try:
            fm = build_features(anchor, raw.prices, daily, weather, capacity, zone)
        except InsufficientHistory:
            continue
        price = np.array([raw.prices.at(t) for t in fm.times])
        p_lastweek = decode(fm.X[:, col["p_lastweek"]], fm)  # back to EUR/MWh
        keep = ~(np.isnan(price) | np.isnan(fm.X[:, col["wind_mean"]]) | np.isnan(p_lastweek))
        n = len(fm.times)
        rows = {
            "X": fm.X,
            "y_price": price,
            "times": np.array(fm.times, dtype=object),
            "h": fm.h,
            "anchor": np.full(n, anchor, dtype=object),
            "target_date": np.array([anchor + timedelta(days=int(h)) for h in fm.h], dtype=object),
            "mean7": fm.mean7,
            "scale": fm.scale,
            "p_lastweek": p_lastweek,
        }
        for k, v in rows.items():
            cols[k].append(v[keep])
    if not cols["X"]:
        return Dataset(**{k: np.empty((0, len(col)) if k == "X" else 0) for k in cols})
    return Dataset(**{k: np.concatenate(v) for k, v in cols.items()})
