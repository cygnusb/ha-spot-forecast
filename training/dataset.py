"""Raw-data loading and the training dataset, built on the shared ``core.features``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.features import (
    FEATURE_NAMES,
    HORIZONS,
    Hourly,
    InsufficientHistory,
    build_features,
    daily_stats,
)
from core.zones import DE_LU

from .constants import OFFSETS

# Open-Meteo previous-runs radiation at lead offsets >= 6 days is wrong before this date
# (shifted/scaled by 0.5-1.6x; ECMWF switches to coarser output steps at 144 h). Masked.
RADIATION_BROKEN = {6: "2025-07-01", 7: "2025-07-01"}
_COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True, eq=False)
class RawData:
    prices: Hourly  # EUR/MWh, hourly means
    weather: dict[int, dict[str, Hourly]]  # offset -> "{point}__{var}" -> series
    capacity: dict[str, dict[str, float]]  # "YYYY-MM" -> cap_* (forward-filled)


@dataclass(frozen=True, eq=False)
class Dataset:
    """One row per (anchor, target hour); all arrays have ``n`` rows."""

    X: np.ndarray  # n x len(FEATURE_NAMES)
    y_price: np.ndarray  # actual price, EUR/MWh
    times: np.ndarray  # target hour start, UTC datetime (object array)
    h: np.ndarray
    anchor: np.ndarray  # date (object array)
    target_date: np.ndarray  # local date (object array)
    mean7: np.ndarray
    scale: np.ndarray
    p_lastweek: np.ndarray  # naive baseline


def _hourly(s: pd.Series) -> Hourly:
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return Hourly(s.index.min().to_pydatetime(), s.reindex(idx).to_numpy(float))


def _load_prices(raw: Path) -> Hourly:
    parts = []
    for f in sorted((raw / "prices" / DE_LU.code).glob("*.json")):
        d = json.loads(f.read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        parts.append(pd.Series(d["price"], index=idx))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()].dropna()
    # 15-min products since 2025-10-01: hourly means throughout
    return _hourly(s.resample("h").mean().dropna())


def _load_weather(raw: Path) -> dict[int, dict[str, Hourly]]:
    point_vars = {
        **dict.fromkeys(DE_LU.points, DE_LU.weather_vars),
        **dict.fromkeys(DE_LU.neighbor_points, DE_LU.neighbor_vars),
    }
    out: dict[int, dict[str, Hourly]] = {o: {} for o in OFFSETS}
    for name, variables in point_vars.items():
        frames = []
        for f in sorted((raw / "weather").glob(f"{name}_*.json")):
            h = json.loads(f.read_text())["hourly"]
            frames.append(
                pd.DataFrame(h).assign(time=lambda x: pd.to_datetime(x["time"], utc=True))
            )
        df = pd.concat(frames).drop_duplicates("time").set_index("time").sort_index()
        for o, until in RADIATION_BROKEN.items():
            df.loc[
                df.index < pd.Timestamp(until, tz="UTC"), f"shortwave_radiation_previous_day{o}"
            ] = np.nan
        for o in OFFSETS:
            for v in variables:
                out[o][f"{name}__{v}"] = _hourly(df[f"{v}_previous_day{o}"])
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


def load_raw(raw: Path) -> RawData:
    return RawData(_load_prices(raw), _load_weather(raw), _load_capacity(raw))


def build_dataset(raw: RawData, first_anchor: date, last_anchor: date) -> Dataset:
    """Feature rows for every anchor in the range, as research ``build_samples`` keeps them.

    Anchors without 30 days of history are skipped; rows without an actual price,
    ``wind_mean`` or ``p_lastweek`` are dropped (research ``dropna``).
    """
    daily = daily_stats(raw.prices, DE_LU)  # once; build_features ignores prices after the anchor
    weather = {h: raw.weather[h + 1] for h in HORIZONS}
    cols: dict[str, list[np.ndarray]] = {k: [] for k in Dataset.__dataclass_fields__}
    for anchor in (
        first_anchor + timedelta(days=i) for i in range((last_anchor - first_anchor).days + 1)
    ):
        capacity = raw.capacity.get(f"{anchor:%Y-%m}", {})
        try:
            fm = build_features(anchor, raw.prices, daily, weather, capacity, DE_LU)
        except InsufficientHistory:
            continue
        price = np.array([raw.prices.at(t) for t in fm.times])
        p_lastweek = fm.X[:, _COL["p_lastweek"]]
        keep = ~(np.isnan(price) | np.isnan(fm.X[:, _COL["wind_mean"]]) | np.isnan(p_lastweek))
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
        return Dataset(**{k: np.empty((0, len(FEATURE_NAMES)) if k == "X" else 0) for k in cols})
    return Dataset(**{k: np.concatenate(v) for k, v in cols.items()})
