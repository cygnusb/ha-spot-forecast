"""Raw-data loading and the training dataset, built on the shared ``core.features``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    needs_capacity,
)
from core.ingest import (
    hourly_prices,
    merge_hourly,
    parse_energy_charts,
    weather_by_horizon_from_previous_runs,
)
from core.zones import DE_LU, ZONES, Zone


@dataclass(frozen=True, eq=False)
class RawData:
    prices: Hourly  # EUR/MWh, hourly means
    weather: dict[int, dict[str, Hourly]]  # offset -> "{point}__{var}" -> series
    capacity: dict[str, dict[str, float]]  # "YYYY-MM" -> cap_* (forward-filled); {} if unused
    zone: Zone = DE_LU
    extra_prices: dict[str, Hourly] = field(default_factory=dict)  # zone.price_inputs by code

    def parent(self) -> RawData:
        """The raw data of the parent zone: its prices, the shared weather and capacity."""
        zone = ZONES[self.zone.parent]
        return RawData(self.extra_prices[zone.code], self.weather, self.capacity, zone,
                       self.extra_prices)  # fmt: skip


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
    # a zone with a parent: the parent zone's rows of the same (anchor, time), see build_dataset
    X_parent: np.ndarray | None = None
    parent_mean7: np.ndarray | None = None
    parent_scale: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(
            **{
                k: None if getattr(self, k) is None else getattr(self, k)[mask]
                for k in self.__dataclass_fields__
            }
        )

    def parent_rows(self) -> Dataset:
        """The parent zone's rows as a dataset (``X``, ``mean7``, ``scale`` of the parent),
        with this zone's targets; for ``decode`` and a parent model's ``predict``."""
        if self.X_parent is None:
            raise ValueError("the dataset has no parent rows")
        return replace(
            self, X=self.X_parent, mean7=self.parent_mean7, scale=self.parent_scale,
            X_parent=None, parent_mean7=None, parent_scale=None,
        )  # fmt: skip


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
    the prices of its price inputs, and the installed capacity if its features (or its
    parent's) use it. The weather covers ``zone.point_sets()``, the parent's points too."""
    capacity = _load_capacity(raw) if needs_capacity(zone) else {}
    extra = {code: _load_prices(raw, ZONES[code]) for code in zone.price_inputs}
    return RawData(_load_prices(raw, zone), _load_weather(raw, zone), capacity, zone, extra)


def build_dataset(raw: RawData, first_anchor: date, last_anchor: date) -> Dataset:
    """Feature rows for every anchor in the range, as research ``build_samples`` keeps them.

    Anchors without 30 days of history are skipped; rows without an actual price,
    ``wind_mean`` or ``p_lastweek`` are dropped (research ``dropna``).

    A zone with a parent also gets the parent zone's rows of the same (anchor, time) in
    ``X_parent`` (with their ``mean7`` and ``scale``), built as for the parent zone itself;
    rows the parent zone does not have are dropped, and their count is printed.
    """
    zone = raw.zone
    col = {name: i for i, name in enumerate(feature_names(zone))}
    daily = daily_stats(raw.prices, zone)  # once; build_features ignores prices after the anchor
    extra = {c: (s, daily_stats(s, ZONES[c])) for c, s in raw.extra_prices.items()}
    weather = {h: raw.weather[h + 1] for h in HORIZONS}
    cols: dict[str, list[np.ndarray]] = {k: [] for k in Dataset.__dataclass_fields__}
    for anchor in (
        first_anchor + timedelta(days=i) for i in range((last_anchor - first_anchor).days + 1)
    ):
        capacity = raw.capacity.get(f"{anchor:%Y-%m}", {})
        try:
            fm = build_features(anchor, raw.prices, daily, weather, capacity, zone, extra)
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
    fields = [k for k in cols if k not in ("X_parent", "parent_mean7", "parent_scale")]
    if not cols["X"]:
        ds = Dataset(**{k: np.empty((0, len(col)) if k == "X" else 0) for k in fields})
    else:
        ds = Dataset(**{k: np.concatenate(cols[k]) for k in fields})
    if zone.parent is None:
        return ds
    return _with_parent(ds, build_dataset(raw.parent(), first_anchor, last_anchor), zone.parent)


def _with_parent(ds: Dataset, parent: Dataset, code: str) -> Dataset:
    """``ds`` with the rows of ``parent`` of the same (anchor, time), without those it lacks."""
    index = {(a, t): i for i, (a, t) in enumerate(zip(parent.anchor, parent.times, strict=True))}
    rows = [index.get((a, t), -1) for a, t in zip(ds.anchor, ds.times, strict=True)]
    rows = np.array(rows, dtype=np.int64)
    keep = rows >= 0
    if not keep.all():
        print(f"  {int((~keep).sum())} of {len(keep)} rows dropped: {code} has no row for them")
    out = ds.subset(keep)
    rows = rows[keep]
    return replace(
        out,
        X_parent=parent.X[rows],
        parent_mean7=parent.mean7[rows],
        parent_scale=parent.scale[rows],
    )
