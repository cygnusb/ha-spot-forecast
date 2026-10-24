"""Forecast assembly from a model file, plus the conformal calibration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .features import (
    Hourly,
    IncompleteAnchorDay,
    InsufficientHistory,
    build_features,
    daily_stats,
    decode,
)
from .model import ModelFile, ModelMismatch
from .zones import ZONES, Zone

__all__ = ["IncompleteAnchorDay"]  # re-exported: raised by run_forecast


@dataclass(frozen=True)
class Slot:
    """One forecast hour; prices in EUR/MWh, quantiles not yet conformally calibrated."""

    time: datetime  # UTC hour start
    h: int
    median: float
    p90_raw: float
    p10_raw: float
    valid: bool  # False when the hour's weather inputs are missing


def run_forecast(
    model: ModelFile,
    anchor: date,
    prices: Hourly,
    weather: Mapping[int, Mapping[str, Hourly]],
    zone: Zone,
    capacity: Mapping[str, float] | None = None,
    extra_prices: Mapping[str, Hourly] | None = None,
) -> list[Slot]:
    """Median, raw P90 and raw P10 for every hour of anchor+1 … anchor+6.

    ``extra_prices`` holds the prices of the zones in ``zone.price_inputs``, by zone code;
    ``weather`` must hold the series of every point in ``zone.point_sets()``. A composite
    model (``model.parent``) also runs its parent model on the parent zone, with that zone's
    prices from ``extra_prices`` and the parent's capacity, and every slot is the blend of
    the two: ``blend`` x own + (1 - ``blend``) x parent, for the median and both quantiles.

    A slot is ``valid`` only when its ``wind_mean`` feature exists (weather present), and
    for a composite model only when the parent's slot of the same hour is valid too.
    Raises ``InsufficientHistory`` when ``prices`` is too short or the prices of a price
    input are missing, ``IncompleteAnchorDay`` (a subclass) when any hourly price of the
    anchor day is missing (for a composite model: also of the parent zone) and
    ``ModelMismatch`` when ``model`` is for another zone.
    """
    if model.zone != zone.slug:
        raise ModelMismatch(f"model {model.version} is for zone {model.zone!r}, not {zone.slug!r}")
    _check_anchor_day(prices, anchor, zone)
    extra = extra_prices or {}
    missing = [c for c in zone.price_inputs if c not in extra]
    if missing:
        raise InsufficientHistory(f"no prices of {', '.join(missing)} for zone {zone.code}")
    fm = build_features(
        anchor,
        prices,
        daily_stats(prices, zone),
        weather,
        model.capacity if capacity is None else capacity,
        zone,
        {c: (extra[c], daily_stats(extra[c], ZONES[c])) for c in zone.price_inputs},
    )
    wind_mean = fm.X[:, model.features.index("wind_mean")]
    median = decode(model.median.predict(fm.X), fm)
    p90 = decode(model.p90.predict(fm.X), fm)
    p10 = decode(model.p10.predict(fm.X), fm)
    slots = [
        Slot(t, int(h), float(m), float(hi), float(lo), not bool(np.isnan(wind)))
        for t, h, m, hi, lo, wind in zip(fm.times, fm.h, median, p90, p10, wind_mean, strict=True)
    ]
    if model.parent is None:
        return slots
    parent_zone = ZONES[zone.parent]
    if parent_zone.code not in extra:
        raise InsufficientHistory(f"no prices of {parent_zone.code} for zone {zone.code}")
    parent = run_forecast(
        model.parent, anchor, extra[parent_zone.code], weather, parent_zone, extra_prices=extra
    )
    return blend_slots(slots, parent, model.blend)


def blend_slots(own: list[Slot], parent: list[Slot], blend: float) -> list[Slot]:
    """``blend`` x ``own`` + (1 - ``blend``) x ``parent``, matched by (time, h).

    A slot is valid when both of its sides are; one without a parent slot is invalid and NaN.
    """
    by_key = {(s.time, s.h): s for s in parent}
    nan = float("nan")
    out = []
    for s in own:
        p = by_key.get((s.time, s.h))
        if p is None:
            out.append(Slot(s.time, s.h, nan, nan, nan, False))
            continue
        out.append(
            Slot(
                s.time,
                s.h,
                blend * s.median + (1 - blend) * p.median,
                blend * s.p90_raw + (1 - blend) * p.p90_raw,
                blend * s.p10_raw + (1 - blend) * p.p10_raw,
                s.valid and p.valid,
            )
        )
    return out


def _check_anchor_day(prices: Hourly, anchor: date, zone: Zone) -> None:
    tz = ZoneInfo(zone.tz)
    day_start = datetime.combine(anchor, time(), tz).astimezone(UTC)
    day_end = datetime.combine(anchor + timedelta(days=1), time(), tz).astimezone(UTC)
    if np.isnan(prices.window(day_start, (day_end - day_start) // timedelta(hours=1))).any():
        raise IncompleteAnchorDay(f"{zone.code} prices of {anchor} are not complete")


def conformal_offset(
    residuals: np.ndarray, default: float, min_n: int = 200, q: float = 0.9
) -> float:
    """``q``-quantile of the residuals, or ``default`` with fewer than ``min_n`` non-NaN values."""
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    return float(np.quantile(r, q)) if len(r) >= min_n else default


def peak_risk(slot: Slot, offset: float) -> float:
    """Calibrated P90, never below the median."""
    return max(slot.p90_raw + offset, slot.median)


def low_price(slot: Slot, offset: float) -> float:
    """Calibrated P10, never above the median."""
    return min(slot.p10_raw + offset, slot.median)
