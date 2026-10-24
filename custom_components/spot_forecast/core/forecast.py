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
    build_features,
    daily_stats,
    decode,
)
from .model import ModelFile, ModelMismatch
from .zones import Zone

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
) -> list[Slot]:
    """Median, raw P90 and raw P10 for every hour of anchor+1 … anchor+6.

    A slot is ``valid`` only when its ``wind_mean`` feature exists (weather present).
    Raises ``InsufficientHistory`` when ``prices`` is too short, ``IncompleteAnchorDay``
    (a subclass) when any hourly price of the anchor day is missing and ``ModelMismatch``
    when ``model`` is for another zone.
    """
    if model.zone != zone.slug:
        raise ModelMismatch(f"model {model.version} is for zone {model.zone!r}, not {zone.slug!r}")
    tz = ZoneInfo(zone.tz)
    day_start = datetime.combine(anchor, time(), tz).astimezone(UTC)
    day_end = datetime.combine(anchor + timedelta(days=1), time(), tz).astimezone(UTC)
    if np.isnan(prices.window(day_start, (day_end - day_start) // timedelta(hours=1))).any():
        raise IncompleteAnchorDay(f"prices of {anchor} are not complete")
    fm = build_features(
        anchor,
        prices,
        daily_stats(prices, zone),
        weather,
        model.capacity if capacity is None else capacity,
        zone,
    )
    wind_mean = fm.X[:, model.features.index("wind_mean")]
    median = decode(model.median.predict(fm.X), fm)
    p90 = decode(model.p90.predict(fm.X), fm)
    p10 = decode(model.p10.predict(fm.X), fm)
    return [
        Slot(t, int(h), float(m), float(hi), float(lo), not bool(np.isnan(wind)))
        for t, h, m, hi, lo, wind in zip(fm.times, fm.h, median, p90, p10, wind_mean, strict=True)
    ]


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
