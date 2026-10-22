"""Forecast assembly from a model file, plus the conformal calibration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .features import Hourly, build_features, daily_stats, decode
from .model import ModelFile
from .zones import Zone


@dataclass(frozen=True)
class Slot:
    """One forecast hour; prices in EUR/MWh, quantiles not yet conformally calibrated."""

    time: datetime  # UTC hour start
    h: int
    median: float
    p90_raw: float
    p10_raw: float


def run_forecast(
    model: ModelFile,
    anchor: date,
    prices: Hourly,
    weather: Mapping[int, Mapping[str, Hourly]],
    zone: Zone,
    capacity: Mapping[str, float] | None = None,
) -> list[Slot]:
    """Median, raw P90 and raw P10 for every hour of anchor+1 … anchor+6.

    Raises ``InsufficientHistory`` when ``prices`` is too short.
    """
    fm = build_features(
        anchor,
        prices,
        daily_stats(prices, zone),
        weather,
        model.capacity if capacity is None else capacity,
        zone,
    )
    median = decode(model.median.predict(fm.X), fm)
    p90 = decode(model.p90.predict(fm.X), fm)
    p10 = decode(model.p10.predict(fm.X), fm)
    return [
        Slot(t, int(h), float(m), float(hi), float(lo))
        for t, h, m, hi, lo in zip(fm.times, fm.h, median, p90, p10, strict=True)
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
