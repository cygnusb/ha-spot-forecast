"""Rolling conformal offsets (synthetic, no research data needed)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from training.report import Predictions, rolling_offsets

ANCHOR = date(2025, 6, 1)


def _predictions(target_dates: list[date], anchors: list[date]) -> Predictions:
    n = len(target_dates)
    obj = np.empty(n, dtype=object)
    zeros = np.zeros(n)
    return Predictions(
        times=obj,
        h=np.ones(n, dtype=int),
        anchor=np.array(anchors, dtype=object),
        target_date=np.array(target_dates, dtype=object),
        train_last_target=obj,
        actual=zeros,
        p_lastweek=zeros,
        median=zeros,
        p90_raw=zeros,
        p10_raw=zeros,
    )


def test_rolling_offset_uses_only_the_56_target_days_before_the_anchor():
    targets, residuals = [], []

    def add(day: date, count: int, value: float) -> None:
        targets.extend([day] * count)
        residuals.extend([value] * count)

    day = ANCHOR - timedelta(days=92)
    while day < ANCHOR - timedelta(days=56):  # older than the window: must not count
        add(day, 5, -100.0)
        day += timedelta(days=1)
    while day < ANCHOR:  # the window: 56 days x 5 = 280 residuals >= 200
        add(day, 5, 1.0)
        day += timedelta(days=1)
    add(ANCHOR, 100, 100.0)  # target day == anchor: not yet known, must not count
    anchors = [t - timedelta(days=1) for t in targets]
    # the row being calibrated: issued on ANCHOR for the next day
    targets.append(ANCHOR + timedelta(days=1))
    anchors.append(ANCHOR)
    residuals.append(np.nan)

    p = _predictions(targets, anchors)
    r = np.array(residuals)
    assert rolling_offsets(p, r, 0.9)[-1] == 1.0
    assert rolling_offsets(p, r, 0.1)[-1] == 1.0


def test_rolling_offset_falls_back_to_zero_with_too_few_residuals():
    targets = [ANCHOR - timedelta(days=1)] * 199 + [ANCHOR + timedelta(days=1)]
    anchors = [t - timedelta(days=1) for t in targets[:-1]] + [ANCHOR]
    r = np.array([5.0] * 199 + [np.nan])
    assert rolling_offsets(_predictions(targets, anchors), r, 0.9)[-1] == 0.0
