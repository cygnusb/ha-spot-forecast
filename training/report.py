"""Accuracy report over walk-forward (out-of-sample) predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np

from core.features import HORIZONS
from core.forecast import conformal_offset

CALIBRATION_DAYS = 56  # trailing target days for the conformal P90 / P10 offsets


@dataclass(frozen=True, eq=False)
class Predictions:
    """Walk-forward predictions, one row per (anchor, target hour); prices in EUR/MWh."""

    times: np.ndarray  # target hour start, UTC datetime (object array)
    h: np.ndarray
    anchor: np.ndarray  # issue day, date (object array)
    target_date: np.ndarray  # local date (object array)
    train_cutoff: np.ndarray  # first target day the predicting model never saw (object array)
    actual: np.ndarray
    p_lastweek: np.ndarray
    median: np.ndarray
    p90_raw: np.ndarray
    p10_raw: np.ndarray

    def subset(self, mask: np.ndarray) -> Predictions:
        return Predictions(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})


def _ordinals(dates: np.ndarray) -> np.ndarray:
    return np.fromiter((d.toordinal() for d in dates), dtype=np.int64, count=len(dates))


def window_offsets(
    target_date: np.ndarray, h: np.ndarray, residual: np.ndarray, end: date, q: float
) -> dict[int, float]:
    """Per horizon: ``conformal_offset`` over target days in ``[end − 56 days, end)``."""
    td = _ordinals(target_date)
    in_window = (td >= (end - timedelta(days=CALIBRATION_DAYS)).toordinal()) & (
        td < end.toordinal()
    )
    return {k: conformal_offset(residual[in_window & (h == k)], 0.0, q=q) for k in HORIZONS}


def rolling_offsets(p: Predictions, residual: np.ndarray, q: float) -> np.ndarray:
    """Per row: the offset known at its issue day (trailing 56 target days before the anchor)."""
    out = np.zeros(len(residual))
    td, an = _ordinals(p.target_date), _ordinals(p.anchor)
    for k in HORIZONS:
        rows = np.flatnonzero(p.h == k)
        order = np.argsort(td[rows], kind="stable")
        td_sorted, res_sorted = td[rows][order], residual[rows][order]
        for a in np.unique(an[rows]):
            lo = np.searchsorted(td_sorted, a - CALIBRATION_DAYS, "left")
            hi = np.searchsorted(td_sorted, a, "left")
            out[rows[an[rows] == a]] = conformal_offset(res_sorted[lo:hi], 0.0, q=q)
    return out


def _mean(x: np.ndarray) -> float | None:
    return float(np.mean(x)) if len(x) else None


def _horizon_metrics(
    p: Predictions, p90_cal: np.ndarray, p10_cal: np.ndarray, peak: float, trailing_from: date
) -> dict[str, float | int | None]:
    err = p.median - p.actual
    mae, naive = _mean(np.abs(err)), _mean(np.abs(p.p_lastweek - p.actual))
    is_peak, is_neg = p.actual > peak, p.actual < 0
    recent = _ordinals(p.target_date) >= trailing_from.toordinal()
    return {
        "rows": len(err),
        "mae": mae,
        "mae_3m": _mean(np.abs(err[recent])),
        "naive_mae": naive,
        "skill": 1 - mae / naive if mae is not None and naive else None,
        "peak_mae": _mean(np.abs(err[is_peak])),
        "peak_bias": _mean(err[is_peak]),
        "p90_exceedance": _mean(p.actual > p90_cal),
        "p10_undercut": _mean(p.actual < p10_cal),
        "neg_p10_below_zero": _mean(p10_cal[is_neg] < 0),
    }


def evaluate(p: Predictions, since: date, cutoff: date, peak_threshold: float) -> dict[str, object]:
    """Report numbers (EUR/MWh, rates as fractions) over rows with anchor ≥ ``since``.

    P90 / P10 are calibrated per issue day with the rolling offset, as the integration
    will do it; all walk-forward rows (also those before ``since``) feed the offsets.
    "all" is the mean over the horizons (as research ``report``).
    """
    p90_cal = np.maximum(p.p90_raw + rolling_offsets(p, p.actual - p.p90_raw, 0.9), p.median)
    p10_cal = np.minimum(p.p10_raw + rolling_offsets(p, p.actual - p.p10_raw, 0.1), p.median)
    y, m = divmod(cutoff.year * 12 + cutoff.month - 1 - 3, 12)
    trailing_from = date(y, m + 1, 1)
    sel = _ordinals(p.anchor) >= since.toordinal()
    horizons = {
        str(k): _horizon_metrics(
            p.subset(sel & (p.h == k)),
            p90_cal[sel & (p.h == k)],
            p10_cal[sel & (p.h == k)],
            peak_threshold,
            trailing_from,
        )
        for k in HORIZONS
    }
    keys = next(iter(horizons.values())).keys()
    overall = {}
    for key in keys:
        vals = [v[key] for v in horizons.values() if v[key] is not None]
        overall[key] = sum(vals) if key == "rows" else (sum(vals) / len(vals) if vals else None)
    return {
        "unit": "EUR/MWh",
        "train_cutoff": cutoff.isoformat(),
        "walk_forward_from": since.isoformat(),
        "trailing_3m_from": trailing_from.isoformat(),
        "peak_threshold": peak_threshold,
        "horizons": horizons,
        "all": overall,
    }


def _ct(v: float | None) -> str:
    return "-" if v is None or math.isnan(v) else f"{v / 10:.2f}"


def _pct(v: float | None) -> str:
    return "-" if v is None or math.isnan(v) else f"{100 * v:.1f} %"


def _num(v: float | None) -> str:
    return "-" if v is None or math.isnan(v) else f"{v:.2f}"


def render_markdown(version: str, metrics: dict) -> str:
    """Markdown report in ct/kWh."""
    head = (
        "| horizon | MAE | MAE last 3 mo | naive MAE | skill | peak MAE | peak bias "
        "| P90 exceeded | P10 undercut | neg. hours with P10 < 0 | rows |"
    )
    lines = [
        f"# Spot Forecast model {version}",
        "",
        f"Walk-forward, out of sample: issue days {metrics['walk_forward_from']} until the "
        f"training cutoff {metrics['train_cutoff']} (target days before it), monthly refit. "
        f"Last 3 months = target days from {metrics['trailing_3m_from']}. "
        "Naive = same hour last week; skill = 1 - MAE / naive MAE. "
        f"Peak hours: actual > {_ct(metrics['peak_threshold'])} ct/kWh, bias = forecast - actual. "
        f"P90 / P10 calibrated with the rolling conformal offset ({CALIBRATION_DAYS} days). "
        "Prices in ct/kWh; *all* = mean over horizons.",
        "",
        head,
        "|" + "---|" * (head.count("|") - 1),
    ]
    rows = [(f"D+{k}", metrics["horizons"][str(k)]) for k in HORIZONS]
    rows.append(("all", metrics["all"]))
    for name, r in rows:
        lines.append(
            f"| {name} | {_ct(r['mae'])} | {_ct(r['mae_3m'])} | {_ct(r['naive_mae'])} "
            f"| {_num(r['skill'])} | {_ct(r['peak_mae'])} | {_ct(r['peak_bias'])} "
            f"| {_pct(r['p90_exceedance'])} | {_pct(r['p10_undercut'])} "
            f"| {_pct(r['neg_p10_below_zero'])} | {r['rows']} |"
        )
    return "\n".join(lines) + "\n"
