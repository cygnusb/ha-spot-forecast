"""Quality gate of the monthly training: the candidate against the published model.

Both are scored on the target days of one full month that neither has trained on: the
candidate by its walk-forward predictions (monthly refit), the previous model by
predicting the same dataset rows with its trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from core.features import HORIZONS, decode
from core.model import ModelFile

from .dataset import Dataset
from .report import Predictions, _ct, _num, accuracy, mean_over_horizons

TOLERANCE = 1.0  # EUR/MWh (0.1 ct/kWh) the candidate's MAE may exceed the previous model's


def _in_month(target_date: np.ndarray, month_start: date) -> np.ndarray:
    end = (month_start + timedelta(days=32)).replace(day=1)
    return np.fromiter(
        (month_start <= d < end for d in target_date), dtype=bool, count=len(target_date)
    )


def month_metrics(p: Predictions, month_start: date) -> dict:
    """``rows``, ``mae``, ``naive_mae`` and ``skill`` (EUR/MWh) over the target days of the
    calendar month; as ``report.evaluate``'s "all": per horizon, then the mean over them."""
    in_month = _in_month(p.target_date, month_start)
    horizons = {str(k): accuracy(p.subset(in_month & (p.h == k))) for k in HORIZONS}
    return mean_over_horizons(horizons)


def previous_model_predictions(model: ModelFile, ds: Dataset, month_start: date) -> Predictions:
    """The dataset rows with a target day in the month, predicted by ``model``.

    Out of sample only if ``model.train_cutoff`` <= ``month_start``; the caller checks that.
    """
    rows = ds.subset(_in_month(ds.target_date, month_start))
    # decode reads only scale and mean7, which the Dataset carries per row like FeatureMatrix
    median, p90, p10 = (
        decode(t.predict(rows.X), rows) for t in (model.median, model.p90, model.p10)
    )
    return Predictions(
        times=rows.times,
        h=rows.h,
        anchor=rows.anchor,
        target_date=rows.target_date,
        train_last_target=np.full(
            len(rows.h), model.train_cutoff - timedelta(days=1), dtype=object
        ),
        actual=rows.y_price,
        p_lastweek=rows.p_lastweek,
        median=median,
        p90_raw=p90,
        p10_raw=p10,
    )


@dataclass(frozen=True)
class GateResult:
    passed: bool
    candidate: dict  # month_metrics of the candidate
    previous: dict | None  # month_metrics of the published model; None on the first run
    reason: str


def evaluate_gate(
    candidate: dict, previous: dict | None, tolerance: float = TOLERANCE
) -> GateResult:
    """Pass if the candidate beats the naive baseline (skill > 0) and its MAE is at most
    the previous model's + ``tolerance``. Without a previous model only skill counts."""

    def result(passed: bool, reason: str) -> GateResult:
        return GateResult(passed, candidate, previous, reason)

    mae, skill = candidate["mae"], candidate["skill"]
    if not candidate["rows"] or mae is None:
        return result(False, "the candidate has no rows in the gate month")
    if skill is None or skill <= 0:
        return result(False, f"candidate skill {_num(skill)} against the naive baseline is not > 0")
    if previous is None or previous["mae"] is None:
        return result(True, f"candidate skill {skill:.3f} > 0; no previous model to compare")
    limit = previous["mae"] + tolerance
    if mae > limit:
        return result(
            False,
            f"candidate MAE {mae:.2f} EUR/MWh is worse than the previous model's "
            f"{previous['mae']:.2f} + {tolerance:.2f} tolerance",
        )
    return result(
        True,
        f"candidate MAE {mae:.2f} EUR/MWh <= previous {previous['mae']:.2f} + "
        f"{tolerance:.2f} tolerance; skill {skill:.3f} > 0",
    )


def render_markdown(result: GateResult, month_start: date) -> str:
    """Gate verdict and the two models' month metrics, prices in ct/kWh."""
    lines = [
        f"## Quality gate: {'passed' if result.passed else 'FAILED'}",
        "",
        f"Target days in {month_start:%Y-%m}, out of sample for both models. {result.reason}.",
        "",
        "| model | MAE | naive MAE | skill | rows |",
        "|---|---|---|---|---|",
    ]
    for name, m in (("candidate", result.candidate), ("previous", result.previous)):
        if m is None:
            lines.append(f"| {name} | - | - | - | - |")
        else:
            lines.append(
                f"| {name} | {_ct(m['mae'])} | {_ct(m['naive_mae'])} | {_num(m['skill'])} "
                f"| {m['rows']} |"
            )
    return "\n".join(lines) + "\n"
