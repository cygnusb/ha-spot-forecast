"""Quality gate of the monthly training (synthetic, no research data needed)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

pytest.importorskip("pandas")
from core.features import FEATURE_NAMES
from training.dataset import Dataset
from training.gate import evaluate_gate, month_metrics, previous_model_predictions
from training.report import Predictions, evaluate

SEPT = date(2025, 9, 1)


def _predictions(rng: np.random.Generator, targets: list[date], h: list[int]) -> Predictions:
    n = len(targets)
    actual = rng.normal(80, 30, n)
    zeros, obj = np.zeros(n), np.empty(n, dtype=object)
    return Predictions(
        times=obj,
        h=np.array(h),
        anchor=np.array([t - timedelta(days=k) for t, k in zip(targets, h, strict=True)]),
        target_date=np.array(targets, dtype=object),
        train_last_target=obj,
        actual=actual,
        p_lastweek=actual + rng.normal(0, 25, n),
        median=actual + rng.normal(0, 15, n),
        p90_raw=zeros,
        p10_raw=zeros,
    )


def test_month_metrics_matches_evaluate_for_full_window():
    rng = np.random.default_rng(0)
    rows = [(SEPT + timedelta(days=d), k) for d in range(30) for k in range(1, 7)]
    rows = [r for r in rows if rng.random() > 0.2]  # unequal rows per horizon
    p = _predictions(rng, *map(list, zip(*rows, strict=True)))
    # every target day is in September and every anchor is >= since: both select all rows
    since = min(p.anchor)
    overall = evaluate(p, since, date(2025, 10, 1), peak_threshold=200.0)["all"]
    got = month_metrics(p, SEPT)
    assert set(got) == {"mae", "naive_mae", "skill", "rows"}
    assert got["rows"] == len(rows) == overall["rows"]
    for key in ("mae", "naive_mae", "skill"):
        assert got[key] == pytest.approx(overall[key])
    assert 0 < got["skill"] < 1


def test_month_metrics_counts_only_target_days_in_the_month():
    rng = np.random.default_rng(1)
    targets = [date(2025, 8, 31), SEPT, date(2025, 9, 30), date(2025, 10, 1)]
    p = _predictions(rng, targets, [1, 1, 1, 1])
    assert month_metrics(p, SEPT)["rows"] == 2
    assert month_metrics(p, date(2025, 11, 1)) == {
        "rows": 0,
        "mae": None,
        "naive_mae": None,
        "skill": None,
    }


def test_previous_model_predictions_decode_the_month_rows(tiny_model_file):
    rng = np.random.default_rng(2)
    targets = [date(2025, 8, 30), date(2025, 8, 31), SEPT, date(2025, 9, 15), date(2025, 10, 1)]
    n = len(targets)
    ds = Dataset(
        X=rng.normal(size=(n, len(FEATURE_NAMES))),
        y_price=rng.normal(80, 30, n),
        times=np.empty(n, dtype=object),
        h=np.array([1, 2, 1, 2, 1]),
        anchor=np.array([t - timedelta(days=1) for t in targets], dtype=object),
        target_date=np.array(targets, dtype=object),
        mean7=rng.normal(80, 10, n),
        scale=rng.uniform(50, 100, n),
        p_lastweek=rng.normal(80, 30, n),
    )
    p = previous_model_predictions(tiny_model_file, ds, SEPT)
    sel = [2, 3]
    np.testing.assert_array_equal(p.target_date, ds.target_date[sel])
    np.testing.assert_array_equal(p.actual, ds.y_price[sel])
    np.testing.assert_array_equal(p.p_lastweek, ds.p_lastweek[sel])
    for name, trees in (("median", "median"), ("p90_raw", "p90"), ("p10_raw", "p10")):
        expected = (
            getattr(tiny_model_file, trees).predict(ds.X[sel]) * ds.scale[sel] + ds.mean7[sel]
        )
        np.testing.assert_allclose(getattr(p, name), expected)
    assert set(p.train_last_target) == {tiny_model_file.train_cutoff - timedelta(days=1)}


def _m(mae: float | None, skill: float | None, rows: int = 4320) -> dict:
    return {"mae": mae, "naive_mae": 30.0, "skill": skill, "rows": rows}


def test_gate_passes_within_tolerance():
    result = evaluate_gate(_m(19.4, 0.3), _m(18.5, 0.35))
    assert result.passed
    assert result.candidate == _m(19.4, 0.3) and result.previous == _m(18.5, 0.35)
    assert evaluate_gate(_m(18.0, 0.3), _m(18.5, 0.35)).passed


def test_gate_fails_when_worse_than_previous():
    result = evaluate_gate(_m(20.0, 0.3), _m(18.5, 0.35))
    assert not result.passed
    assert "20.00" in result.reason and "18.50" in result.reason


@pytest.mark.parametrize("skill", [0.0, -0.1, None])
def test_gate_fails_without_skill(skill):
    result = evaluate_gate(_m(17.0, skill), _m(18.5, 0.35))
    assert not result.passed
    assert "skill" in result.reason


def test_gate_fails_without_rows():
    result = evaluate_gate(_m(None, None, rows=0), None)
    assert not result.passed
    assert "no rows" in result.reason


def test_gate_first_run_checks_skill_only():
    passed = evaluate_gate(_m(40.0, 0.01), None)
    assert passed.passed and passed.previous is None
    assert "no previous" in passed.reason
    assert not evaluate_gate(_m(10.0, -0.01), None).passed
