"""Monthly walk-forward (as training.train.walk_forward), median model only, one variant.

usage: wf.py VARIANT [MAX_ITER]
  VARIANT: base | base_seed1 | hol | dst | both
Writes pred_<VARIANT>.npz (row index into cache, prediction) and a runtime line.
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parents[1] / "data" / "holidays_dst"  # build.py output
FIRST_ANCHOR = date(2025, 10, 1)
CUTOFF = date(2026, 10, 22)


def variant_X(c, variant: str) -> np.ndarray:
    names = list(c["feature_names"])
    X = c["X"].copy()
    if variant in ("dst", "both"):
        X[:, names.index("p_lastday")] = c["p_lastday_dst"]
        X[:, names.index("p_lastweek")] = c["p_lastweek_dst"]
    if variant in ("hol", "both"):
        X = np.column_stack([X, c["hol"]])  # hol_share, hol_bridge, hol_eve, lu_holiday
    return X


def fit(X, y, max_iter: int, seed: int) -> HistGradientBoostingRegressor:
    # same hyperparameters as training.train.fit (median model)
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    ).fit(X, y)


def main() -> None:
    variant = sys.argv[1]
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    seed = 1 if variant == "base_seed1" else 0
    c = np.load(HERE / "cache.npz")
    X = variant_X(c, variant)
    y = (c["y_price"] - c["mean7"]) / c["scale"]
    anchor, td = c["anchor"], c["target_date"]
    idx_all, pred_all, month_all, iters = [], [], [], []
    t0 = time.time()
    month = FIRST_ANCHOR.replace(day=1)
    while month < CUTOFF:
        nxt = (month + timedelta(days=32)).replace(day=1)
        test = np.flatnonzero((anchor >= max(month, FIRST_ANCHOR).toordinal())
                              & (anchor < nxt.toordinal()) & (td < CUTOFF.toordinal()))
        if len(test):
            train = np.flatnonzero(td < month.toordinal())  # keeps cache row order (h, time)
            m = fit(X[train], y[train], max_iter, seed)
            p = m.predict(X[test]) * c["scale"][test] + c["mean7"][test]
            idx_all.append(test)
            pred_all.append(p)
            month_all.append(np.full(len(test), month.toordinal()))
            iters.append(m.n_iter_)
            print(f"{variant} {month:%Y-%m} train={len(train)} test={len(test)} n_iter={m.n_iter_} "
                  f"t={time.time() - t0:.0f}s", flush=True)
        month = nxt
    dt = time.time() - t0
    np.savez(HERE / f"pred_{variant}.npz", idx=np.concatenate(idx_all), pred=np.concatenate(pred_all),
             month=np.concatenate(month_all), runtime=dt, n_iter=np.array(iters), max_iter=max_iter)
    print(f"{variant} done in {dt:.0f}s  n_iter={iters}")


if __name__ == "__main__":
    main()
