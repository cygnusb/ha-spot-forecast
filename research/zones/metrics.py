"""Metrics over the walk-forward predictions (training.report.evaluate), table per run.

python -I metrics.py [pkl ...]   (default: all out/pred_*.pkl; WF_FROM / TAG as in bt.py)
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genfeat  # noqa: E402,F401
import numpy as np  # noqa: E402
from training.report import Predictions, evaluate  # noqa: E402

from bt import CUTOFF, EVAL_FROM, OUT, TAG, WF_TO  # noqa: E402


def load(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    p = Predictions(times=r["times"], h=r["h"], anchor=r["anchor"], target_date=r["target"],
                    train_last_target=r["train_last_target"], actual=r["y"], p_lastweek=r["naive"],
                    median=r["median"], p90_raw=r["p90_raw"], p10_raw=r["median"] * np.nan)
    return r, p


def summarize(path):
    r, p = load(path)
    sel = np.array([a >= EVAL_FROM for a in p.anchor])
    ev = p.subset(sel)
    peak = float(np.percentile(ev.actual, 97))
    m = evaluate(p, EVAL_FROM, min(CUTOFF, WF_TO), peak)
    hz = m["horizons"]
    meanabs = float(np.mean(np.abs(ev.actual)))
    from datetime import timedelta
    pr = genfeat.load_prices(r["meta"]["zone"])
    lastday = np.array([pr.at(t - timedelta(days=int(h))) for t, h in zip(ev.times, ev.h)])
    naive_d = {str(k): float(np.nanmean(np.abs(lastday - ev.actual)[ev.h == k])) for k in range(1, 7)}
    out = {
        "zone": r["meta"]["zone"], "variant": r["meta"]["variant"], "max_iter": r["meta"]["max_iter"],
        "features": r["meta"]["features"], "rows_eval": int(sel.sum()),
        "eval": f"{min(ev.anchor)}..{max(ev.target_date)}",
        "mae_h": {k: hz[k]["mae"] for k in hz}, "naive_h": {k: hz[k]["naive_mae"] for k in hz},
        "skill_h": {k: hz[k]["skill"] for k in hz},
        "naive_d_h": naive_d, "naive_d": float(np.mean(list(naive_d.values()))),
        "mae": m["all"]["mae"], "naive": m["all"]["naive_mae"], "skill": m["all"]["skill"],
        "mae_3m": m["all"]["mae_3m"],
        "p90_exc": m["all"]["p90_exceedance"] if not np.isnan(r["p90_raw"]).all() else float("nan"),
        "mean_abs_price": meanabs, "mean_price": float(np.mean(ev.actual)),
        "std_price": float(np.std(ev.actual)),
        "neg_share": float(np.mean(ev.actual < 0)), "bias": float(np.mean(ev.median - ev.actual)), "peak_mae": m["all"]["peak_mae"],
        "peak_bias": m["all"]["peak_bias"], "peak_thr": peak,
        "build_s": r["meta"]["build_s"], "fit_s": r["meta"]["fit_s"],
        "refit": r["meta"].get("refit_months", 1),
    }
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(OUT, "pred_*.pkl")))
    rows = [summarize(p) for p in paths]
    with open(os.path.join(OUT, f"metrics{TAG}.json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    ct = lambda v: f"{v / 10:.2f}"  # noqa: E731
    print("| zone | variant | it | feats | MAE h1 | h3 | h6 | MAE all | naive | skill | mean|p| | MAE/mean|p| "
          "| MAE 3m | naiveD | skillD | neg% | build+fit s |")
    print("|" + "---|" * 17)
    for o in rows:
        print(f"| {o['zone']} | {o['variant']} | {o['max_iter']} | {o['features']} | {ct(o['mae_h']['1'])} "
              f"| {ct(o['mae_h']['3'])} | {ct(o['mae_h']['6'])} | {ct(o['mae'])} | {ct(o['naive'])} "
              f"| {o['skill']:.2f} | {ct(o['mean_abs_price'])} | {o['mae'] / o['mean_abs_price']:.2f} "
              f"| {ct(o['mae_3m'])} | {ct(o['naive_d'])} | {1 - o['mae'] / o['naive_d']:.2f} | {100 * o['neg_share']:.1f} "
              f"| {o['build_s']:.0f}+{o['fit_s']:.0f} |")


if __name__ == "__main__":
    main()
