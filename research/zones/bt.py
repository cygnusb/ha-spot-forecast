"""Walk-forward backtest for one zone and feature variant (same scheme as training.train).

python -I bt.py ZONE VARIANT [max_iter]
VARIANT: base | noprecip | hydro | noweather | core | core_nocap (core* = DE-LU production features)
env: REFIT_MONTHS (default 3, production 1), P90=1, WF_FROM / WF_TO / TAG for another window
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genfeat  # noqa: E402  (sets sys.path for core/training)
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from zonecfg import RESEARCH_RAW, ZONES  # noqa: E402

CUTOFF = date(2026, 10, 22)  # target days before this (data ends 2026-10-21)
WF_FROM = date.fromisoformat(os.environ.get("WF_FROM", "2025-10-01"))  # walk-forward start (1st of a month)
WF_TO = date.fromisoformat(os.environ.get("WF_TO", CUTOFF.isoformat()))  # issue days before this
STEP = int(os.environ.get("REFIT_MONTHS", "3"))  # refit every STEP months (production: 1)
QUANTS = (("median", None),) if os.environ.get("P90") != "1" else (("median", None), ("p90_raw", 0.9))
EVAL_FROM = WF_FROM  # report window: issue days from here
TAG = os.environ.get("TAG", "")  # suffix of the output files, e.g. "_2025q2q3" for another window
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zones", "out")


def fit(X, y, quantile, max_iter):
    m = HistGradientBoostingRegressor(
        loss="absolute_error" if quantile is None else "quantile", quantile=quantile,
        learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0,
    )
    return m.fit(X, y)


def dataset(zone, variant):
    if variant.startswith("core"):
        from training.dataset import build_dataset as tb, load_raw
        from training.train import build_training_data
        from core.features import FEATURE_NAMES

        raw = load_raw(RESEARCH_RAW)
        ds = build_training_data(raw, CUTOFF)
        names = list(FEATURE_NAMES)
        X = ds.X
        if variant == "core_nodoy":
            keep = [i for i, n in enumerate(names) if n != "doy"]
            X, names = X[:, keep], [names[i] for i in keep]
        if variant == "core_nocap":
            keep = [i for i, n in enumerate(names) if not n.startswith("cap_")]
            X, names = X[:, keep], [names[i] for i in keep]
        return {"X": X, "y": ds.y_price, "times": ds.times, "h": ds.h, "anchor": ds.anchor,
                "target": ds.target_date, "mean7": ds.mean7, "scale": ds.scale,
                "naive": ds.p_lastweek, "names": tuple(names)}
    cfg = ZONES[zone]
    if variant.startswith("deset"):
        from dataclasses import replace
        de = ZONES["DE-LU"]
        cfg = replace(cfg, extra={"raw_extra": tuple(p for p in de.points if p not in cfg.points),
                                  "de_price": variant == "desetp"})
    prices = genfeat.load_prices(zone)
    weather = genfeat.load_weather(cfg, de_extra=(variant in ("plusde", "deset", "desetp")))
    ds = genfeat.build_dataset(cfg, prices, weather, prices.start.date(), CUTOFF - timedelta(days=1),
                               hydro_extra=(variant in ("hydro", "hydnd")), de_extra=(variant in ("plusde", "deset", "desetp")))
    keep = np.flatnonzero(ds["target"] < CUTOFF)
    order = np.lexsort((np.array([t.timestamp() for t in ds["times"][keep]]), ds["h"][keep]))
    idx = keep[order]
    out = {k: (v[idx] if k != "names" else v) for k, v in ds.items()}
    names = list(out["names"])
    if variant in ("noprecip", "noweather", "nodoy", "hydnd"):
        if variant in ("nodoy", "hydnd"):
            drop = lambda n: n == "doy"  # noqa: E731
        elif variant == "noprecip":
            drop = lambda n: "precip" in n  # noqa: E731
        else:
            calendar = {"h", "hour", "dow", "doy", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek"}
            drop = lambda n: not (n in calendar or n.startswith("D_"))  # noqa: E731
        keepc = [i for i, n in enumerate(names) if not drop(n)]
        out["X"] = out["X"][:, keepc]
        out["names"] = tuple(names[i] for i in keepc)
    return out


REL = ("p_lastday", "p_lastweek", "D_mean", "D_min", "D_max", "D_mean30", "D_min7", "D_max7",
       "D_gq_evening14", "D_gq_q90_14")
SPREAD = ("D_std", "D_std7")
# desetprel: DE-LU price levels relative to the zone's own 7-day mean, in its scale units
DE_REL = ("de_p_lastday", "de_p_lastweek", "de_D_mean", "de_D_mean7", "de_D_gq_evening14", "de_D_max")


def relative(ds):
    """Price-level features as (x - mean7) / scale; spreads / scale. D_mean7 stays absolute."""
    X = ds["X"].copy()
    names = list(ds["names"])
    for n in REL:
        if n in names:
            i = names.index(n)
            X[:, i] = (X[:, i] - ds["mean7"]) / ds["scale"]
    for n in DE_REL:
        if n in names:
            i = names.index(n)
            X[:, i] = (X[:, i] - ds["mean7"]) / ds["scale"]
    for n in SPREAD:
        if n in names:
            i = names.index(n)
            X[:, i] = X[:, i] / ds["scale"]
    ds["X"] = X
    return ds


def main():
    zone, variant = sys.argv[1], sys.argv[2]
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    t0 = time.time()
    base_variant = {"rel": "base", "hydrel": "hydnd", "core_rel": "core", "desetprel": "desetp"}.get(variant, variant)
    from core.features import FEATURE_SCHEMA

    if variant == "core_rel" and FEATURE_SCHEMA >= 2:
        sys.exit("core is relative (and has wall-clock lags) since feature schema 2: run `core`; "
                 "the core / core_rel results in ../README.md are on schema 1 (v0.1.2)")
    ds = dataset(zone, base_variant)
    if base_variant != variant:
        ds = relative(ds)
    t_build = time.time() - t0
    print(f"{zone} {variant}: rows={len(ds['y'])} features={ds['X'].shape[1]} build={t_build:.0f}s "
          f"targets {min(ds['target'])}..{max(ds['target'])}", flush=True)
    ytr = (ds["y"] - ds["mean7"]) / ds["scale"]
    preds = {k: np.full(len(ds["y"]), np.nan) for k in ("median", "p90_raw")}
    tl = np.full(len(ds["y"]), None, dtype=object)
    month = WF_FROM
    t1 = time.time()
    while month < WF_TO:
        nxt = month
        for _ in range(STEP):
            nxt = (nxt + timedelta(days=32)).replace(day=1)
        test = np.flatnonzero((ds["anchor"] >= month) & (ds["anchor"] < min(nxt, WF_TO)) & (ds["target"] < CUTOFF))
        if len(test):
            train = np.flatnonzero(ds["target"] < month)
            for k, q in QUANTS:
                m = fit(ds["X"][train], ytr[train], q, max_iter)
                preds[k][test] = m.predict(ds["X"][test]) * ds["scale"][test] + ds["mean7"][test]
            tl[test] = max(ds["target"][train])
            print(f"  {month:%Y-%m}: train={len(train)} test={len(test)} iters={m.n_iter_} "
                  f"{time.time() - t1:.0f}s", flush=True)
        month = nxt
    sel = ~np.isnan(preds["median"])
    res = {k: ds[k][sel] for k in ("times", "h", "anchor", "target", "y", "naive")}
    res.update({k: v[sel] for k, v in preds.items()})
    res["train_last_target"] = tl[sel]
    res["meta"] = {"zone": zone, "variant": variant, "max_iter": max_iter, "rows": int(len(ds["y"])),
                   "features": int(ds["X"].shape[1]), "names": ds["names"], "build_s": t_build, "refit_months": STEP,
                   "fit_s": time.time() - t1}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"pred_{zone}_{variant}_{max_iter}{TAG}.pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"done {zone} {variant} total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
