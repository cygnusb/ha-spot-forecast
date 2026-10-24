"""DK1/DK2: own model vs. own model with DE-LU inputs vs. the DE-LU forecast reused, and blends.

Same rows (target hour and horizon present in every run), issue days from 2025-10-01, MAE per
horizon and over all horizons in ct/kWh. The DE-LU forecast is the production setting
(`core_rel`, monthly refit, 600 iterations).

python -I dk_options.py
"""

import os
import pickle
from datetime import date

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zones", "out")


def load(zone, variant):
    path = f"{OUT}/pred_{zone}_{variant}_600.pkl"
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def keyed(r):
    return {(t, int(h)): i for i, (t, h) in enumerate(zip(r["times"], r["h"]))}


de = load("DE-LU", "core_rel")
kde = keyed(de)
for zone in ("DK1", "DK2"):
    runs = {v: r for v in ("rel", "desetprel") if (r := load(zone, v)) is not None}
    idx = {v: keyed(r) for v, r in runs.items()}
    base, kb = runs["rel"], idx["rel"]
    keys = [k for k in kb if k in kde and all(k in i for i in idx.values())
            and base["anchor"][kb[k]] >= date(2025, 10, 1)]
    y = np.array([base["y"][kb[k]] for k in keys])
    h = np.array([k[1] for k in keys])
    f = {v: np.array([r["median"][idx[v][k]] for k in keys]) for v, r in runs.items()}
    f["DE"] = np.array([de["median"][kde[k]] for k in keys])
    for v in list(runs):
        f[f"{v}+DE"] = (f[v] + f["DE"]) / 2
    print(f"{zone}: rows {len(keys)}")
    print("  variant          " + " ".join(f"D+{k:<4}" for k in range(1, 7)) + "  all    bias")
    for name, p in f.items():
        per_h = [np.mean(np.abs(p - y)[h == k]) / 10 for k in range(1, 7)]
        print(f"  {name:<16} " + " ".join(f"{m:.2f} " for m in per_h)
              + f"  {np.mean(per_h):.2f}  {np.mean(p - y) / 10:+.2f}")
