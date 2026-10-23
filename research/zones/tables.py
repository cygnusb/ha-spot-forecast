"""Markdown tables for the report from out/metrics.json (+ DE-LU blend)."""
import os, sys, json, pickle
import numpy as np
from datetime import date
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zones", "out")
M = {(m["zone"], m["variant"], m["max_iter"]): m for m in json.load(open(f"{OUT}/metrics.json"))}
de = pickle.load(open(f"{OUT}/pred_DE-LU_base_300.pkl", "rb"))
kde = {(t, int(h)): i for i, (t, h) in enumerate(zip(de["times"], de["h"]))}


def blend(z, v):
    r = pickle.load(open(f"{OUT}/pred_{z}_{v}_300.pkl", "rb"))
    pairs = [(i, kde.get((t, int(h)))) for i, (t, h) in enumerate(zip(r["times"], r["h"])) if r["anchor"][i] >= date(2025, 10, 1)]
    a = np.array([i for i, j in pairs if j is not None]); b = np.array([j for i, j in pairs if j is not None])
    y, f, h = r["y"][a], (r["median"][a] + de["median"][b]) / 2, r["h"][a]
    reuse = np.mean([np.mean(np.abs(de["median"][b] - y)[h == k]) for k in range(1, 7)])
    return np.mean([np.mean(np.abs(f - y)[h == k]) for k in range(1, 7)]) / 10, reuse / 10


ct = lambda v: f"{v / 10:.2f}"  # noqa: E731
print("| zone | variant | MAE D+1 | D+3 | D+6 | all | naive wk | naive D | skill | mean abs price | MAE / abs price | bias | DE-LU fc reused | 50/50 blend |")
print("|" + "---|" * 14)
for spec in sys.argv[1:]:
    z, v = spec.split(":")[:2]
    it = int(spec.split(":")[2]) if spec.count(":") == 2 else 300
    m = M.get((z, v, it))
    if m is None:
        print(f"| {z} | {v} | (missing) |"); continue
    b, reuse = blend(z, v) if z != "DE-LU" and it == 300 else (float("nan"), float("nan"))
    print(f"| {z} | {v}{'' if it == 300 else ' (monthly, 600 it)'} | {ct(m['mae_h']['1'])} | {ct(m['mae_h']['3'])} | {ct(m['mae_h']['6'])} "
          f"| **{ct(m['mae'])}** | {ct(m['naive'])} | {ct(m['naive_d'])} | {m['skill']:.2f} | {ct(m['mean_abs_price'])} "
          f"| {m['mae'] / m['mean_abs_price']:.2f} | {m['bias'] / 10:+.2f} | {'' if np.isnan(reuse) else f'{reuse:.2f}'} | {'' if np.isnan(b) else f'{b:.2f}'} |")
