"""Diagnostics: bias-removed (oracle, per horizon) MAE, own vs DE-reuse; monthly gap."""
import os, sys, pickle
import numpy as np
from datetime import date
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zones", "out")
de = pickle.load(open(f"{OUT}/pred_DE-LU_base_300.pkl", "rb"))
kde = {(t, int(h)): i for i, (t, h) in enumerate(zip(de["times"], de["h"]))}
for arg in sys.argv[1:]:
    z, v = arg.split(":")
    r = pickle.load(open(f"{OUT}/pred_{z}_{v}_300.pkl", "rb"))
    pairs = [(i, kde.get((t, int(h)))) for i, (t, h) in enumerate(zip(r["times"], r["h"])) if r["anchor"][i] >= date(2025, 10, 1)]
    a = np.array([i for i, j in pairs if j is not None]); b = np.array([j for i, j in pairs if j is not None])
    y, own, reu, h = r["y"][a], r["median"][a], de["median"][b], r["h"][a]
    def mae(f, debias):
        out = []
        for k in range(1, 7):
            e = (f - y)[h == k]
            out.append(np.mean(np.abs(e - (np.mean(e) if debias else 0))))
        return np.mean(out) / 10
    mo = np.array([f"{t:%Y-%m}" for t in r["target"][a]])
    worse = [m for m in sorted(set(mo)) if np.mean(np.abs(own - y)[mo == m]) > np.mean(np.abs(reu - y)[mo == m])]
    print(f"{z}:{v} raw own {mae(own,0):.2f} DE {mae(reu,0):.2f} | debiased own {mae(own,1):.2f} DE {mae(reu,1):.2f} | own worse in {worse}")
