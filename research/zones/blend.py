"""Own-zone model vs. DE-LU forecast reused vs. 50/50 blend, per horizon (ct/kWh)."""
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
    res = {}
    for name, f in (("own", own), ("DE", reu), ("blend", (own + reu) / 2)):
        res[name] = np.mean([np.mean(np.abs(f - y)[h == k]) for k in range(1, 7)]) / 10
    print(f"{z}:{v} own {res['own']:.2f}  DE-reuse {res['DE']:.2f}  blend {res['blend']:.2f}  bias own {np.mean(own-y)/10:+.2f} DE {np.mean(reu-y)/10:+.2f}")
