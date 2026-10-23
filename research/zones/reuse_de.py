"""MAE of the DE-LU (generic, base) forecast used as-is for another zone, same rows."""
import os, sys, pickle, glob
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zones", "out")
from datetime import date
de = pickle.load(open(f"{OUT}/pred_DE-LU_base_300.pkl", "rb"))
key = lambda r: {(t, int(h)): i for i, (t, h) in enumerate(zip(r["times"], r["h"]))}
kde = key(de)
for z in sys.argv[1:]:
    r = pickle.load(open(f"{OUT}/pred_{z}_base_300.pkl", "rb"))
    idx = [(i, kde.get((t, int(h)))) for i, (t, h) in enumerate(zip(r["times"], r["h"])) if r["anchor"][i] >= date(2025, 10, 1)]
    idx = [(i, j) for i, j in idx if j is not None]
    a, b = np.array([i for i, _ in idx]), np.array([j for _, j in idx])
    own = np.mean(np.abs(r["median"][a] - r["y"][a])); reuse = np.mean(np.abs(de["median"][b] - r["y"][a]))
    print(f"{z}: own model {own/10:.2f} ct, DE-LU forecast reused {reuse/10:.2f} ct, naive {np.mean(np.abs(r['naive'][a]-r['y'][a]))/10:.2f} (rows {len(a)})")
