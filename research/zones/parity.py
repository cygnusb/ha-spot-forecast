import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genfeat
import numpy as np
from datetime import date
from zonecfg import ZONES, RESEARCH_RAW
from training.dataset import load_raw, build_dataset as tb
from core.features import FEATURE_NAMES
t=time.time()
raw = load_raw(RESEARCH_RAW)
core = tb(raw, date(2025, 3, 1), date(2025, 3, 10))
cfg = ZONES["DE-LU"]
prices = genfeat.load_prices("DE-LU"); weather = genfeat.load_weather(cfg)
g = genfeat.build_dataset(cfg, prices, weather, date(2025, 3, 1), date(2025, 3, 10))
print("rows", len(core.y_price), len(g["y"]), time.time()-t)
gn = {n: i for i, n in enumerate(g["names"])}
bad = []
for i, n in enumerate(FEATURE_NAMES):
    if n in gn:
        a, b = core.X[:, i], g["X"][:, gn[n]]
        if not np.allclose(a, b, equal_nan=True): bad.append(n)
print("shared", len(set(FEATURE_NAMES) & set(gn)), "mismatch", bad)
print("only core", sorted(set(FEATURE_NAMES) - set(gn)))
print("only gen", sorted(set(gn) - set(FEATURE_NAMES)))
print(np.allclose(core.y_price, g["y"]), np.allclose(core.scale, g["scale"]))
