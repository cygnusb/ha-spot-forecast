"""Price coupling between zones over the evaluation window (hours, EUR/MWh)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genfeat, numpy as np
from datetime import datetime, UTC
from zonecfg import ZONES
first = datetime(2025, 10, 2, tzinfo=UTC); n = (datetime(2026, 10, 21, tzinfo=UTC) - first).days * 24
P = {z: genfeat.load_prices(z).window(first, n) for z in ZONES}
pairs = {"NL": "DE-LU", "DK1": "DE-LU", "DK2": "DK1", "SE4": "DK2", "NO2": "DK1", "SE3": "SE4", "NO1": "SE3",
         "FI": "SE3", "SE1": "SE2", "SE2": "SE3", "NO3": "SE2", "NO4": "SE1", "NO5": "NO1"}
out = {}
for z, p in P.items():
    d = P["DE-LU"]
    row = {"eq_DE": float(np.mean(np.abs(p - d) < 0.5)), "corr_DE": float(np.corrcoef(p, d)[0, 1]),
           "daily_std_mean": float(np.mean([np.std(p[i:i + 24]) for i in range(0, n, 24)]))}
    if z in pairs:
        q = P[pairs[z]]; row["nb"] = pairs[z]; row["eq_nb"] = float(np.mean(np.abs(p - q) < 0.5))
    out[z] = row
    print(z, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zones", "out", "coupling.json"), "w"), indent=1)
