import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zonecfg import ZONES, weather_dir, price_dir, point_vars, NEW_POINTS
out = {}
for z, c in ZONES.items():
    pts = list(dict.fromkeys(c.points + c.nb))
    size = sum(f.stat().st_size for p in pts for f in weather_dir(p).glob(f"{p}_*.json"))
    cols = sum(len(point_vars(p)) * 6 for p in pts)
    # Open-Meteo weight per day of history (vars/10 per point, days/14)
    w_day = sum(max(len(point_vars(p)) * 6 / 10, 1) for p in pts) / 14
    psize = sum(f.stat().st_size for f in price_dir(z).glob("*.json"))
    new = sum(p in NEW_POINTS for p in pts)
    out[z] = dict(points=len(c.points), nb=len(c.nb), new=new, weather_mb=size / 1e6, price_mb=psize / 1e6,
                  weather_cols=cols, om_weight_per_year=w_day * 365)
    print(z, {k: round(v, 1) if isinstance(v, float) else v for k, v in out[z].items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zones", "out", "volume.json"), "w"), indent=1)
