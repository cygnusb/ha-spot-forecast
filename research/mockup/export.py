import json, sys
import numpy as np, pandas as pd
D = "data/"
p = pd.read_parquet(D + "predictions_default.parquet"); p.index.name = None
q = pd.read_parquet(D + "predictions_p90.parquet"); q.index.name = None
p["time"] = p.index; q["time"] = q.index
m = p.merge(q[["time", "h", "pred"]].rename(columns={"pred": "p90"}), on=["time", "h"], how="left")
# rolling conformal offset per h: trailing 56 target days before issue date
m["res"] = m["price"] - m["p90"]
out_p90 = []
for h, g in m.groupby("h"):
    g = g.sort_values("time").copy()
    daily = g.groupby("target_date")["res"].apply(list)
    off = {}
    dates = sorted(g["issue_date"].unique())
    for d in dates:
        win = daily[(daily.index < d) & (daily.index >= d - pd.Timedelta(days=56))]
        vals = [v for l in win for v in l if pd.notna(v)]
        off[d] = np.quantile(vals, 0.9) if len(vals) > 200 else 0.0
    g["p90c"] = g["p90"] + g["issue_date"].map(off)
    out_p90.append(g)
m = pd.concat(out_p90)
grid = pd.date_range(m["time"].min(), m["time"].max(), freq="h")
actual = m.groupby("time")["price"].first().reindex(grid)
loc = grid.tz_convert("Europe/Berlin")
def arr(s): return [None if pd.isna(v) else round(float(v) / 10, 2) for v in s]
res = {"t": [x.strftime("%Y-%m-%d %H:%M") for x in loc], "actual": arr(actual), "pred": {}, "p90": {}}
for h in range(1, 7):
    g = m[m["h"] == h].groupby("time")
    res["pred"][h] = arr(g["pred"].first().reindex(grid))
    res["p90"][h] = arr(g["p90c"].first().reindex(grid))
json.dump(res, open(sys.argv[1], "w"), separators=(",", ":"))
print(len(grid), loc[0], loc[-1])
