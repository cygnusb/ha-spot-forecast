"""Compare walk-forward predictions of the variants on identical rows.

MAE overall = mean over horizons (training/report.py convention). ct/kWh = EUR/MWh / 10.
Per-month paired deltas vs base; day-block bootstrap CIs on the affected subsets.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "data" / "holidays_dst"  # build.py output
VARIANTS = ["base", "base_seed1", "hol", "dst", "both"]
RNG = np.random.default_rng(0)


def load():
    c = np.load(HERE / "cache.npz")
    preds = {}
    for v in VARIANTS:
        f = HERE / f"pred_{v}.npz"
        if f.exists():
            preds[v] = np.load(f)
    idx = preds["base"]["idx"]
    for v, p in preds.items():
        assert np.array_equal(p["idx"], idx), v
    return c, idx, preds


def mae_h(err: np.ndarray, h: np.ndarray) -> tuple[float, dict[int, float]]:
    per = {k: float(np.mean(np.abs(err[h == k]))) for k in range(1, 7) if (h == k).any()}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def boot_delta(err_v, err_b, day, n=2000):
    """Mean |e_v| - |e_b| with 95 % CI, resampling target days (row clusters)."""
    d = np.abs(err_v) - np.abs(err_b)
    days, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, d)
    cnt = np.bincount(inv)
    stats = []
    for _ in range(n):
        k = RNG.integers(0, len(days), len(days))
        stats.append(s[k].sum() / cnt[k].sum())
    return d.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5), len(days)


def main() -> None:
    c, idx, preds = load()
    y = c["y_price"][idx]
    h = c["h"][idx]
    td = c["target_date"][idx]
    month = preds["base"]["month"]
    hol = c["hol"][idx]
    names = list(c["feature_names"])
    nat_hol = c["X"][idx, names.index("holiday")] == 1
    errs = {v: p["pred"] - y for v, p in preds.items()}

    print(f"rows={len(idx)} target days {date.fromordinal(td.min())}..{date.fromordinal(td.max())}")
    print("\n## Overall MAE (mean over horizons), EUR/MWh [ct/kWh]  + per horizon EUR/MWh")
    print("variant      all              h1     h2     h3     h4     h5     h6   runtime  n_iter")
    for v, e in errs.items():
        a, per = mae_h(e, h)
        rt = float(preds[v]["runtime"])
        print(f"{v:<11} {a:6.2f} [{a / 10:.3f}]  " + " ".join(f"{per[k]:6.2f}" for k in range(1, 7))
              + f"  {rt:6.0f}s  {list(preds[v]['n_iter'])}")

    # subsets
    lag_ld = ~np.isclose(c["X"][idx, names.index("p_lastday")], c["p_lastday_dst"][idx], equal_nan=True)
    lag_lw = ~np.isclose(c["X"][idx, names.index("p_lastweek")], c["p_lastweek_dst"][idx], equal_nan=True)
    dst_rows = lag_ld | lag_lw
    new_hol = (~nat_hol) & ((hol[:, 0] > 0) | (hol[:, 1] > 0) | (hol[:, 2] > 0) | (hol[:, 3] > 0))
    weekday = np.array([date.fromordinal(int(d)).weekday() < 5 for d in td])
    new_hol_wd = new_hol & weekday
    subsets = {
        "new-holiday days (regional/bridge/eve/LU, not national)": new_hol,
        "  of those Mon-Fri": new_hol_wd,
        "  regional share>=0.3 or bridge, Mon-Fri": weekday & ~nat_hol & ((hol[:, 0] >= 0.3) | (hol[:, 1] > 0)),
        "  Dec 24/31": hol[:, 2] > 0,
        "national holidays (reference)": nat_hol,
        "DST-affected rows (lag differs)": dst_rows,
    }
    print("\n## Affected subsets: MAE EUR/MWh per variant; delta vs base with day-bootstrap 95% CI")
    for name, m in subsets.items():
        days = sorted({date.fromordinal(int(d)) for d in td[m]})
        print(f"\n{name}: rows={m.sum()} target days={len(days)}")
        if len(days) <= 20:
            print("   days:", ", ".join(f"{d:%Y-%m-%d}" for d in days))
        for v, e in errs.items():
            a, _ = mae_h(e[m], h[m])
            pooled = float(np.mean(np.abs(e[m])))
            line = f"   {v:<11} MAE(hmean)={a:6.2f}  pooled={pooled:6.2f}"
            if v != "base":
                d, lo, hi, nd = boot_delta(e[m], errs["base"][m], td[m])
                line += f"  delta={d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            print(line)

    print("\n## Per-month MAE (pooled rows, EUR/MWh) and paired delta vs base")
    months = np.unique(month)
    hdr = "month    " + " ".join(f"{v:>11}" for v in errs)
    print(hdr)
    deltas = {v: [] for v in errs if v != "base"}
    for mo in months:
        m = month == mo
        vals = {v: float(np.mean(np.abs(e[m]))) for v, e in errs.items()}
        for v in deltas:
            deltas[v].append(vals[v] - vals["base"])
        print(f"{date.fromordinal(int(mo)):%Y-%m}  " + " ".join(f"{vals[v]:11.2f}" for v in errs))
    print("\nper-month delta vs base: mean, std, months better / total, t (mean / (std/sqrt n))")
    for v, d in deltas.items():
        d = np.array(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"   {v:<11} {d.mean():+.3f}  sd={d.std(ddof=1):.3f}  better={int((d < 0).sum())}/{len(d)}  t={t:+.2f}")

    # whole-period paired bootstrap (by target day) on all rows
    print("\n## All rows: delta vs base, day-bootstrap 95% CI (EUR/MWh)")
    for v in deltas:
        d, lo, hi, nd = boot_delta(errs[v], errs["base"], td)
        print(f"   {v:<11} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  days={nd}")


if __name__ == "__main__":
    sys.exit(main())
