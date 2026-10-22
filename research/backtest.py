"""Walk-forward backtest of a multi-day DE-LU day-ahead price forecast.

Forecast setup (no look-ahead):
  - Issue time: 06:00 Europe/Berlin on day D. Known: prices up to end of D
    (auction on D-1), weather model runs from D-1 or earlier.
  - Targets: every hour of days D+1 .. D+6 (horizon h = 1..6).
  - Weather for D+h comes from Open-Meteo `previous_day{h+1}`, i.e. a run at least
    (h+1)*24h older than the target hour -> always available at issue time.
  - Model is refit monthly on all samples whose target day lies before the test month.

Usage: python backtest.py [--test-start 2025-04-01] [--loss absolute_error] [--drop SUBSTR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fetch import NEIGHBOR_POINTS, NEIGHBOR_VARS, NEIGHBOR_ZONES, OFFSETS, POINTS, RAW, WEATHER_VARS

TZ = "Europe/Berlin"
HORIZONS = range(1, 7)
WIND_POINTS = ["north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt"]
SOLAR_POINTS = ["sachsen_anhalt", "nrw", "hessen", "bawue", "bayern", "uckermark"]
NB_WIND = ["nb_nl", "nb_be", "nb_dk_west", "nb_pl_north", "nb_se_south"]
NB_HYDRO = ["nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_north"]
NB_NORDIC = ["nb_no_south", "nb_se_north", "nb_fi"]
OUT = Path(__file__).parent / "data"
# Open-Meteo previous-runs radiation at lead offsets >= 6 days is wrong before this date
# (shifted/scaled by 0.5-1.6x; ECMWF switches to coarser output steps at 144 h). Masked.
RADIATION_BROKEN = {6: "2025-07-01", 7: "2025-07-01"}


# ---------------------------------------------------------------- loading


def load_prices(zone: str = "DE-LU") -> pd.Series:
    parts = []
    for f in sorted((RAW / "prices" / zone).glob("*.json")):
        d = json.loads(f.read_text())
        parts.append(pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()].dropna()
    # 15-min products since 2025-10-01: use hourly means throughout.
    return s.resample("h").mean().dropna()


def load_weather() -> dict[int, pd.DataFrame]:
    """offset -> DataFrame(index=UTC hour, columns=f'{point}__{var}')"""
    frames: dict[int, list[pd.DataFrame]] = {o: [] for o in OFFSETS}
    point_vars = {**{p: WEATHER_VARS for p in POINTS}, **{p: NEIGHBOR_VARS for p in NEIGHBOR_POINTS}}
    for name, variables in point_vars.items():
        per_point = []
        for f in sorted((RAW / "weather").glob(f"{name}_*.json")):
            h = json.loads(f.read_text())["hourly"]
            per_point.append(pd.DataFrame(h).assign(time=lambda x: pd.to_datetime(x["time"], utc=True)))
        df = pd.concat(per_point).drop_duplicates("time").set_index("time").sort_index()
        for o, until in RADIATION_BROKEN.items():
            df.loc[df.index < pd.Timestamp(until, tz="UTC"), f"shortwave_radiation_previous_day{o}"] = np.nan
        for o in OFFSETS:
            cols = {f"{v}_previous_day{o}": f"{name}__{v}" for v in variables}
            frames[o].append(df[list(cols)].rename(columns=cols))
    return {o: pd.concat(fs, axis=1) for o, fs in frames.items()}


def load_public_power(country: str = "de") -> pd.DataFrame:
    """Actual hourly load and generation (MW). Realised values - never a feature for the target hour."""
    parts = []
    for f in sorted((RAW / "public_power" / country).glob("*.json")):
        d = json.loads(f.read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        parts.append(pd.DataFrame({p["name"]: p["data"] for p in d["production_types"]}, index=idx))
    g = pd.concat(parts).sort_index()
    g = g[~g.index.duplicated()].resample("h").mean()
    return pd.DataFrame(
        {
            "load": g["Load"],
            "wind": g.get("Wind onshore", 0) + g.get("Wind offshore", 0),
            "solar": g.get("Solar", 0),
            "residual": g["Residual load"],
            "cross_border": g["Cross border electricity trading"],
        }
    )


def load_installed() -> pd.DataFrame:
    d = json.loads((RAW / "installed_power.json").read_text())
    idx = pd.PeriodIndex([pd.Period(f"{t[3:]}-{t[:2]}", "M") for t in d["time"]])
    cols = {p["name"]: p["data"] for p in d["production_types"]}
    df = pd.DataFrame(cols, index=idx)
    return pd.DataFrame(
        {
            "cap_solar": df.get("Solar AC", df.get("Solar DC")),
            "cap_wind_on": df["Wind onshore"],
            "cap_wind_off": df["Wind offshore"],
        },
        index=idx,
    ).ffill()


# ---------------------------------------------------------------- features


def clear_sky_ghi(t: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Haurwitz clear-sky irradiance (W/m2), averaged-hour midpoint (Open-Meteo labels the end)."""
    tm = t - pd.Timedelta(minutes=30)
    doy = tm.dayofyear.values
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (tm.hour.values + tm.minute.values / 60 + lon / 15 - 12))
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    cosz = np.clip(cosz, 0, None)
    with np.errstate(divide="ignore"):
        return np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-6)), 0.0)


def weather_features(w: pd.DataFrame) -> pd.DataFrame:
    f = w.copy()
    # cloud-based irradiance estimate: always available, unlike the masked radiation
    est = lambda p: clear_sky_ghi(w.index, *POINTS[p]) * (1 - 0.75 * (w[f"{p}__cloud_cover"] / 100) ** 3.4)  # noqa: E731
    f["solar_est_mean"] = sum(est(p) for p in SOLAR_POINTS) / len(SOLAR_POINTS)
    ws = lambda p: w[f"{p}__wind_speed_100m"] / 3.6  # noqa: E731  m/s
    # crude power curve: cubic between cut-in 3 m/s and rated 12 m/s
    pw = lambda v: (np.clip(v, 3, 12) - 3) ** 3 / 9**3  # noqa: E731
    f["wind_mean"] = sum(ws(p) for p in WIND_POINTS) / len(WIND_POINTS)
    f["wind_power_idx"] = sum(pw(ws(p)) for p in WIND_POINTS) / len(WIND_POINTS)
    f["solar_mean"] = sum(w[f"{p}__shortwave_radiation"] for p in SOLAR_POINTS) / len(SOLAR_POINTS)
    f["temp_mean"] = sum(w[f"{p}__temperature_2m"] for p in POINTS) / len(POINTS)
    f["cloud_mean"] = sum(w[f"{p}__cloud_cover"] for p in POINTS) / len(POINTS)
    # neighbours (prefix nb_ so they can be ablated with --drop nb_)
    f["nb_wind_power_idx"] = sum(pw(ws(p)) for p in NB_WIND) / len(NB_WIND)
    f["nb_temp_fr"] = (w["nb_fr_north__temperature_2m"] + w["nb_fr_south__temperature_2m"]) / 2
    f["nb_temp_nordic"] = sum(w[f"{p}__temperature_2m"] for p in NB_NORDIC) / len(NB_NORDIC)
    f["nb_precip_hydro"] = sum(w[f"{p}__precipitation"] for p in NB_HYDRO)
    f["nb_solar_fr"] = (w["nb_fr_north__shortwave_radiation"] + w["nb_fr_south__shortwave_radiation"]) / 2
    # daily aggregates over the local target day
    day = f.index.tz_convert(TZ).date
    for c in ["wind_power_idx", "solar_mean", "solar_est_mean", "temp_mean", "nb_wind_power_idx", "nb_temp_fr",
              "nb_temp_nordic"]:
        f[f"{c}_day"] = f.groupby(day)[c].transform("mean")
    f["nb_precip_hydro_day"] = f.groupby(day)["nb_precip_hydro"].transform("sum")
    # neighbour raw point columns add nothing over the aggregates (ablation 2026-10-21)
    return f.drop(columns=[c for c in w.columns if c.startswith("nb_")])


def load_gas() -> pd.Series:
    """Daily TTF close (EUR/MWh) by trading date; research only, see fetch_gas()."""
    r = json.loads((RAW / "gas_ttf.json").read_text())["chart"]["result"][0]
    tz = r["meta"]["exchangeTimezoneName"]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).date
    s = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(idx), dtype=float)
    return s[~s.index.duplicated(keep="last")].dropna()


def neighbor_price_daily(nb_prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Per local date: daily mean and 7-day mean of each neighbour zone (prefix nbp_)."""
    cols = {}
    for zone, s in nb_prices.items():
        daily = s.groupby(s.index.tz_convert(TZ).date).mean()
        daily.index = pd.to_datetime(daily.index)
        cols[f"nbp_{zone}_Dmean"] = daily
        cols[f"nbp_{zone}_mean7"] = daily.rolling(7).mean()
    return pd.DataFrame(cols)


def build_samples(
    prices: pd.Series,
    weather: dict[int, pd.DataFrame],
    cap: pd.DataFrame,
    nb_prices: dict[str, pd.Series],
    gas: pd.Series | None = None,
) -> pd.DataFrame:
    local = prices.index.tz_convert(TZ)
    pdf = pd.DataFrame({"price": prices.values, "date": local.date}, index=prices.index)
    daily = pdf.groupby("date")["price"].agg(["mean", "min", "max", "std"])
    daily.index = pd.to_datetime(daily.index)
    daily["mean7"] = daily["mean"].rolling(7).mean()
    daily["std7"] = pdf.groupby("date")["price"].std().rolling(7).mean().values
    daily["mean30"] = daily["mean"].rolling(30).mean()
    daily["min7"] = daily["min"].rolling(7).min()
    daily["max7"] = daily["max"].rolling(7).max()
    # Key-free proxy for the fuel-driven price level: recent evening prices, when gas plants set the price.
    hour = local.hour
    evening = pdf[(hour >= 18) & (hour <= 21)].groupby("date")["price"].median()
    evening.index = pd.to_datetime(evening.index)
    daily["gq_evening14"] = evening.rolling(14).median()
    q = pdf.groupby("date")["price"].quantile(0.9)
    q.index = pd.to_datetime(q.index)
    daily["gq_q90_14"] = q.rolling(14).mean()
    daily = daily.add_prefix("D_")
    nb_daily = neighbor_price_daily(nb_prices)

    de_holidays = holidays.Germany(years=range(2023, 2028))
    out = []
    for h in HORIZONS:
        w = weather_features(weather[h + 1])
        t = w.index
        loc = t.tz_convert(TZ)
        target_date = pd.to_datetime(loc.date)
        issue_date = target_date - pd.Timedelta(days=h)
        s = pd.DataFrame(index=t)
        s["h"] = h
        s["target_date"] = target_date
        s["issue_date"] = issue_date
        s["hour"] = loc.hour
        s["dow"] = loc.dayofweek
        s["doy"] = loc.dayofyear
        s["holiday"] = [d in de_holidays for d in loc.date]
        s["offday"] = s["holiday"] | (s["dow"] >= 5)
        s["issue_dow"] = issue_date.dayofweek
        # price lags known at issue time
        lag_d = t - pd.Timedelta(days=h)  # same hour on day D
        lag_w = t - pd.Timedelta(days=7)  # same hour one week before target (<= D since h <= 6)
        s["p_lastday"] = prices.reindex(lag_d).values
        s["p_lastweek"] = prices.reindex(lag_w).values
        for zone, zs in nb_prices.items():
            s[f"nbp_{zone}_lastday"] = zs.reindex(lag_d).values
        s = s.join(daily, on="issue_date")
        s = s.join(nb_daily, on="issue_date")
        if gas is not None:
            # Last settlement before the 06:00 issue time: the previous trading day's close.
            g = gas.reindex(pd.date_range(gas.index.min(), issue_date.max(), freq="D")).ffill()
            s["gas_ttf"] = g.reindex(issue_date - pd.Timedelta(days=1)).values
            s["gas_chg7"] = s["gas_ttf"] - g.reindex(issue_date - pd.Timedelta(days=8)).values
        s = s.join(w)
        per = issue_date.to_period("M")
        s[cap.columns] = cap.reindex(per).values
        s["price"] = prices.reindex(t).values
        out.append(s)
    df = pd.concat(out)
    return df.dropna(subset=["price", "D_mean30", "wind_mean", "p_lastweek"])


# ---------------------------------------------------------------- model / eval

NON_FEATURES = {"price", "target_date", "issue_date", "y", "_scale"}


def walk_forward(
    df: pd.DataFrame,
    test_start: str,
    loss: str,
    drop: list[str] | None = None,
    quantile: float | None = None,
    min_leaf: int = 40,
    target: str = "diff",
) -> pd.DataFrame:
    df = df.copy()
    # "z": deviation scaled by recent volatility, so peak height can grow with the regime
    # "gas": deviation relative to the gas price, so spikes can scale with fuel cost beyond the training range
    if target == "z":
        scale = df["D_std7"].clip(lower=5)
    elif target == "gas":
        scale = df["gas_ttf"].clip(lower=15) / 40
    elif target == "gq":
        # key-free alternative: scale by the recent evening price level (gas plants usually set it)
        scale = df["D_gq_evening14"].clip(lower=50) / 100
    else:
        scale = 1.0
    df["_scale"] = scale
    df["y"] = (df["price"] - df["D_mean7"]) / df["_scale"]
    features = [c for c in df.columns if c not in NON_FEATURES and not any(d in c for d in drop or [])]
    preds = []
    months = pd.period_range(test_start, df["issue_date"].max(), freq="M")
    for m in months:
        m_start = m.start_time
        train = df[df["target_date"] < m_start]
        test = df[(df["issue_date"] >= m_start) & (df["issue_date"] <= m.end_time)]
        if test.empty:
            continue
        model = HistGradientBoostingRegressor(
            loss="quantile" if quantile else loss,
            quantile=quantile,
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            min_samples_leaf=min_leaf,
            l2_regularization=1.0,
            categorical_features=["dow", "issue_dow"],
            random_state=0,
        )
        model.fit(train[features], train["y"])
        p = test[["h", "target_date", "issue_date", "price", "p_lastday", "p_lastweek"]].copy()
        s = test["_scale"].values
        p["pred"] = model.predict(test[features]) * s + test["D_mean7"].values
        preds.append(p)
        print(f"  {m}: train={len(train):>7} test={len(test):>6}")
    return pd.concat(preds)


def top_k_metrics(p: pd.DataFrame, col: str, k: int = 4) -> pd.Series:
    """Per (issue, target) day: hit rate of the k cheapest / dearest hours and the
    regret (ct/kWh) of picking the forecast's k cheapest hours instead of the true ones."""

    def per_day(g: pd.DataFrame) -> pd.Series:
        true_lo = set(g.nsmallest(k, "price").index)
        pred_lo = g.nsmallest(k, col)
        true_hi = set(g.nlargest(k, "price").index)
        pred_hi = g.nlargest(k, col)
        return pd.Series(
            {
                "hit_cheap": len(true_lo & set(pred_lo.index)) / k,
                "hit_dear": len(true_hi & set(pred_hi.index)) / k,
                "regret_cheap": pred_lo["price"].mean() - g.nsmallest(k, "price")["price"].mean(),
                "regret_dear": g.nlargest(k, "price")["price"].mean() - pred_hi["price"].mean(),
            }
        )

    g = p.reset_index(drop=True).groupby(["h", "issue_date"])
    return g.apply(per_day).groupby("h").mean()


def report(p: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, col in [("model", "pred"), ("naive_lastday", "p_lastday"), ("naive_lastweek", "p_lastweek")]:
        mae = (p[col] - p["price"]).abs().groupby(p["h"]).mean() / 10  # EUR/MWh -> ct/kWh
        tk = top_k_metrics(p.dropna(subset=[col]), col)
        rows[(name, "MAE ct/kWh")] = mae
        rows[(name, "hit cheap4")] = tk["hit_cheap"]
        rows[(name, "hit dear4")] = tk["hit_dear"]
        rows[(name, "regret cheap4 ct")] = tk["regret_cheap"] / 10
        rows[(name, "regret dear4 ct")] = tk["regret_dear"] / 10
        peak = p["price"] > p["price"].quantile(0.97)
        q = p[peak]
        rows[(name, "peak MAE ct")] = (q[col] - q["price"]).abs().groupby(q["h"]).mean() / 10
        rows[(name, "peak bias ct")] = (q[col] - q["price"]).groupby(q["h"]).mean() / 10
    r = pd.DataFrame(rows)
    r.loc["all"] = r.mean()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-04-01")
    ap.add_argument("--loss", default="absolute_error")
    # neighbour prices (nbp_) did not improve MAE in the ablation, so they are off by default
    ap.add_argument("--drop", nargs="*", default=["nbp_", "gas_", "gq_"], help="drop feature columns containing these substrings")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quantile", type=float, default=None, help="fit a quantile model instead (e.g. 0.9)")
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--target", choices=["diff", "z", "gas", "gq"], default="diff")
    ap.add_argument("--oracle-countries", nargs="*", default=[], help="with --oracle: also neighbours' actuals")
    ap.add_argument("--oracle", action="store_true", help="DIAGNOSTIC ONLY: add actual load/generation of the target hour")
    args = ap.parse_args()

    prices = load_prices()
    weather = load_weather()
    cap = load_installed()
    nb_prices = {z: load_prices(z) for z in NEIGHBOR_ZONES}
    df = build_samples(prices, weather, cap, nb_prices, load_gas())
    if args.oracle:
        df = df.join(load_public_power().add_prefix("oracle_"))
        for c in args.oracle_countries:
            nb = load_public_power(c)[["load", "residual", "cross_border"]]
            df = df.join(nb.add_prefix(f"oracle_{c}_"))
    print(f"samples: {len(df)}  issue days {df.issue_date.min().date()}..{df.issue_date.max().date()}")
    p = walk_forward(df, args.test_start, args.loss, args.drop, args.quantile, args.min_leaf, args.target)
    tag = args.tag or args.loss
    p.to_parquet(OUT / f"predictions_{tag}.parquet")
    pd.set_option("display.width", 200)
    r = report(p)
    print(r.round(2).to_string())
    r.round(3).to_csv(OUT / f"report_{tag}.csv")
    month = p["target_date"].dt.to_period("M")
    by_month = pd.DataFrame(
        {
            "model": (p["pred"] - p["price"]).abs().groupby(month).mean() / 10,
            "naive_lastweek": (p["p_lastweek"] - p["price"]).abs().groupby(month).mean() / 10,
            "price_mean": p["price"].groupby(month).mean() / 10,
            "price_std": p["price"].groupby(month).std() / 10,
        }
    )
    print(by_month.round(2).to_string())


if __name__ == "__main__":
    main()
