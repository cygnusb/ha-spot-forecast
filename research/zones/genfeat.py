"""Zone-generic feature builder, mirroring core.features.build_features (feature schema 1).

Reuses the core helpers (Hourly, daily_stats, clear-sky, power curve, ingest parsers) and only
replaces the DE-specific parts: feature list from the zone config, neighbour aggregates without
the hard-coded FR/Nordic points, holiday calendar from the `holidays` package.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache, reduce
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import holidays
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "custom_components", "spot_forecast"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.features import (  # noqa: E402
    _DAILY_NAMES,
    HORIZONS,
    Hourly,
    _clear_sky_ghi,
    _day_mean,
    _wind_power,
    daily_stats,
)
from core.ingest import (  # noqa: E402
    hourly_prices,
    merge_hourly,
    parse_energy_charts,
    weather_by_horizon_from_previous_runs,
)

from zonecfg import NEW_POINTS, RESEARCH_POINTS, ZONES, ZoneCfg, point_vars, price_dir, weather_dir  # noqa: E402

_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)


def coords(p: str) -> tuple[float, float]:
    return RESEARCH_POINTS[p][0] if p in RESEARCH_POINTS else NEW_POINTS[p][0]


# ---------------------------------------------------------------- holidays
@lru_cache(maxsize=None)
def offdays(country: str, year: int) -> frozenset[date]:
    """National holidays plus de-facto Nordic off days (Christmas Eve, NYE, Midsummer Eve)."""
    h = set(holidays.country_holidays(country, years=year).keys())
    if country in ("SE", "NO", "DK", "FI"):
        h |= {date(year, 12, 24), date(year, 12, 31)}
    if country == "SE":  # Midsummer Eve: Friday 19-25 June
        h |= {date(year, 6, d) for d in range(19, 26) if date(year, 6, d).weekday() == 4}
    return frozenset(h)


# ---------------------------------------------------------------- raw data
def _files(directory, pattern):
    return sorted(f for f in directory.glob(pattern) if not f.name.startswith("."))


def load_prices(zone: str) -> Hourly:
    slots = {}
    for f in _files(price_dir(zone), "*.json"):
        for s in parse_energy_charts(json.loads(f.read_text())):
            slots.setdefault(s.start, s)
    return hourly_prices(sorted(slots.values(), key=lambda s: s.start))


def load_point(name: str) -> dict[int, dict[str, Hourly]]:
    """offset h+1 keyed by horizon h -> {"name__var": Hourly}."""
    variables = point_vars(name)
    files = [
        weather_by_horizon_from_previous_runs([json.loads(f.read_text())], [name], variables)
        for f in _files(weather_dir(name), f"{name}_*.json")
    ]
    if not files:
        raise FileNotFoundError(name)
    out = {}
    empty = Hourly(datetime(1970, 1, 1, tzinfo=UTC), np.array([]))
    for h in HORIZONS:
        out[h] = {k: reduce(merge_hourly, reversed([f[h][k] for f in files]), empty) for k in files[0][h]}
    return out


def load_weather(cfg: ZoneCfg, de_extra: bool = False) -> dict[int, dict[str, Hourly]]:
    out = {h: {} for h in HORIZONS}
    extra = ZONES["DE-LU"].points if de_extra or cfg.extra.get("raw_extra") else ()
    for p in dict.fromkeys(cfg.points + cfg.nb + extra):
        for h, series in load_point(p).items():
            out[h].update(series)
    return out


# ---------------------------------------------------------------- features
DE_PRICE = ("de_p_lastday", "de_p_lastweek", "de_D_mean", "de_D_mean7", "de_D_gq_evening14", "de_D_max")
DE_EXTRA = ("de_wind_power_idx", "de_solar_est_mean", "de_temp_mean", "de_wind_power_idx_day",
            "de_solar_est_mean_day", "de_temp_mean_day")


def feature_names(cfg: ZoneCfg, hydro_extra: bool = False, de_extra: bool = False) -> tuple[str, ...]:
    raw = [f"{p}__{v}" for p in cfg.points + cfg.extra.get("raw_extra", ()) for v in point_vars(p)]
    agg = ["wind_mean", "wind_power_idx", "temp_mean"]
    day = ["wind_power_idx", "temp_mean"]
    if cfg.solar:
        agg += ["solar_est_mean", "cloud_mean"]
        day += ["solar_est_mean"]
    if [p for p in cfg.solar if "shortwave_radiation" in point_vars(p)]:
        agg += ["solar_mean"]
        day += ["solar_mean"]
    if cfg.hydro:
        agg += ["precip_hydro"]
    agg += ["nb_wind_power_idx", "nb_temp_mean"]
    day += ["nb_wind_power_idx", "nb_temp_mean"]
    if cfg.nb_hydro:
        agg += ["nb_precip_hydro"]
    names = ["h", "hour", "dow", "doy", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek",
             *(f"D_{n}" for n in _DAILY_NAMES), *raw, *agg, *(f"{c}_day" for c in day)]
    if cfg.hydro:
        names.append("precip_hydro_day")
    if cfg.nb_hydro:
        names.append("nb_precip_hydro_day")
    if hydro_extra:
        names += ["hyd_precip_past14", "hyd_precip_past30", "hyd_precip_past90", "hyd_temp_past14",
                  "hyd_temp_past30", "hyd_precip_fc6"]
    if de_extra:
        names += list(DE_EXTRA)
    if cfg.extra.get("de_price"):
        names += list(DE_PRICE)
    return tuple(names)


def _weather_features(w, times, cfg: ZoneCfg, de_extra: bool = False) -> dict:
    f = {f"{p}__{v}": w(f"{p}__{v}") for p in cfg.points + cfg.extra.get("raw_extra", ()) for v in point_vars(p)}

    def avg(fn, pts):
        return sum(fn(p) for p in pts) / len(pts)

    def est(p):
        return _clear_sky_ghi(times, *coords(p)) * (1 - 0.75 * (w(f"{p}__cloud_cover") / 100) ** 3.4)

    f["wind_mean"] = avg(lambda p: w(f"{p}__wind_speed_100m") / 3.6, cfg.wind)
    f["wind_power_idx"] = avg(lambda p: _wind_power(w(f"{p}__wind_speed_100m")), cfg.wind)
    f["temp_mean"] = avg(lambda p: w(f"{p}__temperature_2m"), cfg.points)
    if cfg.solar:
        f["solar_est_mean"] = avg(est, cfg.solar)
        f["cloud_mean"] = avg(lambda p: w(f"{p}__cloud_cover"),
                              [p for p in cfg.points if "cloud_cover" in point_vars(p)])
        rad = [p for p in cfg.solar if "shortwave_radiation" in point_vars(p)]
        if rad:
            f["solar_mean"] = avg(lambda p: w(f"{p}__shortwave_radiation"), rad)
    if cfg.hydro:
        f["precip_hydro"] = sum(w(f"{p}__precipitation") for p in cfg.hydro)
    f["nb_wind_power_idx"] = avg(lambda p: _wind_power(w(f"{p}__wind_speed_100m")), cfg.nb_wind)
    f["nb_temp_mean"] = avg(lambda p: w(f"{p}__temperature_2m"), cfg.nb)
    if cfg.nb_hydro:
        f["nb_precip_hydro"] = sum(w(f"{p}__precipitation") for p in cfg.nb_hydro)
    if de_extra:
        de = ZONES["DE-LU"]
        f["de_wind_power_idx"] = avg(lambda p: _wind_power(w(f"{p}__wind_speed_100m")), de.wind)
        f["de_solar_est_mean"] = avg(est, de.solar)
        f["de_temp_mean"] = avg(lambda p: w(f"{p}__temperature_2m"), de.points)
    n = len(times)
    for c in ("de_wind_power_idx", "de_solar_est_mean", "de_temp_mean", "wind_power_idx", "solar_mean", "solar_est_mean", "temp_mean", "nb_wind_power_idx", "nb_temp_mean"):
        if c in f:
            f[f"{c}_day"] = np.full(n, _day_mean(f[c]))
    for c in ("precip_hydro", "nb_precip_hydro"):
        if c in f:
            f[f"{c}_day"] = np.full(n, np.nansum(f[c]))
    return f


def _hydro_extra(weather, anchor_end_utc, cfg, n, fc_sum) -> dict:
    """Trailing near-actual precip/temperature from offset-2 forecasts up to the anchor end."""
    pts = tuple(dict.fromkeys(cfg.hydro + cfg.nb_hydro))
    s1 = weather[1]  # horizon 1 = offset 2: issued >= 2 days before its valid time
    out = {}
    for days in (14, 30, 90):
        first = anchor_end_utc - days * _DAY
        tot = sum(np.nansum(s1[f"{p}__precipitation"].window(first, days * 24)) for p in pts)
        out[f"hyd_precip_past{days}"] = tot
    for days in (14, 30):
        first = anchor_end_utc - days * _DAY
        vals = [np.nanmean(s1[f"{p}__temperature_2m"].window(first, days * 24)) for p in pts]
        out[f"hyd_temp_past{days}"] = float(np.mean(vals))
    out["hyd_precip_fc6"] = fc_sum
    return {k: np.full(n, v) for k, v in out.items()}


@dataclass
class Rows:
    X: np.ndarray
    times: list
    h: np.ndarray
    mean7: float
    scale: float


def build_features(anchor, prices, daily, weather, cfg: ZoneCfg, names, hydro_extra=False, de=None):
    a = daily.index(anchor)
    if a is None or np.isnan(daily.mean30[a]) or np.isnan(daily.gq_evening14[a]):
        return None
    d_features = {f"D_{n}": getattr(daily, n)[a] for n in _DAILY_NAMES}
    scale = max(daily.gq_evening14[a], 50.0) / 100.0
    tz = ZoneInfo(cfg.tz)
    known_until = datetime.combine(anchor + _DAY, time(), tz).astimezone(UTC)

    def lag(first, n):
        out = prices.window(first, n)
        out[max((known_until - first) // _HOUR, 0):] = np.nan
        return out

    blocks, times, hs = [], [], []
    per_h = []
    fc_sum = 0.0
    for h in HORIZONS:
        target = anchor + timedelta(days=h)
        first = datetime.combine(target, time(), tz).astimezone(UTC)
        n = (datetime.combine(target + _DAY, time(), tz).astimezone(UTC) - first) // _HOUR
        day_times = [first + i * _HOUR for i in range(n)]
        series = weather.get(h, {})

        def w(key, series=series, first=first, n=n):
            s = series.get(key)
            return np.full(n, np.nan) if s is None else s.window(first, n)

        hol = target in offdays(cfg.country, target.year)
        dow = target.weekday()
        cols = {
            "h": h,
            "hour": np.array([t.astimezone(tz).hour for t in day_times], dtype=float),
            "dow": dow,
            "doy": target.timetuple().tm_yday,
            "holiday": float(hol),
            "offday": float(hol or dow >= 5),
            "issue_dow": anchor.weekday(),
            "p_lastday": lag(first - h * _DAY, n),
            "p_lastweek": lag(first - 7 * _DAY, n),
            **d_features,
            **_weather_features(w, day_times, cfg, "de_wind_power_idx" in names),
        }
        if de is not None:
            dp, dd = de
            ai = dd.index(anchor)
            dl = dp.window(first - h * _DAY, n); dl[max((known_until - (first - h * _DAY)) // _HOUR, 0):] = np.nan
            dw = dp.window(first - 7 * _DAY, n); dw[max((known_until - (first - 7 * _DAY)) // _HOUR, 0):] = np.nan
            cols.update({"de_p_lastday": dl, "de_p_lastweek": dw,
                         "de_D_mean": dd.mean[ai] if ai is not None else np.nan,
                         "de_D_mean7": dd.mean7[ai] if ai is not None else np.nan,
                         "de_D_gq_evening14": dd.gq_evening14[ai] if ai is not None else np.nan,
                         "de_D_max": dd.max[ai] if ai is not None else np.nan})
        if hydro_extra:
            fc_sum += float(np.nansum(sum(w(f"{p}__precipitation") for p in dict.fromkeys(cfg.hydro + cfg.nb_hydro))))
        per_h.append((cols, n))
        times += day_times
        hs.append(np.full(n, h))
    for cols, n in per_h:
        if hydro_extra:
            cols.update(_hydro_extra(weather, known_until, cfg, n, fc_sum))
        blocks.append(np.column_stack([np.broadcast_to(np.asarray(cols[c], dtype=float), n) for c in names]))
    return Rows(np.vstack(blocks), times, np.concatenate(hs), d_features["D_mean7"], scale)


def build_dataset(cfg: ZoneCfg, prices, weather, first_anchor, last_anchor, hydro_extra=False,
                  de_extra=False):
    names = feature_names(cfg, hydro_extra, de_extra)
    col = {n: i for i, n in enumerate(names)}
    daily = daily_stats(prices, SimpleNamespace(tz=cfg.tz, evening_hours=(18, 21)))
    de = None
    if cfg.extra.get("de_price"):
        dp = load_prices("DE-LU")
        de = (dp, daily_stats(dp, SimpleNamespace(tz="Europe/Berlin", evening_hours=(18, 21))))
    out = {k: [] for k in ("X", "y", "times", "h", "anchor", "target", "mean7", "scale", "naive")}
    d = first_anchor
    while d <= last_anchor:
        r = build_features(d, prices, daily, weather, cfg, names, hydro_extra, de)
        if r is not None:
            y = np.array([prices.at(t) for t in r.times])
            naive = r.X[:, col["p_lastweek"]]
            keep = ~(np.isnan(y) | np.isnan(r.X[:, col["wind_mean"]]) | np.isnan(naive))
            n = len(y)
            vals = {
                "X": r.X, "y": y, "times": np.array(r.times, dtype=object), "h": r.h,
                "anchor": np.full(n, d, dtype=object),
                "target": np.array([d + timedelta(days=int(k)) for k in r.h], dtype=object),
                "mean7": np.full(n, r.mean7), "scale": np.full(n, r.scale), "naive": naive,
            }
            for k, v in vals.items():
                out[k].append(v[keep])
        d += _DAY
    ds = {k: np.concatenate(v) for k, v in out.items()}
    ds["names"] = names
    return ds
