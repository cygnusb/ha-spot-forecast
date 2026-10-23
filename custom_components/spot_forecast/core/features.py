"""Model features: a numpy port of ``build_samples`` in research/backtest.py.

Feature schema 2 changes two things against research ``build_samples``, both ported from
the later research scripts: the price lags read the same local wall-clock hour across a DST
switch (research/holidays_dst/build.py ``dst_lags``), and the price levels are relative to
the 7-day mean in units of the target scale (research/zones/bt.py ``relative``).

Any change here that alters a feature value breaks parity with the research backtest
and with every shipped model; ``tests/core/test_features.py`` pins the golden values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .calendar import de_holidays
from .zones import DE_LU, Zone

FEATURE_SCHEMA = 2
HORIZONS = range(1, 7)

_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_DAILY_NAMES = ("mean", "min", "max", "std", "mean7", "std7", "mean30", "min7", "max7",
                "gq_evening14", "gq_q90_14")  # fmt: skip
_DAY_MEANS = ("wind_power_idx", "solar_mean", "solar_est_mean", "temp_mean", "nb_wind_power_idx",
              "nb_temp_fr", "nb_temp_nordic")  # fmt: skip
CAPACITY_NAMES = ("cap_solar", "cap_wind_on", "cap_wind_off")
# Schema 2: price levels as (x - D_mean7) / scale, spreads as x / scale; D_mean7 stays absolute.
RELATIVE_NAMES = ("p_lastday", "p_lastweek", "D_mean", "D_min", "D_max", "D_mean30", "D_min7",
                  "D_max7", "D_gq_evening14", "D_gq_q90_14")  # fmt: skip
SPREAD_NAMES = ("D_std", "D_std7")

FEATURE_NAMES: tuple[str, ...] = (
    "h", "hour", "dow", "doy", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek",
    *(f"D_{n}" for n in _DAILY_NAMES),
    *(f"{p}__{v}" for p in DE_LU.points for v in DE_LU.weather_vars),
    "solar_est_mean", "wind_mean", "wind_power_idx", "solar_mean", "temp_mean", "cloud_mean",
    "nb_wind_power_idx", "nb_temp_fr", "nb_temp_nordic", "nb_precip_hydro", "nb_solar_fr",
    *(f"{n}_day" for n in _DAY_MEANS), "nb_precip_hydro_day",
    *CAPACITY_NAMES,
)  # fmt: skip


_RELATIVE = [FEATURE_NAMES.index(n) for n in RELATIVE_NAMES]
_SPREAD = [FEATURE_NAMES.index(n) for n in SPREAD_NAMES]


class InsufficientHistory(ValueError):
    """The anchor day lacks the 30-day price history the features need."""


class IncompleteAnchorDay(InsufficientHistory):
    """The anchor day's own prices are not all published yet."""


@dataclass(frozen=True, eq=False)
class Hourly:
    """Hourly series starting at ``start`` (UTC, on the hour); NaN marks a missing hour."""

    start: datetime
    values: np.ndarray

    def at(self, t: datetime) -> float:
        """Value of the hour starting at ``t``; NaN outside the series."""
        i, rest = divmod(t - self.start, _HOUR)
        if rest or not 0 <= i < len(self.values):
            return float("nan")
        return float(self.values[i])

    def window(self, first: datetime, n: int) -> np.ndarray:
        """``n`` consecutive hours from ``first``, NaN where the series has no value."""
        out = np.full(n, np.nan)
        i, rest = divmod(first - self.start, _HOUR)
        if rest:
            return out
        lo, hi = max(i, 0), min(i + n, len(self.values))
        if lo < hi:
            out[lo - i : hi - i] = self.values[lo:hi]
        return out


@dataclass(frozen=True, eq=False)
class DailyStats:
    """Per local date: price statistics and their rolling windows (NaN where undefined)."""

    dates: list[date]
    mean: np.ndarray
    min: np.ndarray
    max: np.ndarray
    std: np.ndarray
    mean7: np.ndarray
    std7: np.ndarray
    mean30: np.ndarray
    min7: np.ndarray
    max7: np.ndarray
    gq_evening14: np.ndarray
    gq_q90_14: np.ndarray

    def index(self, d: date) -> int | None:
        i = (d - self.dates[0]).days if self.dates else -1
        return i if 0 <= i < len(self.dates) else None


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """Feature rows for target days anchor+1 … anchor+6, h-major then time.

    The price-level columns (``RELATIVE_NAMES``, ``SPREAD_NAMES``) are in units of
    ``scale``; ``decode`` turns a relative level back into EUR/MWh.
    """

    X: np.ndarray
    times: list[datetime]
    h: np.ndarray
    mean7: np.ndarray
    scale: np.ndarray


def _rolling(values: np.ndarray, rows: np.ndarray, window: int, fn: Callable) -> np.ndarray:
    """pandas ``rolling(window).fn()`` over the rows selected by ``rows``, scattered back.

    As in pandas, a window that is incomplete or contains NaN gives NaN.
    """
    sub = values[rows]
    rolled = np.full(len(sub), np.nan)
    for i in range(window - 1, len(sub)):
        w = sub[i - window + 1 : i + 1]
        if not np.isnan(w).any():
            rolled[i] = fn(w)
    out = np.full(len(values), np.nan)
    out[rows] = rolled
    return out


def daily_stats(prices: Hourly, zone: Zone) -> DailyStats:
    """Daily price statistics by local date, as research ``build_samples`` computes them.

    Rolling windows run over the dates that have prices (research rolls over the rows of
    its daily table), the evening median over the dates that have evening prices.
    """
    tz = ZoneInfo(zone.tz)
    n = len(prices.values)
    if n == 0:
        empty = np.array([])
        return DailyStats([], *([empty] * len(_DAILY_NAMES)))
    local = [(prices.start + i * _HOUR).astimezone(tz) for i in range(n)]
    first = local[0].date()
    dates = [first + timedelta(days=k) for k in range((local[-1].date() - first).days + 1)]
    by_day: list[list[float]] = [[] for _ in dates]
    evening_by_day: list[list[float]] = [[] for _ in dates]
    lo, hi = zone.evening_hours
    for t, v in zip(local, prices.values, strict=True):
        if np.isnan(v):
            continue
        k = (t.date() - first).days
        by_day[k].append(v)
        if lo <= t.hour <= hi:
            evening_by_day[k].append(v)

    def per_day(groups: list[list[float]], fn: Callable, min_count: int = 1) -> np.ndarray:
        return np.array([fn(np.array(g)) if len(g) >= min_count else np.nan for g in groups])

    mean = per_day(by_day, np.mean)
    std = per_day(by_day, lambda g: np.std(g, ddof=1), min_count=2)
    q90 = per_day(by_day, lambda g: np.quantile(g, 0.9))
    evening = per_day(evening_by_day, np.median)
    mn, mx = per_day(by_day, np.min), per_day(by_day, np.max)
    rows = np.array([len(g) > 0 for g in by_day])
    evening_rows = np.array([len(g) > 0 for g in evening_by_day])
    return DailyStats(
        dates=dates,
        mean=mean,
        min=mn,
        max=mx,
        std=std,
        mean7=_rolling(mean, rows, 7, np.mean),
        std7=_rolling(std, rows, 7, np.mean),
        mean30=_rolling(mean, rows, 30, np.mean),
        min7=_rolling(mn, rows, 7, np.min),
        max7=_rolling(mx, rows, 7, np.max),
        gq_evening14=_rolling(evening, evening_rows, 14, np.median),
        gq_q90_14=_rolling(q90, rows, 14, np.mean),
    )


def _clear_sky_ghi(times: list[datetime], lat: float, lon: float) -> np.ndarray:
    """Haurwitz clear-sky irradiance (W/m2) at the hour midpoint (Open-Meteo labels the end)."""
    tm = [t.astimezone(UTC) - timedelta(minutes=30) for t in times]
    doy = np.array([t.timetuple().tm_yday for t in tm])
    hour = np.array([t.hour for t in tm])
    minute = np.array([t.minute for t in tm])
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (hour + minute / 60 + lon / 15 - 12))
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    cosz = np.clip(cosz, 0, None)
    return np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-6)), 0.0)


def _wind_power(kmh: np.ndarray) -> np.ndarray:
    """Crude power curve on 100 m wind: cubic between cut-in 3 m/s and rated 12 m/s."""
    return (np.clip(kmh / 3.6, 3, 12) - 3) ** 3 / 9**3


def _day_mean(x: np.ndarray) -> float:
    """pandas groupby mean: NaN hours are skipped, an all-NaN day gives NaN."""
    ok = ~np.isnan(x)
    return float(x[ok].mean()) if ok.any() else float("nan")


def _weather_features(w: Callable[[str], np.ndarray], times: list[datetime], zone: Zone) -> dict:
    """research ``weather_features`` for the hours of one local target day."""
    f = {f"{p}__{v}": w(f"{p}__{v}") for p in zone.points for v in zone.weather_vars}

    def est(p: str) -> np.ndarray:
        return _clear_sky_ghi(times, *zone.points[p]) * (
            1 - 0.75 * (w(f"{p}__cloud_cover") / 100) ** 3.4
        )

    def avg(fn: Callable[[str], np.ndarray], points: tuple[str, ...] | dict) -> np.ndarray:
        # sum() then divide, in the research order, so values match to the last bit
        return sum(fn(p) for p in points) / len(points)

    f["solar_est_mean"] = avg(est, zone.solar)
    f["wind_mean"] = avg(lambda p: w(f"{p}__wind_speed_100m") / 3.6, zone.wind)
    f["wind_power_idx"] = avg(lambda p: _wind_power(w(f"{p}__wind_speed_100m")), zone.wind)
    f["solar_mean"] = avg(lambda p: w(f"{p}__shortwave_radiation"), zone.solar)
    f["temp_mean"] = avg(lambda p: w(f"{p}__temperature_2m"), zone.points)
    f["cloud_mean"] = avg(lambda p: w(f"{p}__cloud_cover"), zone.points)
    f["nb_wind_power_idx"] = avg(lambda p: _wind_power(w(f"{p}__wind_speed_100m")), zone.nb_wind)
    f["nb_temp_fr"] = (w("nb_fr_north__temperature_2m") + w("nb_fr_south__temperature_2m")) / 2
    f["nb_temp_nordic"] = avg(lambda p: w(f"{p}__temperature_2m"), zone.nb_nordic)
    f["nb_precip_hydro"] = sum(w(f"{p}__precipitation") for p in zone.nb_hydro)
    f["nb_solar_fr"] = (
        w("nb_fr_north__shortwave_radiation") + w("nb_fr_south__shortwave_radiation")
    ) / 2
    n = len(times)
    for c in _DAY_MEANS:
        f[f"{c}_day"] = np.full(n, _day_mean(f[c]))
    # pandas groupby sum skips NaN hours, so an all-NaN day sums to 0 (as in research)
    f["nb_precip_hydro_day"] = np.full(n, np.nansum(f["nb_precip_hydro"]))
    return f


def _wall_clock_index(lag_day: date, hours: np.ndarray, tz: ZoneInfo) -> tuple[datetime, int, list]:
    """Where each local ``hours`` value falls on ``lag_day``: the day's first hour (UTC), its
    length in hours and, per value, the index of that wall-clock hour in the day.

    As research ``dst_lags``: an hour the day has twice (clocks going back) is its first
    occurrence (fold=0), an hour it does not have (clocks going forward) the hour before.
    """
    first = datetime.combine(lag_day, time(), tz).astimezone(UTC)
    n = (datetime.combine(lag_day + _DAY, time(), tz).astimezone(UTC) - first) // _HOUR
    position: dict[int, int] = {}
    for i in range(n):
        position.setdefault((first + i * _HOUR).astimezone(tz).hour, i)
    index = [position[hr] if hr in position else position[hr - 1] for hr in hours.astype(int)]
    return first, n, index


def build_features(
    anchor: date,
    prices: Hourly,
    daily: DailyStats,
    weather: Mapping[int, Mapping[str, Hourly]],
    capacity: Mapping[str, float],
    zone: Zone,
) -> FeatureMatrix:
    """Feature rows for every hour of the local days anchor+1 … anchor+6.

    ``prices`` must reach the end of the anchor day; prices after it are ignored, since they
    are not published at issue time (the price lags read NaN there instead). The lags take
    the same local wall-clock hour on the anchor day (``p_lastday``) and a week before the
    target day (``p_lastweek``). ``weather[h]`` holds the ``"{point}__{var}"`` series for
    target day anchor+h. Missing inputs give NaN features.
    """
    a = daily.index(anchor)
    if a is None or np.isnan(daily.mean30[a]) or np.isnan(daily.gq_evening14[a]):
        raise InsufficientHistory(f"less than 30 days of prices before the end of {anchor}")
    d_features = {f"D_{n}": getattr(daily, n)[a] for n in _DAILY_NAMES}
    scale = max(daily.gq_evening14[a], zone.scale_floor) / zone.scale_divisor
    tz = ZoneInfo(zone.tz)
    known_until = datetime.combine(anchor + _DAY, time(), tz).astimezone(UTC)

    def lag(day: date, hours: np.ndarray) -> np.ndarray:
        """Prices at the local ``hours`` of ``day``; hours after the anchor day read NaN."""
        first, n, index = _wall_clock_index(day, hours, tz)
        out = prices.window(first, n)
        out[max((known_until - first) // _HOUR, 0) :] = np.nan
        return out[index]

    blocks, times, hs = [], [], []
    for h in HORIZONS:
        target = anchor + timedelta(days=h)
        first = datetime.combine(target, time(), tz).astimezone(UTC)
        n = (datetime.combine(target + _DAY, time(), tz).astimezone(UTC) - first) // _HOUR
        day_times = [first + i * _HOUR for i in range(n)]
        series = weather.get(h, {})

        def w(key: str, series=series, first=first, n=n) -> np.ndarray:
            s = series.get(key)
            return np.full(n, np.nan) if s is None else s.window(first, n)

        holiday = target in de_holidays(target.year)
        dow = target.weekday()
        hour = np.array([t.astimezone(tz).hour for t in day_times], dtype=float)
        cols = {
            "h": h,
            "hour": hour,
            "dow": dow,
            "doy": target.timetuple().tm_yday,
            "holiday": float(holiday),
            "offday": float(holiday or dow >= 5),
            "issue_dow": anchor.weekday(),
            "p_lastday": lag(target - h * _DAY, hour),
            "p_lastweek": lag(target - 7 * _DAY, hour),
            **d_features,
            **_weather_features(w, day_times, zone),
            **{c: capacity.get(c, np.nan) for c in CAPACITY_NAMES},
        }
        blocks.append(
            np.column_stack(
                [np.broadcast_to(np.asarray(cols[c], dtype=float), n) for c in FEATURE_NAMES]
            )
        )
        times += day_times
        hs.append(np.full(n, h))

    X = np.vstack(blocks)
    # in the order of research ``relative``: subtract, then divide
    X[:, _RELATIVE] = (X[:, _RELATIVE] - d_features["D_mean7"]) / scale
    X[:, _SPREAD] = X[:, _SPREAD] / scale
    return FeatureMatrix(
        X=X,
        times=times,
        h=np.concatenate(hs),
        mean7=np.full(len(X), d_features["D_mean7"]),
        scale=np.full(len(X), scale),
    )


def decode(y: np.ndarray, fm: FeatureMatrix) -> np.ndarray:
    """Model output or a relative price level (scaled deviation from the 7-day mean) back to
    EUR/MWh."""
    return y * fm.scale + fm.mean7
