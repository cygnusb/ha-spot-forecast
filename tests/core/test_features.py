from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    Hourly,
    InsufficientHistory,
    build_features,
    daily_stats,
    decode,
    feature_names,
)
from core.zones import DE_LU, NL

GOLDEN_ANCHORS = [
    date(2025, 6, 15),
    date(2025, 10, 25),
    date(2025, 10, 26),
    date(2026, 3, 28),
    date(2026, 9, 20),
]
BERLIN = ZoneInfo("Europe/Berlin")


def build(case, prices=None, weather=None):
    prices = case.prices if prices is None else prices
    weather = case.weather if weather is None else weather
    return build_features(
        case.anchor, prices, daily_stats(prices, DE_LU), weather, case.capacity, DE_LU
    )


def col(name):
    return FEATURE_NAMES.index(name)


def replace_series(weather, h, key, values):
    """Copy of ``weather`` with the series ``key`` of horizon ``h`` replaced."""
    out = {hh: dict(series) for hh, series in weather.items()}
    out[h][key] = replace(out[h][key], values=values)
    return out


def test_feature_names_match_research(golden_feature_names):
    assert len(FEATURE_NAMES) == 82
    assert golden_feature_names == FEATURE_NAMES
    assert FEATURE_SCHEMA == 2


@pytest.mark.parametrize("anchor", GOLDEN_ANCHORS, ids=str)
def test_golden_parity(golden, anchor):
    case = golden[anchor]
    fm = build(case)
    assert [t.isoformat() for t in fm.times] == case.times
    np.testing.assert_allclose(fm.X, case.X, rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fm.scale, case.scale, rtol=1e-12)
    np.testing.assert_allclose(fm.mean7, case.mean7, rtol=1e-12)
    np.testing.assert_array_equal(fm.h, fm.X[:, col("h")])


def test_prices_after_anchor_day_are_ignored(golden):
    """Ruling R7: training passes the full price history; lags must not read past the anchor."""
    case = golden[date(2025, 10, 25)]
    extra = np.arange(48, dtype=float) + 1000.0  # two more days of (synthetic) prices
    longer = Hourly(case.prices.start, np.concatenate([case.prices.values, extra]))
    fm = build(case, prices=longer)
    np.testing.assert_allclose(fm.X, case.X, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_dst_end_target_day_has_25_rows(golden):
    fm = build(golden[date(2025, 10, 25)])  # target 2025-10-26 has 25 hours
    assert (fm.h == 1).sum() == 25
    hours = fm.X[fm.h == 1, col("hour")]
    assert list(hours[:4]) == [0, 1, 2, 2]


def test_dst_start_target_day_has_23_rows(golden):
    fm = build(golden[date(2026, 3, 28)])  # target 2026-03-29 has 23 hours
    assert (fm.h == 1).sum() == 23
    assert 2 not in fm.X[fm.h == 1, col("hour")]


def test_insufficient_history_raises(golden):
    case = golden[date(2025, 6, 15)]
    cut = 20 * 24  # the anchor day and the 19 days before it (no DST change in this window)
    short = Hourly(
        case.prices.start + timedelta(hours=len(case.prices.values) - cut),
        case.prices.values[-cut:],
    )
    with pytest.raises(InsufficientHistory):
        build(case, prices=short)


def test_insufficient_history_when_anchor_has_no_prices(golden):
    case = golden[date(2025, 6, 15)]
    with pytest.raises(InsufficientHistory):
        build_features(
            case.anchor + timedelta(days=1),
            case.prices,
            daily_stats(case.prices, DE_LU),
            case.weather,
            {},
            DE_LU,
        )


def _synthetic_prices(anchor: date, days: int, evening: float, other: float) -> Hourly:
    start = datetime(anchor.year, anchor.month, anchor.day, tzinfo=BERLIN) - timedelta(
        days=days - 1
    )
    start = start.astimezone(UTC)
    end = datetime.combine(anchor + timedelta(days=1), datetime.min.time(), BERLIN).astimezone(UTC)
    n = int((end - start) / timedelta(hours=1))
    local_hours = [(start + timedelta(hours=i)).astimezone(BERLIN).hour for i in range(n)]
    values = np.array([evening if 18 <= hr <= 21 else other for hr in local_hours], dtype=float)
    return Hourly(start, values)


RELATIVE = ("p_lastday", "p_lastweek", "D_mean", "D_min", "D_max", "D_mean30", "D_min7", "D_max7",
            "D_gq_evening14", "D_gq_q90_14")  # fmt: skip
SPREADS = ("D_std", "D_std7")


def test_scale_floor():
    anchor = date(2025, 5, 10)
    prices = _synthetic_prices(anchor, days=45, evening=12.0, other=80.0)
    daily = daily_stats(prices, DE_LU)
    assert daily.gq_evening14[daily.dates.index(anchor)] == pytest.approx(12.0)
    fm = build_features(anchor, prices, daily, {}, {}, DE_LU)
    assert fm.scale.min() == pytest.approx(0.5)
    assert fm.scale.max() == pytest.approx(0.5)
    # the relative price levels stay finite with the scale at its floor
    assert np.isfinite(fm.X[:, [col(n) for n in RELATIVE + SPREADS]]).all()
    a = daily.dates.index(anchor)
    assert fm.X[:, col("D_gq_evening14")] == pytest.approx((12.0 - daily.mean7[a]) / 0.5)


def _hour_index_prices(anchor: date, days: int) -> Hourly:
    """Prices that name their hour: the value is the hour's index in the series."""
    start = datetime.combine(anchor - timedelta(days=days), datetime.min.time(), BERLIN)
    start = start.astimezone(UTC)
    end = datetime.combine(anchor + timedelta(days=1), datetime.min.time(), BERLIN).astimezone(UTC)
    return Hourly(start, np.arange((end - start) // timedelta(hours=1), dtype=float))


def test_price_levels_are_relative_to_the_7_day_mean():
    """Schema 2: price levels as (x - D_mean7) / scale, spreads as x / scale."""
    anchor = date(2026, 6, 10)
    prices = _hour_index_prices(anchor, days=45)
    daily = daily_stats(prices, DE_LU)
    fm = build_features(anchor, prices, daily, {}, {}, DE_LU)
    a = daily.dates.index(anchor)
    mean7, scale = daily.mean7[a], max(daily.gq_evening14[a], 50.0) / 100.0
    assert (fm.mean7 == mean7).all() and (fm.scale == scale).all()
    for name in RELATIVE[2:]:
        expected = (getattr(daily, name.removeprefix("D_"))[a] - mean7) / scale
        assert (fm.X[:, col(name)] == expected).all(), name
    for name in SPREADS:
        assert (fm.X[:, col(name)] == getattr(daily, name.removeprefix("D_"))[a] / scale).all()
    assert (fm.X[:, col("D_mean7")] == mean7).all()  # the reference level stays absolute
    # the lags are prices: the same hour on the anchor day / a week before the target day
    for lag, back in (("p_lastday", lambda h: h), ("p_lastweek", lambda h: 7)):
        expected = [
            prices.at(t - timedelta(days=back(int(h)))) for t, h in zip(fm.times, fm.h, strict=True)
        ]
        np.testing.assert_allclose(decode(fm.X[:, col(lag)], fm), expected, rtol=1e-12)


def _lag_hours(fm, prices: Hourly, lag: str, h: int) -> list[str]:
    """The local wall-clock time of the hour each row's price lag was read from."""
    index = decode(fm.X[:, col(lag)], fm)[fm.h == h]
    return [
        (prices.start + timedelta(hours=round(float(i)))).astimezone(BERLIN).strftime("%d %H:%M%z")
        for i in index
    ]


def test_lags_keep_the_wall_clock_hour_when_the_clocks_go_back():
    # target 2026-10-25 has 25 hours (02:00 twice); the anchor day 10-24 has 24
    prices = _hour_index_prices(date(2026, 10, 24), days=45)
    fm = build_features(date(2026, 10, 24), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    assert (fm.h == 1).sum() == 25
    lastday = _lag_hours(fm, prices, "p_lastday", 1)
    assert lastday[:5] == [
        "24 00:00+0200", "24 01:00+0200", "24 02:00+0200", "24 02:00+0200", "24 03:00+0200"
    ]  # fmt: skip
    assert lastday[-1] == "24 23:00+0200"  # the day's last hour has a lag (no NaN)
    assert not np.isnan(fm.X[:, col("p_lastday")]).any()
    assert _lag_hours(fm, prices, "p_lastweek", 1)[-1] == "18 23:00+0200"

    # the lag day itself has 25 hours: 02:00 reads its first 02:00 (CEST)
    prices = _hour_index_prices(date(2026, 10, 25), days=45)
    fm = build_features(date(2026, 10, 25), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    lastday = _lag_hours(fm, prices, "p_lastday", 1)
    assert len(lastday) == 24
    assert lastday[1:4] == ["25 01:00+0200", "25 02:00+0200", "25 03:00+0100"]
    assert lastday[-1] == "25 23:00+0100"
    prices = _hour_index_prices(date(2026, 10, 26), days=45)
    fm = build_features(date(2026, 10, 26), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    week = _lag_hours(fm, prices, "p_lastweek", 6)  # target 11-01, lag day 10-25
    assert week[1:4] == ["25 01:00+0200", "25 02:00+0200", "25 03:00+0100"]


def test_lags_keep_the_wall_clock_hour_when_the_clocks_go_forward():
    # target 2027-03-28 has 23 hours (no 02:00)
    prices = _hour_index_prices(date(2027, 3, 27), days=45)
    fm = build_features(date(2027, 3, 27), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    lastday = _lag_hours(fm, prices, "p_lastday", 1)
    assert len(lastday) == 23
    assert lastday[:3] == ["27 00:00+0100", "27 01:00+0100", "27 03:00+0100"]
    assert lastday[-1] == "27 23:00+0100"

    # the lag day has no 02:00: the target's 02:00 reads the hour before
    prices = _hour_index_prices(date(2027, 3, 28), days=45)
    fm = build_features(date(2027, 3, 28), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    lastday = _lag_hours(fm, prices, "p_lastday", 1)
    assert lastday[1:4] == ["28 01:00+0100", "28 01:00+0100", "28 03:00+0200"]
    assert lastday[-1] == "28 23:00+0200"
    prices = _hour_index_prices(date(2027, 3, 29), days=45)
    fm = build_features(date(2027, 3, 29), prices, daily_stats(prices, DE_LU), {}, {}, DE_LU)
    week = _lag_hours(fm, prices, "p_lastweek", 6)  # target 04-04, lag day 03-28
    assert week[1:4] == ["28 01:00+0100", "28 01:00+0100", "28 03:00+0200"]


def test_missing_weather_gives_nan_not_error(golden):
    case = golden[date(2025, 10, 25)]
    weather = {
        h: {k: s for k, s in series.items() if k != "bayern__temperature_2m"}
        for h, series in case.weather.items()
    }
    fm = build(case, weather=weather)
    for name in ("bayern__temperature_2m", "temp_mean", "temp_mean_day"):
        assert np.isnan(fm.X[:, col(name)]).all()
    assert not np.isnan(fm.X[:, col("wind_mean")]).any()


def test_day_aggregates_skip_missing_hours_like_research(golden):
    """pandas groupby mean skips NaN hours; groupby sum counts an all-NaN day as 0."""
    case = golden[date(2025, 10, 25)]
    temp = case.weather[1]["bayern__temperature_2m"].values.copy()
    temp[:5] = np.nan
    weather = replace_series(case.weather, 1, "bayern__temperature_2m", temp)
    precip = case.weather[3]["nb_alps_ch__precipitation"].values.copy()
    precip[10:] = np.nan
    weather = replace_series(weather, 3, "nb_alps_ch__precipitation", precip)
    weather = replace_series(weather, 2, "nb_alps_ch__precipitation", np.full(24, np.nan))
    fm = build(case, weather=weather)

    h1 = fm.h == 1
    assert np.isnan(fm.X[h1, col("temp_mean")][:5]).all()
    assert fm.X[h1, col("temp_mean_day")] == pytest.approx(np.nanmean(fm.X[h1, col("temp_mean")]))
    h3 = fm.h == 3
    assert fm.X[h3, col("nb_precip_hydro_day")] == pytest.approx(
        np.nansum(fm.X[h3, col("nb_precip_hydro")])
    )
    h2 = fm.h == 2
    assert np.isnan(fm.X[h2, col("nb_precip_hydro")]).all()
    assert (fm.X[h2, col("nb_precip_hydro_day")] == 0.0).all()


def test_daily_rolling_windows_skip_days_without_prices(golden):
    """Research rolls over the rows of its daily table, so a day without prices is skipped."""
    case = golden[date(2025, 6, 15)]
    values = case.prices.values.copy()
    full = daily_stats(case.prices, DE_LU)
    gap = full.dates.index(case.anchor) - 3
    gap_start = int(
        (
            datetime.combine(full.dates[gap], datetime.min.time(), BERLIN).astimezone(UTC)
            - case.prices.start
        )
        / timedelta(hours=1)
    )
    values[gap_start : gap_start + 24] = np.nan
    daily = daily_stats(Hourly(case.prices.start, values), DE_LU)
    assert daily.dates == full.dates
    assert np.isnan(daily.mean[gap]) and np.isnan(daily.mean7[gap])
    a = daily.dates.index(case.anchor)
    kept = [i for i in range(a + 1) if i != gap][-7:]
    assert daily.mean7[a] == pytest.approx(np.mean(full.mean[kept]), rel=1e-12)


def test_hourly_at():
    s = Hourly(datetime(2025, 1, 1, tzinfo=UTC), np.array([1.0, 2.0, 3.0]))
    assert s.at(datetime(2025, 1, 1, 2, tzinfo=UTC)) == 3.0
    assert np.isnan(s.at(datetime(2025, 1, 1, 3, tzinfo=UTC)))
    assert np.isnan(s.at(datetime(2024, 12, 31, 23, tzinfo=UTC)))


def test_decode_inverts_target(golden):
    case = golden[date(2025, 6, 15)]
    fm = build(case)
    y = np.linspace(-1, 1, len(fm.times))
    np.testing.assert_allclose(decode(y, fm), y * case.scale + case.mean7)


# ---------------------------------------------------------------- NL (generic layout)

RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)
AMSTERDAM = ZoneInfo("Europe/Amsterdam")


def test_de_lu_keeps_its_feature_names():
    assert feature_names(DE_LU) is FEATURE_NAMES


def test_nl_feature_names():
    """research/zones/genfeat.py ``feature_names(ZONES["NL"])``, written out."""
    names = feature_names(NL)
    assert names[:20] == (
        "h", "hour", "dow", "doy", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek",
        *(f"D_{n}" for n in ("mean", "min", "max", "std", "mean7", "std7", "mean30", "min7",
                             "max7", "gq_evening14", "gq_q90_14")),
    )  # fmt: skip
    nb = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
    s = ("wind_speed_100m", "temperature_2m", "cloud_cover")
    de = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
    raw = [
        *(f"nb_nl__{v}" for v in nb),
        *(f"nl_borssele__{v}" for v in s),
        *(f"north_sea__{v}" for v in de),
        *(f"nl_{p}__{v}" for p in ("flevoland", "groningen", "randstad", "brabant") for v in s),
    ]
    assert names[20:] == (
        *raw,
        "wind_mean", "wind_power_idx", "temp_mean", "solar_est_mean", "cloud_mean",
        "nb_wind_power_idx", "nb_temp_mean", "nb_precip_hydro",
        "wind_power_idx_day", "temp_mean_day", "solar_est_mean_day", "nb_wind_power_idx_day",
        "nb_temp_mean_day", "nb_precip_hydro_day",
    )  # fmt: skip
    assert len(names) == 57 and not any(n.startswith("cap_") for n in names)


def _nl_inputs(anchor: date) -> tuple[Hourly, dict]:
    """45 days of NL prices to the end of the anchor day, constant weather for every key."""
    start = datetime.combine(anchor - timedelta(days=44), datetime.min.time(), AMSTERDAM)
    start = start.astimezone(UTC)
    end = datetime.combine(anchor + timedelta(days=1), datetime.min.time(), AMSTERDAM)
    n = (end.astimezone(UTC) - start) // timedelta(hours=1)
    prices = Hourly(start, 60 + 20 * np.sin(np.arange(n) / 24 * 2 * np.pi))
    keys = [f"{p}__{v}" for points, vs in NL.point_sets() for p in points for v in vs]
    series = Hourly(start, np.full(n + 24 * 8, 10.0))
    return prices, {h: dict.fromkeys(keys, series) for h in range(1, 7)}


def test_nl_features_on_synthetic_inputs():
    anchor = date(2025, 4, 24)  # D+2 is Koningsdag 2025 (Saturday 26 April)
    prices, weather = _nl_inputs(anchor)
    fm = build_features(anchor, prices, daily_stats(prices, NL), weather, {}, NL)
    names = feature_names(NL)
    c = {n: i for i, n in enumerate(names)}
    assert fm.X.shape == (6 * 24, len(names))
    hol = {int(h): fm.X[fm.h == h, c["holiday"]][0] for h in range(1, 7)}
    assert hol == {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    # local Amsterdam hours, and the generic aggregates of constant inputs
    assert list(fm.X[fm.h == 1, c["hour"]]) == list(range(24))
    assert (fm.X[:, c["temp_mean"]] == 10.0).all() and (fm.X[:, c["nb_temp_mean"]] == 10.0).all()
    assert (fm.X[:, c["nb_precip_hydro_day"]] == 240.0).all()
    np.testing.assert_allclose(fm.X[:, c["wind_mean"]], 10.0 / 3.6)
    # schema 2 relative levels on NL rows too
    lag = [prices.at(t - timedelta(days=7)) for t in fm.times]
    np.testing.assert_allclose(decode(fm.X[:, c["p_lastweek"]], fm), lag, rtol=1e-12)
    assert (fm.X[:, c["D_mean7"]] == fm.mean7).all()


@pytest.fixture(scope="module")
def research_nl():
    """research/zones genfeat and bt, the NL config and its real inputs."""
    if not (RESEARCH / "data" / "zones" / "weather").exists():
        pytest.skip("research zone data missing")
    pytest.importorskip("holidays")
    pytest.importorskip("sklearn")
    saved = list(sys.path)
    sys.path.insert(0, str(RESEARCH / "zones"))
    try:
        import bt
        import genfeat
    finally:
        sys.path[:] = saved
    cfg = genfeat.ZONES["NL"]
    prices = genfeat.load_prices("NL")
    return SimpleNamespace(
        genfeat=genfeat, bt=bt, cfg=cfg, prices=prices, weather=genfeat.load_weather(cfg)
    )


def test_nl_feature_names_match_research(research_nl):
    assert feature_names(NL) == research_nl.genfeat.feature_names(research_nl.cfg)


# away from the DST switches the research UTC lags equal the wall-clock lags
_FAR_FROM_DST = [date(2025, 4, 24), date(2025, 6, 15), date(2025, 12, 20), date(2026, 2, 1),
                 date(2026, 8, 20)]  # fmt: skip
_NEAR_DST = [date(2025, 10, 25), date(2026, 3, 27)]


@pytest.mark.parametrize("anchor", _FAR_FROM_DST + _NEAR_DST, ids=str)
def test_nl_feature_values_match_research(research_nl, anchor):
    """genfeat ``build_features`` + bt ``relative`` on the real data, to the last bit; the lag
    columns only on anchors whose lags do not cross a DST switch (research lags are UTC)."""
    r = research_nl
    names = r.genfeat.feature_names(r.cfg)
    daily = r.genfeat.daily_stats(r.prices, SimpleNamespace(tz=r.cfg.tz, evening_hours=(18, 21)))
    rows = r.genfeat.build_features(anchor, r.prices, daily, r.weather, r.cfg, names)
    n = len(rows.X)
    ds = {"X": rows.X, "names": names, "mean7": np.full(n, rows.mean7)}
    expected = r.bt.relative({**ds, "scale": np.full(n, rows.scale)})["X"]
    fm = build_features(anchor, r.prices, daily_stats(r.prices, NL), r.weather, {}, NL)
    assert fm.times == rows.times
    assert (fm.mean7 == rows.mean7).all() and (fm.scale == rows.scale).all()
    cols = list(range(len(names)))
    if anchor in _NEAR_DST:
        cols = [i for i, name in enumerate(names) if name not in ("p_lastday", "p_lastweek")]
    np.testing.assert_array_equal(fm.X[:, cols], expected[:, cols])
    assert np.isfinite(fm.X[:, names.index("wind_mean")]).all()
