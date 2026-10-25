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
    hydro_history_gaps,
)
from core.zones import DE_LU, DK1, DK2, NL, NO2, point_vars

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


# ---------------------------------------------------------------- DK1, DK2 (generic + DE-LU inputs)

COPENHAGEN = ZoneInfo("Europe/Copenhagen")
_DE_EXTRA = ("de_wind_power_idx", "de_solar_est_mean", "de_temp_mean", "de_wind_power_idx_day",
             "de_solar_est_mean_day", "de_temp_mean_day")  # fmt: skip
_DE_PRICE = ("de_p_lastday", "de_p_lastweek", "de_D_mean", "de_D_mean7", "de_D_gq_evening14",
             "de_D_max")  # fmt: skip


@pytest.mark.parametrize("zone", [DK1, DK2], ids=lambda z: z.code)
def test_dk_feature_names(zone):
    """The generic layout with the DE-LU own points' raw columns after the zone's own, then
    the DE-LU aggregates and the DE-LU price features (research/zones/bt.py "desetp")."""
    names = feature_names(zone)
    generic = feature_names(replace(zone, raw_extra=(), de_extra=False, price_inputs=()))
    own_raw = [f"{p}__{v}" for p in zone.own for v in point_vars(p)]
    de_raw = [f"{p}__{v}" for p in DE_LU.own for v in point_vars(p)]
    assert names[: 20 + len(own_raw)] == generic[: 20 + len(own_raw)]
    assert names[20 + len(own_raw) : 20 + len(own_raw) + len(de_raw)] == tuple(de_raw)
    assert names[20 + len(own_raw) + len(de_raw) :] == (
        *generic[20 + len(own_raw) :], *_DE_EXTRA, *_DE_PRICE
    )  # fmt: skip
    assert len(names) == {"DK1": 102, "DK2": 98}[zone.code]


def _shifted(prices: Hourly, factor: float, offset: float) -> Hourly:
    return Hourly(prices.start, prices.values * factor + offset)


def _dk_inputs(anchor: date):
    """DK prices and DE-LU prices that differ, both naming their hour; no weather."""
    de = _hour_index_prices(anchor, days=45)
    dk = _shifted(de, 0.5, 7.0)
    return dk, {"DE-LU": (de, daily_stats(de, DE_LU))}


def test_dk_price_inputs_are_relative_to_the_own_7_day_mean():
    anchor = date(2026, 6, 10)
    dk, extra = _dk_inputs(anchor)
    daily = daily_stats(dk, DK1)
    fm = build_features(anchor, dk, daily, {}, {}, DK1, extra)
    c = {n: i for i, n in enumerate(feature_names(DK1))}
    de, de_daily = extra["DE-LU"]
    a, i = daily.dates.index(anchor), de_daily.dates.index(anchor)
    for name in ("mean", "mean7", "gq_evening14", "max"):
        expected = (getattr(de_daily, name)[i] - fm.mean7) / fm.scale
        assert (fm.X[:, c[f"de_D_{name}"]] == expected).all(), name
    assert (fm.mean7 == daily.mean7[a]).all()  # the own zone's level, not DE-LU's
    for lag, back in (("de_p_lastday", lambda h: h), ("de_p_lastweek", lambda h: 7)):
        expected = [
            de.at(t - timedelta(days=back(int(h)))) for t, h in zip(fm.times, fm.h, strict=True)
        ]
        np.testing.assert_allclose(decode(fm.X[:, c[lag]], fm), expected, rtol=1e-12)


_DST_ANCHORS = [date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 26), date(2027, 3, 27),
                date(2027, 3, 28), date(2027, 3, 29)]  # fmt: skip


@pytest.mark.parametrize("anchor", _DST_ANCHORS, ids=str)
def test_dk_de_price_lags_equal_the_de_lu_lags_across_dst(anchor):
    """A DK row's DE-LU price lags are the lags of the DE-LU row of the same UTC hour."""
    dk, extra = _dk_inputs(anchor)
    de, de_daily = extra["DE-LU"]
    fm = build_features(anchor, dk, daily_stats(dk, DK1), {}, {}, DK1, extra)
    ref = build_features(anchor, de, de_daily, {}, {}, DE_LU)
    assert fm.times == ref.times and (fm.h == ref.h).all()
    c = {n: i for i, n in enumerate(feature_names(DK1))}
    for lag in ("p_lastday", "p_lastweek"):
        got = decode(fm.X[:, c[f"de_{lag}"]], fm)
        expected = decode(ref.X[:, col(lag)], ref)
        np.testing.assert_allclose(got, expected, rtol=1e-12, err_msg=lag)
        assert not np.isnan(got).any()


@pytest.mark.parametrize("extra", [None, {}, {"NL": None}], ids=["none", "empty", "other"])
def test_dk_without_de_lu_prices_raises(extra):
    anchor = date(2026, 6, 10)
    dk, _ = _dk_inputs(anchor)
    with pytest.raises(InsufficientHistory, match="DE-LU"):
        build_features(anchor, dk, daily_stats(dk, DK2), {}, {}, DK2, extra)


def test_dk_with_short_de_lu_prices_gives_nan_de_features():
    """As research: DE-LU statistics the anchor day lacks are NaN features, not an error
    (the forecast checks the DE-LU prices through the parent model, see core.forecast)."""
    anchor = date(2026, 6, 10)
    dk, extra = _dk_inputs(anchor)
    de, _ = extra["DE-LU"]
    short = Hourly(de.start, de.values[: 24 * 20])
    fm = build_features(anchor, dk, daily_stats(dk, DK1), {}, {}, DK1,
                        {"DE-LU": (short, daily_stats(short, DE_LU))})  # fmt: skip
    c = {n: i for i, n in enumerate(feature_names(DK1))}
    assert np.isnan(fm.X[:, [c[n] for n in _DE_PRICE]]).all()


def test_de_lu_and_nl_ignore_extra_prices(golden):
    case = golden[date(2025, 6, 15)]
    fm = build(case)
    de = {"DE-LU": (case.prices, daily_stats(case.prices, DE_LU))}
    again = build_features(
        case.anchor, case.prices, daily_stats(case.prices, DE_LU), case.weather, case.capacity,
        DE_LU, de,
    )  # fmt: skip
    np.testing.assert_array_equal(fm.X, again.X)


def test_dk_de_extra_aggregates_on_synthetic_inputs():
    anchor = date(2026, 6, 10)
    dk, extra = _dk_inputs(anchor)
    keys = [f"{p}__{v}" for points, vs in DK1.point_sets() for p in points for v in vs]
    series = Hourly(dk.start, np.full(len(dk.values) + 24 * 8, 10.0))
    weather = {h: dict.fromkeys(keys, series) for h in range(1, 7)}
    fm = build_features(anchor, dk, daily_stats(dk, DK1), weather, {}, DK1, extra)
    c = {n: i for i, n in enumerate(feature_names(DK1))}
    assert (fm.X[:, c["de_temp_mean"]] == 10.0).all()
    assert (fm.X[:, c["de_temp_mean_day"]] == 10.0).all()
    assert (fm.X[:, c["de_wind_power_idx"]] == fm.X[:, c["wind_power_idx"]]).all()
    assert (fm.X[:, c["bayern__temperature_2m"]] == 10.0).all()
    assert fm.X[fm.h == 1, c["de_solar_est_mean"]].max() > 0  # June daylight


@pytest.fixture(scope="module", params=["DK1", "DK2"])
def research_dk(request):
    """research/zones genfeat and bt with variant "desetp" of the zone, and its real inputs."""
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
    code = request.param
    de = genfeat.ZONES["DE-LU"]
    base = genfeat.ZONES[code]
    # bt.py ``dataset(zone, "desetp")``
    cfg = replace(base, extra={"raw_extra": tuple(p for p in de.points if p not in base.points),
                               "de_price": True})  # fmt: skip
    dp = genfeat.load_prices("DE-LU")
    berlin = SimpleNamespace(tz="Europe/Berlin", evening_hours=(18, 21))
    return SimpleNamespace(
        genfeat=genfeat,
        bt=bt,
        cfg=cfg,
        zone={"DK1": DK1, "DK2": DK2}[code],
        prices=genfeat.load_prices(code),
        weather=genfeat.load_weather(cfg, de_extra=True),
        de=(dp, genfeat.daily_stats(dp, berlin)),
    )


def test_dk_feature_names_match_research(research_dk):
    r = research_dk
    assert feature_names(r.zone) == r.genfeat.feature_names(r.cfg, de_extra=True)


@pytest.mark.parametrize("anchor", _FAR_FROM_DST + _NEAR_DST, ids=str)
def test_dk_feature_values_match_research(research_dk, anchor):
    """genfeat ``build_features`` (variant "desetp") + bt ``relative`` on the real data, to the
    last bit; the four lag columns only away from the DST switches (research lags are UTC)."""
    r = research_dk
    names = r.genfeat.feature_names(r.cfg, de_extra=True)
    daily = r.genfeat.daily_stats(r.prices, SimpleNamespace(tz=r.cfg.tz, evening_hours=(18, 21)))
    rows = r.genfeat.build_features(anchor, r.prices, daily, r.weather, r.cfg, names, False, r.de)
    n = len(rows.X)
    ds = {"X": rows.X, "names": names, "mean7": np.full(n, rows.mean7)}
    expected = r.bt.relative({**ds, "scale": np.full(n, rows.scale)})["X"]
    de_prices = r.de[0]
    extra = {"DE-LU": (de_prices, daily_stats(de_prices, DE_LU))}
    fm = build_features(
        anchor, r.prices, daily_stats(r.prices, r.zone), r.weather, {}, r.zone, extra
    )
    assert fm.times == rows.times
    assert (fm.mean7 == rows.mean7).all() and (fm.scale == rows.scale).all()
    cols = list(range(len(names)))
    if anchor in _NEAR_DST:
        lags = ("p_lastday", "p_lastweek", "de_p_lastday", "de_p_lastweek")
        cols = [i for i, name in enumerate(names) if name not in lags]
    np.testing.assert_array_equal(fm.X[:, cols], expected[:, cols])
    assert np.isfinite(fm.X[:, names.index("wind_mean")]).all()
    assert np.isfinite(fm.X[:, names.index("de_D_mean")]).all()


# ---------------------------------------------------------------- NO2 (generic + hydro features)

OSLO = ZoneInfo("Europe/Oslo")
_HYDRO_EXTRA = ("hyd_precip_past14", "hyd_precip_past30", "hyd_precip_past90", "hyd_temp_past14",
                "hyd_temp_past30", "hyd_precip_fc6")  # fmt: skip


def test_no2_feature_names():
    """research ``meta["names"]`` of pred_NO2_hydrel_600.pkl, written out: no ``doy``, no solar
    columns, the own hydro sums, and the six hydro features last."""
    n = ("wind_speed_100m", "temperature_2m", "precipitation")
    assert feature_names(NO2) == (
        "h", "hour", "dow", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek",
        *(f"D_{n}" for n in ("mean", "min", "max", "std", "mean7", "std7", "mean30", "min7",
                             "max7", "gq_evening14", "gq_q90_14")),
        "nb_no_south__wind_speed_100m", "nb_no_south__shortwave_radiation",
        "nb_no_south__temperature_2m", "nb_no_south__precipitation",
        *(f"{p}__{v}" for p in ("no2_kristiansand", "no2_stavanger", "no2_sirdal") for v in n),
        "wind_mean", "wind_power_idx", "temp_mean", "precip_hydro", "nb_wind_power_idx",
        "nb_temp_mean", "nb_precip_hydro",
        "wind_power_idx_day", "temp_mean_day", "nb_wind_power_idx_day", "nb_temp_mean_day",
        "precip_hydro_day", "nb_precip_hydro_day",
        *_HYDRO_EXTRA,
    )  # fmt: skip
    assert len(feature_names(NO2)) == 51


def _no2_inputs(anchor: date, precip: float = 0.5):
    """45 days of NO2 prices to the end of the anchor day; constant weather for every key, the
    hydro history with precipitation ``precip`` and temperature 4.0 from 100 days before."""
    start = datetime.combine(anchor - timedelta(days=44), datetime.min.time(), OSLO)
    start = start.astimezone(UTC)
    end = datetime.combine(anchor + timedelta(days=1), datetime.min.time(), OSLO)
    n = (end.astimezone(UTC) - start) // timedelta(hours=1)
    prices = Hourly(start, 60 + 20 * np.sin(np.arange(n) / 24 * 2 * np.pi))
    keys = [f"{p}__{v}" for points, vs in NO2.point_sets() for p in points for v in vs]
    series = Hourly(start, np.full(n + 24 * 8, 10.0))
    weather = {h: dict.fromkeys(keys, series) for h in range(1, 7)}
    hstart = start - timedelta(days=60)
    hn = n + 24 * 62
    history = {}
    for p in NO2.hydro_points:
        history[f"{p}__precipitation"] = Hourly(hstart, np.full(hn, precip))
        history[f"{p}__temperature_2m"] = Hourly(hstart, np.full(hn, 4.0))
    return prices, weather, history


def _c(name):
    return feature_names(NO2).index(name)


def test_no2_features_on_synthetic_inputs():
    anchor = date(2026, 5, 15)  # D+2 is Grunnlovsdag (Sunday 17 May)
    prices, weather, history = _no2_inputs(anchor)
    fm = build_features(anchor, prices, daily_stats(prices, NO2), weather, {}, NO2,
                        hydro_history=history)  # fmt: skip
    assert fm.X.shape == (6 * 24, 51)
    hol = {int(h): fm.X[fm.h == h, _c("holiday")][0] for h in range(1, 7)}
    assert hol == {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    assert list(fm.X[fm.h == 1, _c("hour")]) == list(range(24))  # local Oslo hours
    # own hydro: the hourly sum over the three hydro points, the day's sum of it
    assert (fm.X[:, _c("precip_hydro")] == 30.0).all()
    assert (fm.X[:, _c("precip_hydro_day")] == 720.0).all()
    assert (fm.X[:, _c("nb_precip_hydro")] == 30.0).all()
    assert (fm.X[:, _c("temp_mean")] == 10.0).all()
    # the six points' trailing sums of 0.5 per hour, real hours back from the anchor's end
    np.testing.assert_allclose(fm.X[:, _c("hyd_precip_past14")], 6 * 14 * 24 * 0.5)
    np.testing.assert_allclose(fm.X[:, _c("hyd_precip_past30")], 6 * 30 * 24 * 0.5)
    np.testing.assert_allclose(fm.X[:, _c("hyd_precip_past90")], 6 * 90 * 24 * 0.5)
    assert (fm.X[:, _c("hyd_temp_past14")] == 4.0).all()
    assert (fm.X[:, _c("hyd_temp_past30")] == 4.0).all()
    # the forecast's precipitation of the six points over all six target days, on every row
    assert (fm.X[:, _c("hyd_precip_fc6")] == 6 * 6 * 24 * 10.0).all()


def test_no2_hydro_windows_count_real_hours_across_the_fall_back():
    """Anchor 2026-10-25 (Europe/Oslo falls back that night): the windows end at the end of the
    anchor day, 23:00 UTC, and reach back 336 / 720 / 2160 real hours, not local days."""
    anchor = date(2026, 10, 25)
    prices, weather, _ = _no2_inputs(anchor)
    known_until = datetime(2026, 10, 25, 23, tzinfo=UTC)
    start = known_until - timedelta(days=100)
    hours = [start + i * timedelta(hours=1) for i in range(100 * 24)]  # up to known_until
    back = np.array([(known_until - t) / timedelta(hours=1) for t in hours])  # 2400 … 1
    history = {}
    for k, p in enumerate(NO2.hydro_points):
        # 1 inside the 14 days, 10 up to 30 days, 100 up to 90, NaN before
        precip = np.select([back <= 336, back <= 720, back <= 2160], [1, 10, 100], np.nan)
        temp = np.select([back <= 336, back <= 720], [k, 2 * k], -50.0)
        history[f"{p}__precipitation"] = Hourly(start, precip)
        history[f"{p}__temperature_2m"] = Hourly(start, temp)
    fm = build_features(anchor, prices, daily_stats(prices, NO2), weather, {}, NO2,
                        hydro_history=history)  # fmt: skip
    assert len(fm.times) == 6 * 24  # the fall-back day is the anchor, not a target
    assert fm.X[0, _c("hyd_precip_past14")] == 6 * 336
    assert fm.X[0, _c("hyd_precip_past30")] == 6 * (336 + 10 * 384)
    assert fm.X[0, _c("hyd_precip_past90")] == 6 * (336 + 10 * 384 + 100 * 1440)
    # 14 days: each point's mean is k; 30 days: (336 k + 384 * 2k) / 720
    assert fm.X[0, _c("hyd_temp_past14")] == pytest.approx(np.mean(range(6)))
    assert fm.X[0, _c("hyd_temp_past30")] == pytest.approx(np.mean(range(6)) * (336 + 768) / 720)
    assert hydro_history_gaps(history, anchor, NO2) == []
    assert hydro_history_gaps(history, anchor + timedelta(days=1), NO2) == [
        f"{p}__{v}" for p in NO2.hydro_points for v in ("precipitation", "temperature_2m")
    ]  # the history ends at 23:00 UTC: the next anchor's window misses its last day


def test_no2_hydro_history_gaps_name_the_series_with_a_hole():
    anchor = date(2026, 5, 15)
    _, _, history = _no2_inputs(anchor)
    assert hydro_history_gaps(history, anchor, NO2) == []
    s = history["no5_sogn__precipitation"]
    values = s.values.copy()
    values[len(values) - 24 * 50] = np.nan  # an hour 48 days before the anchor's end
    history["no5_sogn__precipitation"] = Hourly(s.start, values)
    del history["no2_sirdal__temperature_2m"]
    assert hydro_history_gaps(history, anchor, NO2) == [
        "no2_sirdal__temperature_2m",
        "no5_sogn__precipitation",
    ]


def test_no2_hydro_features_skip_missing_hours_like_research():
    """``nansum`` and ``nanmean`` as research: a hole lowers the sum (which is why a live run
    checks ``hydro_history_gaps`` first); a point without any temperature makes the mean NaN."""
    anchor = date(2026, 5, 15)
    prices, weather, history = _no2_inputs(anchor)
    s = history["no5_sogn__precipitation"]
    values = s.values.copy()
    values[-24 * 3] = np.nan  # inside every window
    history["no5_sogn__precipitation"] = Hourly(s.start, values)
    del history["no5_bergen__temperature_2m"]
    fm = build_features(anchor, prices, daily_stats(prices, NO2), weather, {}, NO2,
                        hydro_history=history)  # fmt: skip
    assert fm.X[0, _c("hyd_precip_past14")] == 6 * 14 * 24 * 0.5 - 0.5
    assert np.isnan(fm.X[0, _c("hyd_temp_past14")])


def test_no2_needs_the_hydro_history():
    anchor = date(2026, 5, 15)
    prices, weather, _ = _no2_inputs(anchor)
    with pytest.raises(ValueError, match="NO2 needs the hydro history"):
        build_features(anchor, prices, daily_stats(prices, NO2), weather, {}, NO2)


def test_other_zones_ignore_the_hydro_history(golden):
    case = golden[date(2025, 6, 15)]
    plain = build(case)
    with_history = build_features(
        case.anchor, case.prices, daily_stats(case.prices, DE_LU), case.weather, case.capacity,
        DE_LU, hydro_history={},
    )  # fmt: skip
    np.testing.assert_array_equal(plain.X, with_history.X)


@pytest.fixture(scope="module")
def research_no2():
    """research/zones genfeat and bt with variant "hydnd" of NO2, and its real inputs."""
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
    cfg = genfeat.ZONES["NO2"]
    return SimpleNamespace(
        genfeat=genfeat,
        bt=bt,
        cfg=cfg,
        prices=genfeat.load_prices("NO2"),
        weather=genfeat.load_weather(cfg),
    )


def test_no2_feature_names_match_research(research_no2):
    r = research_no2
    names = r.genfeat.feature_names(r.cfg, hydro_extra=True)
    assert feature_names(NO2) == tuple(n for n in names if n != "doy")


# 2024-04-20: an early anchor, whose 90-day window starts before the data (2024-03)
@pytest.mark.parametrize("anchor", [date(2024, 4, 20), *_FAR_FROM_DST, *_NEAR_DST], ids=str)
def test_no2_feature_values_match_research(research_no2, anchor):
    """genfeat ``build_features`` with ``hydro_extra``, the ``doy`` column dropped and bt
    ``relative`` (bt.py variant "hydrel"), on the real data, to the last bit; the lag columns
    only away from the DST switches (research lags are UTC)."""
    r = research_no2
    names = r.genfeat.feature_names(r.cfg, hydro_extra=True)
    daily = r.genfeat.daily_stats(r.prices, SimpleNamespace(tz=r.cfg.tz, evening_hours=(18, 21)))
    rows = r.genfeat.build_features(anchor, r.prices, daily, r.weather, r.cfg, names, True)
    keep = [i for i, name in enumerate(names) if name != "doy"]
    names = tuple(names[i] for i in keep)
    n = len(rows.X)
    ds = {"X": rows.X[:, keep], "names": names, "mean7": np.full(n, rows.mean7)}
    expected = r.bt.relative({**ds, "scale": np.full(n, rows.scale)})["X"]
    fm = build_features(
        anchor, r.prices, daily_stats(r.prices, NO2), r.weather, {}, NO2,
        hydro_history=r.weather[1],
    )  # fmt: skip
    assert fm.times == rows.times
    assert (fm.mean7 == rows.mean7).all() and (fm.scale == rows.scale).all()
    cols = list(range(len(names)))
    if anchor in _NEAR_DST:
        cols = [i for i, name in enumerate(names) if name not in ("p_lastday", "p_lastweek")]
    np.testing.assert_array_equal(fm.X[:, cols], expected[:, cols])
    assert np.isfinite(fm.X[:, names.index("wind_mean")]).all()
    assert np.isfinite(fm.X[:, [names.index(c) for c in _HYDRO_EXTRA]]).all()
