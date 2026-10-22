from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
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
)
from core.zones import DE_LU

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
    assert FEATURE_SCHEMA == 1


@pytest.mark.parametrize("anchor", GOLDEN_ANCHORS, ids=str)
def test_golden_parity(golden, anchor):
    case = golden[anchor]
    fm = build(case)
    assert [t.isoformat() for t in fm.times] == case.times
    # Known research look-ahead: when the clocks fall back between anchor and target day, the
    # target day's last hour minus h days is 00:00 local of anchor+1. Research reads that price
    # from its full series although it is not published at issue time; the price window here
    # ends with the anchor day, so the lag is NaN. Exactly these cells are exempt from parity.
    end = case.prices.start + len(case.prices.values) * timedelta(hours=1)
    lookahead = [i for i, t in enumerate(fm.times) if t - timedelta(days=int(fm.h[i])) >= end]
    assert len(lookahead) == (6 if anchor == date(2025, 10, 25) else 0)
    expected = case.X.copy()
    expected[lookahead, col("p_lastday")] = np.nan
    np.testing.assert_allclose(fm.X, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fm.scale, case.scale, rtol=1e-12)
    np.testing.assert_allclose(fm.mean7, case.mean7, rtol=1e-12)
    np.testing.assert_array_equal(fm.h, fm.X[:, col("h")])


def test_prices_after_anchor_day_are_ignored(golden):
    """Ruling R7: training passes the full price history; lags must not read past the anchor."""
    case = golden[date(2025, 10, 25)]
    extra = np.arange(48, dtype=float) + 1000.0  # two more days of (synthetic) prices
    longer = Hourly(case.prices.start, np.concatenate([case.prices.values, extra]))
    fm = build(case, prices=longer)
    end = case.prices.start + len(case.prices.values) * timedelta(hours=1)
    lookahead = [i for i, t in enumerate(fm.times) if t - timedelta(days=int(fm.h[i])) >= end]
    assert len(lookahead) == 6
    assert np.isnan(fm.X[lookahead, col("p_lastday")]).all()
    expected = case.X.copy()
    expected[lookahead, col("p_lastday")] = np.nan
    np.testing.assert_allclose(fm.X, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


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


def test_scale_floor():
    anchor = date(2025, 5, 10)
    prices = _synthetic_prices(anchor, days=45, evening=12.0, other=80.0)
    daily = daily_stats(prices, DE_LU)
    assert daily.gq_evening14[daily.dates.index(anchor)] == pytest.approx(12.0)
    fm = build_features(anchor, prices, daily, {}, {}, DE_LU)
    assert fm.scale.min() == pytest.approx(0.5)
    assert fm.scale.max() == pytest.approx(0.5)


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
