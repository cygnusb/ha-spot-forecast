from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from core.features import Hourly, InsufficientHistory, build_features, daily_stats, decode
from core.forecast import (
    IncompleteAnchorDay,
    Slot,
    conformal_offset,
    low_price,
    peak_risk,
    run_forecast,
)
from core.zones import DE_LU

T = datetime(2026, 1, 1, tzinfo=UTC)


def test_run_forecast_matches_manual(golden, tiny_model_file):
    m = tiny_model_file
    case = golden[date(2025, 6, 15)]
    slots = run_forecast(m, case.anchor, case.prices, case.weather, DE_LU, case.capacity)
    assert len(slots) == len(case.times) and {s.h for s in slots} == {1, 2, 3, 4, 5, 6}
    fm = build_features(
        case.anchor,
        case.prices,
        daily_stats(case.prices, DE_LU),
        case.weather,
        case.capacity,
        DE_LU,
    )
    assert [s.time for s in slots] == fm.times
    for name, got in (("median", "median"), ("p90", "p90_raw"), ("p10", "p10_raw")):
        expected = decode(getattr(m, name).predict(fm.X), fm)
        np.testing.assert_allclose([getattr(s, got) for s in slots], expected)


def test_run_forecast_capacity_defaults_to_model(golden, tiny_model_file):
    case = golden[date(2025, 6, 15)]
    a = run_forecast(tiny_model_file, case.anchor, case.prices, case.weather, DE_LU)
    b = run_forecast(
        tiny_model_file, case.anchor, case.prices, case.weather, DE_LU, tiny_model_file.capacity
    )
    assert a == b


def test_conformal_offset_uses_default_below_200():
    assert conformal_offset(np.ones(199), default=7.0) == 7.0


def test_conformal_offset_quantile():
    r = np.arange(1000, dtype=float)
    assert conformal_offset(r, default=0) == pytest.approx(np.quantile(r, 0.9))


def test_conformal_offset_low_quantile():
    r = np.arange(1000, dtype=float)
    assert conformal_offset(r, default=0, q=0.1) == pytest.approx(np.quantile(r, 0.1))


def test_conformal_offset_ignores_nan():
    r = np.concatenate([np.arange(200, dtype=float), np.full(50, np.nan)])
    assert conformal_offset(r, default=0) == pytest.approx(np.quantile(np.arange(200), 0.9))
    assert (
        conformal_offset(np.concatenate([np.ones(150), np.full(100, np.nan)]), default=7.0) == 7.0
    )


def test_peak_risk_never_below_median():
    assert peak_risk(Slot(T, 1, median=100, p90_raw=80, p10_raw=50, valid=True), offset=5) == 100


def test_low_price_never_above_median():
    assert low_price(Slot(T, 1, median=10, p90_raw=30, p10_raw=20, valid=True), offset=-5) == 10


def test_missing_weather_hours_are_invalid(golden, tiny_model_file):
    case = golden[date(2025, 6, 15)]
    ok = run_forecast(tiny_model_file, case.anchor, case.prices, case.weather, DE_LU, case.capacity)
    assert all(s.valid for s in ok)
    target = ok[5]
    weather = {h: dict(series) for h, series in case.weather.items()}
    for p in DE_LU.wind:  # blank that hour at every wind point
        key = f"{p}__wind_speed_100m"
        s = weather[target.h][key]
        values = s.values.copy()
        values[int((target.time - s.start) / timedelta(hours=1))] = np.nan
        weather[target.h][key] = Hourly(s.start, values)
    slots = run_forecast(tiny_model_file, case.anchor, case.prices, weather, DE_LU, case.capacity)
    assert [s.valid for s in slots] == [s.time != target.time for s in slots]


def test_incomplete_anchor_day_raises(golden, tiny_model_file):
    case = golden[date(2025, 6, 15)]
    values = case.prices.values.copy()
    # last hour of the anchor day (local midnight 2025-06-15 +1 day = 2025-06-15 22:00 UTC)
    i = int((datetime(2025, 6, 15, 21, tzinfo=UTC) - case.prices.start) / timedelta(hours=1))
    values[i] = np.nan
    prices = Hourly(case.prices.start, values)
    with pytest.raises(IncompleteAnchorDay) as e:
        run_forecast(tiny_model_file, case.anchor, prices, case.weather, DE_LU, case.capacity)
    assert isinstance(e.value, InsufficientHistory)


def test_model_of_another_zone_raises(golden, tiny_model_file):
    from core.model import ModelMismatch
    from core.zones import NL

    case = golden[date(2025, 6, 15)]
    with pytest.raises(ModelMismatch, match="is for zone 'de_lu', not 'nl'"):
        run_forecast(tiny_model_file, case.anchor, case.prices, case.weather, NL)
