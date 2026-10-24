from dataclasses import replace
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


# ---------------------------------------------------------------- composite models (DK1)


def _dk1_inputs(case):
    """DK1 prices (DE-LU's, shifted) and the golden weather plus constant series for the
    points the DE-LU case does not have."""
    from core.zones import DK1

    dk_prices = Hourly(case.prices.start, case.prices.values * 0.8 + 3.0)
    filler = Hourly(case.prices.start, np.full(len(case.prices.values) + 24 * 8, 7.0))
    keys = [f"{p}__{v}" for points, vs in DK1.point_sets() for p in points for v in vs]
    weather = {
        h: {k: series.get(k, filler) for k in keys} | series for h, series in case.weather.items()
    }
    return dk_prices, weather


def test_composite_forecast_is_the_blend_of_both_models(golden, tiny_dk1_model_file):
    from core.zones import DK1

    case = golden[date(2025, 6, 15)]
    dk_prices, weather = _dk1_inputs(case)
    m = tiny_dk1_model_file
    slots = run_forecast(
        m, case.anchor, dk_prices, weather, DK1, extra_prices={"DE-LU": case.prices}
    )
    own = run_forecast(
        replace(m, parent=None), case.anchor, dk_prices, weather, DK1,
        extra_prices={"DE-LU": case.prices},
    )  # fmt: skip
    parent = run_forecast(m.parent, case.anchor, case.prices, weather, DE_LU)
    assert [(s.time, s.h) for s in slots] == [(s.time, s.h) for s in own]
    assert [(s.time, s.h) for s in parent] == [(s.time, s.h) for s in own]
    for name in ("median", "p90_raw", "p10_raw"):
        got = np.array([getattr(s, name) for s in slots])
        expected = 0.5 * np.array([getattr(s, name) for s in own]) + 0.5 * np.array(
            [getattr(s, name) for s in parent]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        assert not np.allclose(got, [getattr(s, name) for s in own])  # the parent counts
    assert all(s.valid for s in slots)


def test_composite_slot_is_invalid_when_the_parent_slot_is(golden, tiny_dk1_model_file):
    from core.zones import DK1

    case = golden[date(2025, 6, 15)]
    dk_prices, weather = _dk1_inputs(case)
    target = run_forecast(
        tiny_dk1_model_file, case.anchor, dk_prices, weather, DK1,
        extra_prices={"DE-LU": case.prices},
    )[7]  # fmt: skip
    # blank that hour at every DE-LU wind point that is not a DK1 wind point
    weather = {h: dict(series) for h, series in weather.items()}
    for p in set(DE_LU.wind) - set(DK1.wind):
        key = f"{p}__wind_speed_100m"
        s = weather[target.h][key]
        values = s.values.copy()
        values[int((target.time - s.start) / timedelta(hours=1))] = np.nan
        weather[target.h][key] = Hourly(s.start, values)
    own = run_forecast(
        replace(tiny_dk1_model_file, parent=None), case.anchor, dk_prices, weather, DK1,
        extra_prices={"DE-LU": case.prices},
    )  # fmt: skip
    assert all(s.valid for s in own)
    slots = run_forecast(
        tiny_dk1_model_file, case.anchor, dk_prices, weather, DK1,
        extra_prices={"DE-LU": case.prices},
    )  # fmt: skip
    assert [s.valid for s in slots] == [(s.time, s.h) != (target.time, target.h) for s in slots]


@pytest.mark.parametrize("extra", [None, {}, {"NL": None}], ids=["none", "empty", "other"])
def test_composite_without_parent_prices_raises(golden, tiny_dk1_model_file, extra):
    from core.zones import DK1

    case = golden[date(2025, 6, 15)]
    dk_prices, weather = _dk1_inputs(case)
    with pytest.raises(InsufficientHistory, match="DE-LU"):
        run_forecast(tiny_dk1_model_file, case.anchor, dk_prices, weather, DK1, extra_prices=extra)


def test_composite_with_incomplete_parent_anchor_day_raises(golden, tiny_dk1_model_file):
    """Short DE-LU prices never give a forecast from NaN DE-LU features."""
    from core.zones import DK1

    case = golden[date(2025, 6, 15)]
    dk_prices, weather = _dk1_inputs(case)
    i = int((datetime(2025, 6, 15, 21, tzinfo=UTC) - case.prices.start) / timedelta(hours=1))
    short = Hourly(case.prices.start, case.prices.values[:i])
    with pytest.raises(IncompleteAnchorDay, match="DE-LU prices"):
        run_forecast(
            tiny_dk1_model_file, case.anchor, dk_prices, weather, DK1, extra_prices={"DE-LU": short}
        )
    with pytest.raises(InsufficientHistory):
        run_forecast(
            tiny_dk1_model_file, case.anchor, dk_prices, weather, DK1,
            extra_prices={"DE-LU": Hourly(case.prices.start, case.prices.values[i - 24 * 20 : i])},
        )  # fmt: skip


def test_blend_slots_unmatched_slot_is_invalid():
    from core.forecast import blend_slots

    own = [
        Slot(T, 1, 10.0, 20.0, 5.0, True),
        Slot(T + timedelta(hours=1), 1, 10.0, 20.0, 5.0, True),
    ]
    parent = [Slot(T, 1, 30.0, 40.0, 1.0, True), Slot(T, 2, 99.0, 99.0, 99.0, True)]
    out = blend_slots(own, parent, 0.25)
    assert out[0] == Slot(T, 1, 25.0, 35.0, 2.0, True)
    assert not out[1].valid and np.isnan(out[1].median)
