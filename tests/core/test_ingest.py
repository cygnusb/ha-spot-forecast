from datetime import UTC, datetime, timedelta
import os
from pathlib import Path

import numpy as np
import pytest

from core.ingest import (
    MarketSlot,
    hourly_prices,
    merge_hourly,
    parse_energy_charts,
    parse_open_meteo,
    weather_by_horizon_from_previous_runs,
)

T0 = datetime(2026, 10, 22, tzinfo=UTC)
RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)


def _doc(prices, step=900, start=T0):
    t = int(start.timestamp())
    return {
        "unix_seconds": [t + i * step for i in range(len(prices))],
        "price": prices,
        "unit": "EUR / MWh",
    }


def test_energy_charts_quarter_hours_to_hourly_mean():
    slots = parse_energy_charts(_doc([10, 20, 30, 40, 5, 5, 5, 5]))
    assert slots[0] == MarketSlot(T0, T0 + timedelta(minutes=15), 10.0)
    assert slots[-1].end == T0 + timedelta(hours=2)
    h = hourly_prices(slots)
    assert h.start == T0
    np.testing.assert_allclose(h.values, [25, 5])


def test_energy_charts_skips_none():
    slots = parse_energy_charts(_doc([10, None, 30, 40]))
    assert [s.price for s in slots] == [10, 30, 40]
    assert slots[0].end == T0 + timedelta(minutes=15)  # None slot does not extend the first
    assert hourly_prices(slots).values[0] == pytest.approx(80 / 3)


def test_hourly_prices_gap_is_nan():
    slots = parse_energy_charts({**_doc([1, 2, 3], step=3600), "price": [1, None, 3]})
    np.testing.assert_array_equal(hourly_prices(slots).values, [1, np.nan, 3])


def test_merge_hourly_b_wins_on_overlap():
    a = hourly_prices(parse_energy_charts(_doc([1, 2, 3], step=3600)))
    b = hourly_prices(
        parse_energy_charts(_doc([20, 30, 40], step=3600, start=T0 + timedelta(hours=1)))
    )
    m = merge_hourly(a, b)
    assert m.start == T0
    np.testing.assert_array_equal(m.values, [1, 20, 30, 40])


def _meteo(var, base, n=4, suffix=""):
    times = [(T0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]
    return {"hourly": {"time": times, f"{var}{suffix}": [base + i for i in range(n)]}}


def test_parse_open_meteo_multi_location_keys():
    docs = [_meteo("wind_speed_100m", 10), _meteo("wind_speed_100m", 100)]
    out = parse_open_meteo(docs, ["a", "b"], ["wind_speed_100m"])
    assert set(out) == {"a__wind_speed_100m", "b__wind_speed_100m"}
    assert out["a__wind_speed_100m"].start == T0
    np.testing.assert_array_equal(out["b__wind_speed_100m"].values, [100, 101, 102, 103])


def test_parse_open_meteo_suffix_and_none():
    doc = _meteo("cloud_cover", 1, suffix="_previous_day3")
    doc["hourly"]["cloud_cover_previous_day3"][1] = None
    out = parse_open_meteo([doc], ["a"], ["cloud_cover"], "_previous_day3")
    np.testing.assert_array_equal(out["a__cloud_cover"].values, [1, np.nan, 3, 4])


def _previous_runs(start):
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(48)]
    hourly = {"time": times}
    for o in range(2, 8):
        hourly[f"shortwave_radiation_previous_day{o}"] = [1.0] * 48
        hourly[f"cloud_cover_previous_day{o}"] = [2.0] * 48
    return {"hourly": hourly}


def test_previous_runs_mask_radiation_before_2025_07():
    # the window straddles the cutoff: first 24 h on 2025-06-30, last 24 h on 2025-07-01
    doc = _previous_runs(datetime(2025, 6, 30, tzinfo=UTC))
    out = weather_by_horizon_from_previous_runs(
        [doc], ["p"], ["shortwave_radiation", "cloud_cover"]
    )
    assert sorted(out) == [1, 2, 3, 4, 5, 6]
    for h in (1, 2, 3, 4):  # offsets 2..5 are fine
        assert not np.isnan(out[h]["p__shortwave_radiation"].values).any()
    for h in (5, 6):  # offsets 6, 7 masked before the cutoff only
        v = out[h]["p__shortwave_radiation"].values
        assert np.isnan(v[:24]).all() and (v[24:] == 1).all()
    assert not np.isnan(out[6]["p__cloud_cover"].values).any()
    after = weather_by_horizon_from_previous_runs(
        [_previous_runs(datetime(2025, 7, 1, tzinfo=UTC))], ["p"], ["shortwave_radiation"]
    )
    assert not np.isnan(after[6]["p__shortwave_radiation"].values).any()


def test_ingest_matches_training_load_raw():
    raw = RESEARCH / "data" / "raw"
    if not raw.exists():
        pytest.skip("research raw data missing")
    pytest.importorskip("pandas")
    import json

    from core.zones import DE_LU
    from training.dataset import load_raw

    month = "2025-09"
    price_files = sorted((raw / "prices" / DE_LU.code).glob("*.json"))
    slots = [s for f in price_files for s in parse_energy_charts(json.loads(f.read_text()))]
    mine = hourly_prices(
        sorted({s.start: s for s in reversed(slots)}.values(), key=lambda s: s.start)
    )
    full = load_raw(raw)
    assert mine.start == full.prices.start
    np.testing.assert_array_equal(mine.values, full.prices.values)

    name = next(iter(DE_LU.points))
    doc = json.loads(next((raw / "weather").glob(f"{name}_{month}.json")).read_text())
    got = weather_by_horizon_from_previous_runs([doc], [name], DE_LU.weather_vars)
    for h, series in got.items():
        for key, s in series.items():
            ref = full.weather[h + 1][key]
            np.testing.assert_array_equal(
                s.window(s.start, len(s.values)), ref.window(s.start, len(s.values))
            )
