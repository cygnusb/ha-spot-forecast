from datetime import UTC, datetime, timedelta
import json
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
from core.zones import DE_LU, N_VARS, NB_VARS, NO2

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


# ---------------------------------------------------------------- NO2 hydro history

HA_FIXTURES = Path(__file__).resolve().parents[2] / "tests_ha" / "fixtures"
_HYDRO_VARS = ("precipitation", "temperature_2m")


def _hydro_history():
    """The runtime hydro history: the recorded live ``past_days`` body, parsed as
    ``sources.open_meteo.fetch_hydro_history`` does it."""
    body = json.loads((HA_FIXTURES / "open_meteo_hydro_no2.json").read_text())
    return parse_open_meteo(body, list(NO2.hydro_points), _HYDRO_VARS, "_previous_day2")


def _same_hours(a, b):
    first = max(a.start, b.start)
    end = min(
        a.start + len(a.values) * timedelta(hours=1), b.start + len(b.values) * timedelta(hours=1)
    )
    n = (end - first) // timedelta(hours=1)
    return a.window(first, n), b.window(first, n)


def test_runtime_hydro_history_equals_the_previous_runs_horizon_1_series():
    """The ``past_days`` body against the ``_previous_day2`` columns of the previous-runs
    recordings of the same points (2026-10-01 … 10-22): what training reads as ``weather[1]``."""

    def prev(name, points, variables):
        docs = json.loads((HA_FIXTURES / name).read_text())
        return weather_by_horizon_from_previous_runs(docs, points, variables)[1]

    n_points = [p for points, v in NO2.point_sets() if v == N_VARS for p in points]
    reference = prev("open_meteo_prev_no2_n.json", n_points, N_VARS)
    reference |= prev("open_meteo_prev_nb.json", list(DE_LU.nb), NB_VARS)
    history = _hydro_history()
    assert set(history) == {f"{p}__{v}" for p in NO2.hydro_points for v in _HYDRO_VARS}
    for key, series in history.items():
        live, archived = _same_hours(series, reference[key])
        assert len(live) == 22 * 24 and not np.isnan(live).any(), key
        np.testing.assert_array_equal(live, archived, err_msg=key)


def test_runtime_hydro_history_equals_the_training_data():
    """The same against the training raw files of the research data, over the 90 days the
    recording shares with them (data to 2026-10-21)."""
    if not (RESEARCH / "data" / "zones" / "weather").exists():
        pytest.skip("research zone data missing")
    weather = {}
    for p in NO2.hydro_points:
        base = RESEARCH / "data" / ("raw" if p.startswith("nb_") else "zones") / "weather"
        files = sorted(f for f in base.glob(f"{p}_2026-*.json") if not f.name.startswith("."))
        for f in files:
            doc = json.loads(f.read_text())
            for key, s in weather_by_horizon_from_previous_runs([doc], [p], ("precipitation",
                    "temperature_2m"))[1].items():  # fmt: skip
                weather[key] = merge_hourly(s, weather[key]) if key in weather else s
    for key, series in _hydro_history().items():
        live, archived = _same_hours(series, weather[key])
        ok = ~np.isnan(archived)
        assert ok.sum() >= 90 * 24, key
        np.testing.assert_array_equal(live[ok], archived[ok], err_msg=key)
