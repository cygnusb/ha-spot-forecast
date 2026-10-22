"""Dataset parity with the research backtest (needs the research raw data)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
import os
from pathlib import Path
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pytest

pytest.importorskip("pandas")
from core.features import FEATURE_NAMES, Hourly, build_features, daily_stats
from core.zones import DE_LU
from training.dataset import build_dataset, load_raw

RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)
RESEARCH_RAW = RESEARCH / "data" / "raw"

pytestmark = pytest.mark.skipif(not RESEARCH_RAW.exists(), reason="research raw data missing")


@pytest.fixture(scope="module")
def raw():
    return load_raw(RESEARCH_RAW)


@pytest.fixture(scope="module")
def research_rows():
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    sys.path.insert(0, str(RESEARCH))
    try:
        import backtest as bt
    finally:
        sys.path.remove(str(RESEARCH))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas deprecations inside research code
        df = bt.build_samples(bt.load_prices(), bt.load_weather(), bt.load_installed(), {}, None)
    import pandas as pd

    df = df[
        (df["issue_date"] >= pd.Timestamp("2025-09-01"))
        & (df["issue_date"] <= pd.Timestamp("2025-09-30"))
    ]
    return df.assign(_t=df.index).sort_values(["issue_date", "h", "_t"])


def test_dataset_matches_research_rows(raw, research_rows):
    ds = build_dataset(raw, date(2025, 9, 1), date(2025, 9, 30))
    assert len(ds.y_price) == len(research_rows)
    np.testing.assert_allclose(
        ds.X,
        research_rows[list(FEATURE_NAMES)].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )
    np.testing.assert_allclose(ds.y_price, research_rows["price"].to_numpy(float))
    np.testing.assert_allclose(ds.p_lastweek, research_rows["p_lastweek"].to_numpy(float))
    assert [t.isoformat() for t in ds.times] == [t.isoformat() for t in research_rows["_t"]]
    assert list(ds.h) == list(research_rows["h"])
    assert list(ds.anchor) == [d.date() for d in research_rows["issue_date"]]
    assert list(ds.target_date) == [d.date() for d in research_rows["target_date"]]


def test_shared_daily_stats_equal_per_anchor(raw):
    """One daily_stats over the full history gives the features of per-anchor computation
    on prices cut at the end of the anchor day (what the integration will have)."""
    weather = {h: raw.weather[h + 1] for h in range(1, 7)}
    shared = daily_stats(raw.prices, DE_LU)
    for anchor in (date(2025, 3, 30), date(2025, 9, 12), date(2025, 10, 26)):
        end = datetime.combine(anchor + timedelta(days=1), time(), ZoneInfo(DE_LU.tz)).astimezone(
            UTC
        )
        cut = Hourly(
            raw.prices.start,
            raw.prices.values[: int((end - raw.prices.start) / timedelta(hours=1))],
        )
        cap = raw.capacity.get(f"{anchor:%Y-%m}", {})
        a = build_features(anchor, raw.prices, shared, weather, cap, DE_LU)
        b = build_features(anchor, cut, daily_stats(cut, DE_LU), weather, cap, DE_LU)
        np.testing.assert_allclose(a.X, b.X, equal_nan=True)


def test_skips_anchors_without_history(raw):
    ds = build_dataset(raw, date(2024, 2, 1), date(2024, 3, 20))
    assert len(ds.y_price) == 0 or min(ds.anchor) >= date(2024, 3, 1)


def test_full_range_is_complete_and_fast(raw):
    import time as clock

    t0 = clock.perf_counter()
    ds = build_dataset(raw, date(2024, 1, 1), date(2026, 10, 21))
    elapsed = clock.perf_counter() - t0
    assert elapsed < 120
    assert ds.X.shape == (len(ds.y_price), len(FEATURE_NAMES))
    assert not np.isnan(ds.y_price).any() and not np.isnan(ds.p_lastweek).any()
    print(f"full range: {len(ds.y_price)} rows, {elapsed:.1f}s")
