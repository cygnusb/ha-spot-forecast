"""Write the golden feature fixture from the research backtest.

Runs the research ``build_samples`` on the raw data and stores, per golden anchor,
the inputs ``core.features.build_features`` needs and the feature rows research
produced. ``tests/core/test_features.py`` checks parity against this file.

Usage: python training/make_golden.py   (env SPOT_RESEARCH_DIR, default <repo>/research)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
import gzip
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "golden_features.json.gz"
ANCHORS = [
    date(2025, 6, 15),
    date(2025, 10, 25),
    date(2025, 10, 26),
    date(2026, 3, 28),
    date(2026, 9, 20),
]
HISTORY_DAYS = 45
TZ = ZoneInfo("Europe/Berlin")

RESEARCH = Path(os.environ.get("SPOT_RESEARCH_DIR", REPO / "research"))
sys.path.insert(0, str(RESEARCH))
import backtest as bt  # noqa: E402
from fetch import NEIGHBOR_POINTS, NEIGHBOR_VARS, POINTS, WEATHER_VARS  # noqa: E402

# Feature order, written independently of core.features so the fixture checks FEATURE_NAMES.
FEATURE_NAMES = [
    "h", "hour", "dow", "doy", "holiday", "offday", "issue_dow", "p_lastday", "p_lastweek",
    "D_mean", "D_min", "D_max", "D_std", "D_mean7", "D_std7", "D_mean30", "D_min7", "D_max7",
    "D_gq_evening14", "D_gq_q90_14",
    *[f"{p}__{v}" for p in POINTS for v in WEATHER_VARS],
    "solar_est_mean", "wind_mean", "wind_power_idx", "solar_mean", "temp_mean", "cloud_mean",
    "nb_wind_power_idx", "nb_temp_fr", "nb_temp_nordic", "nb_precip_hydro", "nb_solar_fr",
    "wind_power_idx_day", "solar_mean_day", "solar_est_mean_day", "temp_mean_day",
    "nb_wind_power_idx_day", "nb_temp_fr_day", "nb_temp_nordic_day", "nb_precip_hydro_day",
    "cap_solar", "cap_wind_on", "cap_wind_off",
]  # fmt: skip
WEATHER_KEYS = [f"{p}__{v}" for p in POINTS for v in WEATHER_VARS] + [
    f"{p}__{v}" for p in NEIGHBOR_POINTS for v in NEIGHBOR_VARS
]


def local_midnight_utc(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, time(), TZ).astimezone(UTC))


def floats(values) -> list[float | None]:
    """JSON-safe floats: NaN becomes null."""
    return [None if math.isnan(v) else float(v) for v in np.asarray(values, dtype=float)]


def hourly(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    return {"start": start.isoformat(), "values": floats(series.reindex(idx).to_numpy())}


def case(
    anchor: date, prices: pd.Series, weather: dict, cap: pd.DataFrame, df: pd.DataFrame
) -> dict:
    rows = df[df["issue_date"] == pd.Timestamp(anchor)]
    rows = rows.assign(_t=rows.index).sort_values(["h", "_t"])
    weather_out = {}
    for h in range(1, 7):
        start, end = (
            local_midnight_utc(anchor + timedelta(days=h)),
            local_midnight_utc(anchor + timedelta(days=h + 1)),
        )
        n_hours = int((end - start) / pd.Timedelta(hours=1))
        kept = int((rows["h"] == h).sum())
        if kept != n_hours:
            raise SystemExit(f"{anchor} h={h}: research kept {kept} of {n_hours} hours")
        w = weather[h + 1]
        weather_out[str(h)] = {k: hourly(w[k], start, end) for k in WEATHER_KEYS}
    per = pd.Period(anchor, "M")
    cap_row = cap.reindex([per]).iloc[0]
    scale = rows["D_gq_evening14"].clip(lower=50) / 100
    return {
        "anchor": anchor.isoformat(),
        "prices": hourly(
            prices,
            local_midnight_utc(anchor - timedelta(days=HISTORY_DAYS)),
            local_midnight_utc(anchor + timedelta(days=1)),
        ),
        "weather": weather_out,
        "capacity": {c: floats([cap_row[c]])[0] for c in cap.columns},
        "times": [t.isoformat() for t in rows.index],
        "X": [floats(r) for r in rows[FEATURE_NAMES].astype(float).to_numpy()],
        "mean7": floats(rows["D_mean7"].to_numpy()),
        "scale": floats(scale.to_numpy()),
    }


def main() -> None:
    prices = bt.load_prices()
    weather = bt.load_weather()
    cap = bt.load_installed()
    df = bt.build_samples(prices, weather, cap, {}, None)
    payload = {
        "feature_names": FEATURE_NAMES,
        "cases": [case(a, prices, weather, cap, df) for a in ANCHORS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    OUT.write_bytes(gzip.compress(raw, mtime=0))
    for c in payload["cases"]:
        print(c["anchor"], len(c["times"]), "rows, capacity", c["capacity"])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
