#!/usr/bin/env python3
"""Write the look-back data of the project website: forecast against the actual price.

The forecasts are the bundled walk-forward backfill (custom_components/spot_forecast/
models/<zone>.backfill.json.gz): out-of-sample predictions for the 180 days before the
model's training cutoff, made by models that had not seen those days. They go through
the same ``Ledger`` the integration uses, so the site shows exactly what the look-back
card would show after an install with the longest backfill window. The actual prices
come from the research price files (Energy-Charts, Bundesnetzagentur | SMARD.de,
CC BY 4.0), loaded as tools/card_scenarios.py loads them. No weather is needed.

The output is committed (site/data/), because the site is built in CI without the
research data:

    SPOT_RESEARCH_DIR=… .venv/bin/python tools/site_data.py

Format: ``t`` local wall-clock hours ("YYYY-MM-DD HH:MM"), ``actual`` and
``forecast["1"…"6"]`` in ct/kWh on that grid (null where unknown), plus the model
version, the zone and the peak threshold in ct/kWh.

``trailing`` holds the figures of the model report's trailing period (its
``trailing_3m_from`` to the cutoff) per horizon, in ct/kWh: ``mae`` (the same rows as
the report's ``mae_3m``, which ``main`` checks), and the naive baseline the report has
for the full period only, ``naive_mae`` (the price of the same hour a week before, as in
training/report.py) and ``skill`` = 1 − mae ÷ naive_mae.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import card_scenarios as cs

from core.features import HORIZONS, Hourly
from core.ingest import hourly_prices
from core.ledger import GAP_GRACE_DAYS, Ledger
from core.zones import DE_LU, Zone

OUT = cs.REPO / "site" / "data"
BACKFILL_DAYS = 180  # what the training ships before its cutoff
_HOUR = timedelta(hours=1)


def backtest(zone: Zone = DE_LU) -> dict:
    model = cs._model()
    cutoff = model.train_cutoff
    first = cutoff - timedelta(days=BACKFILL_DAYS)
    start, end = cs._midnight(first, zone), cs._midnight(cutoff, zone)

    hourly = hourly_prices(list(cs._market(zone.code)))
    i = (start - hourly.start) // _HOUR
    n = (end - start) // _HOUR
    if i < 0 or i + n > len(hourly.values):
        raise SystemExit(f"research prices do not cover {first} … {cutoff}")
    ledger = Ledger()
    ledger.record_actual(Hourly(start, hourly.values[i : i + n]))
    for run in cs._backfill_runs(zone, model, first, cutoff):
        ledger.add_run(run)
    rows = ledger.finalize(
        zone.tz,
        today=cutoff + timedelta(days=GAP_GRACE_DAYS),
        zone_id=zone.slug,
        default_p90=model.p90_offsets,
        default_p10=model.p10_offsets,
    )

    hours = [start + k * _HOUR for k in range(n)]
    index = {t: k for k, t in enumerate(hours)}
    prefix = f"spot_forecast:{zone.slug}_"
    series = {"price": [None] * n} | {f"forecast_h{h}": [None] * n for h in HORIZONS}
    for r in rows:
        name = r.statistic_id.removeprefix(prefix)
        if name in series and r.start in index:
            series[name][index[r.start]] = r.value * 100  # EUR/kWh -> ct/kWh
    tz = ZoneInfo(zone.tz)
    local = [t.astimezone(tz) for t in hours]
    trailing_from = date.fromisoformat(model.metrics["trailing_3m_from"])
    recent = [k for k, t in enumerate(local) if t.date() >= trailing_from]
    trailing = {}
    for h in HORIZONS:
        actual, forecast = series["price"], series[f"forecast_h{h}"]
        pairs = [
            (abs(forecast[k] - actual[k]), abs(actual[k - 168] - actual[k]))
            for k in recent
            if k >= 168 and None not in (actual[k], forecast[k], actual[k - 168])
        ]
        mae = sum(e for e, _ in pairs) / len(pairs)
        naive = sum(e for _, e in pairs) / len(pairs)
        trailing[str(h)] = {
            "mae": round(mae, 3),
            "naive_mae": round(naive, 3),
            "skill": round(1 - mae / naive, 4),
        }
    rounded = {k: [None if v is None else round(v, 2) for v in vs] for k, vs in series.items()}
    return {
        "zone": zone.code,
        "model_version": model.version,
        "from": first.isoformat(),
        "until": (cutoff - timedelta(days=1)).isoformat(),
        "peak_threshold": round(model.peak_threshold / 10, 1),
        "source": (
            f"Prices: {zone.price_attribution}. Forecasts: {model.version} walk-forward backfill."
        ),
        "trailing": {
            "from": trailing_from.isoformat(),
            "until": (cutoff - timedelta(days=1)).isoformat(),
            "horizons": trailing,
        },
        "t": [t.strftime("%Y-%m-%d %H:%M") for t in local],
        "actual": rounded["price"],
        "forecast": {str(h): rounded[f"forecast_h{h}"] for h in HORIZONS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT / f"backtest_{DE_LU.slug}.json")
    out = parser.parse_args().out
    doc = backtest()
    # the trailing MAE must be the report's own figure, or the naive baseline next to it
    # would be measured on other rows
    for h, m in cs._model().metrics["horizons"].items():
        if abs(doc["trailing"]["horizons"][h]["mae"] - m["mae_3m"] / 10) > 0.005:
            raise SystemExit(f"D+{h}: trailing MAE differs from the model report's mae_3m")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    known = sum(v is not None for v in doc["actual"])
    print(f"{out}: {len(doc['t'])} hours, {known} with a price, {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
