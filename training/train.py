"""Training CLI: model file, out-of-sample backfill and accuracy report.

python -m training.train --raw DIR --out DIR --cutoff YYYY-MM-DD
    [--walk-forward-from YYYY-MM-DD] [--backfill-days 180] [--capacity-month YYYY-MM]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import gzip
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from core.features import FEATURE_NAMES, FEATURE_SCHEMA, HORIZONS, Hourly
from core.model import FORMAT, ModelFile, dump_model
from core.zones import DE_LU

from .dataset import Dataset, RawData, build_dataset, load_raw
from .export import trees_from_sklearn
from .report import Predictions, evaluate, render_markdown, window_offsets

ZONE = "de_lu"
BACKFILL_FORMAT = 1
PEAK_PERCENTILE = 97
QUANTILES = {"median": None, "p90_raw": 0.9, "p10_raw": 0.1}


class DataLag(ValueError):
    """Input data for the cutoff is not published yet; rerunning later fixes it."""


@dataclass(frozen=True, eq=False)
class TrainResult:
    """What one training run produced, for ``training.ci``."""

    version: str
    model_path: Path
    backfill_path: Path
    metrics_path: Path
    report_path: Path
    model: ModelFile
    predictions: Predictions  # walk-forward, out of sample
    dataset: Dataset  # all training rows (target days before the cutoff)


def build_training_data(raw: RawData, cutoff: date) -> Dataset:
    """All rows with a target day before ``cutoff``, in research row order (horizon, time).

    The row order matters only for the early-stopping validation split, which then
    matches research ``walk_forward``.
    """
    ds = build_dataset(raw, raw.prices.start.date(), cutoff - timedelta(days=1))
    keep = np.flatnonzero(ds.target_date < cutoff)
    order = np.lexsort((np.array([t.timestamp() for t in ds.times[keep]]), ds.h[keep]))
    return ds.subset(keep[order])


def fit(ds: Dataset, quantile: float | None, max_iter: int = 600) -> HistGradientBoostingRegressor:
    """Median (``None``) or quantile model on the scaled target, all features numeric."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error" if quantile is None else "quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=0,
    )
    return model.fit(ds.X, (ds.y_price - ds.mean7) / ds.scale)


def _predict(model: HistGradientBoostingRegressor, ds: Dataset) -> np.ndarray:
    return model.predict(ds.X) * ds.scale + ds.mean7


def walk_forward(ds: Dataset, first_anchor: date, cutoff: date, max_iter: int = 600) -> Predictions:
    """Monthly refit as research ``walk_forward``: the model for month m is fitted on
    targets before the 1st of m and predicts the issue days in m (from ``first_anchor``),
    targets before ``cutoff`` only. Every prediction is out of sample."""
    parts: list[Predictions] = []
    month = first_anchor.replace(day=1)
    while month < cutoff:
        nxt = (month + timedelta(days=32)).replace(day=1)
        test_idx = np.flatnonzero(
            (ds.anchor >= max(month, first_anchor)) & (ds.anchor < nxt) & (ds.target_date < cutoff)
        )
        if len(test_idx):
            train = ds.subset(np.flatnonzero(ds.target_date < month))
            test = ds.subset(test_idx)
            preds = {k: _predict(fit(train, q, max_iter), test) for k, q in QUANTILES.items()}
            parts.append(
                Predictions(
                    times=test.times,
                    h=test.h,
                    anchor=test.anchor,
                    target_date=test.target_date,
                    train_last_target=np.full(len(test_idx), max(train.target_date), dtype=object),
                    actual=test.y_price,
                    p_lastweek=test.p_lastweek,
                    **preds,
                )
            )
            print(f"  {month:%Y-%m}: train={len(train.y_price):>7} test={len(test_idx):>6}")
        month = nxt
    if not parts:
        raise ValueError(f"no walk-forward rows between {first_anchor} and {cutoff}")
    fields = Predictions.__dataclass_fields__
    return Predictions(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in fields})


def _local_midnight_utc(d: date) -> datetime:
    return datetime.combine(d, time(), ZoneInfo(DE_LU.tz)).astimezone(UTC)


def peak_threshold(prices: Hourly, cutoff: date) -> float:
    """97th percentile of the hourly prices in the 365 days before ``cutoff``."""
    first = _local_midnight_utc(cutoff - timedelta(days=365))
    hours = int((_local_midnight_utc(cutoff) - first) / timedelta(hours=1))
    return float(np.nanpercentile(prices.window(first, hours), PEAK_PERCENTILE))


def _last_valid_hour(s: Hourly) -> datetime | None:
    idx = np.flatnonzero(~np.isnan(s.values))
    return s.start + timedelta(hours=int(idx[-1])) if len(idx) else None


def check_data_reaches(raw: RawData, cutoff: date) -> None:
    """Fail if prices or any weather series end before the day ``cutoff - 1``.

    A gap at the end would otherwise only drop rows (no actual price, no history) and the
    model would silently be trained on less data than its version name promises.
    """
    last_day = cutoff - timedelta(days=1)
    need = _local_midnight_utc(last_day)
    series = {"prices": raw.prices}
    for offset, by_name in sorted(raw.weather.items()):
        series.update({f"weather {name} (offset {offset})": s for name, s in by_name.items()})
    for name, s in series.items():
        last = _last_valid_hour(s)
        if last is None or last < need:
            ends = "has no data" if last is None else f"ends {last:%Y-%m-%d %H:%M} UTC"
            raise ValueError(
                f"{name} {ends}, before {last_day} (the day before cutoff {cutoff}); "
                "refetch the raw data or choose an earlier cutoff"
            )


def capacity_at(raw: RawData, cutoff: date, month: str | None = None) -> dict[str, float]:
    """Installed capacity of the last training anchor's month (the day before the cutoff),
    or of ``month`` ("YYYY-MM") when given."""
    month = month or f"{cutoff - timedelta(days=1):%Y-%m}"
    if month not in raw.capacity:
        raise DataLag(
            f"installed_power has no value for {month}; Energy-Charts publishes monthly with a "
            "lag — rerun after it appears or pass --capacity-month"
        )
    return dict(raw.capacity[month])


def backfill(p: Predictions, version: str, first: date, cutoff: date) -> dict:
    keep = (p.target_date >= first) & (p.target_date < cutoff)
    q = p.subset(keep)
    order = np.lexsort((q.h, np.array([t.timestamp() for t in q.times])))
    rows = [
        [q.times[i].isoformat(), int(q.h[i])]
        + [round(float(v), 2) for v in (q.median[i], q.p90_raw[i], q.p10_raw[i])]
        for i in order
    ]
    return {"format": BACKFILL_FORMAT, "model_version": version, "rows": rows}


def _month(value: str) -> str:
    """argparse type for "YYYY-MM"."""
    try:
        return f"{datetime.strptime(value, '%Y-%m'):%Y-%m}"
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM month: {value!r}") from None


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m training.train", description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="raw-data directory (fetch output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        required=True,
        help="train on target days before this date",
    )
    ap.add_argument(
        "--walk-forward-from",
        type=date.fromisoformat,
        default=None,
        help="first issue day of the report (default: cutoff - backfill days)",
    )
    ap.add_argument("--backfill-days", type=int, default=180)
    ap.add_argument(
        "--capacity-month",
        type=_month,
        default=None,
        help="take the installed capacity of this month (default: the month before the cutoff)",
    )
    return ap.parse_args([str(a) for a in argv])


def run(argv: list[str], max_iter: int = 600) -> TrainResult:
    """Train, write model file, backfill, report. ``max_iter`` is for fast tests only.

    Raises ``DataLag`` before any training when the installed capacity is not published yet.
    """
    args = _parse(argv)
    cutoff = args.cutoff
    backfill_from = cutoff - timedelta(days=args.backfill_days)
    report_from = args.walk_forward_from or backfill_from
    version = f"{ZONE}-{cutoff:%Y-%m}"

    raw = load_raw(args.raw)
    check_data_reaches(raw, cutoff)
    capacity = capacity_at(raw, cutoff, args.capacity_month)
    ds = build_training_data(raw, cutoff)
    print(f"rows: {len(ds.y_price)}  target days {min(ds.target_date)}..{max(ds.target_date)}")
    # earliest issue day whose D+6 target is in the backfill window
    first_anchor = min(report_from, backfill_from - timedelta(days=max(HORIZONS)))
    wf = walk_forward(ds, first_anchor, cutoff, max_iter)

    threshold = peak_threshold(raw.prices, cutoff)
    metrics = evaluate(wf, report_from, cutoff, threshold)
    final = {k: trees_from_sklearn(fit(ds, q, max_iter)) for k, q in QUANTILES.items()}
    model = ModelFile(
        format=FORMAT,
        zone=ZONE,
        version=version,
        feature_schema=FEATURE_SCHEMA,
        features=tuple(FEATURE_NAMES),
        train_cutoff=cutoff,
        created=datetime.now(UTC),
        median=final["median"],
        p90=final["p90_raw"],
        p10=final["p10_raw"],
        p90_offsets=window_offsets(wf.target_date, wf.h, wf.actual - wf.p90_raw, cutoff, 0.9),
        p10_offsets=window_offsets(wf.target_date, wf.h, wf.actual - wf.p10_raw, cutoff, 0.1),
        peak_threshold=threshold,
        capacity=capacity,
        metrics=metrics,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    result = TrainResult(
        version=version,
        model_path=out / f"{version}.json.gz",
        backfill_path=out / f"{version}.backfill.json.gz",
        metrics_path=out / f"{version}.metrics.json",
        report_path=out / f"{version}.report.md",
        model=model,
        predictions=wf,
        dataset=ds,
    )
    result.model_path.write_bytes(dump_model(model))
    bf = backfill(wf, version, backfill_from, cutoff)
    result.backfill_path.write_bytes(gzip.compress(json.dumps(bf).encode(), mtime=0))
    result.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    report = render_markdown(version, metrics)
    result.report_path.write_text(report)
    print(report)
    return result


def main() -> None:
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
