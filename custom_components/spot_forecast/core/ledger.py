"""Forecast ledger: the runs the user saw, calibration, statistic rows and accuracy.

Prices are EUR/MWh inside the ledger; statistic rows and ``accuracy`` are EUR/kWh.
A run's slot with horizon ``h`` targets the local day ``anchor + h``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
import gzip
import json
import math
from zoneinfo import ZoneInfo

import numpy as np

from .features import HORIZONS, Hourly
from .forecast import Slot, conformal_offset, low_price, peak_risk

CALIBRATION_DAYS = 56  # trailing target days used for the conformal offsets
BACKFILL_FORMAT = 1
ISSUE_HOUR = 6  # local hour a backfilled run is deemed to have been issued
_HOUR = timedelta(hours=1)
_PREFIX = "spot_forecast"
PRICE = "price"  # series key in ``Ledger.finalized``; forecast series are "h1" … "h6"
SERIES = (PRICE, *(f"h{h}" for h in HORIZONS))  # what a fully finalized day has
GAP_GRACE_DAYS = 2  # a day with missing hours is written once it is this many days past


@dataclass(frozen=True)
class Run:
    """One forecast issued for ``anchor`` (local date); slots hold raw EUR/MWh quantiles."""

    anchor: date
    issued_at: datetime  # UTC-aware
    model_version: str
    slots: tuple[Slot, ...]
    backfill: bool


@dataclass(frozen=True)
class StatRow:
    """One hourly long-term-statistic value; ``start`` is the UTC hour start, value EUR/kWh."""

    statistic_id: str
    start: datetime
    value: float


def _kwh(eur_per_mwh: float) -> float:
    return round(eur_per_mwh / 1000, 5)


def _nan_to_none(x: float) -> float | None:
    return None if math.isnan(x) else x


def _none_to_nan(x: float | None) -> float:
    return math.nan if x is None else x


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def _day_hours(d: date, tz: ZoneInfo) -> list[datetime]:
    """UTC hour starts of local day ``d`` (23, 24 or 25 of them)."""
    start = datetime.combine(d, time(), tz).astimezone(UTC)
    end = datetime.combine(d + timedelta(days=1), time(), tz).astimezone(UTC)
    return [start + i * _HOUR for i in range((end - start) // _HOUR)]


@dataclass
class Ledger:
    runs: dict[date, Run] = field(default_factory=dict)  # last run per anchor
    # (target day, series) pairs whose statistics are written; series is PRICE or "h<h>"
    finalized: set[tuple[date, str]] = field(default_factory=set)
    actual: dict[datetime, float] = field(default_factory=dict)  # UTC hour -> EUR/MWh

    def copy(self) -> Ledger:
        """A ledger that can be changed without touching this one (runs and slots are frozen)."""
        return Ledger(dict(self.runs), set(self.finalized), dict(self.actual))

    def add_run(self, run: Run) -> None:
        """Store ``run`` unless the stored run of its anchor takes precedence.

        What the user saw wins: a backfill run never replaces a live run, and a live run
        always replaces a backfill run. Between runs of one kind the later issued wins; a
        backfill run also on a tie, so a re-import with a wider window completes a stored one.
        """
        old = self.runs.get(run.anchor)
        if (
            old is None
            or (old.backfill and not run.backfill)
            or (old.backfill == run.backfill and run.issued_at > old.issued_at)
            or (old.backfill and run.backfill and run.issued_at == old.issued_at)
        ):
            self.runs[run.anchor] = run

    def record_actual(self, hourly: Hourly) -> None:
        """Store every known (non-NaN) hour; later values replace earlier ones."""
        for i, v in enumerate(hourly.values):
            if not np.isnan(v):
                self.actual[hourly.start + i * _HOUR] = float(v)

    def offsets(
        self, anchor: date, default_p90: Mapping[int, float], default_p10: Mapping[int, float]
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Conformal offsets per h from stored rows whose target day is in
        [anchor - 56 d, anchor) and whose actual is known (EUR/MWh)."""
        lo = anchor - timedelta(days=CALIBRATION_DAYS)
        res90: dict[int, list[float]] = {h: [] for h in HORIZONS}
        res10: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for run in self.runs.values():
            if not lo - timedelta(days=max(HORIZONS)) <= run.anchor < anchor:
                continue
            for s in run.slots:
                target = run.anchor + timedelta(days=s.h)
                actual = self.actual.get(s.time)
                if not s.valid or actual is None or not lo <= target < anchor or s.h not in res90:
                    continue
                res90[s.h].append(actual - s.p90_raw)
                res10[s.h].append(actual - s.p10_raw)
        p90 = {h: conformal_offset(np.array(res90[h]), default_p90[h], q=0.9) for h in HORIZONS}
        p10 = {h: conformal_offset(np.array(res10[h]), default_p10[h], q=0.1) for h in HORIZONS}
        return p90, p10

    def finalize(
        self,
        tz: str,
        *,
        today: date,
        zone_id: str,
        default_p90: Mapping[int, float],
        default_p10: Mapping[int, float],
    ) -> list[StatRow]:
        """Statistic rows not yet written, for every target day with all hourly actuals.

        A day with missing hours is written as it is once it lies ``GAP_GRACE_DAYS`` before
        ``today`` (local): the price rows of the hours that exist, the forecast rows in full.
        Tracked per series: a day's price rows go out once; its forecast, peak-risk and low
        rows for horizon ``h`` go out once a run with anchor ``day - h`` holds slots for that
        horizon (so runs added or completed later, e.g. by the backfill, still produce their
        rows). ``default_*`` are the model's offsets, used by ``offsets`` while a horizon has
        too few residuals. Days whose every series is written are not looked at again.
        """
        zone = ZoneInfo(tz)
        by_anchor: dict[date, tuple[dict[int, float], dict[int, float]]] = {}
        rows: list[StatRow] = []
        for day in self._open_days(zone):
            hours = _day_hours(day, zone)
            known = [t for t in hours if t in self.actual]
            if not known:
                continue
            if len(known) < len(hours) and day > today - timedelta(days=GAP_GRACE_DAYS):
                continue
            if (day, PRICE) not in self.finalized:
                rows += [
                    StatRow(f"{_PREFIX}:{zone_id}_price", t, _kwh(self.actual[t])) for t in known
                ]
                self.finalized.add((day, PRICE))
            span = set(hours)
            for h in HORIZONS:
                run = self.runs.get(day - timedelta(days=h))
                if (
                    run is None
                    or (day, f"h{h}") in self.finalized
                    or not any(s.h == h for s in run.slots)
                ):
                    continue
                if run.anchor not in by_anchor:
                    by_anchor[run.anchor] = self.offsets(run.anchor, default_p90, default_p10)
                o90, o10 = by_anchor[run.anchor]
                for s in run.slots:
                    if s.h != h or not s.valid or s.time not in span:
                        continue
                    cal = {
                        f"forecast_h{h}": s.median,
                        f"peak_risk_h{h}": peak_risk(s, o90[h]),
                        f"low_h{h}": low_price(s, o10[h]),
                    }
                    rows += [
                        StatRow(f"{_PREFIX}:{zone_id}_{name}", s.time, _kwh(v))
                        for name, v in cal.items()
                    ]
                self.finalized.add((day, f"h{h}"))
        return rows

    def _open_days(self, zone: ZoneInfo) -> list[date]:
        """Local days from the first to the last actual hour with a series not yet written."""
        if not self.actual:
            return []
        first = min(self.actual).astimezone(zone).date()
        last = max(self.actual).astimezone(zone).date()
        days = (first + timedelta(days=i) for i in range((last - first).days + 1))
        return [d for d in days if any((d, s) not in self.finalized for s in SERIES)]

    def accuracy(self, now: datetime, peak_threshold: float, *, tz: str, days: int = 30) -> dict:
        """Median-forecast accuracy over target days in [today - days, today) (local), EUR/kWh.

        ``peak_threshold`` is EUR/MWh; ``peak_mae`` covers hours with actual > it
        (as in training/report.py).
        """
        today = now.astimezone(ZoneInfo(tz)).date()
        lo = today - timedelta(days=days)
        errs: dict[int, list[float]] = {}
        peak: list[float] = []
        for run in self.runs.values():
            for s in run.slots:
                actual = self.actual.get(s.time)
                if not s.valid or actual is None:
                    continue
                if not lo <= run.anchor + timedelta(days=s.h) < today:
                    continue
                e = abs(s.median - actual)
                errs.setdefault(s.h, []).append(e)
                if actual > peak_threshold:
                    peak.append(e)
        allerr = [e for v in errs.values() for e in v]
        return {
            "mae": _kwh(float(np.mean(allerr))) if allerr else None,
            "mae_h": {h: _kwh(float(np.mean(v))) for h, v in sorted(errs.items())},
            "peak_mae": _kwh(float(np.mean(peak))) if peak else None,
            "n": len(allerr),
        }

    def prune(self, keep_days: int) -> None:
        """Drop runs, actuals and finalized days older than ``keep_days`` before the newest one."""
        newest = [max(self.actual)] if self.actual else []
        newest += [r.issued_at for r in self.runs.values()]
        if not newest:
            return
        cutoff = max(newest).date() - timedelta(days=keep_days)
        cut_t = datetime.combine(cutoff, time(), UTC)
        self.runs = {a: r for a, r in self.runs.items() if a >= cutoff}
        self.actual = {t: v for t, v in self.actual.items() if t >= cut_t}
        self.finalized = {f for f in self.finalized if f[0] >= cutoff}

    def to_json(self) -> dict:
        return {
            "runs": [
                {
                    "anchor": r.anchor.isoformat(),
                    "issued_at": r.issued_at.isoformat(),
                    "model_version": r.model_version,
                    "backfill": r.backfill,
                    "slots": [
                        [
                            s.time.isoformat(),
                            s.h,
                            _nan_to_none(s.median),
                            _nan_to_none(s.p90_raw),
                            _nan_to_none(s.p10_raw),
                            s.valid,
                        ]
                        for s in r.slots
                    ],
                }
                for _, r in sorted(self.runs.items())
            ],
            "finalized": [[d.isoformat(), k] for d, k in sorted(self.finalized)],
            "actual": [[t.isoformat(), v] for t, v in sorted(self.actual.items())],
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Ledger:
        runs = {}
        for r in d["runs"]:
            slots = tuple(
                Slot(
                    _utc(t),
                    int(h),
                    _none_to_nan(m),
                    _none_to_nan(hi),
                    _none_to_nan(lo),
                    bool(valid),
                )
                for t, h, m, hi, lo, valid in r["slots"]
            )
            run = Run(
                date.fromisoformat(r["anchor"]),
                _utc(r["issued_at"]),
                r["model_version"],
                slots,
                bool(r["backfill"]),
            )
            runs[run.anchor] = run
        return cls(
            runs,
            {(date.fromisoformat(x), str(k)) for x, k in d["finalized"]},
            {_utc(t): float(v) for t, v in d["actual"]},
        )


def runs_from_backfill_file(data: bytes, model_version: str, tz: str) -> list[Run]:
    """Parse the training backfill (gzip JSON, format 1) into one run per anchor day.

    The runs carry the model version the file names, else ``model_version``: a
    downloaded model without its own file uses the bundled one, made by another model.
    """
    doc = json.loads(gzip.decompress(data))
    if doc.get("format") != BACKFILL_FORMAT:
        raise ValueError(f"unsupported backfill format {doc.get('format')!r}")
    version = doc.get("model_version") or model_version
    zone = ZoneInfo(tz)
    slots: dict[date, list[Slot]] = {}
    for t_iso, h, median, p90, p10 in doc["rows"]:
        t = _utc(t_iso)
        anchor = t.astimezone(zone).date() - timedelta(days=int(h))
        slots.setdefault(anchor, []).append(
            Slot(t, int(h), float(median), float(p90), float(p10), True)
        )
    return [
        Run(
            anchor,
            datetime.combine(anchor, time(ISSUE_HOUR), zone).astimezone(UTC),
            version,
            tuple(slots[anchor]),
            True,
        )
        for anchor in sorted(slots)
    ]
