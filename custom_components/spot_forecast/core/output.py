"""Output model: what the entities show.

Builds the SBP-compatible ``prices`` attribute (market, then forecast), the
current price, the same with a fixed surcharge (the total price), 48 h risk
windows for the binary sensors and per-day summaries.
Prices leave this module in EUR/kWh; the core works in EUR/MWh.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .forecast import Slot, low_price, peak_risk
from .ingest import MarketSlot

_DECIMALS = 5
_MAX_SLOT = timedelta(hours=1)  # longest entry; caps the length across gaps in the list


def _eur_kwh(eur_mwh: float) -> float:
    return round(eur_mwh / 1000.0, _DECIMALS)


def prices_attribute(
    market: Sequence[MarketSlot],
    forecast: Sequence[Slot],
    p90_off: dict[int, float],
    p10_off: dict[int, float],
    tz: str,
    start: datetime,
) -> list[dict]:
    """Market slots from ``start`` at native resolution, then valid forecast hours after them.

    ``time`` is ISO in market-local time with offset, so the two 02:00 of a
    fall-back day differ by offset. Uniqueness is by instant.
    """
    zone = ZoneInfo(tz)
    by_instant: dict[datetime, dict] = {}
    market_end = None
    for m in sorted(market, key=lambda s: s.start):
        if market_end is None or m.end > market_end:
            market_end = m.end
        if m.start < start:
            continue
        by_instant[m.start] = {
            "time": m.start.astimezone(zone).isoformat(),
            "price": _eur_kwh(m.price),
            "source": "market",
        }
    for s in sorted(forecast, key=lambda s: s.time):
        if not s.valid or s.time < start:
            continue
        if market_end is not None and s.time < market_end:
            continue
        by_instant[s.time] = {
            "time": s.time.astimezone(zone).isoformat(),
            "price": _eur_kwh(s.median),
            "source": "forecast",
            "p90": _eur_kwh(peak_risk(s, p90_off[s.h])),
            "p10": _eur_kwh(low_price(s, p10_off[s.h])),
            "h": s.h,
        }
    return [by_instant[k] for k in sorted(by_instant)]


def with_surcharge(entries: list[dict], surcharge: float) -> list[dict]:
    """``prices`` entries with ``surcharge`` (EUR/kWh) added to ``price``, ``p90`` and ``p10``."""
    return [
        {
            **e,
            **{k: round(e[k] + surcharge, _DECIMALS) for k in ("price", "p90", "p10") if k in e},
        }
        for e in entries
    ]


def current_price(market: Sequence[MarketSlot], now: datetime) -> float | None:
    """Price (EUR/kWh) of the market slot running at ``now``; None if there is none."""
    for m in market:
        if m.start <= now < m.end:
            return _eur_kwh(m.price)
    return None


def _as_utc_delta(a: datetime, b: datetime) -> timedelta:
    # Subtracting aware datetimes sharing one tzinfo ignores DST; go through UTC.
    return a.astimezone(UTC) - b.astimezone(UTC)


def _lengths(entries: list[dict]) -> list[tuple[datetime, timedelta]]:
    starts = [datetime.fromisoformat(e["time"]) for e in entries]
    out: list[tuple[datetime, timedelta]] = []
    for i, s in enumerate(starts):
        if i + 1 < len(starts):
            out.append((s, min(_as_utc_delta(starts[i + 1], s), _MAX_SLOT)))
        else:
            out.append((s, out[-1][1] if out else timedelta(hours=1)))
    return out


def flag_window(
    entries: list[dict],
    now: datetime,
    hours: int = 48,
    *,
    key: str,
    threshold: float,
    above: bool,
) -> tuple[list[dict], bool]:
    """Entries in [now - own length, now + hours) whose value crosses ``threshold`` (EUR/kWh).

    Market entries compare ``price``, forecast entries compare ``entry[key]``.
    ``above`` flags value >= threshold, otherwise value < threshold. Returns the
    flagged entries in time order and ``certain`` (any flagged entry is market).
    """
    end = now + timedelta(hours=hours)
    flagged: list[dict] = []
    for entry, (start, length) in zip(entries, _lengths(entries), strict=True):
        if start + length <= now or start >= end:
            continue
        value = entry["price"] if entry["source"] == "market" else entry[key]
        if (value >= threshold) if above else (value < threshold):
            flagged.append(entry)
    flagged.sort(key=lambda e: datetime.fromisoformat(e["time"]).timestamp())
    return flagged, any(e["source"] == "market" for e in flagged)


def day_summaries(entries: list[dict], tz: str, today: date, days: int = 7) -> list[dict]:
    """Per local day from ``today``: mean/min/max of ``price``, p90_max/p10_min, source.

    ``mean`` is weighted by slot length (entry length = next start minus start, capped at 1 h),
    so mixed 15-min/60-min days are not biased. ``p90_max``/``p10_min`` use
    forecast entries only. Days without entries get None values.
    """
    zone = ZoneInfo(tz)
    per_day: dict[date, list[tuple[dict, float]]] = {}
    for entry, (start, length) in zip(entries, _lengths(entries), strict=True):
        per_day.setdefault(start.astimezone(zone).date(), []).append(
            (entry, length.total_seconds())
        )
    out = []
    for offset in range(days):
        day = today + timedelta(days=offset)
        items = per_day.get(day, [])
        summary: dict = {
            "date": day.isoformat(),
            "mean": None,
            "min": None,
            "max": None,
            "p90_max": None,
            "p10_min": None,
            "source": None,
        }
        if items:
            prices = [e["price"] for e, _ in items]
            total = sum(w for _, w in items)
            summary["mean"] = round(sum(e["price"] * w for e, w in items) / total, _DECIMALS)
            summary["min"] = min(prices)
            summary["max"] = max(prices)
            fc = [e for e, _ in items if e["source"] == "forecast"]
            if fc:
                summary["p90_max"] = max(e["p90"] for e in fc)
                summary["p10_min"] = min(e["p10"] for e in fc)
            sources = {e["source"] for e, _ in items}
            summary["source"] = sources.pop() if len(sources) == 1 else "mixed"
        out.append(summary)
    return out
