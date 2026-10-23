"""Home Assistant states and statistics for the card screenshots, made by the real core code.

Each scenario is a fixed past moment on research data. The forecast comes from
``core.forecast.run_forecast`` with the bundled model, the ``prices`` attribute from
``core.output.prices_attribute`` and the long-term statistics from ``Ledger.finalize``
over the backfill, as the integration builds them: the bundled backfill file for target
days before the model's training cutoff, forecasts recomputed from archived weather for
the days after it. Nothing is hand-written: a hand-made price curve would drift away from
what the integration shows.

The screenshot moments lie after the bundled model's training cutoff, so every forecast
on them is out of sample, as on a real install: a forecast for days the model was
trained on would look better than the integration can be.

Weather for a past anchor comes from the Open-Meteo previous-runs archive, as in training:
horizon ``h`` reads the run made ``h + 1`` days before the target day.

    scenario("default") -> {"now", "now_ms", "states", "statistics", "history"}

Research data is read from ``SPOT_RESEARCH_DIR`` (default ``research/``), see
research/README.md. The scenarios are slow to build (the weather archive is large), so
``gen_card_screenshots.py`` caches them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "custom_components" / "spot_forecast"), str(REPO)]

from const import (  # noqa: E402
    DEFAULT_BACKFILL_DAYS,
    DOMAIN,
    MAX_BACKFILL_DAYS,
    PRICE_HISTORY_DAYS,
)

from core.features import HORIZONS, Hourly, InsufficientHistory  # noqa: E402
from core.forecast import run_forecast  # noqa: E402
from core.ingest import MarketSlot, hourly_prices, parse_energy_charts  # noqa: E402
from core.ledger import ISSUE_HOUR, Ledger, Run, runs_from_backfill_file  # noqa: E402
from core.model import ModelFile, load_model  # noqa: E402
from core.output import current_price, flag_window, prices_attribute  # noqa: E402
from core.schedule import choose_anchor  # noqa: E402
from core.zones import DE_LU, Zone  # noqa: E402

RESEARCH = Path(os.environ.get("SPOT_RESEARCH_DIR", REPO / "research"))
MODELS = REPO / "custom_components" / "spot_forecast" / "models"
PRICE_SENSOR = "sensor.spot_forecast_de_lu_price"
PEAK_SENSOR = "binary_sensor.spot_forecast_de_lu_peak_risk"
# Where "sun.sun" is computed: the middle of Germany. Only the night shading uses it.
SUN_LAT, SUN_LON = 51.16, 10.45
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_EXPIRED = timedelta(hours=40)  # past the 36 h after which the forecast is dropped


@dataclass(frozen=True)
class Spec:
    now: datetime  # local wall clock of the screenshot
    backfill_days: int  # target days of the backfill window (option backfill_days)
    expired: bool = False  # the last successful run is too old: market prices only
    surcharge: float = 0.0  # option "grid fees and taxes", EUR/kWh


def _local(*args: int) -> datetime:
    return datetime(*args, tzinfo=ZoneInfo(DE_LU.tz))


SCENARIOS: dict[str, Spec] = {
    # After the day-ahead publication: market through tomorrow (10-15), forecast for six
    # days (10-16 … 10-21), and the look-back of an install with the longest backfill
    # window. Two weeks after the bundled model's cutoff (2026-10-01), so the forecast is
    # out of sample; anchor 10-15 is the latest the research weather archive covers for
    # all six days (it lacks the run for D+6 of anchor 10-16).
    "default": Spec(_local(2026, 10, 14, 14, 30), backfill_days=MAX_BACKFILL_DAYS),
    # 2026-10-25 (fall-back) is past the research data; the 2025 fall-back day is the
    # nearest one with prices and archived weather. It is "today", in 15-minute slots.
    "dst": Spec(_local(2025, 10, 26, 14, 30), backfill_days=0),
    # The same fall-back day as the first forecast day, in hourly forecast slots.
    "dst_forecast": Spec(_local(2025, 10, 24, 14, 30), backfill_days=0),
    # The weather source failed for two days: the forecast is dropped, prices remain.
    "stale": Spec(_local(2026, 10, 14, 14, 30), backfill_days=MAX_BACKFILL_DAYS, expired=True),
    # Installed today with the default backfill window.
    "fresh_install": Spec(_local(2026, 10, 14, 14, 30), backfill_days=DEFAULT_BACKFILL_DAYS),
    # The default day with grid fees and taxes set: the forecast card's "mit Abgaben" view.
    "fees": Spec(_local(2026, 10, 14, 14, 30), backfill_days=0, surcharge=0.187),
}


@cache
def _model() -> ModelFile:
    return load_model((MODELS / f"{DE_LU.slug}.json.gz").read_bytes())


@cache
def _market(code: str) -> tuple[MarketSlot, ...]:
    """Every day-ahead slot in the research price files of zone ``code``, native resolution."""
    slots: dict[datetime, MarketSlot] = {}
    for f in sorted((RESEARCH / "data" / "raw" / "prices" / code).glob("*.json")):
        for s in parse_energy_charts(json.loads(f.read_text())):
            slots.setdefault(s.start, s)
    return tuple(slots[k] for k in sorted(slots))


@cache
def _weather():
    """Previous-runs weather by forecast offset, exactly as training loads it."""
    from training.dataset import _load_weather

    return _load_weather(RESEARCH / "data" / "raw", DE_LU)


def _midnight(day: date, zone: Zone) -> datetime:
    return datetime.combine(day, time(), ZoneInfo(zone.tz)).astimezone(UTC)


def _sun(now: datetime) -> dict:
    """``sun.sun`` as Home Assistant shows it at ``now``: the next rising and setting."""

    def events(day: date) -> tuple[datetime, datetime]:
        # NOAA sunrise equation (https://en.wikipedia.org/wiki/Sunrise_equation)
        n = day.toordinal() - date(2000, 1, 1).toordinal()
        j_star = n - SUN_LON / 360
        m = math.radians((357.5291 + 0.98560028 * j_star) % 360)
        c = 1.9148 * math.sin(m) + 0.02 * math.sin(2 * m) + 0.0003 * math.sin(3 * m)
        lam = math.radians((math.degrees(m) + c + 180 + 102.9372) % 360)
        transit = j_star + 0.0053 * math.sin(m) - 0.0069 * math.sin(2 * lam)
        dec = math.asin(math.sin(lam) * math.sin(math.radians(23.4397)))
        lat = math.radians(SUN_LAT)
        cos_w = (math.sin(math.radians(-0.833)) - math.sin(lat) * math.sin(dec)) / (
            math.cos(lat) * math.cos(dec)
        )
        w = math.degrees(math.acos(cos_w)) / 360
        epoch = datetime(2000, 1, 1, 12, tzinfo=UTC)
        return epoch + timedelta(days=transit - w), epoch + timedelta(days=transit + w)

    today = now.astimezone(UTC).date()
    days = [today + timedelta(days=i) for i in range(-1, 3)]
    rises = [events(d)[0] for d in days]
    sets = [events(d)[1] for d in days]
    next_rising = min(t for t in rises if t > now)
    next_setting = min(t for t in sets if t > now)
    above = next_setting < next_rising
    return {
        "entity_id": "sun.sun",
        "state": "above_horizon" if above else "below_horizon",
        "attributes": {
            "next_rising": next_rising.isoformat(),
            "next_setting": next_setting.isoformat(),
            "rising": not above,
            "friendly_name": "Sun",
        },
        "last_changed": now.astimezone(UTC).isoformat(),
        "last_updated": now.astimezone(UTC).isoformat(),
    }


def _state(entity_id: str, state: str, attributes: dict, now: datetime) -> dict:
    stamp = now.astimezone(UTC).isoformat()
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": attributes,
        "last_changed": stamp,
        "last_updated": stamp,
    }


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def _statistics(rows) -> dict[str, list[dict]]:
    """Ledger rows as ``recorder/statistics_during_period`` returns them (start/end in ms)."""
    out: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: (r.statistic_id, r.start)):
        out.setdefault(r.statistic_id, []).append(
            {
                "start": _ms(r.start),
                "end": _ms(r.start + _HOUR),
                "mean": r.value,
                "min": r.value,
                "max": r.value,
                "state": r.value,
            }
        )
    return out


def _backfill_runs(zone: Zone, model: ModelFile, first: date, today: date) -> list[Run]:
    """The bundled walk-forward runs whose target days lie in [first, today)."""
    data = (MODELS / f"{zone.slug}.backfill.json.gz").read_bytes()
    runs = []
    for run in runs_from_backfill_file(data, model.version, zone.tz):
        slots = tuple(s for s in run.slots if first <= run.anchor + timedelta(days=s.h) < today)
        if slots:
            runs.append(replace(run, slots=slots))
    return runs


def _recomputed_runs(
    zone: Zone, model: ModelFile, hourly: Hourly, weather, first: date, today: date
) -> list[Run]:
    """Runs for the target days in [max(cutoff, first), today), forecast from archived weather.

    What the integration's backfill does on the device for the days after the training
    cutoff (``backfill._recompute``): each anchor sees the prices published by its end and
    the previous-runs weather of training.
    """
    start = max(model.train_cutoff, first)
    runs = []
    for i in range((today - start).days + max(HORIZONS)):
        anchor = start + (i - max(HORIZONS)) * _DAY
        end = _midnight(anchor + _DAY, zone)
        known = Hourly(hourly.start, hourly.values[: max((end - hourly.start) // _HOUR, 0)])
        try:
            slots = run_forecast(model, anchor, known, {h: weather[h + 1] for h in HORIZONS}, zone)
        except InsufficientHistory:
            continue
        kept = tuple(s for s in slots if start <= anchor + timedelta(days=s.h) < today)
        if kept:
            issued = datetime.combine(anchor, time(ISSUE_HOUR), ZoneInfo(zone.tz)).astimezone(UTC)
            runs.append(Run(anchor, issued, model.version, kept, True))
    return runs


def _merge(runs: list[Run]) -> list[Run]:
    """One run per anchor: an anchor before the cutoff has bundled and recomputed slots."""
    merged: dict[date, Run] = {}
    for run in runs:
        old = merged.get(run.anchor)
        merged[run.anchor] = run if old is None else replace(old, slots=old.slots + run.slots)
    return [merged[a] for a in sorted(merged)]


def build(spec: Spec, zone: Zone = DE_LU) -> dict:
    model = _model()
    now = spec.now.astimezone(UTC)
    today = spec.now.date()

    # What the price source has published by now: through tomorrow after ~13:00.
    published_until = _midnight(today + 2 * _DAY if spec.now.hour >= 13 else today + _DAY, zone)
    market = [m for m in _market(zone.code) if m.start < published_until]
    anchor = choose_anchor(now, market, zone.tz)
    if (anchor - today).days not in (0, 1):
        raise ValueError(f"research prices do not cover {today}")
    hourly = hourly_prices(market)

    # The ledger holds the prices of the last PRICE_HISTORY_DAYS, or of the backfill window.
    ledger = Ledger()
    first = today - timedelta(days=max(spec.backfill_days, PRICE_HISTORY_DAYS))
    i = max((_midnight(first, zone) - hourly.start) // _HOUR, 0)
    ledger.record_actual(Hourly(hourly.start + i * _HOUR, hourly.values[i:]))
    weather = _weather()
    if spec.backfill_days:
        first = today - timedelta(days=spec.backfill_days)
        bundled = _backfill_runs(zone, model, first, min(today, model.train_cutoff))
        for run in _merge(bundled + _recomputed_runs(zone, model, hourly, weather, first, today)):
            ledger.add_run(run)

    slots = run_forecast(model, anchor, hourly, {h: weather[h + 1] for h in HORIZONS}, zone)
    issued_at = now - (_EXPIRED if spec.expired else timedelta(minutes=10))
    ledger.add_run(Run(anchor, issued_at, model.version, tuple(slots), False))
    if spec.expired:
        # the weather source has failed since: the last run is too old to show
        slots = []

    p90_off, p10_off = ledger.offsets(anchor, model.p90_offsets, model.p10_offsets)
    rows = ledger.finalize(
        zone.tz,
        today=today,
        domain=DOMAIN,
        zone_id=zone.slug,
        default_p90=model.p90_offsets,
        default_p10=model.p10_offsets,
    )
    entries = prices_attribute(market, slots, p90_off, p10_off, zone.tz, _midnight(today, zone))
    if sum(s.valid for s in slots) != len(slots):
        raise ValueError(f"weather archive incomplete for anchor {anchor}")

    threshold = round(round(model.peak_threshold / 10, 1) / 100, 5)  # EUR/kWh, as the sensor
    flagged, certain = flag_window(entries, now, key="p90", threshold=threshold, above=True)
    price = current_price(market, now)
    states = {
        PRICE_SENSOR: _state(
            PRICE_SENSOR,
            str(price),
            {
                "prices": entries,
                "unit": "EUR/kWh",
                "issued_at": issued_at.isoformat(),
                "anchor_day": anchor.isoformat(),
                "model_version": model.version,
                "stale": spec.expired,
                "surcharge": spec.surcharge,
                "state_class": "measurement",
                "unit_of_measurement": "EUR/kWh",
                "attribution": f"Prices: {zone.price_attribution}; weather: Open-Meteo (CC BY 4.0)",
                "friendly_name": f"Spot Forecast {zone.code} Preis",
            },
            spec.now,
        ),
        PEAK_SENSOR: _state(
            PEAK_SENSOR,
            "on" if flagged else "off",
            {
                "slots": flagged,
                "certain": certain,
                "threshold": threshold,
                "friendly_name": f"Spot Forecast {zone.code} Preisspitzen-Risiko",
            },
            spec.now,
        ),
        "sun.sun": _sun(now),
    }
    # The sensor's own recorded history (state only; ``prices`` is not recorded),
    # in the compressed form of history/history_during_period.
    history = [
        {"s": str(round(m.price / 1000, 5)), "lu": m.start.timestamp()}
        for m in market
        if now - 3 * _DAY <= m.start <= now
    ]
    return {
        "now": spec.now.isoformat(),
        "now_ms": _ms(now),
        "time_zone": zone.tz,
        "local_today": today.isoformat(),
        "states": states,
        "statistics": _statistics(rows),
        "history": {PRICE_SENSOR: history},
    }


def scenario(name: str) -> dict:
    """The states and statistics of scenario ``name`` (see ``SCENARIOS``)."""
    return build(SCENARIOS[name])
