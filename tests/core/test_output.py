from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from sbp_adapter_copy import detect_adapter, epex_matches, nordpool_matches, price_factor_from_attrs

from core.forecast import Slot
from core.ingest import MarketSlot
from core.output import current_price, day_summaries, flag_window, prices_attribute

TZ = "Europe/Berlin"
BERLIN = ZoneInfo(TZ)
P90 = dict.fromkeys(range(1, 8), 10.0)
P10 = dict.fromkeys(range(1, 8), -10.0)


def _local_midnight(y, m, d):
    return datetime(y, m, d, tzinfo=BERLIN)


def _market(start_utc, count, minutes=15, price=50.0):
    out = []
    for i in range(count):
        s = start_utc + timedelta(minutes=minutes * i)
        out.append(MarketSlot(s, s + timedelta(minutes=minutes), price + i))
    return out


def _forecast(start_utc, count, valid=True, median=60.0):
    return [
        Slot(
            start_utc + timedelta(hours=i),
            h=1 + i // 24,
            median=median,
            p90_raw=median + 20,
            p10_raw=median - 20,
            valid=valid,
        )
        for i in range(count)
    ]


def _instants(entries):
    return [datetime.fromisoformat(e["time"]).astimezone(UTC) for e in entries]


def test_prices_attribute_market_then_forecast_no_gap_no_overlap():
    # market: local 2026-03-10 00:00 .. 12:00 (UTC+1) ; forecast hours start earlier too
    m0 = datetime(2026, 3, 9, 23, tzinfo=UTC)
    market = _market(m0, 48)  # 12 h
    forecast = _forecast(m0 + timedelta(hours=6), 24)  # first 6 overlap
    start = _local_midnight(2026, 3, 10)
    out = prices_attribute(market, forecast, P90, P10, TZ, start)
    assert [e["source"] for e in out] == ["market"] * 48 + ["forecast"] * 18
    inst = _instants(out)
    assert inst == sorted(set(inst))
    assert inst[48] == m0 + timedelta(hours=12)
    f = out[48]
    assert f["price"] == 0.06
    assert f["p90"] == 0.09 and f["p10"] == 0.03 and f["h"] == 1
    assert out[0] == {"time": "2026-03-10T00:00:00+01:00", "price": 0.05, "source": "market"}


def test_prices_attribute_drops_market_before_start():
    m0 = datetime(2026, 3, 9, 22, tzinfo=UTC)
    out = prices_attribute(_market(m0, 8), [], P90, P10, TZ, _local_midnight(2026, 3, 10))
    assert len(out) == 4
    assert out[0]["time"] == "2026-03-10T00:00:00+01:00"


def test_prices_attribute_skips_invalid_forecast_slots():
    m0 = datetime(2026, 3, 9, 23, tzinfo=UTC)
    f = _forecast(m0, 3)
    f[1] = Slot(f[1].time, f[1].h, 1.0, 2.0, 0.0, False)
    out = prices_attribute([], f, P90, P10, TZ, _local_midnight(2026, 3, 10))
    assert len(out) == 2
    assert _instants(out) == [f[0].time, f[2].time]


def test_prices_attribute_prices_rounded_to_5_decimals():
    m = [
        MarketSlot(
            datetime(2026, 3, 9, 23, tzinfo=UTC), datetime(2026, 3, 10, tzinfo=UTC), 33.3333333
        )
    ]
    out = prices_attribute(m, [], P90, P10, TZ, _local_midnight(2026, 3, 10))
    assert out[0]["price"] == 0.03333


def test_prices_attribute_dst_fall_back_market():
    m0 = datetime(2026, 10, 24, 22, tzinfo=UTC)  # local midnight, CEST
    market = _market(m0, 100)
    out = prices_attribute(market, [], P90, P10, TZ, _local_midnight(2026, 10, 25))
    assert len(out) == 100
    times = [e["time"] for e in out]
    assert len(set(times)) == 100
    assert "2026-10-25T02:00:00+02:00" in times and "2026-10-25T02:00:00+01:00" in times
    assert len(set(_instants(out))) == 100


def test_prices_attribute_dst_fall_back_forecast():
    market = _market(datetime(2026, 10, 23, 22, tzinfo=UTC), 96)  # all of 2026-10-24
    forecast = _forecast(datetime(2026, 10, 24, 22, tzinfo=UTC), 25)
    out = prices_attribute(market, forecast, P90, P10, TZ, _local_midnight(2026, 10, 24))
    fc = [e for e in out if e["source"] == "forecast"]
    assert len(fc) == 25
    assert len({e["time"] for e in fc}) == 25
    assert len(set(_instants(out))) == len(out)


def test_prices_attribute_is_parsed_by_sbp_entsoe_adapter():
    m0 = datetime(2026, 3, 9, 23, tzinfo=UTC)
    market = _market(m0, 8)  # 2 h quarter-hourly
    forecast = _forecast(m0 + timedelta(hours=2), 5)
    out = prices_attribute(market, forecast, P90, P10, TZ, _local_midnight(2026, 3, 10))
    attrs = {"prices": out, "unit": "EUR/kWh"}
    assert not nordpool_matches(attrs)
    assert not epex_matches(attrs)
    adapter = detect_adapter(attrs)
    assert adapter is not None and adapter.name == "entsoe"
    assert price_factor_from_attrs(attrs) == 1.0
    now = datetime(2026, 3, 9, 23, tzinfo=UTC).astimezone(BERLIN)
    slots = adapter.parse(attrs, now)
    assert len(slots) == len(out) == 13
    assert slots[7].hours == 0.25
    assert slots[8].hours == 1.0
    assert [s.price for s in slots] == [e["price"] for e in out]


def test_current_price():
    m0 = datetime(2026, 3, 10, 12, tzinfo=UTC)
    market = _market(m0, 4)
    assert current_price(market, m0 + timedelta(minutes=20)) == 0.051
    assert current_price(market, m0 + timedelta(minutes=15)) == 0.051  # start inclusive
    assert current_price(market, m0 + timedelta(hours=1)) is None  # end exclusive
    assert current_price(market, m0 - timedelta(minutes=1)) is None


def _e(local_iso, price, source="forecast", **extra):
    return {"time": local_iso, "price": price, "source": source, **extra}


def test_flag_window_peak_counts_market_and_forecast():
    now = datetime(2026, 3, 10, 12, 10, tzinfo=BERLIN)
    entries = [
        _e("2026-03-10T11:45:00+01:00", 0.30, "market"),  # ended before now
        _e("2026-03-10T12:00:00+01:00", 0.25, "market"),  # running slot, flagged
        _e("2026-03-10T12:15:00+01:00", 0.10, "market"),
        _e("2026-03-10T13:00:00+01:00", 0.10, "forecast", p90=0.25, p10=0.0),  # flagged
        _e("2026-03-10T14:00:00+01:00", 0.30, "forecast", p90=0.20, p10=0.0),  # >= flagged
        _e("2026-03-10T15:00:00+01:00", 0.10, "forecast", p90=0.19, p10=0.0),
        _e("2026-03-12T14:00:00+01:00", 0.10, "forecast", p90=0.90, p10=0.0),  # beyond 48 h
    ]
    flagged, certain = flag_window(entries, now, key="p90", threshold=0.20, above=True)
    assert [e["time"] for e in flagged] == [
        "2026-03-10T12:00:00+01:00",
        "2026-03-10T13:00:00+01:00",
        "2026-03-10T14:00:00+01:00",
    ]
    assert certain is True
    flagged, certain = flag_window(entries[3:], now, key="p90", threshold=0.20, above=True)
    assert len(flagged) == 2 and certain is False
    assert flag_window(entries, now, key="p90", threshold=5.0, above=True) == ([], False)


def test_flag_window_negative_uses_p10():
    now = datetime(2026, 3, 10, 12, 0, tzinfo=BERLIN)
    entries = [
        _e("2026-03-10T12:00:00+01:00", 0.0, "market"),  # 0 is not negative
        _e("2026-03-10T13:00:00+01:00", 0.05, "forecast", p90=0.1, p10=-0.01),  # flagged via p10
        _e("2026-03-10T14:00:00+01:00", -0.02, "forecast", p90=0.1, p10=-0.05),
        _e("2026-03-10T15:00:00+01:00", 0.05, "forecast", p90=0.1, p10=0.0),
    ]
    flagged, certain = flag_window(entries, now, key="p10", threshold=0.0, above=False)
    assert [e["time"][11:13] for e in flagged] == ["13", "14"]
    assert certain is False
    entries.insert(1, _e("2026-03-10T12:15:00+01:00", -0.01, "market"))
    flagged, certain = flag_window(entries, now, key="p10", threshold=0.0, above=False)
    assert certain is True and flagged[0]["source"] == "market"


def test_day_summaries_seven_days():
    today = date(2026, 3, 10)
    entries = []
    # day 0: market only, hourly 1.0 then 3.0 (mean 2.0)
    entries += [
        _e("2026-03-10T00:00:00+01:00", 0.10, "market"),
        _e("2026-03-10T01:00:00+01:00", 0.30, "market"),
    ]
    # day 1: mixed: 1 h market at 0.10, 1 h forecast with p90/p10
    entries += [
        _e("2026-03-11T22:00:00+01:00", 0.10, "market"),
        _e("2026-03-11T23:00:00+01:00", 0.20, "forecast", p90=0.4, p10=-0.1),
    ]
    entries += [_e("2026-03-12T10:00:00+01:00", 0.20, "forecast", p90=0.5, p10=0.0)]
    out = day_summaries(entries, TZ, today)
    assert [d["date"] for d in out] == [f"2026-03-{d}" for d in range(10, 17)]
    assert out[0] == {
        "date": "2026-03-10",
        "mean": 0.2,
        "min": 0.1,
        "max": 0.3,
        "p90_max": None,
        "p10_min": None,
        "source": "market",
    }
    assert out[1]["source"] == "mixed" and out[1]["mean"] == 0.15
    assert out[1]["p90_max"] == 0.4 and out[1]["p10_min"] == -0.1
    assert out[2]["source"] == "forecast"
    assert out[3] == {
        "date": "2026-03-13",
        "mean": None,
        "min": None,
        "max": None,
        "p90_max": None,
        "p10_min": None,
        "source": None,
    }
    assert len(out) == 7


def test_day_summaries_weights_by_slot_length():
    # 15-min slots at 0.0 for 1 h, then two hourly slots at 0.4: weighted 0.8/3 (unweighted 0.133)
    entries = [
        _e(f"2026-03-10T0{i // 4}:{(i % 4) * 15:02d}:00+01:00", 0.0, "market") for i in range(4)
    ]
    entries.append(_e("2026-03-10T01:00:00+01:00", 0.4, "market"))
    entries.append(_e("2026-03-10T02:00:00+01:00", 0.4, "market"))
    out = day_summaries(entries, TZ, date(2026, 3, 10), days=1)
    assert out[0]["mean"] == 0.26667
