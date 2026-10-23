"""Offline checks of the fetch helpers (no network)."""

from datetime import UTC, date, datetime, timedelta
import json

import pytest

pytest.importorskip("requests")
from training import fetch
from training.fetch import _is_complete, month_chunks


def test_month_chunks_clip_to_range():
    assert list(month_chunks(date(2024, 3, 5), date(2024, 5, 10))) == [
        (date(2024, 3, 5), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 5, 1), date(2024, 5, 10)),
    ]
    assert list(month_chunks(date(2024, 1, 1), date(2024, 12, 31), months=6))[1] == (
        date(2024, 7, 1),
        date(2024, 12, 31),
    )


def _hours(first: date, last: datetime) -> list[datetime]:
    t, out = datetime(first.year, first.month, first.day, tzinfo=UTC), []
    while t <= last:
        out.append(t)
        t += timedelta(hours=1)
    return out


def _price_doc(first: date, last: datetime, tail_nulls: int = 0) -> dict:
    """Energy-Charts layout; the API includes 00:00 UTC after the last requested day."""
    times = _hours(first, last)
    prices = [50.0] * (len(times) - tail_nulls) + [None] * tail_nulls
    return {"unix_seconds": [int(t.timestamp()) for t in times], "price": prices}


def _weather_doc(first: date, last: datetime) -> dict:
    """Open-Meteo layout with ``timezone=GMT``: hours labelled at their start."""
    times = _hours(first, last)
    return {
        "utc_offset_seconds": 0,
        "hourly": {
            "time": [f"{t:%Y-%m-%dT%H:%M}" for t in times],
            "wind_speed_100m_previous_day2": [5.0] * len(times),
        },
    }


def _write(path, doc) -> None:
    path.write_text(json.dumps(doc))


def test_chunk_is_complete_only_when_the_file_reaches_the_chunk_end(tmp_path):
    f = tmp_path / "x.json"
    end = date(2024, 6, 1)
    assert not _is_complete(f, date(2024, 1, 31), end)  # missing file
    _write(f, {})
    assert not _is_complete(f, date(2024, 1, 31), end)  # no data at all
    f.write_text("{trunc")
    assert not _is_complete(f, date(2024, 1, 31), end)  # unreadable: refetch, do not crash

    _write(f, _price_doc(date(2024, 1, 1), datetime(2024, 2, 1, tzinfo=UTC)))
    assert _is_complete(f, date(2024, 1, 31), end)
    # a chunk that touches the last few days is refetched even when complete
    assert not _is_complete(f, date(2024, 5, 30), end)

    _write(f, _price_doc(date(2024, 1, 1), datetime(2024, 1, 21, tzinfo=UTC)))
    assert not _is_complete(f, date(2024, 1, 31), end)  # fetched while the chunk was open
    _write(f, _price_doc(date(2024, 1, 1), datetime(2024, 2, 1, tzinfo=UTC), tail_nulls=30))
    assert not _is_complete(f, date(2024, 1, 31), end)  # timestamps without prices

    _write(f, _weather_doc(date(2024, 1, 1), datetime(2024, 1, 31, 23, tzinfo=UTC)))
    assert _is_complete(f, date(2024, 1, 31), end)
    _write(f, _weather_doc(date(2024, 1, 1), datetime(2024, 1, 21, 23, tzinfo=UTC)))
    assert not _is_complete(f, date(2024, 1, 31), end)


def test_chunk_fetched_while_open_is_refetched_later(tmp_path, monkeypatch):
    """Review scenario: a run on 2026-10-22 caches the open 2026-07 half-year; a run on
    2027-01-06 must fetch it again instead of keeping a gap from 2026-10-22 on."""
    today = {"end": date(2026, 10, 21)}
    requested: list[str] = []

    def fake_get(url, params, retries=5):
        start = date.fromisoformat(params["start"][:10])
        stop = datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
        available = datetime.combine(today["end"] + timedelta(days=1), datetime.min.time(), UTC)
        requested.append(params["start"][:10])
        return _price_doc(start, min(stop, available))

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    fetch.fetch_prices(tmp_path, today["end"], start=date(2026, 3, 1))
    assert "2026-07-01" in requested

    requested.clear()
    today["end"] = date(2027, 1, 5)
    fetch.fetch_prices(tmp_path, today["end"], start=date(2026, 3, 1))
    assert requested == ["2026-07-01", "2027-01-01"]  # the 2026-01 half-year stays cached
    doc = json.loads((tmp_path / "prices" / "DE-LU" / "2026-07_6m.json").read_text())
    assert datetime.fromtimestamp(doc["unix_seconds"][-1], UTC) == datetime(2027, 1, 1, tzinfo=UTC)


class _Ok:
    status_code = 200

    @staticmethod
    def json() -> dict:
        return {"ok": True}


def test_get_retries_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise fetch.requests.Timeout("read timed out")
        return _Ok()

    monkeypatch.setattr(fetch.requests, "get", flaky)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    assert fetch._get("https://example.org", {}) == {"ok": True}
    assert len(calls) == 2


def test_get_gives_up_after_the_last_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_args, **_kwargs):
        raise fetch.requests.ConnectionError("unreachable")

    monkeypatch.setattr(fetch.requests, "get", down)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    with pytest.raises(fetch.requests.ConnectionError):
        fetch._get("https://example.org", {}, retries=3)
