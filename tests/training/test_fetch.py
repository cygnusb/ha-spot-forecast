"""Offline checks of the fetch helpers (no network)."""

from datetime import UTC, date, datetime, timedelta
import json

import pytest

pytest.importorskip("requests")
from core.zones import DE_LU, NL
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
    fetch.fetch_prices(tmp_path, today["end"], DE_LU, start=date(2026, 3, 1))
    assert "2026-07-01" in requested

    requested.clear()
    today["end"] = date(2027, 1, 5)
    fetch.fetch_prices(tmp_path, today["end"], DE_LU, start=date(2026, 3, 1))
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


def test_weather_sets_fetch_every_point_once():
    """DE-LU's two requests as before, then only the five NL points it does not share."""
    sets = fetch.weather_sets([DE_LU, NL])
    assert sets[:2] == DE_LU.point_sets()
    assert len(sets) == 3
    assert list(sets[2][0]) == ["nl_borssele", "nl_flevoland", "nl_groningen", "nl_randstad",
                                "nl_brabant"]  # fmt: skip
    assert sets[2][1] == ("wind_speed_100m", "temperature_2m", "cloud_cover")
    names = [p for points, _ in fetch.weather_sets([NL, DE_LU]) for p in points]
    assert len(names) == len(set(names)) == 27


def test_fetch_all_by_zone(tmp_path, monkeypatch):
    """Prices per zone into prices/<code>, installed capacity only for a zone that uses it."""
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params, retries=5):
        calls.append((url.rsplit("/", 1)[-1], params))
        if url.endswith("/price"):
            return _price_doc(date(2026, 9, 1), datetime(2026, 9, 2, tzinfo=UTC))
        return {"time": [], "production_types": []}

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch, "fetch_weather", lambda *a, **k: None)
    fetch.fetch_all(tmp_path, date(2026, 9, 30), [NL])
    assert {c[0] for c in calls} == {"price"} and {c[1]["bzn"] for c in calls} == {"NL"}
    assert (tmp_path / "prices" / "NL").is_dir() and not (
        tmp_path / "installed_power.json"
    ).exists()
    calls.clear()
    fetch.fetch_all(tmp_path, date(2026, 9, 30), [DE_LU, NL])
    assert calls[0][0] == "installed_power"
    assert [c[1]["bzn"] for c in calls if c[0] == "price"][-1] == "NL"


def test_zones_option():
    assert fetch._zones("NL,DE-LU,NL") == [NL, DE_LU]
    with pytest.raises(Exception, match="unknown zones"):
        fetch._zones("NL,AT")
