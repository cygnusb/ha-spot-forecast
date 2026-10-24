"""Key-free HTTP sources and the bundled model, against recorded payloads."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import gzip
import json
import re

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)
from yarl import URL

from custom_components.spot_forecast.core.ingest import parse_open_meteo
from custom_components.spot_forecast.core.zones import DE_LU, NO2
from custom_components.spot_forecast.model_loader import load_bundled
from custom_components.spot_forecast.sources import energy_charts, http, open_meteo
from custom_components.spot_forecast.sources.energy_charts import fetch_prices
from custom_components.spot_forecast.sources.http import (
    SourceError,
    SourceRateLimited,
    get_body,
)
from custom_components.spot_forecast.sources.open_meteo import (
    check_hydro_history,
    fetch_forecast,
    fetch_hydro_history,
    fetch_previous_runs,
)

EC_URL = "https://api.energy-charts.info/price"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TEST_URL = "https://example.invalid/data"


@pytest.fixture
def pauses(monkeypatch) -> list[float]:
    """Record the back-off and pacing pauses instead of sleeping."""
    recorded: list[float] = []

    async def pause(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(http, "pause", pause)
    return recorded


def _latitudes(points) -> str:
    return ",".join(str(lat) for lat, _ in points.values())


async def test_fetch_prices_parses_fixture(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, recorded
) -> None:
    aioclient_mock.get(
        EC_URL,
        params={"bzn": "DE-LU", "start": "2026-10-19", "end": "2026-10-22"},
        json=recorded("energy_charts_price.json"),
    )

    slots = await fetch_prices(
        async_get_clientsession(hass),
        "DE-LU",
        date(2026, 10, 19),
        date(2026, 10, 22),
        run_blocking=hass.async_add_executor_job,
    )

    assert aioclient_mock.call_count == 1
    assert len(slots) == 4 * 96
    assert slots[0].start == datetime(2026, 10, 18, 22, tzinfo=UTC)
    assert slots[-1].end == datetime(2026, 10, 22, 22, tzinfo=UTC)
    assert all(s.end - s.start == timedelta(minutes=15) for s in slots)
    assert all(isinstance(s.price, float) for s in slots)


async def test_fetch_prices_invalid_body_is_source_error(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(EC_URL, text="<html>maintenance</html>")

    with pytest.raises(SourceError, match="invalid JSON"):
        await fetch_prices(
            async_get_clientsession(hass),
            "DE-LU",
            date(2026, 10, 19),
            date(2026, 10, 22),
            run_blocking=hass.async_add_executor_job,
        )


async def test_fetch_forecast_two_requests_keys(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, recorded
) -> None:
    common = {"models": "ecmwf_ifs025", "forecast_days": "8", "timezone": "GMT"}
    aioclient_mock.get(
        FORECAST_URL,
        params={"latitude": _latitudes(DE_LU.points), **common},
        json=recorded("open_meteo_de.json"),
    )
    aioclient_mock.get(
        FORECAST_URL,
        params={"latitude": _latitudes(DE_LU.neighbor_points), **common},
        json=recorded("open_meteo_nb.json"),
    )

    weather = await fetch_forecast(
        async_get_clientsession(hass), DE_LU, run_blocking=hass.async_add_executor_job
    )

    assert aioclient_mock.call_count == 2
    hourly = [URL(str(url)).query["hourly"] for _, url, _, _ in aioclient_mock.mock_calls]
    assert hourly == [",".join(DE_LU.weather_vars), ",".join(DE_LU.neighbor_vars)]
    expected = {f"{p}__{v}" for p in DE_LU.points for v in DE_LU.weather_vars} | {
        f"{p}__{v}" for p in DE_LU.neighbor_points for v in DE_LU.neighbor_vars
    }
    assert set(weather) == expected
    series = weather["north_sea__wind_speed_100m"]
    assert series.start == datetime(2026, 10, 22, tzinfo=UTC)
    assert len(series.values) == 8 * 24
    assert "nb_nl__precipitation" in weather


async def test_get_body_retries_429_then_succeeds(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    statuses = iter([429, 200])

    async def respond(method, url, data):
        status = next(statuses)
        return AiohttpClientMockResponse(
            method, url, status=status, json={"ok": True} if status == 200 else None
        )

    aioclient_mock.get(TEST_URL, side_effect=respond)

    result = await get_body(async_get_clientsession(hass), TEST_URL, {"a": "1"})

    assert json.loads(result) == {"ok": True}
    assert aioclient_mock.call_count == 2
    assert pauses == [30]


async def test_get_body_gives_up_after_three_attempts(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=503)

    with pytest.raises(SourceError) as err:
        await get_body(async_get_clientsession(hass), TEST_URL, {})

    assert not isinstance(err.value, SourceRateLimited)
    assert aioclient_mock.call_count == 3
    assert pauses == [2, 4]


async def test_get_body_rate_limited_raises_subclass(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=429)

    with pytest.raises(SourceRateLimited):
        await get_body(async_get_clientsession(hass), TEST_URL, {})

    assert pauses == [30, 30]


async def test_get_body_client_error_is_not_retried(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=400)

    with pytest.raises(SourceError):
        await get_body(async_get_clientsession(hass), TEST_URL, {})

    assert aioclient_mock.call_count == 1
    assert pauses == []


@pytest.mark.parametrize(
    ("status", "retry_after", "expected"),
    [
        (429, "90", [90]),
        (429, "5", [30]),  # never less than after a 429 without the header
        (429, "3600", [120]),  # capped
        (429, "Wed, 21 Oct 2026 07:28:00 GMT", [30]),  # the date form is not read
        (503, "10", [10]),
        (503, "1", [2]),  # the back-off when it is longer
        (500, "10", [2]),  # only 429 and 503
    ],
)
async def test_get_body_honours_retry_after(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    pauses: list[float],
    status: int,
    retry_after: str,
    expected: list[float],
) -> None:
    statuses = iter([status, 200])

    async def respond(method, url, data):
        code = next(statuses)
        if code == 200:
            return AiohttpClientMockResponse(method, url, json={"ok": True})
        return AiohttpClientMockResponse(
            method, url, status=code, headers={"Retry-After": retry_after}
        )

    aioclient_mock.get(TEST_URL, side_effect=respond)

    assert json.loads(await get_body(async_get_clientsession(hass), TEST_URL, {})) == {"ok": True}
    assert pauses == expected


async def test_open_meteo_400_reason_is_in_the_error(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    reason = "Cannot initialize WeatherVariable from invalid String value cloud_cover_previous_day9"
    aioclient_mock.get(FORECAST_URL, status=400, json={"error": True, "reason": reason})

    with pytest.raises(SourceError, match="HTTP 400: " + re.escape(reason)):
        await fetch_forecast(
            async_get_clientsession(hass), DE_LU, run_blocking=hass.async_add_executor_job
        )

    assert aioclient_mock.call_count == 1
    assert pauses == []


async def test_client_error_without_a_reason_keeps_the_plain_message(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=403, text="<html>" + "x" * 100_000 + "</html>")

    with pytest.raises(SourceError) as err:
        await get_body(async_get_clientsession(hass), TEST_URL, {})

    assert str(err.value) == f"{TEST_URL}: HTTP 403"


@pytest.mark.parametrize("source", ["prices", "forecast", "previous_runs", "hydro"])
async def test_sources_limit_the_body(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    pauses: list[float],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    session, run_blocking = async_get_clientsession(hass), hass.async_add_executor_job
    aioclient_mock.get(re.compile("^https://"), content=b"[" + b"0," * 1000 + b"0]")
    if source == "prices":
        monkeypatch.setattr(energy_charts, "MAX_PRICES_BYTES", 1000)
        fetch = fetch_prices(
            session, "DE-LU", date(2026, 10, 19), date(2026, 10, 22), run_blocking=run_blocking
        )
    else:
        monkeypatch.setattr(open_meteo, "MAX_WEATHER_BYTES", 1000)
        if source == "forecast":
            fetch = fetch_forecast(session, DE_LU, run_blocking=run_blocking)
        elif source == "hydro":
            fetch = fetch_hydro_history(session, NO2, run_blocking=run_blocking)
        else:
            fetch = fetch_previous_runs(
                session, DE_LU, date(2026, 10, 1), date(2026, 10, 2), run_blocking=run_blocking
            )

    with pytest.raises(SourceError, match="limit"):
        await fetch


async def test_get_body_within_limit(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, content=b"x" * 200_000)

    assert await get_body(async_get_clientsession(hass), TEST_URL, {}, 200_000) == b"x" * 200_000


async def test_get_body_rejects_a_declared_length_above_the_limit(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, content=b"x" * 10, headers={"Content-Length": "1000"})

    with pytest.raises(SourceError, match="limit"):
        await get_body(async_get_clientsession(hass), TEST_URL, {}, 999)

    assert aioclient_mock.call_count == 1  # not retried
    assert pauses == []


async def test_get_body_stops_reading_past_the_limit(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    """No Content-Length (chunked): the body is counted while it is read."""
    aioclient_mock.get(TEST_URL, content=b"x" * 200_001)

    with pytest.raises(SourceError, match="limit"):
        await get_body(async_get_clientsession(hass), TEST_URL, {}, 200_000)

    assert aioclient_mock.call_count == 1
    assert pauses == []


async def test_fetch_previous_runs_chunks_per_month(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    async def respond(method, url, data):
        """One document per requested point, every requested column constant 1.0."""
        q = url.query
        first = datetime.fromisoformat(q["start_date"])
        hours = ((datetime.fromisoformat(q["end_date"]) - first).days + 1) * 24
        hourly = {"time": [(first + timedelta(hours=i)).isoformat() for i in range(hours)]}
        hourly |= {column: [1.0] * hours for column in q["hourly"].split(",")}
        docs = [{"hourly": hourly} for _ in q["latitude"].split(",")]
        return AiohttpClientMockResponse(method, url, json=docs)

    aioclient_mock.get(re.compile(re.escape(PREVIOUS_URL)), side_effect=respond)

    weather = await fetch_previous_runs(
        async_get_clientsession(hass),
        DE_LU,
        date(2026, 9, 29),
        date(2026, 10, 2),
        run_blocking=hass.async_add_executor_job,
    )

    ranges = [
        (URL(str(url)).query["start_date"], URL(str(url)).query["end_date"])
        for _, url, _, _ in aioclient_mock.mock_calls
    ]
    assert ranges == [("2026-09-29", "2026-09-30")] * 2 + [("2026-10-01", "2026-10-02")] * 2
    assert pauses == [3, 3, 3]
    assert set(weather) == set(range(1, 7))
    columns = URL(str(aioclient_mock.mock_calls[0][1])).query["hourly"].split(",")
    assert "wind_speed_100m_previous_day2" in columns
    assert "cloud_cover_previous_day7" in columns
    series = weather[3]["bayern__temperature_2m"]
    assert series.start == datetime(2026, 9, 29, tzinfo=UTC)
    assert len(series.values) == 4 * 24
    assert (series.values == 1.0).all()
    assert "nb_fi__precipitation" in weather[6]


def test_bundled_model_loads() -> None:
    model, backfill = load_bundled("DE-LU")

    assert model.zone == DE_LU.slug
    assert model.peak_threshold > 0
    assert model.train_cutoff <= date(2026, 10, 22)
    assert gzip.decompress(backfill)


async def test_fetch_hydro_history_live_is_the_recorded_past_days_body(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float], recorded
) -> None:
    """One request; the ``_previous_day2`` columns of the six hydro points, in their order."""
    body = recorded("open_meteo_hydro_no2.json")
    aioclient_mock.get(PREVIOUS_URL, json=body)

    history = await fetch_hydro_history(
        async_get_clientsession(hass), NO2, run_blocking=hass.async_add_executor_job
    )

    assert aioclient_mock.call_count == 1 and pauses == []
    variables = ("precipitation", "temperature_2m")
    expected = parse_open_meteo(body, list(NO2.hydro_points), variables, "_previous_day2")
    assert list(history) == [f"{p}__{v}" for p in NO2.hydro_points for v in variables]
    for key, series in history.items():
        assert series.start == expected[key].start == datetime(2026, 7, 24, tzinfo=UTC)
        assert (series.values == expected[key].values).all()
    check_hydro_history(history, date(2026, 10, 24), NO2)  # recorded on 2026-10-24: complete


async def test_fetch_hydro_history_range_chunks_per_month(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    async def respond(method, url, data):
        q = url.query
        first = datetime.fromisoformat(q["start_date"])
        hours = ((datetime.fromisoformat(q["end_date"]) - first).days + 1) * 24
        hourly = {"time": [(first + timedelta(hours=i)).isoformat() for i in range(hours)]}
        hourly |= {column: [2.0] * hours for column in q["hourly"].split(",")}
        return AiohttpClientMockResponse(
            method, url, json=[{"hourly": hourly} for _ in q["latitude"].split(",")]
        )

    aioclient_mock.get(re.compile(re.escape(PREVIOUS_URL)), side_effect=respond)

    history = await fetch_hydro_history(
        async_get_clientsession(hass),
        NO2,
        date(2026, 7, 30),
        date(2026, 9, 2),
        run_blocking=hass.async_add_executor_job,
    )

    queries = [URL(str(url)).query for _, url, _, _ in aioclient_mock.mock_calls]
    assert [(q["start_date"], q["end_date"]) for q in queries] == [
        ("2026-07-30", "2026-07-31"), ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-02"),
    ]  # fmt: skip
    assert all("past_days" not in q and "forecast_days" not in q for q in queries)
    assert queries[0]["hourly"] == "precipitation_previous_day2,temperature_2m_previous_day2"
    assert pauses == [3, 3]
    series = history["no5_bergen__temperature_2m"]
    assert series.start == datetime(2026, 7, 30, tzinfo=UTC)
    assert len(series.values) == 35 * 24 and (series.values == 2.0).all()


def test_check_hydro_history_names_the_zone_and_the_series(recorded) -> None:
    body = recorded("open_meteo_hydro_no2.json")
    variables = ("precipitation", "temperature_2m")
    history = parse_open_meteo(body, list(NO2.hydro_points), variables, "_previous_day2")
    check_hydro_history(history, date(2026, 10, 22), NO2)
    s = history["no2_stavanger__temperature_2m"]
    s.values[(datetime(2026, 9, 1, tzinfo=UTC) - s.start) // timedelta(hours=1)] = float("nan")
    with pytest.raises(SourceError, match=r"^NO2: .*no2_stavanger__temperature_2m$"):
        check_hydro_history(history, date(2026, 10, 22), NO2)
    with pytest.raises(SourceError, match="before the end of 2026-10-26"):
        check_hydro_history(history, date(2026, 10, 26), NO2)  # after the recording
