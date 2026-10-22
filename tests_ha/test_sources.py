"""Key-free HTTP sources and the bundled model, against recorded payloads."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import gzip
import re

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)
from yarl import URL

from custom_components.spot_forecast.core.zones import DE_LU
from custom_components.spot_forecast.model_loader import load_bundled
from custom_components.spot_forecast.sources import http
from custom_components.spot_forecast.sources.energy_charts import fetch_prices
from custom_components.spot_forecast.sources.http import (
    SourceError,
    SourceRateLimited,
    get_json,
)
from custom_components.spot_forecast.sources.open_meteo import (
    fetch_forecast,
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
        async_get_clientsession(hass), "DE-LU", date(2026, 10, 19), date(2026, 10, 22)
    )

    assert aioclient_mock.call_count == 1
    assert len(slots) == 4 * 96
    assert slots[0].start == datetime(2026, 10, 18, 22, tzinfo=UTC)
    assert slots[-1].end == datetime(2026, 10, 22, 22, tzinfo=UTC)
    assert all(s.end - s.start == timedelta(minutes=15) for s in slots)
    assert all(isinstance(s.price, float) for s in slots)


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

    weather = await fetch_forecast(async_get_clientsession(hass), DE_LU)

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


async def test_get_json_retries_429_then_succeeds(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    statuses = iter([429, 200])

    async def respond(method, url, data):
        status = next(statuses)
        return AiohttpClientMockResponse(
            method, url, status=status, json={"ok": True} if status == 200 else None
        )

    aioclient_mock.get(TEST_URL, side_effect=respond)

    result = await get_json(async_get_clientsession(hass), TEST_URL, {"a": "1"})

    assert result == {"ok": True}
    assert aioclient_mock.call_count == 2
    assert pauses == [30]


async def test_get_json_gives_up_after_three_attempts(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=503)

    with pytest.raises(SourceError) as err:
        await get_json(async_get_clientsession(hass), TEST_URL, {})

    assert not isinstance(err.value, SourceRateLimited)
    assert aioclient_mock.call_count == 3
    assert pauses == [2, 4]


async def test_get_json_rate_limited_raises_subclass(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=429)

    with pytest.raises(SourceRateLimited):
        await get_json(async_get_clientsession(hass), TEST_URL, {})

    assert pauses == [30, 30]


async def test_get_json_client_error_is_not_retried(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, pauses: list[float]
) -> None:
    aioclient_mock.get(TEST_URL, status=400)

    with pytest.raises(SourceError):
        await get_json(async_get_clientsession(hass), TEST_URL, {})

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
        async_get_clientsession(hass), DE_LU, date(2026, 9, 29), date(2026, 10, 2)
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
