"""Fixtures for the tests that run against a real Home Assistant.

Separate from `tests/`, which is pure Python and runs without Home Assistant. Never
collect both suites in one pytest run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta
import json
import logging
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

# The tests import `custom_components.spot_forecast` the way Home Assistant loads it,
# so the repo root has to be on sys.path; pytest only adds the test directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_components.spot_forecast.core.zones import DE_LU
from custom_components.spot_forecast.sources import http

pytest_plugins = "pytest_homeassistant_custom_component"

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Silence the SQL echo the plugin turns on.

    It sets `sqlalchemy.engine` to INFO at import; the recorder thread then logs its
    last statements after pytest's capture of a test has ended, straight onto the
    terminal between the progress dots.
    """
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@pytest.fixture
def recorded() -> Callable[[str], Any]:
    """Loader of a recorded API payload from `tests_ha/fixtures` (see its README)."""
    return lambda name: json.loads((FIXTURES / name).read_text())


@pytest.fixture
def hass_with_recorder(recorder_mock, enable_custom_integrations, hass) -> HomeAssistant:
    """A Home Assistant with the recorder that can load this custom integration.

    The parameter order is load-bearing: `recorder_mock` refuses to run once `hass`
    exists, so it must be requested before anything that pulls `hass` in.
    """
    return hass


EC_URL = "https://api.energy-charts.info/price"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BERLIN = ZoneInfo("Europe/Berlin")
RECORDED_DAYS = 4  # 2026-10-19 … 2026-10-22, the part both price fixtures share
HISTORY_BLOCKS = 10  # 40 days of tiled history before 2026-10-19


def _with_history(doc: dict) -> dict:
    """The recorded document with ``HISTORY_BLOCKS`` copies of its first four days before it.

    The integration fetches 40 days of prices (``run_forecast`` needs 30); the recording
    holds four. The copies are shifted back in whole 4-day blocks, all within CEST, so the
    quarter-hour grid stays intact. The fixture files themselves stay unchanged.
    """
    n = RECORDED_DAYS * 96
    stamps, prices = [], []
    for block in range(HISTORY_BLOCKS, 0, -1):
        stamps += [u - block * RECORDED_DAYS * 86400 for u in doc["unix_seconds"][:n]]
        prices += doc["price"][:n]
    return {"unix_seconds": stamps + doc["unix_seconds"], "price": prices + doc["price"]}


class Sources:
    """Serves the recorded payloads, or errors, for Energy-Charts and the Open-Meteo forecast."""

    def __init__(self, mock: AiohttpClientMocker, recorded: Callable[[str], Any]) -> None:
        self._mock = mock
        self._recorded = recorded

    def serve(
        self,
        prices: str = "energy_charts_price.json",
        *,
        prices_status: int = 200,
        weather_status: int = 200,
        price_overrides: dict[datetime, float] | None = None,
    ) -> None:
        """Replace all registered responses (and the call log).

        ``price_overrides`` sets the price (EUR/MWh) of the quarter hours starting at the
        given instants. Prices are cut to the requested ``start``/``end`` market days.
        """
        self._mock.clear_requests()
        if prices_status != 200:
            self._mock.get(EC_URL, status=prices_status)
        else:
            doc = _with_history(self._recorded(prices))
            for at, price in (price_overrides or {}).items():
                doc["price"][doc["unix_seconds"].index(int(at.timestamp()))] = price

            async def respond(method, url, data):
                first, last = (date.fromisoformat(url.query[k]) for k in ("start", "end"))
                lo = datetime.combine(first, time(), BERLIN).timestamp()
                hi = datetime.combine(last + timedelta(days=1), time(), BERLIN).timestamp()
                pairs = [
                    (u, p)
                    for u, p in zip(doc["unix_seconds"], doc["price"], strict=True)
                    if lo <= u < hi
                ]
                body = {"unix_seconds": [u for u, _ in pairs], "price": [p for _, p in pairs]}
                return AiohttpClientMockResponse(method, url, json=body)

            self._mock.get(EC_URL, side_effect=respond)
        if weather_status != 200:
            self._mock.get(FORECAST_URL, status=weather_status)
        else:
            for points, name in (
                (DE_LU.points, "open_meteo_de.json"),
                (DE_LU.neighbor_points, "open_meteo_nb.json"),
            ):
                latitudes = ",".join(str(lat) for lat, _ in points.values())
                self._mock.get(
                    FORECAST_URL, params={"latitude": latitudes}, json=self._recorded(name)
                )

    def fail_all(self) -> None:
        """Every source answers HTTP 500."""
        self.serve(prices_status=500, weather_status=500)


@pytest.fixture
def sources(aioclient_mock: AiohttpClientMocker, recorded, monkeypatch) -> Sources:
    """Recorded sources as of 2026-10-22, before the day-ahead auction; no back-off sleeps."""

    async def no_pause(seconds: float) -> None:
        return None

    monkeypatch.setattr(http, "pause", no_pause)
    served = Sources(aioclient_mock, recorded)
    served.serve()
    return served
