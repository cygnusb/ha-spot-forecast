"""Day-ahead prices from Energy-Charts (Fraunhofer ISE; data from SMARD, CC BY 4.0)."""

from __future__ import annotations

from datetime import date

import aiohttp

from ..const import MAX_PRICES_BYTES
from ..core.features import Hourly
from ..core.ingest import MarketSlot, hourly_prices, parse_energy_charts
from ..core.zones import Zone
from .http import RunBlocking, SourceError, decode_json, get_body

PRICE_URL = "https://api.energy-charts.info/price"


async def fetch_prices(
    session: aiohttp.ClientSession,
    zone: str,
    start: date,
    end: date,
    *,
    run_blocking: RunBlocking,
) -> list[MarketSlot]:
    """Market slots (EUR/MWh) of the market days ``start`` through ``end``, both inclusive.

    The API reads date-only bounds as whole days in the zone's local time. Slots
    without a price yet are left out. The body is decoded and parsed via ``run_blocking``.
    """
    body = await get_body(
        session,
        PRICE_URL,
        {"bzn": zone, "start": start.isoformat(), "end": end.isoformat()},
        MAX_PRICES_BYTES,
    )
    return await run_blocking(_parse, body)


async def fetch_price_inputs(
    session: aiohttp.ClientSession,
    zone: Zone,
    start: date,
    end: date,
    *,
    run_blocking: RunBlocking,
) -> dict[str, Hourly]:
    """The hourly prices (EUR/MWh) of the zones in ``zone.price_inputs``, by zone code, for
    the market days ``start`` through ``end``; one request each, one after the other."""
    out: dict[str, Hourly] = {}
    for code in zone.price_inputs:
        market = await fetch_prices(session, code, start, end, run_blocking=run_blocking)
        out[code] = await run_blocking(hourly_prices, market)
    return out


def _parse(body: bytes) -> list[MarketSlot]:
    """Slots of an Energy-Charts body (blocking)."""
    doc = decode_json(body, PRICE_URL)
    try:
        return parse_energy_charts(doc)
    except (KeyError, TypeError, ValueError) as e:
        raise SourceError(f"unexpected Energy-Charts payload: {type(e).__name__}: {e}") from e
