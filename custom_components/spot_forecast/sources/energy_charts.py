"""Day-ahead prices from Energy-Charts (Fraunhofer ISE; data from SMARD, CC BY 4.0)."""

from __future__ import annotations

from datetime import date

import aiohttp

from ..core.ingest import MarketSlot, parse_energy_charts
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
        session, PRICE_URL, {"bzn": zone, "start": start.isoformat(), "end": end.isoformat()}
    )
    return await run_blocking(_parse, body)


def _parse(body: bytes) -> list[MarketSlot]:
    """Slots of an Energy-Charts body (blocking)."""
    doc = decode_json(body, PRICE_URL)
    try:
        return parse_energy_charts(doc)
    except (KeyError, TypeError, ValueError) as e:
        raise SourceError(f"unexpected Energy-Charts payload: {type(e).__name__}: {e}") from e
