"""Persistence of the ledger, recent market slots and the last successful run."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import STORE_KEY, STORE_VERSION
from .core.ingest import MarketSlot
from .core.ledger import Ledger

SAVE_DELAY = 10  # seconds


class ForecastStore:
    """One entry's data across restarts, so an API outage at start still shows a forecast."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store: Store[dict[str, Any]] = Store(
            hass, STORE_VERSION, STORE_KEY.format(entry_id=entry_id)
        )
        self._pending: dict[str, Any] | None = None

    async def async_load(self) -> tuple[Ledger, list[MarketSlot], datetime | None]:
        """The stored ledger, market slots and last success; empty ones if nothing is stored."""
        data = await self._store.async_load()
        if data is None:
            return Ledger(), [], None
        market = [
            MarketSlot(datetime.fromisoformat(start), datetime.fromisoformat(end), float(price))
            for start, end, price in data["market"]
        ]
        last = data["last_success"]
        return (
            Ledger.from_json(data["ledger"]),
            market,
            None if last is None else datetime.fromisoformat(last),
        )

    @callback
    def async_save(
        self, ledger: Ledger, market: list[MarketSlot], last_success: datetime | None
    ) -> None:
        """Save a snapshot taken now, written ``SAVE_DELAY`` seconds later."""
        snapshot = {
            "ledger": ledger.to_json(),
            "market": [[m.start.isoformat(), m.end.isoformat(), m.price] for m in market],
            "last_success": None if last_success is None else last_success.isoformat(),
        }
        self._pending = snapshot
        self._store.async_delay_save(lambda: snapshot, SAVE_DELAY)

    async def async_flush(self) -> None:
        """Write a pending snapshot now (on unload; a reload reads the file right after)."""
        if self._pending is not None:
            snapshot, self._pending = self._pending, None
            await self._store.async_save(snapshot)
