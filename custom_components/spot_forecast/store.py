"""Persistence of the ledger, recent market slots, the last successful run and the backfill."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import STORE_KEY, STORE_VERSION
from .core.ingest import MarketSlot
from .core.ledger import Ledger

_LOGGER = logging.getLogger(__name__)

SAVE_DELAY = 10  # seconds


class ForecastStore:
    """One entry's data across restarts, so an API outage at start still shows a forecast."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store: Store[dict[str, Any]] = Store(
            hass, STORE_VERSION, STORE_KEY.format(entry_id=entry_id)
        )
        self._pending: dict[str, Any] | None = None

    async def async_load(self) -> tuple[Ledger, list[MarketSlot], datetime | None, int]:
        """The stored ledger, market slots, last success and backfill window done (days).

        Empty ones if nothing is stored. A payload that cannot be read (damaged, or an
        unknown layout) is dropped with a warning instead of failing setup on every retry.
        A payload without ``backfill_done_days`` (written before the backfill) reads as 0.
        """
        data = await self._store.async_load()
        if data is None:
            return Ledger(), [], None, 0
        try:
            market = [
                MarketSlot(datetime.fromisoformat(start), datetime.fromisoformat(end), float(price))
                for start, end, price in data["market"]
            ]
            last = data["last_success"]
            return (
                Ledger.from_json(data["ledger"]),
                market,
                None if last is None else datetime.fromisoformat(last),
                int(data.get("backfill_done_days", 0)),
            )
        except Exception as err:  # noqa: BLE001 - any unreadable layout starts empty
            _LOGGER.warning(
                "Stored forecast data %s cannot be read, starting empty: %s: %s",
                self._store.key,
                type(err).__name__,
                err,
            )
            return Ledger(), [], None, 0

    @callback
    def async_save(
        self,
        ledger: Ledger,
        market: list[MarketSlot],
        last_success: datetime | None,
        backfill_done_days: int,
    ) -> None:
        """Save a snapshot taken now, written ``SAVE_DELAY`` seconds later."""
        snapshot = {
            "ledger": ledger.to_json(),
            "market": [[m.start.isoformat(), m.end.isoformat(), m.price] for m in market],
            "last_success": None if last_success is None else last_success.isoformat(),
            "backfill_done_days": backfill_done_days,
        }
        self._pending = snapshot

        def written() -> dict[str, Any]:
            if self._pending is snapshot:
                self._pending = None
            return snapshot

        self._store.async_delay_save(written, SAVE_DELAY)

    async def async_flush(self) -> None:
        """Write a pending snapshot now (on unload; a reload reads the file right after)."""
        if self._pending is not None:
            snapshot, self._pending = self._pending, None
            await self._store.async_save(snapshot)
