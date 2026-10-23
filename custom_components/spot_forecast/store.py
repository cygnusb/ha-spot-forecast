"""Persistence of the ledger, recent market slots, the last run and the backfill window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import inspect
import logging
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import STORE_KEY, STORE_VERSION
from .core.ingest import MarketSlot
from .core.ledger import Ledger

_LOGGER = logging.getLogger(__name__)

SAVE_DELAY = 10  # seconds


@dataclass(frozen=True)
class StoredState:
    """What survives a restart. Nothing in it may change once it is handed to the Store."""

    ledger: Ledger
    market: tuple[MarketSlot, ...]
    last_success: datetime | None
    backfill_done_days: int  # the widest backfill window completed
    last_attempt_failed: bool  # keeps an outage visible as ``stale`` across a restart


EMPTY = StoredState(Ledger(), (), None, 0, False)


def _thread_safe_store_kwargs() -> dict[str, Any]:
    """Ask for serialising in a worker thread where the running HA lets the caller choose.

    From 2025.12 the Store serialises in the event loop unless told otherwise; before that
    it always called ``data_func`` and serialised in a worker thread.
    """
    if "serialize_in_event_loop" in inspect.signature(Store.__init__).parameters:
        return {"serialize_in_event_loop": False}
    return {}


class ForecastStore:
    """One entry's data across restarts, so an API outage at start still shows a forecast.

    The snapshot is turned into JSON in a worker thread, so the ledger handed to
    ``async_save`` must never change afterwards: the coordinator swaps in new ledgers, it
    never changes one it has saved.
    """

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._hass = hass
        self._store: Store[dict[str, Any]] = Store(
            hass, STORE_VERSION, STORE_KEY.format(entry_id=entry_id), **_thread_safe_store_kwargs()
        )
        self._pending: Callable[[], dict[str, Any]] | None = None

    async def async_load(self) -> StoredState:
        """The stored state, ``EMPTY`` if nothing is stored.

        A payload that cannot be read (damaged, or an unknown layout) is dropped with a
        warning instead of failing setup on every retry. Keys added later
        (``backfill_done_days``, ``last_attempt_failed``) read as 0 and False when missing.
        """
        data = await self._store.async_load()
        if data is None:
            return EMPTY
        try:
            return await self._hass.async_add_executor_job(_from_json, data)
        except Exception as err:  # noqa: BLE001 - any unreadable layout starts empty
            _LOGGER.warning(
                "Stored forecast data %s cannot be read, starting empty: %s: %s",
                self._store.key,
                type(err).__name__,
                err,
            )
            return EMPTY

    @callback
    def async_save(self, state: StoredState) -> None:
        """Save ``state`` ``SAVE_DELAY`` seconds later, turned into JSON off the loop."""

        def data() -> dict[str, Any]:  # runs in a worker thread
            self._hass.loop.call_soon_threadsafe(self._written, data)
            return _to_json(state)

        self._pending = data
        self._store.async_delay_save(data, SAVE_DELAY)

    @callback
    def _written(self, data: Callable[[], dict[str, Any]]) -> None:
        """``data`` went to the file: nothing to flush unless a newer snapshot came since."""
        if self._pending is data:
            self._pending = None

    async def async_flush(self) -> None:
        """Write a pending snapshot now (on unload; a reload reads the file right after)."""
        if self._pending is not None:
            data, self._pending = self._pending, None
            await self._store.async_save(await self._hass.async_add_executor_job(data))

    async def async_remove(self) -> None:
        """Delete the file (the entry is removed)."""
        self._pending = None
        await self._store.async_remove()


def _to_json(state: StoredState) -> dict[str, Any]:
    """The Store payload of ``state`` (blocking)."""
    return {
        "ledger": state.ledger.to_json(),
        "market": [[m.start.isoformat(), m.end.isoformat(), m.price] for m in state.market],
        "last_success": None if state.last_success is None else state.last_success.isoformat(),
        "backfill_done_days": state.backfill_done_days,
        "last_attempt_failed": state.last_attempt_failed,
    }


def _from_json(data: dict[str, Any]) -> StoredState:
    """``StoredState`` of a Store payload (blocking); raises on any unreadable layout."""
    market = tuple(
        MarketSlot(datetime.fromisoformat(start), datetime.fromisoformat(end), float(price))
        for start, end, price in data["market"]
    )
    last = data["last_success"]
    return StoredState(
        Ledger.from_json(data["ledger"]),
        market,
        None if last is None else datetime.fromisoformat(last),
        int(data.get("backfill_done_days", 0)),
        bool(data.get("last_attempt_failed", False)),
    )
