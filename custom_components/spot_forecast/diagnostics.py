"""Config-entry diagnostics: model, run state, ledger sizes, backfill and accuracy.

Nothing secret exists (no account, no key). The ledger is summarized by its sizes only,
and the ``prices`` attribute is cut to its first 48 entries.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

if TYPE_CHECKING:
    from . import SpotForecastConfigEntry

ENTRIES_SHOWN = 48


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: SpotForecastConfigEntry
) -> dict[str, Any]:
    """Diagnostics of one loaded entry."""
    coordinator = entry.runtime_data.coordinator
    model = coordinator.model
    data = coordinator.data
    ledger = coordinator.ledger
    result = coordinator.backfill_result
    return {
        "options": dict(entry.options),
        "model": {
            "version": model.version,
            "train_cutoff": model.train_cutoff.isoformat(),
            "created": model.created.isoformat(),
        },
        "last_success": None if data.last_success is None else data.last_success.isoformat(),
        "stale": data.stale,
        "anchor": data.anchor.isoformat(),
        "issued_at": None if data.issued_at is None else data.issued_at.isoformat(),
        "ledger": {
            "runs": len(ledger.runs),
            "backfill_runs": sum(run.backfill for run in ledger.runs.values()),
            "finalized_days": len({day for day, _ in ledger.finalized}),
            "actual_hours": len(ledger.actual),
        },
        "backfill": {
            "done_days": coordinator.backfill_done_days,
            "result": None if result is None else asdict(result),
        },
        "accuracy": data.accuracy,
        "entries": data.entries[:ENTRIES_SHOWN],
    }
