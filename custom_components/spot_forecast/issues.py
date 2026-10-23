"""Repairs issue helpers: the ``model_outdated`` warning (not fixable from the UI).

The same issue warns, as ``model_future``, of a model whose training cutoff is more than
``MAX_CUTOFF_AHEAD`` after today: a wrong clock, or a wrong model file.

Not named ``repairs.py``: Home Assistant loads that module as the integration's repairs
platform and rejects one without ``async_create_fix_flow``.
"""

from __future__ import annotations

from datetime import date

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN, MODEL_MAX_AGE_DAYS
from .core.manifest import MAX_CUTOFF_AHEAD, cutoff_ahead
from .core.model import ModelFile

MODEL_OUTDATED = "model_outdated"
MODEL_FUTURE = "model_future"  # translation key of the same issue


def _issue_id(entry_id: str) -> str:
    """One issue per entry (one entry per zone), so unloading one zone keeps the others'."""
    return f"{MODEL_OUTDATED}_{entry_id}"


@callback
def async_check_model_age(
    hass: HomeAssistant, entry_id: str, model: ModelFile, today: date
) -> None:
    """Raise the issue while ``today`` is too far from the training cutoff, else clear it."""
    if cutoff_ahead(model.train_cutoff, today):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _issue_id(entry_id),
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key=MODEL_FUTURE,
            translation_placeholders={
                "version": model.version,
                "cutoff": model.train_cutoff.isoformat(),
                "max_ahead": str(MAX_CUTOFF_AHEAD.days),
            },
        )
    elif (today - model.train_cutoff).days > MODEL_MAX_AGE_DAYS:
        ir.async_create_issue(
            hass,
            DOMAIN,
            _issue_id(entry_id),
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key=MODEL_OUTDATED,
            translation_placeholders={
                "version": model.version,
                "cutoff": model.train_cutoff.isoformat(),
                "max_age": str(MODEL_MAX_AGE_DAYS),
            },
        )
    else:
        async_delete_model_issue(hass, entry_id)


@callback
def async_delete_model_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Remove the issue (it is gone already if it was never raised)."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
