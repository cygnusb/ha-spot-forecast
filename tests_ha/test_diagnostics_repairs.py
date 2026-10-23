"""The ``model_outdated`` repairs issue and the config-entry diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

from conftest import SetUp
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
import pytest

import custom_components.spot_forecast as integration
from custom_components.spot_forecast.const import CONF_AUTO_UPDATE, CONF_BACKFILL_DAYS, DOMAIN
from custom_components.spot_forecast.diagnostics import async_get_config_entry_diagnostics
from custom_components.spot_forecast.model_loader import load_bundled


async def test_model_outdated_issue_created_and_cleared(
    hass_with_recorder: HomeAssistant, set_up: SetUp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today is 2026-10-22: a cutoff 61 days back raises the issue, 60 days back does not."""
    hass = hass_with_recorder
    model, backfill = load_bundled("DE-LU")
    old = replace(model, train_cutoff=date(2026, 8, 22))
    monkeypatch.setattr(integration, "load_active", lambda zone, _: (old, backfill, "bundled"))
    entry = await set_up()
    issue_id = f"model_outdated_{entry.entry_id}"
    registry = ir.async_get(hass)

    issue = registry.async_get_issue(DOMAIN, issue_id)
    assert issue is not None
    assert issue.translation_key == "model_outdated"
    assert issue.severity is ir.IssueSeverity.WARNING
    assert not issue.is_fixable
    assert issue.translation_placeholders == {
        "version": model.version,
        "cutoff": "2026-08-22",
        "max_age": "60",
    }

    coordinator = entry.runtime_data.coordinator
    coordinator.model = replace(model, train_cutoff=date(2026, 8, 23))
    await coordinator.async_refresh()  # a run re-evaluates the model's age
    assert registry.async_get_issue(DOMAIN, issue_id) is None

    coordinator.model = old
    await coordinator.async_refresh()
    assert registry.async_get_issue(DOMAIN, issue_id) is not None
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert registry.async_get_issue(DOMAIN, issue_id) is None


async def test_no_model_outdated_issue_for_a_current_model(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    entry = await set_up()
    issue_id = f"model_outdated_{entry.entry_id}"
    assert ir.async_get(hass_with_recorder).async_get_issue(DOMAIN, issue_id) is None


async def test_diagnostics_contains_model_and_ledger(
    hass_with_recorder: HomeAssistant, set_up: SetUp
) -> None:
    """Called directly: the frozen clock makes the HTTP API reject the test client's token.

    The JSON round trip with the plain encoder proves the download would not fall back to
    HA's lossy encoder for anything.
    """
    hass = hass_with_recorder
    entry = await set_up({CONF_BACKFILL_DAYS: 10})
    model = entry.runtime_data.model

    diag = json.loads(json.dumps(await async_get_config_entry_diagnostics(hass, entry)))

    assert diag["options"] == {CONF_BACKFILL_DAYS: 10, CONF_AUTO_UPDATE: False}
    assert diag["model"] == {
        "version": model.version,
        "source": "bundled",
        "train_cutoff": "2026-10-01",
        "created": model.created.isoformat(),
    }
    assert diag["last_success"] == "2026-10-22T08:00:00+00:00"
    assert diag["stale"] is False
    assert diag["anchor"] == "2026-10-22"
    ledger = diag["ledger"]
    assert ledger["runs"] == 17  # 16 backfilled anchors and today's live run
    assert ledger["backfill_runs"] == 16
    assert ledger["finalized_days"] > 0
    assert ledger["actual_hours"] > 0
    assert diag["backfill"]["state"] == "done"
    assert diag["backfill"]["done_days"] == 10
    assert diag["backfill"]["result"]["recomputed"] == 16
    assert diag["backfill"]["result"]["error"] is None
    assert set(diag["backfill"]["result"]) == {"imported", "recomputed", "finalized_days", "error"}
    assert diag["accuracy"]["n"] > 0
    entries = diag["entries"]
    assert len(entries) == 48
    assert entries[0]["time"] == "2026-10-22T00:00:00+02:00"
