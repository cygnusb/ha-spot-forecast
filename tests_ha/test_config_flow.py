"""Config flow, options flow and entry setup against a real Home Assistant."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.spot_forecast.const import (
    CONF_AUTO_UPDATE,
    CONF_BACKFILL_DAYS,
    CONF_PEAK_THRESHOLD,
    CONF_SURCHARGE,
    CONF_ZONE,
    DOMAIN,
)

CONFIG_FLOW = Path(__file__).parent.parent / "custom_components" / DOMAIN / "config_flow.py"


@pytest.fixture
def recorded_sources(sources, freezer) -> None:
    """Setting an entry up runs the forecast: serve the recorded payloads at their date."""
    freezer.move_to(datetime(2026, 10, 22, 10, tzinfo=ZoneInfo("Europe/Berlin")))


def _entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN, title="Spot Forecast DE-LU", unique_id="DE-LU", data={CONF_ZONE: "DE-LU"}
    )


def _default(schema, key: str):
    for marker in schema.schema:
        if marker == key:
            return marker.default()
    raise AssertionError(f"{key} not in schema")


async def test_user_flow_creates_entry(
    hass_with_recorder: HomeAssistant, recorded_sources: None
) -> None:
    hass = hass_with_recorder
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_ZONE: "DE-LU"})
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Spot Forecast DE-LU"
    assert result["data"] == {CONF_ZONE: "DE-LU"}
    entry = result["result"]
    assert entry.unique_id == "DE-LU"
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.model.zone == "de_lu"


async def test_second_entry_aborts(hass_with_recorder: HomeAssistant) -> None:
    hass = hass_with_recorder
    _entry().add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_ZONE: "DE-LU"})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_flow_saves_threshold_and_backfill(
    hass_with_recorder: HomeAssistant, recorded_sources: None
) -> None:
    hass = hass_with_recorder
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    model = entry.runtime_data.model

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    # default comes from the loaded model, EUR/MWh -> ct/kWh
    assert _default(result["data_schema"], CONF_PEAK_THRESHOLD) == round(
        model.peak_threshold / 10, 1
    )
    assert _default(result["data_schema"], CONF_BACKFILL_DAYS) == 60
    assert _default(result["data_schema"], CONF_AUTO_UPDATE) is True
    assert _default(result["data_schema"], CONF_SURCHARGE) == 0.0

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_SURCHARGE: 0,
            CONF_PEAK_THRESHOLD: 22.5,
            CONF_BACKFILL_DAYS: 90.0,
            CONF_AUTO_UPDATE: False,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options == {
        CONF_PEAK_THRESHOLD: 22.5,
        CONF_BACKFILL_DAYS: 90,
        CONF_AUTO_UPDATE: False,
    }
    assert isinstance(entry.options[CONF_BACKFILL_DAYS], int)
    # the update listener reloaded the entry
    assert entry.state is ConfigEntryState.LOADED


async def test_options_flow_saves_the_surcharge(
    hass_with_recorder: HomeAssistant, recorded_sources: None
) -> None:
    hass = hass_with_recorder
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_SURCHARGE: 0.1872, CONF_BACKFILL_DAYS: 60, CONF_AUTO_UPDATE: True},
    )
    await hass.async_block_till_done()

    assert entry.options[CONF_SURCHARGE] == 0.187  # 0.1 ct
    assert entry.runtime_data.coordinator.surcharge == 0.187
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert _default(result["data_schema"], CONF_SURCHARGE) == 0.187
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_options_flow_threshold_at_the_model_default_follows_the_model(
    hass_with_recorder: HomeAssistant, recorded_sources: None
) -> None:
    """Saving the model's own threshold removes the option, so a newer model's applies."""
    hass = hass_with_recorder
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Spot Forecast DE-LU",
        unique_id="DE-LU",
        data={CONF_ZONE: "DE-LU"},
        options={CONF_PEAK_THRESHOLD: 22.5, CONF_BACKFILL_DAYS: 60, CONF_AUTO_UPDATE: True},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    default = round(entry.runtime_data.model.peak_threshold / 10, 1)
    assert default != 22.5

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert _default(result["data_schema"], CONF_PEAK_THRESHOLD) == 22.5
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_PEAK_THRESHOLD: default, CONF_BACKFILL_DAYS: 60, CONF_AUTO_UPDATE: True},
    )
    await hass.async_block_till_done()

    assert entry.options == {CONF_BACKFILL_DAYS: 60, CONF_AUTO_UPDATE: True}
    assert entry.runtime_data.coordinator.peak_threshold == default
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_options_flow_falls_back_without_runtime_data(
    hass_with_recorder: HomeAssistant,
) -> None:
    hass = hass_with_recorder
    entry = _entry()
    entry.add_to_hass(hass)  # not set up: no runtime_data

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert _default(result["data_schema"], CONF_PEAK_THRESHOLD) == 15.0


async def test_setup_and_unload_entry(
    hass_with_recorder: HomeAssistant, recorded_sources: None
) -> None:
    hass = hass_with_recorder
    entry = _entry()
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.backfill[:2] == b"\x1f\x8b"  # raw gzip

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED


def test_options_flow_does_not_set_config_entry_attribute() -> None:
    """OptionsFlow.config_entry is read-only from HA 2024.12 on; keep our own reference."""
    tree = ast.parse(CONFIG_FLOW.read_text())
    options_flows = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Attribute) and b.attr == "OptionsFlow")
            or (isinstance(b, ast.Name) and b.id == "OptionsFlow")
            for b in node.bases
        )
    ]
    assert options_flows, "no OptionsFlow subclass in config_flow.py"

    assigned = {
        target.attr
        for cls in options_flows
        for node in ast.walk(cls)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert "_entry" in assigned
    assert "config_entry" not in assigned
