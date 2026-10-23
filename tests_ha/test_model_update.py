"""Automatic model updates from the ``models`` release, against a real Home Assistant.

The release is served by ``conftest.Sources.release``; a newer model is the bundled one
with a later training cutoff. Downloads land in ``<tmp_path>/.storage/spot_forecast``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import gzip
import hashlib
import json
import logging
from pathlib import Path

from conftest import MODELS_BASE, T10, SetUp, Sources, advance
from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.spot_forecast import model_loader, model_update
from custom_components.spot_forecast.const import (
    CONF_AUTO_UPDATE,
    CONF_ZONE,
    DOMAIN,
    MANIFEST_URL,
    MODEL_DIR,
)
from custom_components.spot_forecast.core.features import FEATURE_SCHEMA
from custom_components.spot_forecast.core.ledger import runs_from_backfill_file
from custom_components.spot_forecast.core.model import ModelFile, dump_model
from custom_components.spot_forecast.diagnostics import async_get_config_entry_diagnostics
from custom_components.spot_forecast.model_loader import load_active, load_bundled
from custom_components.spot_forecast.model_update import check_time

MODEL_VERSION = "sensor.spot_forecast_de_lu_model_version"
BUNDLED, BUNDLED_BACKFILL = load_bundled("DE-LU")


def _model(cutoff: date) -> ModelFile:
    """The bundled model, as if trained up to ``cutoff``."""
    return replace(
        BUNDLED,
        version=f"de_lu-{cutoff:%Y-%m}",
        train_cutoff=cutoff,
        peak_threshold=BUNDLED.peak_threshold + 100,
    )


def _backfill(model: ModelFile) -> bytes:
    """The bundled backfill under another gzip header: same rows, other bytes."""
    return gzip.compress(gzip.decompress(BUNDLED_BACKFILL), mtime=model.train_cutoff.day)


def _published(
    model: ModelFile,
    *,
    feature_schema: int = FEATURE_SCHEMA,
    model_sha: str | None = None,
    data: bytes | None = None,
    base: str = MODELS_BASE,
) -> dict[str, bytes]:
    """The release files of a manifest publishing ``model`` for DE-LU (shape: training).

    ``data`` replaces the model file's bytes (the manifest then has their checksum), and
    ``base`` is where the manifest says the assets are.
    """
    data, backfill = dump_model(model) if data is None else data, _backfill(model)

    def asset(name: str, body: bytes, sha: str | None = None) -> dict:
        digest = sha or hashlib.sha256(body).hexdigest()
        return {"name": name, "url": base + name, "sha256": digest, "size": len(body)}

    entry = {
        "version": model.version,
        "feature_schema": feature_schema,
        "train_cutoff": model.train_cutoff.isoformat(),
        "created": model.created.isoformat(),
        "model": asset(f"{model.version}.json.gz", data, model_sha),
        "backfill": asset(f"{model.version}.backfill.json.gz", backfill),
        "metrics": {"mae_h1": 17.0, "mae": 19.0, "skill": 0.4, "p90_exceedance": 0.1},
    }
    manifest = {"format": 1, "updated": "2026-11-03T02:10:00+00:00", "zones": {"de_lu": entry}}
    return {
        MANIFEST_URL: json.dumps(manifest).encode(),
        MODELS_BASE + f"{model.version}.json.gz": data,
        MODELS_BASE + f"{model.version}.backfill.json.gz": backfill,
    }


def _model_dir(tmp_path: Path) -> Path:
    return tmp_path / ".storage" / MODEL_DIR


def _write(directory: Path, model: ModelFile, backfill: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{model.version}.json.gz").write_bytes(dump_model(model))
    if backfill:
        (directory / f"{model.version}.backfill.json.gz").write_bytes(_backfill(model))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Warnings and errors about models (not HA's custom-integration notice or a failed run)."""
    loggers = (model_update.__name__, model_loader.__name__)
    return [r for r in caplog.records if r.levelno >= logging.WARNING and r.name in loggers]


async def test_update_switches_model_and_next_run_uses_it(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    tmp_path: Path,
) -> None:
    hass = hass_with_recorder
    newer = _model(date(2026, 11, 1))
    sources.release = _published(newer)
    entry = await set_up({CONF_AUTO_UPDATE: True})
    coordinator = entry.runtime_data.coordinator
    state = hass.states.get(MODEL_VERSION)
    assert state.state == BUNDLED.version
    assert state.attributes["source"] == "bundled"

    await advance(hass, freezer, T10 + timedelta(minutes=5))  # the check after setup

    state = hass.states.get(MODEL_VERSION)
    assert state.state == "de_lu-2026-11"
    assert state.attributes["source"] == "downloaded"
    assert state.attributes["train_cutoff"] == "2026-11-01"
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["model"]["version"] == "de_lu-2026-11"
    assert diag["model"]["source"] == "downloaded"
    assert coordinator.peak_threshold == round(newer.peak_threshold / 10, 1)
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert device.model == "de_lu-2026-11"
    assert entry.runtime_data.backfill == _backfill(newer)
    assert sorted(p.name for p in _model_dir(tmp_path).iterdir()) == [
        "de_lu-2026-11.backfill.json.gz",
        "de_lu-2026-11.json.gz",
    ]
    run = coordinator.ledger.runs[date(2026, 10, 22)]
    assert run.model_version == BUNDLED.version  # what was shown stays as it was

    await advance(hass, freezer, T10 + timedelta(minutes=15))  # the next scheduled run

    run = coordinator.ledger.runs[date(2026, 10, 22)]
    assert run.issued_at == T10 + timedelta(minutes=15)
    assert run.model_version == "de_lu-2026-11"

    # a restart starts with the downloaded model and does not fetch it again
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.runtime_data.coordinator.model.version == "de_lu-2026-11"
    assert entry.runtime_data.coordinator.model_source == "downloaded"
    assert await entry.runtime_data.updater.async_check() == "current"


async def test_sha_mismatch_keeps_current_model(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    newer = _model(date(2026, 11, 1))
    sources.release = _published(newer, model_sha="0" * 64)
    entry = await set_up({CONF_AUTO_UPDATE: True})
    updater = entry.runtime_data.updater

    assert await updater.async_check() == "failed"
    assert await updater.async_check() == "failed"  # retried, not remembered

    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    assert hass_with_recorder.states.get(MODEL_VERSION).state == BUNDLED.version
    assert sources.release_calls(MODELS_BASE + "de_lu-2026-11.json.gz") == 2
    assert not _model_dir(tmp_path).exists() or not any(_model_dir(tmp_path).iterdir())
    warnings = _warnings(caplog)
    assert len(warnings) == 1  # one per version and HA run
    assert "de_lu-2026-11" in warnings[0].getMessage()
    assert "SHA-256" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"not a model", "corrupt"),  # checksum right, but it does not load
        (dump_model(_model(date(2026, 12, 1))), "holds de_lu-2026-12"),  # not what is listed
    ],
    ids=["unloadable", "other_version"],
)
async def test_model_failing_its_checks_keeps_current_model(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    data: bytes,
    message: str,
) -> None:
    sources.release = _published(_model(date(2026, 11, 1)), data=data)
    entry = await set_up({CONF_AUTO_UPDATE: True})

    assert await entry.runtime_data.updater.async_check() == "failed"

    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    assert not _model_dir(tmp_path).exists()
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert message in warnings[0].getMessage()


@pytest.mark.parametrize(
    "base",
    [
        "https://example.org/releases/download/models/",
        MODELS_BASE + "../../",  # climbs out of the release
        MODELS_BASE + "%2e%2e%2f",  # the same, encoded
        MODELS_BASE + "x?y=",  # a query smuggled into the name
    ],
)
async def test_assets_outside_the_release_are_not_fetched(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    caplog: pytest.LogCaptureFixture,
    base: str,
) -> None:
    newer = _model(date(2026, 11, 1))
    sources.release = _published(newer, base=base)
    entry = await set_up({CONF_AUTO_UPDATE: True})

    assert await entry.runtime_data.updater.async_check() == "failed"

    assert sources.release_calls() == sources.release_calls(MANIFEST_URL) == 1
    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "is not in the release" in warnings[0].getMessage()


async def test_oversized_model_is_not_used(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(model_update, "MAX_MODEL_BYTES", 1000)
    sources.release = _published(_model(date(2026, 11, 1)))
    entry = await set_up({CONF_AUTO_UPDATE: True})

    assert await entry.runtime_data.updater.async_check() == "failed"

    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    assert "limit" in _warnings(caplog)[0].getMessage()


async def test_schema_mismatch_is_ignored_quietly(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    caplog: pytest.LogCaptureFixture,
) -> None:
    newer = _model(date(2026, 11, 1))
    sources.release = _published(newer, feature_schema=FEATURE_SCHEMA + 1)
    entry = await set_up({CONF_AUTO_UPDATE: True})

    assert await entry.runtime_data.updater.async_check() == "skipped_schema"

    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    assert sources.release_calls() == sources.release_calls(MANIFEST_URL) == 1
    assert _warnings(caplog) == []


async def test_manifest_404_is_quiet(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No ``models`` release yet; the check after setup and the daily one both ask for it."""
    hass = hass_with_recorder
    entry = await set_up({CONF_AUTO_UPDATE: True})
    assert sources.release_calls() == 0

    assert await entry.runtime_data.updater.async_check() == "unavailable"
    await advance(hass, freezer, T10 + timedelta(minutes=5))
    assert sources.release_calls(MANIFEST_URL) == 2
    await advance(hass, freezer, T10 + timedelta(days=1, hours=1))  # past the daily time
    assert sources.release_calls(MANIFEST_URL) == 3

    assert entry.runtime_data.coordinator.model.version == BUNDLED.version
    assert _warnings(caplog) == []


async def test_auto_update_off_never_fetches(
    hass_with_recorder: HomeAssistant,
    set_up: SetUp,
    sources: Sources,
    freezer: FrozenDateTimeFactory,
) -> None:
    hass = hass_with_recorder
    sources.release = _published(_model(date(2026, 11, 1)))
    entry = await set_up({CONF_AUTO_UPDATE: False})

    await advance(hass, freezer, T10 + timedelta(minutes=5))
    await advance(hass, freezer, T10 + timedelta(days=1, hours=1))

    assert sources.release_calls() == 0
    assert entry.runtime_data.coordinator.model.version == BUNDLED.version


async def test_keeps_three_downloaded_versions(
    hass_with_recorder: HomeAssistant, set_up: SetUp, sources: Sources, tmp_path: Path
) -> None:
    directory = _model_dir(tmp_path)
    for month in (7, 8, 9):
        _write(directory, _model(date(2026, month, 1)))
    sources.release = _published(_model(date(2026, 11, 1)))
    entry = await set_up({CONF_AUTO_UPDATE: True})
    assert entry.runtime_data.coordinator.model_source == "bundled"  # all older than it

    assert await entry.runtime_data.updater.async_check() == "updated"

    assert sorted(p.name for p in directory.iterdir()) == [
        f"de_lu-2026-{month:02d}{suffix}"
        for month in (8, 9, 11)
        for suffix in (".backfill.json.gz", ".json.gz")
    ]


async def test_removing_the_entry_deletes_its_zone_models(
    hass_with_recorder: HomeAssistant, set_up: SetUp, tmp_path: Path
) -> None:
    hass = hass_with_recorder
    directory = _model_dir(tmp_path)
    _write(directory, _model(date(2026, 11, 1)))
    _write(directory, replace(_model(date(2026, 11, 1)), zone="at", version="at-2026-11"))
    entry = await set_up(entry_id="first")
    other = MockConfigEntry(domain=DOMAIN, entry_id="second", data={CONF_ZONE: "DE-LU"})
    other.add_to_hass(hass)

    assert await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()
    assert len(list(directory.glob("de_lu-*"))) == 2  # the other entry still uses them

    assert await hass.config_entries.async_remove(other.entry_id)
    await hass.async_block_till_done()
    assert sorted(p.name for p in directory.iterdir()) == [
        "at-2026-11.backfill.json.gz",
        "at-2026-11.json.gz",
    ]


def test_load_active_prefers_newest_by_cutoff(tmp_path: Path) -> None:
    """Bundled cutoff 2026-10-01: a later download wins, an earlier one or a tie does not."""
    assert BUNDLED.train_cutoff == date(2026, 10, 1)
    model, backfill, source = load_active("DE-LU", tmp_path / "missing")
    assert (model.version, backfill, source) == (BUNDLED.version, BUNDLED_BACKFILL, "bundled")

    newer, older = tmp_path / "newer", tmp_path / "older"
    _write(newer, _model(date(2026, 11, 1)))
    _write(newer, _model(date(2026, 9, 1)))
    (newer / "de_lu-2026-12.json.gz").write_bytes(b"not a model")
    model, backfill, source = load_active("DE-LU", newer)
    assert (model.version, source) == ("de_lu-2026-11", "downloaded")
    assert backfill == _backfill(_model(date(2026, 11, 1)))
    assert not (newer / "de_lu-2026-12.json.gz").exists()  # unusable: deleted

    _write(older, _model(date(2026, 9, 1)))
    _write(older, _model(date(2026, 10, 1)))  # a tie
    model, backfill, source = load_active("DE-LU", older)
    assert (model.version, backfill, source) == (BUNDLED.version, BUNDLED_BACKFILL, "bundled")


def test_load_active_stops_at_the_newest_usable_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for month in (11, 12):
        _write(tmp_path, _model(date(2026, month, 1)))
    loaded, load_checked = [], model_loader.load_checked

    def counting(data: bytes, slug: str) -> ModelFile:
        loaded.append(model := load_checked(data, slug))
        return model

    monkeypatch.setattr(model_loader, "load_checked", counting)

    model, _, source = load_active("DE-LU", tmp_path)

    assert (model.version, source) == ("de_lu-2026-12", "downloaded")
    assert [m.version for m in loaded] == ["de_lu-2026-12"]


@pytest.mark.parametrize("backfill", [None, b"not gzip", gzip.compress(b'{"format": 2}')])
def test_load_active_without_usable_downloaded_backfill_uses_bundled(
    tmp_path: Path, backfill: bytes | None
) -> None:
    """The bundled rows keep the bundled model's version: it is in their file."""
    _write(tmp_path, _model(date(2026, 11, 1)), backfill=False)
    if backfill is not None:
        (tmp_path / "de_lu-2026-11.backfill.json.gz").write_bytes(backfill)
    model, backfill, source = load_active("DE-LU", tmp_path)
    assert (model.version, backfill, source) == ("de_lu-2026-11", BUNDLED_BACKFILL, "downloaded")
    runs = runs_from_backfill_file(backfill, model.version, "Europe/Berlin")
    assert {run.model_version for run in runs} == {BUNDLED.version}


def test_load_active_deletes_another_schema_or_zone(tmp_path: Path) -> None:
    _write(tmp_path, replace(_model(date(2026, 11, 1)), feature_schema=FEATURE_SCHEMA + 1))
    _write(tmp_path, replace(_model(date(2026, 12, 1)), zone="at"))
    model, _, source = load_active("DE-LU", tmp_path)
    assert (model.version, source) == (BUNDLED.version, "bundled")
    assert list(tmp_path.iterdir()) == []


def test_load_active_survives_any_error_of_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, _model(date(2026, 11, 1)))

    def broken(data: bytes, slug: str) -> ModelFile:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(model_loader, "load_checked", broken)
    model, _, source = load_active("DE-LU", tmp_path)
    assert (model.version, source) == (BUNDLED.version, "bundled")
    assert (tmp_path / "de_lu-2026-11.json.gz").exists()  # not known to be bad: kept


def test_check_time_is_stable_and_in_the_night() -> None:
    times = {entry_id: check_time(entry_id) for entry_id in ("a", "b", "c", "d", "e", "f")}
    assert all(3 <= hour <= 5 and 0 <= minute < 60 for hour, minute in times.values())
    assert check_time("a") == times["a"]
    assert len(set(times.values())) > 1
