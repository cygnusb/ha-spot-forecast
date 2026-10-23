"""Automatic model updates from the ``models`` release, checked daily.

A published model is used only if its feature schema is this code's, it is trained on
newer data than the active model, both files match their SHA-256 in the manifest, and the
model loads and passes the sample check. It is then kept in ``<config>/.storage/spot_forecast``
(never in ``custom_components``) and the coordinator switches to it. A failed check keeps
the current model; the next daily check tries the same version again.

A version in the manifest's ``revoked`` list is never downloaded; its files are deleted, and
if it is the active model the coordinator switches to the next usable one (the newest other
download, or the bundled model).
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from functools import partial
import hashlib
import logging
from pathlib import Path

from aiohttp import ClientSession
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HassJob, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_call_later, async_track_time_change
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey

from .const import (
    CONF_AUTO_UPDATE,
    DEFAULT_AUTO_UPDATE,
    DOMAIN,
    MANIFEST_URL,
    MAX_BACKFILL_BYTES,
    MAX_DOWNLOADED_MODELS,
    MAX_MANIFEST_BYTES,
    MAX_MODEL_BYTES,
    MODEL_DIR,
)
from .coordinator import SpotForecastCoordinator
from .core.features import FEATURE_SCHEMA
from .core.manifest import (
    MAX_CUTOFF_AHEAD,
    ManifestEntry,
    cutoff_ahead,
    parse_manifest,
    wants_download,
)
from .core.model import ModelFile, ModelMismatch
from .model_loader import (
    MODEL_SUFFIX,
    SOURCE_DOWNLOADED,
    backfill_path,
    check_backfill,
    downloaded_models,
    load_active,
    load_checked,
)
from .sources.http import SourceError, SourceNotFound, decode_json, get_body

_LOGGER = logging.getLogger(__name__)

UPDATED = "updated"
CURRENT = "current"
SKIPPED_SCHEMA = "skipped_schema"
FAILED = "failed"
UNAVAILABLE = "unavailable"
REVOKED = "revoked"

FIRST_CHECK_DELAY = timedelta(minutes=5)  # after setup: a fresh install need not wait a day
# versions whose failed check was logged as a warning in this HA run (later: debug only)
WARNED: HassKey[set[str]] = HassKey(f"{DOMAIN}_model_warnings")


class Rejected(Exception):
    """A downloaded file does not match the manifest."""


def model_dir(hass: HomeAssistant) -> Path:
    """Where downloaded models are kept: ``<config>/.storage/spot_forecast``."""
    return Path(hass.config.path(".storage", MODEL_DIR))


def check_time(entry_id: str) -> tuple[int, int]:
    """The local (hour, minute) of the entry's daily check: 03:00–05:59, fixed per entry.

    Derived from the entry id (not the salted ``hash``), so it is stable across restarts
    and spreads the installations' requests.
    """
    n = int.from_bytes(hashlib.sha256(entry_id.encode()).digest()[:8], "big")
    return 3 + n % 3, n // 3 % 60


class ModelUpdater:
    """Checks the release manifest once after setup and then daily, if the option is on."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        coordinator: SpotForecastCoordinator,
        manifest_url: str = MANIFEST_URL,
    ) -> None:
        self._manifest_url = manifest_url
        # assets are fetched only from the release the manifest belongs to
        self._release_base = manifest_url.rsplit("/", 1)[0] + "/"
        self._hass = hass
        self._entry = entry
        self._coordinator = coordinator
        self._slug = coordinator.zone.slug
        self._lock = asyncio.Lock()  # checks never overlap
        self._warned = hass.data.setdefault(WARNED, set())

    @callback
    def async_start(self) -> None:
        """Schedule the checks; they stop on unload. Nothing happens with the option off."""
        if not self._entry.options.get(CONF_AUTO_UPDATE, DEFAULT_AUTO_UPDATE):
            return
        hour, minute = check_time(self._entry.entry_id)
        self._entry.async_on_unload(
            async_track_time_change(self._hass, self._on_due, hour=hour, minute=minute, second=0)
        )
        self._entry.async_on_unload(
            async_call_later(
                self._hass,
                FIRST_CHECK_DELAY,
                HassJob(
                    self._on_due, f"{DOMAIN} {self._slug} model check", cancel_on_shutdown=True
                ),
            )
        )

    @callback
    def _on_due(self, _now: datetime) -> None:
        # a background task: unload cancels a download in flight instead of waiting for it
        self._entry.async_create_background_task(
            self._hass, self.async_check(), f"{DOMAIN} {self._slug} model check"
        )

    async def async_check(self) -> str:
        """Check the manifest and switch to a newer model; returns the outcome.

        One of ``UPDATED``, ``CURRENT``, ``SKIPPED_SCHEMA`` (the published model needs
        another version of this code), ``FAILED`` (it failed a check, see the log),
        ``UNAVAILABLE`` (no manifest, or nothing published for the zone) and ``REVOKED``
        (the active model was revoked and replaced, and nothing newer was installed).
        """
        async with self._lock:
            try:
                return await self._async_check()
            except Exception:  # a bug: keep the current model, try again tomorrow
                _LOGGER.exception("Model update check failed")
                return FAILED

    async def _async_check(self) -> str:
        session = async_get_clientsession(self._hass)
        url = self._manifest_url
        try:
            # at most MAX_MANIFEST_BYTES: small enough to decode on the event loop
            body = await get_body(session, url, {}, MAX_MANIFEST_BYTES)
            manifest = parse_manifest(decode_json(body, url))
        except SourceNotFound:
            _LOGGER.debug("No models published yet: %s", url)
            return UNAVAILABLE
        except (SourceError, ValueError) as e:
            _LOGGER.debug("Model manifest not usable: %s", e)
            return UNAVAILABLE
        replaced = await self._async_drop_revoked(manifest.revoked)
        entry = manifest.entries.get(self._slug)
        outcome = await self._async_check_entry(session, entry, manifest.revoked)
        return REVOKED if replaced and outcome != UPDATED else outcome

    async def _async_drop_revoked(self, revoked: frozenset[str]) -> bool:
        """Delete revoked downloads; True if the active model was one and was replaced."""
        if not revoked:
            return False
        directory = model_dir(self._hass)
        removed = await self._hass.async_add_executor_job(
            remove_revoked, self._slug, directory, revoked
        )
        for version in removed:
            _LOGGER.warning("Deleted the downloaded model %s: it was revoked", version)
        active = self._coordinator.model
        if active.version not in revoked or self._coordinator.model_source != SOURCE_DOWNLOADED:
            return False
        model, backfill, source = await self._hass.async_add_executor_job(
            partial(load_active, self._coordinator.zone.code, directory, revoked=revoked)
        )
        await self._coordinator.set_model(model, source, backfill)
        _LOGGER.warning(
            "The active model %s was revoked; using the %s model %s",
            active.version,
            source,
            model.version,
        )
        return True

    async def _async_check_entry(
        self, session: ClientSession, entry: ManifestEntry | None, revoked: frozenset[str]
    ) -> str:
        """Download and switch to ``entry`` (the zone's published model) if it is wanted."""
        if entry is None:
            _LOGGER.debug("No model published for %s", self._slug)
            return UNAVAILABLE
        if entry.feature_schema != FEATURE_SCHEMA:
            _LOGGER.debug(
                "Published model %s has feature schema %s, this code %s: not used",
                entry.version,
                entry.feature_schema,
                FEATURE_SCHEMA,
            )
            return SKIPPED_SCHEMA
        if entry.version in revoked:
            _LOGGER.debug("Published model %s is revoked: not used", entry.version)
            return CURRENT
        today = dt_util.utcnow().date()
        if cutoff_ahead(entry.train_cutoff, today):
            error = Rejected(
                f"its training cutoff {entry.train_cutoff} is more than "
                f"{MAX_CUTOFF_AHEAD.days} days after today ({today})"
            )
            self._warn_once(entry.version, error)
            return FAILED
        if not wants_download(
            entry,
            self._coordinator.model.train_cutoff,
            FEATURE_SCHEMA,
            today=today,
            revoked=revoked,
        ):
            return CURRENT
        for asset_url in (entry.model_url, entry.backfill_url):
            name = asset_url[len(self._release_base) :]
            if (
                not asset_url.startswith(self._release_base)
                or not name
                or any(c in name for c in "/%?#\\")
            ):
                error = Rejected(f"{asset_url} is not in the release {self._release_base}")
                self._warn_once(entry.version, error)
                return FAILED
        try:
            model_bytes = await get_body(session, entry.model_url, {}, MAX_MODEL_BYTES)
            backfill = await get_body(session, entry.backfill_url, {}, MAX_BACKFILL_BYTES)
            model = await self._hass.async_add_executor_job(
                install, entry, self._slug, model_bytes, backfill, model_dir(self._hass), today
            )
        except (SourceError, Rejected, ModelMismatch, OSError) as e:
            self._warn_once(entry.version, e)
            return FAILED
        await self._coordinator.set_model(model, SOURCE_DOWNLOADED, backfill)
        _LOGGER.info("Using the downloaded model %s", model.version)
        return UPDATED

    def _warn_once(self, version: str, error: Exception) -> None:
        log = _LOGGER.debug if version in self._warned else _LOGGER.warning
        self._warned.add(version)
        log("Model %s not used, keeping the current one: %s", version, error)


def install(
    entry: ManifestEntry,
    slug: str,
    model_bytes: bytes,
    backfill: bytes,
    directory: Path,
    today: date,
) -> ModelFile:
    """Verify the downloaded pair, store it and drop old versions; returns the model.

    Raises ``Rejected`` on a checksum or manifest mismatch or an unusable backfill file,
    and ``ModelMismatch`` if the model does not load, fails the sample check or has a
    training cutoff too far after ``today``; nothing is written then. Blocking.
    """
    for name, data, digest in (
        ("model", model_bytes, entry.model_sha256),
        ("backfill", backfill, entry.backfill_sha256),
    ):
        if hashlib.sha256(data).hexdigest() != digest:
            raise Rejected(f"SHA-256 of the {name} file does not match the manifest")
    model = load_checked(model_bytes, slug, today=today)
    if (model.version, model.train_cutoff) != (entry.version, entry.train_cutoff):
        raise Rejected(f"the file holds {model.version} (cutoff {model.train_cutoff})")
    try:
        check_backfill(backfill)
    except ValueError as e:
        raise Rejected(str(e)) from e
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry.version}{MODEL_SUFFIX}"
    _write_atomic(backfill_path(path), backfill)  # first: a model never lacks its backfill
    _write_atomic(path, model_bytes)
    others = [p for p in downloaded_models(slug, directory) if p != path]
    for old in others[: max(0, len(others) + 1 - MAX_DOWNLOADED_MODELS)]:
        _remove_pair(old)
    return model


def remove_models(slug: str, directory: Path) -> None:
    """Delete the downloaded files of zone ``slug``, and the directory once empty (blocking)."""
    if not directory.is_dir():
        return
    # models, backfill files, and temporary files of an interrupted write
    for pattern in (f"{slug}-*{MODEL_SUFFIX}", f".{slug}-*.tmp"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)
    if not any(directory.iterdir()):
        directory.rmdir()


def remove_revoked(slug: str, directory: Path, revoked: frozenset[str]) -> list[str]:
    """Delete the downloads of zone ``slug`` whose version is in ``revoked`` (blocking).

    Returns the versions deleted. Only files found on disk are touched: no path is built
    from the manifest's strings.
    """
    removed = []
    for path in downloaded_models(slug, directory):
        version = path.name.removesuffix(MODEL_SUFFIX)
        if version in revoked:
            _remove_pair(path)
            removed.append(version)
    return removed


def _remove_pair(model_path: Path) -> None:
    backfill_path(model_path).unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
