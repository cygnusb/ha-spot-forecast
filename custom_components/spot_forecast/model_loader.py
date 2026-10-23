"""The models of a zone: the one bundled with the integration and the downloaded ones."""

from __future__ import annotations

import logging
from pathlib import Path

from .core.model import ModelCorrupt, ModelFile, ModelMismatch, load_model, sample_check
from .core.zones import ZONES

MODELS_DIR = Path(__file__).parent / "models"
SOURCE_BUNDLED = "bundled"
SOURCE_DOWNLOADED = "downloaded"
MODEL_SUFFIX = ".json.gz"
BACKFILL_SUFFIX = ".backfill.json.gz"

_LOGGER = logging.getLogger(__name__)


def load_bundled(zone: str) -> tuple[ModelFile, bytes]:
    """The bundled model of ``zone`` (a zone code) and the raw bytes of its backfill file.

    Blocking file I/O: run it in the executor. Raises ``KeyError`` for an unknown zone,
    ``OSError`` if a file is missing and ``ModelMismatch`` if the model does not fit.
    """
    slug = ZONES[zone].slug
    model = load_model((MODELS_DIR / f"{slug}{MODEL_SUFFIX}").read_bytes())
    if model.zone != slug:  # training writes the slug, e.g. "de_lu"
        raise ModelMismatch(f"bundled model {slug} is for zone {model.zone!r}, not {slug!r}")
    backfill = (MODELS_DIR / f"{slug}{BACKFILL_SUFFIX}").read_bytes()
    return model, backfill


def downloaded_models(slug: str, storage_dir: Path) -> list[Path]:
    """The downloaded model files of zone ``slug`` (``{version}.json.gz``), oldest first.

    None when ``storage_dir`` does not exist or cannot be read.
    """
    try:
        found = list(storage_dir.glob(f"{slug}-*{MODEL_SUFFIX}"))
    except OSError:
        return []
    return sorted(p for p in found if not p.name.endswith(BACKFILL_SUFFIX))


def backfill_path(model_path: Path) -> Path:
    """Where the backfill file downloaded with ``model_path`` is kept."""
    return model_path.with_name(model_path.name.removesuffix(MODEL_SUFFIX) + BACKFILL_SUFFIX)


def load_checked(data: bytes, slug: str) -> ModelFile:
    """``load_model`` plus the zone and the sample check (``ModelMismatch`` if one fails)."""
    model = load_model(data)
    if model.zone != slug:
        raise ModelMismatch(f"model {model.version} is for zone {model.zone!r}, not {slug!r}")
    sample_check(model)
    return model


def load_active(zone_code: str, storage_dir: Path) -> tuple[ModelFile, bytes, str]:
    """The model to use for ``zone_code``, its backfill bytes and its source.

    The newest by training cutoff of the bundled model and the downloaded ones in
    ``storage_dir`` that load, fit this code (feature schema) and pass the sample check;
    a tie goes to the bundled model. A downloaded model without its backfill file uses
    the bundled backfill. Unusable downloads are skipped; only a bundled model that
    cannot be loaded raises (as ``load_bundled``). Blocking file I/O: run it in the
    executor.
    """
    model, backfill = load_bundled(zone_code)
    active = (model, backfill, SOURCE_BUNDLED)
    slug = ZONES[zone_code].slug
    for path in downloaded_models(slug, storage_dir):
        try:
            candidate = load_checked(path.read_bytes(), slug)
        except ModelCorrupt as e:
            _LOGGER.warning("Ignoring the downloaded model %s: %s", path.name, e)
            continue
        except (OSError, ModelMismatch) as e:  # e.g. made for an older feature schema
            _LOGGER.debug("Ignoring the downloaded model %s: %s", path.name, e)
            continue
        if candidate.train_cutoff <= active[0].train_cutoff:
            continue
        try:
            active = (candidate, backfill_path(path).read_bytes(), SOURCE_DOWNLOADED)
        except OSError:
            active = (candidate, backfill, SOURCE_DOWNLOADED)
    return active
