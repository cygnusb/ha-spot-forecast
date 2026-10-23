"""The models of a zone: the one bundled with the integration and the downloaded ones."""

from __future__ import annotations

import logging
from pathlib import Path

from .core.ledger import runs_from_backfill_file
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

    Versions are ``{slug}-YYYY-MM`` of the training cutoff, so this is also cutoff order.

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
    a tie goes to the bundled model. Downloads are tried newest first, and the first
    usable one newer than the bundled model is taken. A download that does not load or
    fails a check is deleted; only a bundled model that cannot be loaded raises (as
    ``load_bundled``). A downloaded model whose backfill file is missing or unreadable
    uses the bundled one, whose rows keep the bundled model's version (it is in the
    file). Blocking file I/O: run it in the executor.
    """
    bundled, bundled_backfill = load_bundled(zone_code)
    slug = ZONES[zone_code].slug
    for path in reversed(downloaded_models(slug, storage_dir)):
        candidate = _load_download(path, slug)
        if candidate is None or candidate.train_cutoff <= bundled.train_cutoff:
            continue
        backfill = _read_backfill(backfill_path(path))
        if backfill is None:
            _LOGGER.warning(
                "The backfill file of the downloaded model %s is missing or unreadable; "
                "using the bundled one",
                candidate.version,
            )
            backfill = bundled_backfill
        return candidate, backfill, SOURCE_DOWNLOADED
    return bundled, bundled_backfill, SOURCE_BUNDLED


def _load_download(path: Path, slug: str) -> ModelFile | None:
    """The downloaded model at ``path``, or None (deleted when it cannot be used)."""
    try:
        return load_checked(path.read_bytes(), slug)
    except ModelMismatch as e:  # corrupt, or e.g. made for another feature schema
        log = _LOGGER.warning if isinstance(e, ModelCorrupt) else _LOGGER.info
        log("Deleting the unusable downloaded model %s: %s", path.name, e)
        try:
            path.unlink(missing_ok=True)
            backfill_path(path).unlink(missing_ok=True)
        except OSError as err:
            _LOGGER.warning("Cannot delete %s: %s", path.name, err)
    except OSError as e:
        _LOGGER.warning("Cannot read the downloaded model %s: %s", path.name, e)
    except Exception:  # never let a downloaded file stop the setup
        _LOGGER.exception("Ignoring the downloaded model %s", path.name)
    return None


def check_backfill(data: bytes) -> None:
    """Raise ``ValueError`` unless ``data`` parses as a backfill file, rows included.

    The parser the backfill pass uses, so a file that passes cannot break it later.
    """
    try:
        runs_from_backfill_file(data, "", "UTC")
    except Exception as e:  # not gzip, not JSON, nested too deep, another format or shape
        raise ValueError(f"backfill file is unusable: {type(e).__name__}: {e}") from e


def _read_backfill(path: Path) -> bytes | None:
    """The backfill file's raw bytes if ``check_backfill`` passes, else None."""
    try:
        data = path.read_bytes()
        check_backfill(data)
    except (OSError, ValueError):  # missing, unreadable, or any error of the parser
        return None
    return data
