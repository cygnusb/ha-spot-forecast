"""The model and backfill file bundled with the integration, one pair per zone."""

from __future__ import annotations

from pathlib import Path

from .core.model import ModelFile, ModelMismatch, load_model
from .core.zones import ZONES

MODELS_DIR = Path(__file__).parent / "models"


def load_bundled(zone: str) -> tuple[ModelFile, bytes]:
    """The bundled model of ``zone`` (a zone code) and the raw bytes of its backfill file.

    Blocking file I/O: run it in the executor. Raises ``KeyError`` for an unknown zone,
    ``OSError`` if a file is missing and ``ModelMismatch`` if the model does not fit.
    """
    slug = ZONES[zone].slug
    model = load_model((MODELS_DIR / f"{slug}.json.gz").read_bytes())
    if model.zone != slug:  # training writes the slug, e.g. "de_lu"
        raise ModelMismatch(f"bundled model {slug} is for zone {model.zone!r}, not {slug!r}")
    backfill = (MODELS_DIR / f"{slug}.backfill.json.gz").read_bytes()
    return model, backfill
