"""Manifest of the rolling ``models`` release: the published model per zone.

``manifest.json`` (``format`` 1) has ``updated``, ``zones`` (slug -> ``manifest_entry``)
and optionally ``revoked``: a list of model versions (``de_lu-2026-11``) that must not be
used any more. Clients from this version on never download a revoked version, delete it if
they have it and fall back to the next usable model; release builds do not bundle it; the
training gate does not compare against it. Older clients ignore the key.

To revoke a model: download ``manifest.json`` from the ``models`` release, add the version to
``"revoked"`` (create the list if it is missing), keep everything else as it is, and upload
it again (``gh release upload models manifest.json --clobber``). The model's assets stay in
the release: a published version is never replaced, and the next training publishes a new
one. ``merge_manifest`` keeps the list on every later run.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from core.model import ModelFile

FORMAT = 1


def _asset(name: str, data: bytes, base_url: str) -> dict:
    return {
        "name": name,
        "url": f"{base_url.rstrip('/')}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def manifest_entry(
    model_bytes: bytes, backfill_bytes: bytes, model: ModelFile, base_url: str
) -> dict:
    """One zone's entry; the assets are named after ``model.version``. Metrics are the
    walk-forward report's: D+1 MAE, and MAE, skill and P90 exceedance over all horizons."""
    overall = model.metrics["all"]
    return {
        "version": model.version,
        "feature_schema": model.feature_schema,
        "train_cutoff": model.train_cutoff.isoformat(),
        "created": model.created.isoformat(),
        "model": _asset(f"{model.version}.json.gz", model_bytes, base_url),
        "backfill": _asset(f"{model.version}.backfill.json.gz", backfill_bytes, base_url),
        "metrics": {
            "mae_h1": model.metrics["horizons"]["1"]["mae"],
            "mae": overall["mae"],
            "skill": overall["skill"],
            "p90_exceedance": overall["p90_exceedance"],
        },
    }


def revoked_versions(manifest: dict | None) -> list[str]:
    """The ``revoked`` list of ``manifest`` (empty without one); ``ValueError`` if malformed.

    Optional, a list of model versions (``de_lu-2026-11``) that must not be used any more.
    """
    revoked = manifest.get("revoked", []) if manifest is not None else []
    if not isinstance(revoked, list) or not all(isinstance(v, str) for v in revoked):
        raise ValueError(f"manifest: revoked {revoked!r} is not a list of versions")
    return revoked


def merge_manifest(old: dict | None, slug: str, entry: dict) -> dict:
    """``old`` (``None`` on the first run) with the zone ``slug`` set to ``entry``; its
    ``revoked`` list, if any, is kept."""
    if old is not None and old.get("format") != FORMAT:
        raise ValueError(f"unsupported manifest format {old.get('format')!r}")
    zones = dict(old["zones"]) if old is not None else {}
    zones[slug] = entry
    merged = {"format": FORMAT, "updated": datetime.now(UTC).isoformat(), "zones": zones}
    if old is not None and "revoked" in old:
        merged["revoked"] = list(revoked_versions(old))
    return merged
