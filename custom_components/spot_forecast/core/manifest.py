"""The manifest of the rolling ``models`` release: which model is published for each zone.

Written by ``training/manifest.py``; read here in exactly that shape (``format`` 1). The
optional top-level ``revoked`` lists versions that must not be used (clients before it
ignore the key).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Any

FORMAT = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
# a training cutoff later than today plus this is refused: a wrong file, or a wrong clock
MAX_CUTOFF_AHEAD = timedelta(days=7)


@dataclass(frozen=True)
class ManifestEntry:
    """One zone's published model and backfill file."""

    version: str
    feature_schema: int
    train_cutoff: date
    model_url: str
    model_sha256: str
    backfill_url: str
    backfill_sha256: str


@dataclass(frozen=True)
class Manifest:
    """The published entries by zone slug, and the revoked versions of all zones."""

    entries: dict[str, ManifestEntry]
    revoked: frozenset[str]


def parse_manifest(doc: Any) -> Manifest:
    """The manifest; ``ValueError`` on a wrong ``format`` or a malformed entry or list.

    The version must be ``{slug}-YYYY-MM`` of the zone it is listed under (it becomes a
    file name), the URLs https and the checksums 64 lowercase hex digits. ``revoked``,
    if present, is a list of version strings.
    """
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        found = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise ValueError(f"unsupported manifest format {found!r}")
    revoked = doc.get("revoked", [])
    if not isinstance(revoked, list) or not all(isinstance(v, str) for v in revoked):
        raise ValueError(f"malformed manifest: revoked {revoked!r} is not a list of versions")
    try:
        entries = {slug: _entry(slug, e) for slug, e in doc["zones"].items()}
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed manifest: {type(e).__name__}: {e}") from e
    return Manifest(entries, frozenset(revoked))


def _entry(slug: str, e: dict[str, Any]) -> ManifestEntry:
    version = e["version"]
    pattern = re.escape(slug) + r"-[0-9]{4}-[0-9]{2}"  # e.g. de_lu-2026-11; names the files
    if not isinstance(version, str) or not re.fullmatch(pattern, version):
        raise ValueError(f"{slug}: version {version!r} is not {slug}-YYYY-MM")
    schema = e["feature_schema"]
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise TypeError(f"{slug}: feature_schema {schema!r} is not an integer")
    return ManifestEntry(
        version=version,
        feature_schema=schema,
        train_cutoff=date.fromisoformat(e["train_cutoff"]),
        model_url=_url(slug, e["model"]["url"]),
        model_sha256=_sha256(slug, e["model"]["sha256"]),
        backfill_url=_url(slug, e["backfill"]["url"]),
        backfill_sha256=_sha256(slug, e["backfill"]["sha256"]),
    )


def _url(slug: str, url: Any) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"{slug}: asset URL {url!r} is not https")
    return url


def _sha256(slug: str, digest: Any) -> str:
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError(f"{slug}: {digest!r} is not a SHA-256 hex digest")
    return digest


def cutoff_ahead(cutoff: date, today: date) -> bool:
    """Whether a training ``cutoff`` is more than ``MAX_CUTOFF_AHEAD`` after ``today``."""
    return cutoff > today + MAX_CUTOFF_AHEAD


def wants_download(
    entry: ManifestEntry,
    active_cutoff: date,
    feature_schema: int,
    *,
    today: date,
    revoked: frozenset[str] = frozenset(),
) -> bool:
    """Whether ``entry`` fits this code's feature schema and is trained on newer data.

    Never for a revoked version or a cutoff more than ``MAX_CUTOFF_AHEAD`` after ``today``.
    """
    return (
        entry.feature_schema == feature_schema
        and entry.train_cutoff > active_cutoff
        and entry.version not in revoked
        and not cutoff_ahead(entry.train_cutoff, today)
    )
