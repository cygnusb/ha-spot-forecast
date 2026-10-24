"""Bundle the published models into an integration release.

python tools/fetch_published_model.py MODELS_DIR [--manifest-url URL]

Reads the manifest of the rolling ``models`` release. For every zone whose model is bundled in
MODELS_DIR (``{slug}.json.gz``), it downloads the published model and backfill and checks them
as the integration would before it uses a download: both SHA-256 checksums, the model loads
(``load_model``, zone, sample check) with the version, feature schema and cutoff the manifest
names, the backfill parses. Only then are they written over ``{slug}.backfill.json.gz`` and
``{slug}.json.gz``, the backfill first: a model is never bundled without its own backfill (the
other way round the bundled model still loads, and backfill rows carry their model version).

A zone is left alone when its published version is listed in the manifest's ``revoked``, has
another feature schema than the bundled one (the integration at this commit could not load
it) or is not trained on newer data. Asset URLs must be https and in the release the manifest
comes from.

No manifest yet (HTTP 404: no ``models`` release) keeps the committed files. Any other failure
exits 1 and replaces nothing for the zone it happened in. Needs numpy (the integration's
``core``).
"""

from __future__ import annotations

import argparse
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components" / "spot_forecast"))  # as card_scenarios.py

from core.ledger import runs_from_backfill_file  # noqa: E402
from core.model import ModelFile, ModelMismatch, load_model, sample_check  # noqa: E402
from core.zones import zone_by_slug  # noqa: E402

MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
MANIFEST_FORMAT = 1
TIMEOUT = 60


class FetchError(Exception):
    """The published model cannot be bundled; the release must not go out."""


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _verified(asset: dict, prefix: str) -> bytes:
    url = asset["url"]
    if not url.startswith(prefix) or "/" in url.removeprefix(prefix):
        raise FetchError(f"{asset['name']}: URL {url} is not an asset of {prefix}")
    try:
        data = _download(url)
    except urllib.error.URLError as e:
        raise FetchError(f"{asset['name']}: download failed: {e}") from e
    digest = hashlib.sha256(data).hexdigest()
    if digest != asset["sha256"]:
        raise FetchError(f"{asset['name']}: SHA-256 {digest} != manifest {asset['sha256']}")
    return data


def _checked_model(data: bytes, slug: str, entry: dict) -> ModelFile:
    """The model as the integration loads a download (``model_loader.load_checked``), and
    the manifest entry's version and cutoff."""
    try:
        model = load_model(data)
        sample_check(model)
    except ModelMismatch as e:
        raise FetchError(f"{slug}: published {entry['version']} does not load: {e}") from e
    if model.zone != slug:
        raise FetchError(f"{slug}: {model.version} is for zone {model.zone!r}, not {slug!r}")
    if model.version != entry["version"]:
        raise FetchError(f"{slug}: model file is {model.version}, manifest {entry['version']}")
    if model.train_cutoff.isoformat() != entry["train_cutoff"]:
        raise FetchError(
            f"{slug}: {model.version} has cutoff {model.train_cutoff}, "
            f"manifest {entry['train_cutoff']}"
        )
    return model


def _check_backfill(data: bytes, slug: str, version: str) -> None:
    tz = zone_by_slug(slug).tz
    try:
        runs = runs_from_backfill_file(data, version, tz)
    except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError) as e:
        raise FetchError(f"{slug}: backfill of {version} does not parse: {e}") from e
    if not runs or {r.model_version for r in runs} != {version}:
        raise FetchError(f"{slug}: backfill of {version} has no rows of that model")


def _bundled_header(path: Path) -> tuple[int, date]:
    doc = json.loads(gzip.decompress(path.read_bytes()))
    return doc["feature_schema"], date.fromisoformat(doc["train_cutoff"])


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def update_zone(
    models_dir: Path, slug: str, entry: dict, prefix: str, revoked: frozenset[str]
) -> bool:
    """Replace the bundled pair of ``slug`` with the published one; whether it did.

    ``prefix`` is the release URL every asset must be in.
    """
    model_path = models_dir / f"{slug}.json.gz"
    if not model_path.exists():
        print(f"{slug}: not bundled, skipped")
        return False
    if entry["version"] in revoked:
        print(f"{slug}: published {entry['version']} is revoked: keeping the bundled one")
        return False
    schema, cutoff = _bundled_header(model_path)
    published = date.fromisoformat(entry["train_cutoff"])
    if entry["feature_schema"] != schema:
        print(
            f"{slug}: published {entry['version']} has feature schema "
            f"{entry['feature_schema']}, the bundled model {schema}: keeping the bundled one"
        )
        return False
    if published <= cutoff:
        print(f"{slug}: published {entry['version']} is not newer than the bundled model")
        return False
    model = _verified(entry["model"], prefix)
    backfill = _verified(entry["backfill"], prefix)
    _checked_model(model, slug, entry)
    _check_backfill(backfill, slug, entry["version"])
    _write_atomic(models_dir / f"{slug}.backfill.json.gz", backfill)
    _write_atomic(model_path, model)
    print(f"{slug}: bundled {entry['version']}")
    return True


def fetch(models_dir: Path, manifest_url: str) -> list[str]:
    """The slugs whose bundled model was replaced."""
    if not manifest_url.startswith("https://"):
        raise FetchError(f"manifest: URL {manifest_url} is not https")
    try:
        raw = _download(manifest_url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("no published manifest: keeping the committed models")
            return []
        raise FetchError(f"manifest: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"manifest: {e.reason}") from e
    try:
        manifest = json.loads(raw)
    except ValueError as e:
        raise FetchError(f"manifest: not JSON: {e}") from e
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        found = manifest.get("format") if isinstance(manifest, dict) else type(manifest).__name__
        raise FetchError(f"manifest: unsupported format {found!r}")
    revoked = manifest.get("revoked", [])
    if not isinstance(revoked, list) or not all(isinstance(v, str) for v in revoked):
        raise FetchError(f"manifest: revoked {revoked!r} is not a list of versions")
    prefix = manifest_url.rsplit("/", 1)[0] + "/"
    try:
        return [
            slug
            for slug, entry in sorted(manifest["zones"].items())
            if update_zone(models_dir, slug, entry, prefix, frozenset(revoked))
        ]
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise FetchError(f"manifest: malformed: {type(e).__name__}: {e}") from e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models_dir", type=Path)
    ap.add_argument("--manifest-url", default=MANIFEST_URL)
    args = ap.parse_args(argv)
    try:
        fetch(args.models_dir, args.manifest_url)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
