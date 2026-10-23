"""Bundle the published models into an integration release.

python tools/fetch_published_model.py MODELS_DIR [--manifest-url URL]

Reads the manifest of the rolling ``models`` release. For every zone whose model is bundled in
MODELS_DIR (``{slug}.json.gz``), it downloads the published model and backfill, verifies both
SHA-256 checksums and writes them over ``{slug}.json.gz`` and ``{slug}.backfill.json.gz``. A
zone is left alone when the published model has another feature schema than the bundled one
(the integration at this commit could not load it) or is not trained on newer data.

No manifest yet (HTTP 404: no ``models`` release) keeps the committed files. Any other failure
exits 1 and replaces nothing for the zone it happened in. Standard library only.
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

MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
MANIFEST_FORMAT = 1
TIMEOUT = 60


class FetchError(Exception):
    """The published model cannot be bundled; the release must not go out."""


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _verified(asset: dict) -> bytes:
    try:
        data = _download(asset["url"])
    except urllib.error.URLError as e:
        raise FetchError(f"{asset['name']}: download failed: {e}") from e
    digest = hashlib.sha256(data).hexdigest()
    if digest != asset["sha256"]:
        raise FetchError(f"{asset['name']}: SHA-256 {digest} != manifest {asset['sha256']}")
    return data


def _bundled_header(path: Path) -> tuple[int, date]:
    doc = json.loads(gzip.decompress(path.read_bytes()))
    return doc["feature_schema"], date.fromisoformat(doc["train_cutoff"])


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def update_zone(models_dir: Path, slug: str, entry: dict) -> bool:
    """Replace the bundled pair of ``slug`` with the published one; whether it did."""
    model_path = models_dir / f"{slug}.json.gz"
    if not model_path.exists():
        print(f"{slug}: not bundled, skipped")
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
    model = _verified(entry["model"])
    backfill = _verified(entry["backfill"])
    _write_atomic(model_path, model)
    _write_atomic(models_dir / f"{slug}.backfill.json.gz", backfill)
    print(f"{slug}: bundled {entry['version']}")
    return True


def fetch(models_dir: Path, manifest_url: str) -> list[str]:
    """The slugs whose bundled model was replaced."""
    try:
        raw = _download(manifest_url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("no published manifest: keeping the committed models")
            return []
        raise FetchError(f"manifest: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"manifest: {e.reason}") from e
    manifest = json.loads(raw)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise FetchError(f"manifest: unsupported format {manifest.get('format')!r}")
    return [
        slug
        for slug, entry in sorted(manifest["zones"].items())
        if update_zone(models_dir, slug, entry)
    ]


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
