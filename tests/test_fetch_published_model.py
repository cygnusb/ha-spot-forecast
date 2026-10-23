"""tools/fetch_published_model.py: the published model replaces the bundled one at release."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.error

import pytest

from tools import fetch_published_model as fpm

MANIFEST_URL = "https://example.test/models/manifest.json"
BASE = "https://example.test/models"


def _model(schema: int = 1, cutoff: str = "2026-11-01", tag: str = "new") -> bytes:
    doc = {"feature_schema": schema, "train_cutoff": cutoff, "tag": tag}
    return gzip.compress(json.dumps(doc).encode(), mtime=0)


def _asset(name: str, data: bytes) -> dict:
    return {
        "name": name,
        "url": f"{BASE}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _entry(model: bytes, backfill: bytes, schema: int = 1, cutoff: str = "2026-11-01") -> dict:
    version = f"de_lu-{cutoff[:7]}"
    return {
        "version": version,
        "feature_schema": schema,
        "train_cutoff": cutoff,
        "model": _asset(f"{version}.json.gz", model),
        "backfill": _asset(f"{version}.backfill.json.gz", backfill),
    }


class FakeWeb:
    """``urlopen`` over a dict of URL -> bytes; other URLs answer ``status`` (404 by default)."""

    def __init__(self, files: dict[str, bytes], status: int = 404):
        self.files = files
        self.status = status
        self.requested: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.requested.append(url)
        if url in self.files:
            return io.BytesIO(self.files[url])
        raise urllib.error.HTTPError(url, self.status, "nope", {}, None)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "de_lu.json.gz").write_bytes(_model(cutoff="2026-10-01", tag="old"))
    (d / "de_lu.backfill.json.gz").write_bytes(b"old backfill")
    return d


def _published(entry: dict, model: bytes, backfill: bytes) -> dict[str, bytes]:
    manifest = {"format": 1, "updated": "2026-11-03T02:00:00+00:00", "zones": {"de_lu": entry}}
    return {
        MANIFEST_URL: json.dumps(manifest).encode(),
        entry["model"]["url"]: model,
        entry["backfill"]["url"]: backfill,
    }


def _run(monkeypatch, web: FakeWeb, models_dir: Path) -> int:
    monkeypatch.setattr(fpm.urllib.request, "urlopen", web)
    return fpm.main([str(models_dir), "--manifest-url", MANIFEST_URL])


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_fetch_published_model_replaces_bundled_files(monkeypatch, models_dir):
    model, backfill = _model(), b"new backfill"
    web = FakeWeb(_published(_entry(model, backfill), model, backfill))
    assert _run(monkeypatch, web, models_dir) == 0
    assert _snapshot(models_dir) == {
        "de_lu.backfill.json.gz": backfill,
        "de_lu.json.gz": model,
    }


def test_fetch_published_model_verifies_sha(monkeypatch, models_dir, capsys):
    model, backfill = _model(), b"new backfill"
    files = _published(_entry(model, backfill), model, backfill)
    files[f"{BASE}/de_lu-2026-11.backfill.json.gz"] = b"truncated"
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb(files), models_dir) == 1
    # Neither file is replaced, not even the model whose checksum was fine, and no temp file
    # is left behind to end up in the ZIP.
    assert _snapshot(models_dir) == before
    assert "SHA-256" in capsys.readouterr().err


def test_fetch_published_model_keeps_files_without_manifest(monkeypatch, models_dir, capsys):
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb({}), models_dir) == 0
    assert _snapshot(models_dir) == before
    assert "keeping the committed models" in capsys.readouterr().out


@pytest.mark.parametrize("status", [403, 500])
def test_fetch_published_model_fails_on_other_manifest_errors(monkeypatch, models_dir, status):
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb({}, status=status), models_dir) == 1
    assert _snapshot(models_dir) == before


def test_fetch_published_model_fails_on_missing_asset(monkeypatch, models_dir):
    model, backfill = _model(), b"new backfill"
    files = _published(_entry(model, backfill), model, backfill)
    del files[f"{BASE}/de_lu-2026-11.json.gz"]
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb(files), models_dir) == 1
    assert _snapshot(models_dir) == before


@pytest.mark.parametrize(
    ("schema", "cutoff"),
    [(2, "2026-11-01"), (1, "2026-10-01"), (1, "2026-09-01")],
    ids=["other-feature-schema", "same-cutoff", "older-cutoff"],
)
def test_fetch_published_model_keeps_bundled_when_not_usable(
    monkeypatch, models_dir, schema, cutoff
):
    model, backfill = _model(schema=schema, cutoff=cutoff), b"new backfill"
    web = FakeWeb(_published(_entry(model, backfill, schema, cutoff), model, backfill))
    before = _snapshot(models_dir)
    assert _run(monkeypatch, web, models_dir) == 0
    assert _snapshot(models_dir) == before
    assert web.requested == [MANIFEST_URL]


def test_fetch_published_model_skips_zones_not_bundled(monkeypatch, models_dir):
    model, backfill = _model(), b"new backfill"
    files = _published(_entry(model, backfill), model, backfill)
    manifest = json.loads(files[MANIFEST_URL])
    manifest["zones"] = {"nl": manifest["zones"]["de_lu"]}
    files[MANIFEST_URL] = json.dumps(manifest).encode()
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb(files), models_dir) == 0
    assert _snapshot(models_dir) == before


def test_fetch_published_model_rejects_unknown_manifest_format(monkeypatch, models_dir):
    web = FakeWeb({MANIFEST_URL: json.dumps({"format": 2, "zones": {}}).encode()})
    assert _run(monkeypatch, web, models_dir) == 1
