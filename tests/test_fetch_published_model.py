"""tools/fetch_published_model.py: the published model replaces the bundled one at release."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.error

import pytest

from core.features import FEATURE_SCHEMA
from core.model import dump_model
from tools import fetch_published_model as fpm

MANIFEST_URL = "https://example.test/models/manifest.json"
BASE = "https://example.test/models"


TINY = None  # tests/conftest.py's tiny_model_file, the base of every model file here


@pytest.fixture(autouse=True)
def _tiny(tiny_model_file):
    global TINY
    TINY = tiny_model_file


def _model(cutoff: str = "2026-11-01", **changes) -> bytes:
    """A real model file (the tiny test model) with this cutoff and version."""
    model = replace(TINY, version=f"de_lu-{cutoff[:7]}", train_cutoff=date.fromisoformat(cutoff))
    return dump_model(replace(model, **changes))


def _backfill(version: str = "de_lu-2026-11") -> bytes:
    rows = [["2026-10-31T23:00:00+00:00", 1, 80.0, 90.0, 70.0]]
    doc = {"format": 1, "model_version": version, "rows": rows}
    return gzip.compress(json.dumps(doc).encode(), mtime=0)


def _asset(name: str, data: bytes) -> dict:
    return {
        "name": name,
        "url": f"{BASE}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _entry(
    model: bytes, backfill: bytes, schema: int = FEATURE_SCHEMA, cutoff: str = "2026-11-01"
) -> dict:
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
    (d / "de_lu.json.gz").write_bytes(_model(cutoff="2026-10-01"))
    (d / "de_lu.backfill.json.gz").write_bytes(_backfill("de_lu-2026-10"))
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
    model, backfill = _model(), _backfill()
    web = FakeWeb(_published(_entry(model, backfill), model, backfill))
    assert _run(monkeypatch, web, models_dir) == 0
    assert _snapshot(models_dir) == {
        "de_lu.backfill.json.gz": backfill,
        "de_lu.json.gz": model,
    }


def test_fetch_published_model_verifies_sha(monkeypatch, models_dir, capsys):
    model, backfill = _model(), _backfill()
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
    model, backfill = _model(), _backfill()
    files = _published(_entry(model, backfill), model, backfill)
    del files[f"{BASE}/de_lu-2026-11.json.gz"]
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb(files), models_dir) == 1
    assert _snapshot(models_dir) == before


@pytest.mark.parametrize(
    ("schema", "cutoff"),
    [
        (FEATURE_SCHEMA + 1, "2026-11-01"),
        (FEATURE_SCHEMA - 1, "2026-11-01"),  # published before this commit's schema bump
        (FEATURE_SCHEMA, "2026-10-01"),
        (FEATURE_SCHEMA, "2026-09-01"),
    ],
    ids=["newer-feature-schema", "older-feature-schema", "same-cutoff", "older-cutoff"],
)
def test_fetch_published_model_keeps_bundled_when_not_usable(
    monkeypatch, models_dir, schema, cutoff
):
    model = _model(cutoff=cutoff, feature_schema=schema)
    backfill = _backfill(f"de_lu-{cutoff[:7]}")
    web = FakeWeb(_published(_entry(model, backfill, schema, cutoff), model, backfill))
    before = _snapshot(models_dir)
    assert _run(monkeypatch, web, models_dir) == 0
    assert _snapshot(models_dir) == before
    assert web.requested == [MANIFEST_URL]


def test_fetch_published_model_skips_zones_not_bundled(monkeypatch, models_dir):
    model, backfill = _model(), _backfill()
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


def _manifest_doc(files: dict[str, bytes]) -> dict:
    return json.loads(files[MANIFEST_URL])


def _set_manifest(files: dict[str, bytes], doc: dict) -> None:
    files[MANIFEST_URL] = json.dumps(doc).encode()


def test_fetch_published_model_skips_a_revoked_version(monkeypatch, models_dir, capsys):
    model, backfill = _model(), _backfill()
    files = _published(_entry(model, backfill), model, backfill)
    _set_manifest(files, {**_manifest_doc(files), "revoked": ["de_lu-2026-11"]})
    before = _snapshot(models_dir)
    web = FakeWeb(files)
    assert _run(monkeypatch, web, models_dir) == 0
    assert _snapshot(models_dir) == before
    assert web.requested == [MANIFEST_URL]
    assert "revoked" in capsys.readouterr().out


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/models/de_lu-2026-11.json.gz",
        "https://elsewhere.test/models/de_lu-2026-11.json.gz",
        "https://example.test/models/sub/de_lu-2026-11.json.gz",
        "https://example.test/models/../other/de_lu-2026-11.json.gz",
    ],
    ids=["http", "other-host", "sub-path", "dot-dot"],
)
def test_fetch_published_model_only_downloads_from_the_release(monkeypatch, models_dir, url):
    model, backfill = _model(), _backfill()
    entry = _entry(model, backfill)
    entry["model"]["url"] = url
    files = _published(entry, model, backfill)
    before = _snapshot(models_dir)
    web = FakeWeb(files)
    assert _run(monkeypatch, web, models_dir) == 1
    assert _snapshot(models_dir) == before
    assert url not in web.requested


def _flipped(data: bytes) -> bytes:
    i = len(data) // 2
    return data[:i] + bytes([data[i] ^ 0xFF]) + data[i + 1 :]


@pytest.mark.parametrize(
    ("case", "why"),
    [
        ("corrupt model", "does not load"),
        ("implausible model", "does not load"),
        ("other zone", "is for zone 'at'"),
        ("other version", "manifest de_lu-2026-11"),
        ("other cutoff", "has cutoff 2026-11-02"),
        ("corrupt backfill", "does not parse"),
        ("backfill of another model", "has no rows of that model"),
    ],
)
def test_fetch_published_model_checks_the_files_as_the_integration(
    monkeypatch, models_dir, capsys, case, why
):
    # The checksums match the manifest: only loading the files shows they are unusable.
    model, backfill = _model(), _backfill()
    if case == "corrupt model":
        model = _flipped(model)
    elif case == "implausible model":
        median = replace(TINY.median, value=TINY.median.value * 1e6)
        model = _model(median=median)
    elif case == "other zone":
        model = _model(zone="at")
    elif case == "other version":
        model = _model(version="de_lu-2026-10")
    elif case == "other cutoff":
        model = _model(train_cutoff=date(2026, 11, 2))
    elif case == "corrupt backfill":
        backfill = _flipped(backfill)
    elif case == "backfill of another model":
        backfill = _backfill("de_lu-2026-10")
    files = _published(_entry(model, backfill), model, backfill)
    before = _snapshot(models_dir)
    assert _run(monkeypatch, FakeWeb(files), models_dir) == 1
    assert _snapshot(models_dir) == before
    assert why in capsys.readouterr().err


def test_fetch_published_model_writes_the_backfill_first(monkeypatch, models_dir):
    model, backfill = _model(), _backfill()
    written = []
    real = fpm._write_atomic

    def record(path, data):
        written.append(path.name)
        real(path, data)

    monkeypatch.setattr(fpm, "_write_atomic", record)
    web = FakeWeb(_published(_entry(model, backfill), model, backfill))
    assert _run(monkeypatch, web, models_dir) == 0
    assert written == ["de_lu.backfill.json.gz", "de_lu.json.gz"]


@pytest.mark.parametrize(
    "manifest",
    [
        {"format": 1, "zones": {}, "revoked": "de_lu-2026-11"},
        {"format": 1, "zones": {"de_lu": {"version": "de_lu-2026-11"}}},
        ["format", 1],
    ],
    ids=["revoked-not-a-list", "entry-incomplete", "not-an-object"],
)
def test_fetch_published_model_rejects_a_malformed_manifest(monkeypatch, models_dir, manifest):
    web = FakeWeb({MANIFEST_URL: json.dumps(manifest).encode()})
    before = _snapshot(models_dir)
    assert _run(monkeypatch, web, models_dir) == 1
    assert _snapshot(models_dir) == before
