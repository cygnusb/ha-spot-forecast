"""Reading the ``models`` release manifest, in the exact shape ``training.manifest`` writes."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import hashlib

import pytest

from core.manifest import ManifestEntry, parse_manifest, wants_download
from training.manifest import manifest_entry, merge_manifest

BASE = "https://example.org/releases/download/models/"
METRICS = {
    "horizons": {str(h): {"mae": 17.0 + h, "skill": 0.5} for h in range(1, 7)},
    "all": {"mae": 19.21, "skill": 0.45, "p90_exceedance": 0.12, "rows": 25920},
}


@pytest.fixture
def doc(tiny_model_file) -> dict:
    model = replace(tiny_model_file, metrics=METRICS)
    return merge_manifest(None, "de_lu", manifest_entry(b"model", b"backfill", model, BASE))


def test_parse_manifest_and_wants_download(doc, tiny_model_file):
    entries = parse_manifest(doc)
    assert entries == {
        "de_lu": ManifestEntry(
            version="de_lu-2026-11",
            feature_schema=tiny_model_file.feature_schema,
            train_cutoff=date(2026, 11, 1),
            model_url=BASE + "de_lu-2026-11.json.gz",
            model_sha256=hashlib.sha256(b"model").hexdigest(),
            backfill_url=BASE + "de_lu-2026-11.backfill.json.gz",
            backfill_sha256=hashlib.sha256(b"backfill").hexdigest(),
        )
    }
    entry = entries["de_lu"]
    schema = tiny_model_file.feature_schema
    assert wants_download(entry, date(2026, 10, 1), schema)
    assert not wants_download(entry, date(2026, 11, 1), schema)  # same cutoff: not newer
    assert not wants_download(entry, date(2026, 12, 1), schema)
    assert not wants_download(entry, date(2026, 10, 1), schema + 1)


def test_parse_manifest_rejects_wrong_format(doc):
    with pytest.raises(ValueError, match="format"):
        parse_manifest({**doc, "format": 2})
    with pytest.raises(ValueError, match="format"):
        parse_manifest([doc])


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e.pop("model"),
        lambda e: e.update(train_cutoff="soon"),
        lambda e: e.update(version="../../de_lu-2026-11"),
        lambda e: e.update(version="at-2026-11"),  # not the zone it is listed under
        lambda e: e["model"].update(sha256="abc"),
        lambda e: e["backfill"].update(url="http://example.org/x.json.gz"),
    ],
)
def test_parse_manifest_rejects_malformed_entries(doc, change):
    change(doc["zones"]["de_lu"])
    with pytest.raises(ValueError):
        parse_manifest(doc)
