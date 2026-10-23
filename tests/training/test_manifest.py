"""Model-release manifest (no research data needed)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib

import pytest

from training.manifest import manifest_entry, merge_manifest, revoked_versions

BASE = "https://example.org/releases/download/models/"
METRICS = {
    "horizons": {str(h): {"mae": 17.0 + h, "skill": 0.5} for h in range(1, 7)},
    "all": {"mae": 19.21, "skill": 0.45, "p90_exceedance": 0.12, "rows": 25920},
}


@pytest.fixture
def model(tiny_model_file):
    return replace(tiny_model_file, metrics=METRICS)


def test_manifest_entry_sha_and_urls(model):
    entry = manifest_entry(b"model bytes", b"backfill", model, BASE)
    assert entry["version"] == "de_lu-2026-11"
    assert entry["feature_schema"] == model.feature_schema
    assert entry["train_cutoff"] == "2026-11-01"
    assert datetime.fromisoformat(entry["created"]) == model.created
    assert entry["model"] == {
        "name": "de_lu-2026-11.json.gz",
        "url": "https://example.org/releases/download/models/de_lu-2026-11.json.gz",
        "sha256": hashlib.sha256(b"model bytes").hexdigest(),
        "size": 11,
    }
    assert entry["backfill"] == {
        "name": "de_lu-2026-11.backfill.json.gz",
        "url": "https://example.org/releases/download/models/de_lu-2026-11.backfill.json.gz",
        "sha256": hashlib.sha256(b"backfill").hexdigest(),
        "size": 8,
    }
    assert entry["metrics"] == {"mae_h1": 18.0, "mae": 19.21, "skill": 0.45, "p90_exceedance": 0.12}


def test_merge_manifest_keeps_other_zones(model):
    new = manifest_entry(b"m", b"b", model, BASE)
    old = {
        "format": 1,
        "updated": "2026-10-03T02:10:00+00:00",
        "zones": {"de_lu": {"version": "de_lu-2026-10"}, "at": {"version": "at-2026-10"}},
    }
    merged = merge_manifest(old, "de_lu", new)
    assert merged["format"] == 1
    assert merged["zones"] == {"de_lu": new, "at": {"version": "at-2026-10"}}
    assert datetime.fromisoformat(merged["updated"]) > datetime.fromisoformat(old["updated"])
    assert old["zones"]["de_lu"] == {"version": "de_lu-2026-10"}  # not mutated


def test_merge_manifest_first_run_and_wrong_format(model):
    new = manifest_entry(b"m", b"b", model, BASE)
    assert merge_manifest(None, "de_lu", new)["zones"] == {"de_lu": new}
    with pytest.raises(ValueError, match="format"):
        merge_manifest({"format": 2, "zones": {}}, "de_lu", new)


@pytest.mark.parametrize(
    ("manifest", "revoked"),
    [(None, []), ({"format": 1}, []), ({"revoked": ["de_lu-2026-09"]}, ["de_lu-2026-09"])],
)
def test_revoked_versions(manifest, revoked):
    assert revoked_versions(manifest) == revoked


def test_merge_manifest_keeps_the_revoked_list(model):
    new = manifest_entry(b"m", b"b", model, BASE)
    old = {
        "format": 1,
        "updated": "x",
        "zones": {"de_lu": {"version": "de_lu-2026-10"}},
        "revoked": ["de_lu-2026-09"],
    }
    merged = merge_manifest(old, "de_lu", new)
    assert merged["revoked"] == ["de_lu-2026-09"]
    assert "revoked" not in merge_manifest(None, "de_lu", new)
    assert merge_manifest({**old, "revoked": []}, "de_lu", new)["revoked"] == []


@pytest.mark.parametrize("revoked", ["de_lu-2026-09", [1], {"de_lu-2026-09": True}])
def test_merge_manifest_rejects_a_malformed_revoked_list(model, revoked):
    # Never dropped silently: a revoked model would come back for every client.
    old = {"format": 1, "updated": "x", "zones": {}, "revoked": revoked}
    with pytest.raises(ValueError, match="revoked"):
        merge_manifest(old, "de_lu", manifest_entry(b"m", b"b", model, BASE))
    with pytest.raises(ValueError, match="revoked"):
        revoked_versions(old)
