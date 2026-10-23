"""tools/check_model_age.py: the daily signal that the monthly training stopped."""

from __future__ import annotations

from datetime import date
import io
import json
import urllib.error

import pytest

from tools import check_model_age as cma

URL = "https://example.test/models/manifest.json"
TODAY = date(2026, 12, 20)


def _manifest(**cutoffs: str) -> bytes:
    zones = {
        slug: {"version": f"{slug}-{cutoff[:7]}", "train_cutoff": cutoff}
        for slug, cutoff in cutoffs.items()
    }
    return json.dumps({"format": 1, "updated": "x", "zones": zones}).encode()


def _run(monkeypatch, body: bytes | None = None, status: int = 404) -> int:
    def fake(url: str, timeout: float):
        assert url == URL
        if body is None:
            raise urllib.error.HTTPError(url, status, "nope", {}, None)
        return io.BytesIO(body)

    monkeypatch.setattr(cma.urllib.request, "urlopen", fake)
    return cma.main(["--manifest-url", URL], today=TODAY)


def test_fresh_models_pass(monkeypatch, capsys):
    assert _run(monkeypatch, _manifest(de_lu="2026-12-01", at="2026-11-05")) == 0
    assert "::error::" not in capsys.readouterr().out


def test_limit_is_inclusive(monkeypatch):
    assert _run(monkeypatch, _manifest(de_lu="2026-11-05")) == 0  # 45 days


def test_a_stale_zone_fails_and_is_named(monkeypatch, capsys):
    assert _run(monkeypatch, _manifest(de_lu="2026-12-01", at="2026-11-04")) == 1  # 46 days
    out = capsys.readouterr().out
    assert "::error::at: published model at-2026-11 has train_cutoff 2026-11-04, 46 days" in out
    assert "de_lu" not in out


def test_no_manifest_yet_is_quiet(monkeypatch, capsys):
    assert _run(monkeypatch) == 0
    assert "::notice::No published manifest yet" in capsys.readouterr().out


@pytest.mark.parametrize("status", [403, 500])
def test_other_download_errors_fail(monkeypatch, status):
    assert _run(monkeypatch, status=status) == 1
