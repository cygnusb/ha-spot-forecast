"""tools/check_model_age.py: the daily signal that the monthly training stopped."""

from __future__ import annotations

from datetime import date
import io
import json
import urllib.error
import urllib.request

import pytest

from tools import check_model_age as cma

URL = "https://example.test/models/manifest.json"
API = "https://api.github.com/repos/o/r/releases/tags/models"
TODAY = date(2026, 12, 20)


def _manifest(**cutoffs: str) -> bytes:
    zones = {
        slug: {"version": f"{slug}-{cutoff[:7]}", "train_cutoff": cutoff}
        for slug, cutoff in cutoffs.items()
    }
    return json.dumps({"format": 1, "updated": "x", "zones": zones}).encode()


class FakeWeb:
    """``urlopen`` for the manifest (``body`` or HTTP ``status``) and the release API."""

    def __init__(self, body: bytes | None, status: int = 404, release: int = 404):
        self.body, self.status, self.release = body, status, release
        self.headers: dict[str, str] = {}

    def __call__(self, url, timeout: float):
        if isinstance(url, urllib.request.Request):
            assert url.full_url == API
            self.headers = dict(url.header_items())
            if self.release == 200:
                return io.BytesIO(b"{}")
            raise urllib.error.HTTPError(API, self.release, "nope", {}, None)
        assert url == URL
        if self.body is None:
            raise urllib.error.HTTPError(url, self.status, "nope", {}, None)
        return io.BytesIO(self.body)


def _run(monkeypatch, body: bytes | None = None, status: int = 404, release: int = 404) -> int:
    web = FakeWeb(body, status, release)
    monkeypatch.setattr(cma.urllib.request, "urlopen", web)
    return cma.main(["--manifest-url", URL, "--release-api-url", API], today=TODAY)


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


def test_a_cutoff_in_the_future_fails(monkeypatch, capsys):
    assert _run(monkeypatch, _manifest(de_lu="2026-12-27")) == 0  # 7 days ahead
    assert _run(monkeypatch, _manifest(de_lu="2026-12-28")) == 1  # 8 days ahead
    assert "8 days in the future" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"format": 2, "zones": {}}).encode(),
        json.dumps(["not", "a", "manifest"]).encode(),
        b"<html>",
        json.dumps({"format": 1, "zones": {"de_lu": {"version": "x"}}}).encode(),
    ],
    ids=["other-format", "not-an-object", "not-json", "no-cutoff"],
)
def test_a_malformed_manifest_fails(monkeypatch, capsys, body):
    assert _run(monkeypatch, body) == 1
    assert "::error::" in capsys.readouterr().out


def test_no_release_yet_is_quiet(monkeypatch, capsys):
    assert _run(monkeypatch) == 0
    assert "::notice::No models release yet" in capsys.readouterr().out


def test_no_manifest_in_an_existing_release_fails(monkeypatch, capsys):
    assert _run(monkeypatch, release=200) == 1
    assert "exists but has no manifest.json" in capsys.readouterr().out


def test_the_release_check_sends_the_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "t0ken")
    web = FakeWeb(None)
    monkeypatch.setattr(cma.urllib.request, "urlopen", web)
    assert cma.main(["--manifest-url", URL, "--release-api-url", API], today=TODAY) == 0
    assert web.headers["Authorization"] == "Bearer t0ken"


@pytest.mark.parametrize("release", [403, 500])
def test_a_failed_release_check_fails(monkeypatch, release):
    assert _run(monkeypatch, release=release) == 1


@pytest.mark.parametrize("status", [403, 500])
def test_other_download_errors_fail(monkeypatch, status):
    assert _run(monkeypatch, status=status) == 1


def test_future_limit_matches_the_integration():
    """The check is stdlib-only, so it keeps its own copy of the integration's limit."""
    from core.manifest import MAX_CUTOFF_AHEAD

    assert MAX_CUTOFF_AHEAD.days == cma.FUTURE_DAYS
