"""The project website: it builds in every language, no text is missing a translation,
the look-back data it ships stays small, and the accuracy tables are the model's."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import sys

import pytest

pytest.importorskip("jinja2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_site  # noqa: E402

MODEL = REPO / "custom_components" / "spot_forecast" / "models" / "de_lu.json.gz"


def test_site_builds_all_languages(tmp_path):
    build_site.build(tmp_path)
    for code, _, _ in build_site.LANGUAGES:
        page = tmp_path / build_site.page_path(code) / "index.html"
        html = page.read_text(encoding="utf-8")
        assert f'<html lang="{code}">' in html
        # every language page links every other one
        for other, _, _ in build_site.LANGUAGES:
            assert (
                f'hreflang="{other}" href="{build_site.BASE_URL}{build_site.page_path(other)}"'
                in html
            )
        assert "{{" not in html and "{%" not in html
    for shipped in (
        "assets/icon.svg",
        "assets/screenshots/forecast_light.png",
        "assets/screenshots/forecast_dark.png",
        "assets/backtest.json",
        "sitemap.xml",
    ):
        assert (tmp_path / shipped).is_file(), shipped


def test_translations_complete():
    texts = {
        code: json.loads((build_site.SITE / "i18n" / f"{code}.json").read_text(encoding="utf-8"))
        for code, _, _ in build_site.LANGUAGES
    }
    reference = build_site._keys(texts[build_site.DEFAULT])
    for code, table in texts.items():
        assert build_site._keys(table) == reference, code


def test_translation_gap_fails_the_build(tmp_path, monkeypatch):
    i18n = tmp_path / "i18n"
    i18n.mkdir()
    for code, _, _ in build_site.LANGUAGES:
        table = json.loads((build_site.SITE / "i18n" / f"{code}.json").read_text("utf-8"))
        if code != build_site.DEFAULT:
            table["hero"].pop("title")
        (i18n / f"{code}.json").write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(build_site, "SITE", tmp_path)
    with pytest.raises(SystemExit, match=r"de\.json lacks hero\.title"):
        build_site.load_texts()


def test_backtest_json_size():
    data = build_site.BACKTEST
    assert data.stat().st_size < 1.5 * 1024 * 1024
    doc = json.loads(data.read_text(encoding="utf-8"))
    n = len(doc["t"])
    assert n > 24 * 150  # about six months, hourly
    assert len(doc["actual"]) == n
    assert sorted(doc["forecast"]) == [str(h) for h in range(1, 7)]
    assert all(len(series) == n for series in doc["forecast"].values())
    model = json.loads(gzip.decompress(MODEL.read_bytes()))
    assert doc["model_version"] == model["version"]


@pytest.mark.parametrize("page", ["README.md", "docs/accuracy.md"])
def test_accuracy_tables_match_model(page):
    """The accuracy tables are written by hand; they must show the bundled model's metrics."""
    metrics = json.loads(gzip.decompress(MODEL.read_bytes()))["metrics"]
    # the per-horizon table; docs/accuracy.md has a peaks table with D+ rows after it
    text = (REPO / page).read_text(encoding="utf-8").split("## Peaks")[0]
    for h, m in metrics["horizons"].items():
        row = re.search(rf"^\| D\+{h} \|(.*)\|$", text, re.MULTILINE)
        assert row, f"{page} has no accuracy row for D+{h}"
        cells = [c.strip() for c in row.group(1).split("|")]
        assert cells[:4] == [
            f"{m['mae'] / 10:.2f}",
            f"{m['mae_3m'] / 10:.2f}",
            f"{m['naive_mae'] / 10:.2f}",
            f"{m['skill']:.0%}",
        ], f"D+{h}"
