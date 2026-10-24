"""The project website: it builds in every language, no text is missing a translation,
the look-back data it ships stays small, and the accuracy tables are the model's."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import gzip
from itertools import pairwise
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

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


def _ct(x: float) -> str:
    return f"{x:.2f}"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _expected_rows(kind: str) -> dict[str, list[str]]:
    """Per horizon, the cells a hand-written accuracy table must show."""
    metrics = json.loads(gzip.decompress(MODEL.read_bytes()))["metrics"]["horizons"]
    trailing = json.loads(build_site.BACKTEST.read_text(encoding="utf-8"))["trailing"]
    rows = {}
    for h, m in metrics.items():
        t = trailing["horizons"][h]
        full = [_ct(m["mae"] / 10), _ct(m["naive_mae"] / 10), _pct(m["skill"])]
        last3 = [_ct(m["mae_3m"] / 10), _ct(t["naive_mae"]), _pct(t["skill"])]
        rows[h] = {
            "readme": [*full, *last3, _pct(m["p90_exceedance"])],
            "full": [*full, _pct(m["p90_exceedance"]), _pct(m["neg_p10_below_zero"])],
            "last3": last3,
        }[kind]
    return rows


def _table_rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for match in re.finditer(r"^\| D\+(\d) \|(.*)\|$", text, re.MULTILINE):
        rows.setdefault(match.group(1), [c.strip() for c in match.group(2).split("|")])
    return rows


def test_accuracy_tables_match_model():
    """The accuracy tables are written by hand; they must show the bundled model's metrics,
    and for the last three months the naive figures of the look-back data."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert _table_rows(readme) == _expected_rows("readme")
    docs = (REPO / "docs" / "accuracy.md").read_text(encoding="utf-8")
    full, rest = docs.split("### Last three months")
    assert _table_rows(full) == _expected_rows("full")
    assert _table_rows(rest.split("## Peaks")[0]) == _expected_rows("last3")


def test_backtest_times_are_unique_instants_across_the_fall_back():
    import site_data

    # 2026-10-25 in Berlin: 25 hours, 02:00 twice (first +02:00, then +01:00)
    start = datetime(2026, 10, 24, 22, tzinfo=UTC)
    hours = [start + timedelta(hours=k) for k in range(25)]
    t = site_data.timestamps(hours, ZoneInfo("Europe/Berlin"))
    assert len(set(t)) == 25
    assert t[:4] == [
        "2026-10-25T00:00+02:00",
        "2026-10-25T01:00+02:00",
        "2026-10-25T02:00+02:00",
        "2026-10-25T02:00+01:00",
    ]
    assert t[-1] == "2026-10-25T23:00+01:00"
    assert [datetime.fromisoformat(s) for s in t] == hours


def test_backtest_json_times_carry_their_offset():
    doc = json.loads(build_site.BACKTEST.read_text(encoding="utf-8"))
    pattern = re.compile(r"\d{4}-\d\d-\d\dT\d\d:00[+-]\d\d:\d\d")
    assert all(pattern.fullmatch(t) for t in doc["t"])
    instants = [datetime.fromisoformat(t) for t in doc["t"]]
    assert all(b - a == timedelta(hours=1) for a, b in pairwise(instants))


def test_trailing_errors_without_rows_fail_clearly():
    import site_data

    n = 200
    actual = [10.0] * n
    with pytest.raises(SystemExit, match="D\\+3: no trailing hour"):
        site_data.trailing_errors(3, actual, [None] * n, list(range(n)))
    got = site_data.trailing_errors(1, actual, [11.0] * n, list(range(n)))
    assert got["mae"] == 1.0 and got["naive_mae"] == 0.0 and got["skill"] is None


def _bundled_zones() -> list[dict]:
    return [
        json.loads(gzip.decompress(p.read_bytes()))
        for p in sorted(build_site.MODELS.glob("*.json.gz"))
        if not p.name.endswith(".backfill.json.gz")
    ]


def test_zone_map_highlights_every_bundled_model(tmp_path):
    build_site.build(tmp_path)
    models = _bundled_zones()
    assert models  # DE-LU at least
    for code, _, _ in build_site.LANGUAGES:
        html = (tmp_path / build_site.page_path(code) / "index.html").read_text(encoding="utf-8")
        section = html.split('<section id="zones"')[1].split("</section>")[0]
        highlighted = re.findall(
            r'<a class="zone" href="#zone-([a-z0-9_]+)" data-zone="([A-Z0-9-]+)">', section
        )
        assert sorted(slug for slug, _ in highlighted) == sorted(m["zone"] for m in models)
        assert section.count('<path class="land"') == 1
        for m in models:
            row = section.split(f'<tr id="zone-{m["zone"]}">')[1].split("</tr>")[0]
            cells = re.findall(r"<td>([0-9.,]+)</td>", row)
            assert len(cells) == 6
            if code == "en":
                expected = [
                    f"{m['metrics']['horizons'][str(h)]['mae'] / 10:.2f}" for h in range(1, 7)
                ]
                assert cells == expected


def test_zone_map_data_is_small_and_has_the_zones():
    doc = json.loads(build_site.ZONE_MAP.read_text(encoding="utf-8"))
    assert build_site.ZONE_MAP.stat().st_size < 64 * 1024
    assert set(doc["zones"]) == {"DE-LU", "NL", "DK1", "DK2", "NO2"}
    assert doc["zones"]["NO2"]["approximate"] is True
    assert not any(z["approximate"] for code, z in doc["zones"].items() if code != "NO2")
    assert "Natural Earth" in doc["source"] and "v5.1.2" in doc["source"]
    for zone in doc["zones"].values():
        assert re.fullmatch(r"(M[-0-9. ]+l[-0-9. ]+z)+", zone["path"])


def test_a_bundled_zone_without_a_map_shape_fails_the_build(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    doc = {"zone": "at", "version": "at-2026-10", "metrics": {"horizons": {}}}
    (models / "at.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
    monkeypatch.setattr(build_site, "MODELS", models)
    with pytest.raises(SystemExit, match="no shape for the bundled zone at"):
        build_site.zone_figures()


def test_zone_tables_match_the_bundled_models():
    """README's zone table: per bundled model its MAE D+1 … D+6 and skill over all horizons.
    The NL section of docs/accuracy.md: NL's report per horizon."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Supported zones")[1].split("\n## ")[0]
    rows = {
        m.group(1): [c.strip() for c in m.group(2).split("|")]
        for m in re.finditer(r"^\| ([A-Z][A-Z0-9-]*) \|(.*)\|$", section, re.MULTILINE)
    }
    models = {m["zone"]: m for m in _bundled_zones()}
    assert sorted(rows) == sorted(code.upper().replace("_", "-") for code in models)
    for slug, model in models.items():
        metrics = model["metrics"]
        expected = [_ct(metrics["horizons"][str(h)]["mae"] / 10) for h in range(1, 7)]
        assert rows[slug.upper().replace("_", "-")][1:] == [
            *expected,
            _pct(metrics["all"]["skill"]),
        ]

    docs = (REPO / "docs" / "accuracy.md").read_text(encoding="utf-8")
    nl = docs.split("## Netherlands (NL)")[1].split("\n## ")[0]
    horizons = models["nl"]["metrics"]["horizons"]
    assert _table_rows(nl) == {
        h: [
            _ct(m["mae"] / 10),
            _ct(m["mae_3m"] / 10),
            _ct(m["naive_mae"] / 10),
            _pct(m["skill"]),
            _pct(m["p90_exceedance"]),
            _pct(m["neg_p10_below_zero"]),
        ]
        for h, m in horizons.items()
    }
