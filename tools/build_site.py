#!/usr/bin/env python3
"""Build the project page in English and German.

site/index.html.j2 is the page, site/i18n/<lang>.json its text. English goes to the
root of the output directory, German to de/. Each page links its translation
(hreflang). The accuracy table is read from the bundled model's ``metrics``, so the page
always states the figures of the model a fresh install runs; the look-back chart loads
site/data/backtest_<zone>.json (tools/site_data.py) from assets/. A sitemap index at
the root lists these pages and the one MkDocs writes for /docs/.

The texts are trusted HTML from this repository, so they are not escaped. A key
missing from a translation fails the build, so a new sentence on the page cannot go
out in English only.

Usage:
    python3 tools/build_site.py _site
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import shutil
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined, pass_context
from markupsafe import Markup

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"
ASSETS = REPO / "assets"
MODEL = REPO / "custom_components" / "spot_forecast" / "models" / "de_lu.json.gz"
BACKTEST = SITE / "data" / "backtest_de_lu.json"
BASE_URL = "https://cygnusb.github.io/ha-spot-forecast/"
REPO_URL = "https://github.com/cygnusb/ha-spot-forecast"
DEFAULT = "en"

# Code, name in its own language, og:locale. The order is the menu's.
LANGUAGES = [
    ("en", "English", "en_GB"),
    ("de", "Deutsch", "de_DE"),
]


def page_path(lang: str) -> str:
    """The page's path below the site root: '' for English, 'de/' for German."""
    return "" if lang == DEFAULT else f"{lang}/"


def load_texts() -> dict[str, dict]:
    texts = {
        code: json.loads((SITE / "i18n" / f"{code}.json").read_text(encoding="utf-8"))
        for code, _, _ in LANGUAGES
    }
    reference = _keys(texts[DEFAULT])
    problems = []
    for code, table in texts.items():
        keys = _keys(table)
        if missing := sorted(reference - keys):
            problems.append(f"{code}.json lacks {', '.join(missing)}")
        if extra := sorted(keys - reference):
            problems.append(f"{code}.json has keys en.json does not: {', '.join(extra)}")
    if problems:
        raise SystemExit("\n".join(problems))
    return {code: _markup(table) for code, table in texts.items()}


def _keys(table, prefix=""):
    keys = set()
    for key, value in table.items():
        if isinstance(value, dict):
            keys |= _keys(value, f"{prefix}{key}.")
        else:
            keys.add(f"{prefix}{key}")
    return keys


def _markup(value):
    if isinstance(value, dict):
        return {key: _markup(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_markup(item) for item in value]
    return Markup(value) if isinstance(value, str) else value


def model_figures() -> dict:
    """The bundled model's accuracy in ct/kWh (its metrics are EUR/MWh) and fractions.

    The report has the naive error and the skill for the full period only; those of its
    trailing three months come from the look-back data (tools/site_data.py), measured on
    the same hours as the report's ``mae_3m``.
    """
    model = json.loads(gzip.decompress(MODEL.read_bytes()))
    metrics = model["metrics"]
    trailing = json.loads(BACKTEST.read_text(encoding="utf-8"))["trailing"]
    rows = [
        {
            "h": int(h),
            "mae": m["mae"] / 10,
            "mae_3m": m["mae_3m"] / 10,
            "naive": m["naive_mae"] / 10,
            "skill": m["skill"],
            "p90": m["p90_exceedance"],
            "peak_bias": m["peak_bias"] / 10,
            "naive_3m": trailing["horizons"][h]["naive_mae"],
            "skill_3m": trailing["horizons"][h]["skill"],
        }
        for h, m in sorted(metrics["horizons"].items(), key=lambda item: int(item[0]))
    ]
    return {
        "version": model["version"],
        "cutoff": model["train_cutoff"],
        "walk_forward_from": metrics["walk_forward_from"],
        "trailing_3m_from": metrics["trailing_3m_from"],
        "until": (date.fromisoformat(model["train_cutoff"]) - timedelta(days=1)).isoformat(),
        "threshold": round(metrics["peak_threshold"] / 10, 1),
        "rows": rows,
    }


def build(out: Path) -> None:
    texts = load_texts()
    env = Environment(
        loader=FileSystemLoader(SITE),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    env.filters["num"] = _num
    env.filters["pct"] = _pct
    env.filters["month"] = _month
    env.filters["mon"] = _mon
    template = env.get_template("index.html.j2")
    figures = model_figures()
    languages = [
        {
            "code": code,
            "name": name,
            "locale": locale,
            "url": BASE_URL + page_path(code),
            "path": page_path(code),
        }
        for code, name, locale in LANGUAGES
    ]
    for lang in languages:
        # Relative to the page: the English page sits at the root.
        root = "" if lang["code"] == DEFAULT else "../"
        t = texts[lang["code"]]
        html = template.render(
            t=t,
            lang=lang,
            languages=languages,
            default_url=BASE_URL,
            repo_url=REPO_URL,
            root=root,
            og_image=BASE_URL + "assets/screenshots/forecast_light.png",
            json_ld=_json_ld(t, lang),
            m=figures,
            chart_text=Markup(
                json.dumps(t["chart"]["js"], ensure_ascii=False).replace("</", "<\\/")
            ),
        )
        target = out / lang["path"] / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")

    _copy_assets(out)
    (out / "sitemap-pages.xml").write_text(_page_sitemap(languages), encoding="utf-8")
    (out / "sitemap.xml").write_text(_sitemap_index(), encoding="utf-8")


@pass_context
def _num(context, value: float, digits: int = 1) -> str:
    """A figure as the page's language writes it: 1.8 in English, 1,8 in German."""
    text = f"{value:.{digits}f}"
    return text if context["lang"]["code"] == DEFAULT else text.replace(".", ",")


@pass_context
def _pct(context, value: float) -> str:
    """A fraction as a whole percentage: 47% in English, 47 % in German."""
    return f"{value:.0%}" if context["lang"]["code"] == DEFAULT else f"{value * 100:.0f} %"


@pass_context
def _month(context, iso: str) -> str:
    """'2026-07-01' as 'July 2026' / 'Juli 2026', from the page's month names."""
    d = date.fromisoformat(iso)
    return f"{context['t']['months'][d.month - 1]} {d.year}"


@pass_context
def _mon(context, iso: str) -> str:
    """'2026-07-01' as 'Jul 2026' / 'Juli 2026', for table headings."""
    d = date.fromisoformat(iso)
    return f"{context['t']['months_short'][d.month - 1]} {d.year}"


def _json_ld(t, lang) -> Markup:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "Spot Forecast",
        "description": str(t["meta"]["description"]),
        "url": lang["url"],
        "inLanguage": lang["code"],
        "applicationCategory": "UtilitiesApplication",
        "applicationSubCategory": "Home Assistant integration",
        "operatingSystem": "Home Assistant",
        "license": "https://opensource.org/licenses/MIT",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
        "codeRepository": REPO_URL,
        "screenshot": BASE_URL + "assets/screenshots/forecast_light.png",
    }
    # </ cannot end the script element early.
    return Markup(json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))


def _copy_assets(out: Path) -> None:
    target = out / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("icon.svg", "logo.svg", "logo.png"):
        shutil.copy2(ASSETS / name, target / name)
    for folder in ("screenshots", "fonts"):
        shutil.copytree(ASSETS / folder, target / folder, dirs_exist_ok=True)
    shutil.copy2(BACKTEST, target / "backtest.json")


def _page_sitemap(languages) -> str:
    today = date.today().isoformat()
    links = (
        "".join(
            f'    <xhtml:link rel="alternate" hreflang="{lang["code"]}" href="{lang["url"]}"/>\n'
            for lang in languages
        )
        + f'    <xhtml:link rel="alternate" hreflang="x-default" href="{BASE_URL}"/>\n'
    )
    entries = "".join(
        f"  <url>\n    <loc>{lang['url']}</loc>\n    <lastmod>{today}</lastmod>\n{links}  </url>\n"
        for lang in languages
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
        ' xmlns:xhtml="http://www.w3.org/1999/xhtml">\n'
        f"{entries}</urlset>\n"
    )


def _sitemap_index() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <sitemap><loc>{BASE_URL}sitemap-pages.xml</loc></sitemap>\n"
        f"  <sitemap><loc>{BASE_URL}docs/sitemap.xml</loc></sitemap>\n"
        "</sitemapindex>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="output directory, e.g. _site")
    build(parser.parse_args().out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
