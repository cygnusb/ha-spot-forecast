"""The shipped Plotly Graph Card dashboards and the scenarios their screenshots are made from.

The screenshots themselves (tools/gen_card_screenshots.py) are the visual test; these
checks keep the YAML honest about the ids it reads and the theme it draws with, and
make sure the scenarios still come out of the real core code.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CARDS = {
    "forecast": REPO / "examples" / "plotly_forecast.yaml",
    "lookback": REPO / "examples" / "plotly_lookback.yaml",
}
RESEARCH = Path(os.environ.get("SPOT_RESEARCH_DIR", REPO / "research"))

PRICE_SENSOR = "sensor.spot_forecast_de_lu_price"
PRICE_STAT = "spot_forecast:de_lu_price"
FORECAST_STATS = {f"spot_forecast:de_lu_forecast_h{h}" for h in range(1, 7)}
# Everything a card may read: the integration's ids (Global Constraints) and sun.sun.
ALLOWED_IDS = {
    PRICE_SENSOR,
    PRICE_STAT,
    *FORECAST_STATS,
    "binary_sensor.spot_forecast_de_lu_peak_risk",
    "sun.sun",
}
# entity ids as the card code writes them: quoted strings (JS variables like `sun.x` aside)
_ID = re.compile(
    r"""(?<=["'])(?:(?:sensor|binary_sensor|sun)\.[a-z0-9_]+|spot_forecast:[a-z0-9_]+)(?=["'])""",
)


def _load(name: str) -> dict:
    return yaml.safe_load(CARDS[name].read_text(encoding="utf-8"))


def _entity_ids(card: dict) -> set[str]:
    return {e["entity"] for e in card["entities"] if isinstance(e, dict) and e.get("entity")}


@pytest.mark.parametrize("name", sorted(CARDS))
def test_card_yaml_parses_and_references_real_ids(name):
    card = _load(name)
    assert card["type"] == "custom:plotly-graph"
    assert _entity_ids(card) <= ALLOWED_IDS
    # ids inside the $fn / $ex code count too, comments excluded
    code = "\n".join(
        line.split(" #", 1)[0]
        for line in CARDS[name].read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    referenced = set(_ID.findall(code))
    assert referenced <= ALLOWED_IDS, referenced - ALLOWED_IDS
    if name == "forecast":
        # a visible trace bound to the sensor, so the card redraws when the sensor changes
        assert PRICE_SENSOR in _entity_ids(card)
        assert PRICE_STAT in _entity_ids(card)
        assert "binary_sensor.spot_forecast_de_lu_peak_risk" in referenced
        assert "sun.sun" in referenced
    else:
        assert {PRICE_STAT, *FORECAST_STATS} <= _entity_ids(card)


_HARD_INK = re.compile(
    r"#000(?:000)?\b|#fff(?:fff)?\b|\bblack\b|\bwhite\b"
    r"|rgba?\(\s*0\s*,\s*0\s*,\s*0\b|rgba?\(\s*255\s*,\s*255\s*,\s*255\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("name", sorted(CARDS))
def test_cards_use_theme_variables_for_ink(name):
    text = CARDS[name].read_text(encoding="utf-8")
    assert not _HARD_INK.findall(text), _HARD_INK.findall(text)
    # ink, secondary text and the card background come from the HA theme
    for var in ("primary-text-color", "secondary-text-color", "card-background-color"):
        assert f"css_vars['{var}']" in text, var


@pytest.mark.skipif(not (RESEARCH / "data" / "raw").exists(), reason="research raw data missing")
def test_scenarios_build_from_core():
    from tools.card_scenarios import scenario

    s = scenario("default")
    prices = s["states"][PRICE_SENSOR]["attributes"]["prices"]
    sources = {p["source"] for p in prices}
    assert sources == {"market", "forecast"}
    forecast = [p for p in prices if p["source"] == "forecast"]
    assert all({"p90", "p10", "h"} <= p.keys() for p in forecast)
    assert {p["h"] for p in forecast} == set(range(1, 7))
    assert all(p["p10"] <= p["price"] <= p["p90"] for p in forecast)
    assert s["states"]["binary_sensor.spot_forecast_de_lu_peak_risk"]["attributes"]["threshold"]
    assert s["states"]["sun.sun"]["attributes"]["next_rising"]
    for statistic_id in (PRICE_STAT, *FORECAST_STATS):
        rows = s["statistics"][statistic_id]
        assert len(rows) > 24 * 30, statistic_id
        assert {"start", "mean", "state"} <= rows[0].keys()
    # the ledger finalized yesterday's prices, which the forecast card shows from statistics
    assert s["statistics"][PRICE_STAT][-1]["start"] >= s["now_ms"] - 24 * 3600 * 1000

    stale = scenario("stale")
    entries = stale["states"][PRICE_SENSOR]["attributes"]["prices"]
    assert entries and {p["source"] for p in entries} == {"market"}
    assert stale["states"][PRICE_SENSOR]["attributes"]["stale"] is True
