#!/usr/bin/env python3
"""Render the example Plotly cards in the real Plotly Graph Card, with real model output.

The card YAML in examples/ is loaded with PyYAML and mounted in the released
plotly-graph-card bundle (pinned below, downloaded once into tools/.cache/). A stub
``hass`` object serves the states and long-term statistics of a scenario from
tools/card_scenarios.py, which the real core code computes from research data. So the
screenshots are a test of the YAML: what you see is what the card draws on a real
instance with the integration installed.

Every shot is rendered in a light and a dark Home Assistant-like theme. The card sits
on a dashboard background with a margin around it, and the screenshot is taken of that
frame element.

Usage:
    npm install --prefix /tmp playwright@1.63.0   # its browser revision is in the cache
    .venv/bin/python tools/gen_card_screenshots.py [--theme light|dark|both]
        [--scenario NAME|all] [--sizes] [--switch-test] [--rebuild]

    SPOT_RESEARCH_DIR   research data (default research/), see research/README.md
    PLAYWRIGHT_DIR      the playwright package (default /tmp/node_modules/playwright)

The default scenario writes the README images assets/screenshots/{card}_{theme}.png.
Other scenarios, --sizes (360/600/1000 px) and --switch-test (look-back horizon
buttons across a card refresh) write into tools/.cache/shots/ for checking by eye.
The run fails on a card error message, a page error or an unknown websocket call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import urllib.request

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CARD_REPO = "dbuezas/lovelace-plotly-graph-card"
CARD_VERSION = "v4.0.2"  # the version the cards are written for
CACHE = REPO / "tools" / ".cache"
BUNDLE = CACHE / f"plotly-graph-card-{CARD_VERSION}"
SCENARIO_CACHE = CACHE / "scenarios"
WORK = CACHE / "shots"
OUT_DIR = REPO / "assets" / "screenshots"
CARDS = {
    "forecast": REPO / "examples" / "plotly_forecast.yaml",
    "lookback": REPO / "examples" / "plotly_lookback.yaml",
}
PLAYWRIGHT_DIR = os.environ.get("PLAYWRIGHT_DIR", "/tmp/node_modules/playwright")
CARD_WIDTH = 600  # CSS px of a two-column section; rendered at deviceScaleFactor 2
SIZE_WIDTHS = [360, 600, 1000]
MARGIN = 24  # dashboard visible around the card, CSS px
PORT = 18412


# ---------------------------------------------------------------------------
# The card bundle and the scenarios
# ---------------------------------------------------------------------------


def ensure_bundle() -> None:
    """Download every file of the pinned release once (the card loads its chunks lazily)."""
    if (BUNDLE / "plotly-graph-card.js").exists():
        return
    api = f"https://api.github.com/repos/{CARD_REPO}/releases/tags/{CARD_VERSION}"
    with urllib.request.urlopen(api, timeout=30) as r:
        assets = json.load(r)["assets"]
    tmp = BUNDLE.with_suffix(".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        print(f"downloading {asset['name']}")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as r:
            (tmp / asset["name"]).write_bytes(r.read())
    tmp.rename(BUNDLE)


def load_scenario(name: str, rebuild: bool) -> dict:
    """The scenario from the cache, rebuilt when asked or when its inputs are newer."""
    path = SCENARIO_CACHE / f"{name}.json"
    inputs = [
        REPO / "tools" / "card_scenarios.py",
        *(REPO / "custom_components" / "spot_forecast" / "core").glob("*.py"),
        *(REPO / "custom_components" / "spot_forecast" / "models").glob("*.gz"),
    ]
    if (
        not rebuild
        and path.exists()
        and path.stat().st_mtime > max(p.stat().st_mtime for p in inputs)
    ):
        return json.loads(path.read_text())
    from tools.card_scenarios import scenario

    print(f"building scenario {name}")
    data = scenario(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# HTML harness
# ---------------------------------------------------------------------------

# Home Assistant's default theme (light and dark) as far as the card reads it: the theme
# colours and fonts it passes to Plotly and the `css_vars` of the YAML, plus the
# dashboard and <ha-card> around it. Roboto is not installed here; the fallback keeps
# the renders reproducible on this machine.
_COMMON = {
    "ha-font-family-body": "Roboto, 'Helvetica Neue', Arial, sans-serif",
    "ha-font-size-s": "12px",
    "ha-font-weight-normal": "400",
    "ha-card-border-radius": "12px",
    "red-color": "#f44336",
    "pink-color": "#e91e63",
    "purple-color": "#926bc7",
    "deep-purple-color": "#6e41ab",
    "indigo-color": "#3f51b5",
    "blue-color": "#2196f3",
    "light-blue-color": "#03a9f4",
    "cyan-color": "#00bcd4",
    "teal-color": "#009688",
    "green-color": "#4caf50",
    "light-green-color": "#8bc34a",
    "lime-color": "#cddc39",
    "yellow-color": "#ffeb3b",
    "amber-color": "#ffc107",
    "orange-color": "#ff9800",
    "deep-orange-color": "#ff6f22",
    "brown-color": "#795548",
    "light-grey-color": "#bdbdbd",
    "grey-color": "#9e9e9e",
    "dark-grey-color": "#606060",
    "blue-grey-color": "#607d8b",
    "black-color": "#000000",
    "white-color": "#ffffff",
}
THEMES = {
    "light": {
        **_COMMON,
        "primary-background-color": "#fafafa",
        "lovelace-background": "#fafafa",
        "card-background-color": "#ffffff",
        "primary-text-color": "#212121",
        "secondary-text-color": "#727272",
        "divider-color": "rgba(0, 0, 0, 0.12)",
        "primary-color": "#03a9f4",
        "accent-color": "#ff9800",
        "error-color": "#db4437",
        "warning-color": "#ffa600",
        "success-color": "#43a047",
        "info-color": "#039be5",
        "disabled-color": "#bdbdbd",
    },
    "dark": {
        **_COMMON,
        "primary-background-color": "#111111",
        "lovelace-background": "#111111",
        "card-background-color": "#1c1c1c",
        "primary-text-color": "#e1e1e1",
        "secondary-text-color": "#9b9b9b",
        "divider-color": "rgba(225, 225, 225, 0.12)",
        "primary-color": "#03a9f4",
        "accent-color": "#ff9800",
        "error-color": "#db4437",
        "warning-color": "#ffa600",
        "success-color": "#43a047",
        "info-color": "#039be5",
        "disabled-color": "#6f6f6f",
    },
}

HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
{theme_vars}
  }}
  body {{ margin: 0; background: var(--lovelace-background); font-family: var(--ha-font-family-body); }}
  .frame {{ display: inline-block; padding: {margin}px; background: var(--lovelace-background); }}
  .slot {{ width: {width}px; }}
</style>
</head>
<body>
<div class="frame"><div class="slot" id="mount"></div></div>
<script>
  // Just enough of Home Assistant's <ha-card>: the frame, radius and border. The
  // Plotly card draws its own background (paper_bgcolor) inside it.
  customElements.define("ha-card", class extends HTMLElement {{
    constructor() {{
      super();
      this.attachShadow({{ mode: "open" }}).innerHTML = `<style>
        :host {{ display: block; border-radius: var(--ha-card-border-radius);
                 border: 1px solid var(--divider-color);
                 background: var(--card-background-color); color: var(--primary-text-color); }}
      </style><slot></slot>`;
    }}
  }});
  window.__problems = [];
  window.__ws = [];
  const scenario = {scenario_json};
  const inRange = (t, a, b) => t >= Date.parse(a) && (b === undefined || t < Date.parse(b));
  const listeners = {{}};
  // The connection is the card's key for sharing requests: one object for the page.
  const connection = {{
    subscribeMessage: async (callback, message) => {{
      window.__ws.push(message.type);
      if (message.type !== "history/stream") window.__problems.push("unknown subscription " + message.type);
      return async () => {{}};
    }},
    subscribeEvents: async (callback, event) => {{
      (listeners[event] ??= []).push(callback);
      return async () => {{}};
    }},
    addEventListener() {{}},
    removeEventListener() {{}},
  }};
  window.__fireEvent = (event) => (listeners[event] || []).forEach((cb) => cb({{ event_type: event }}));
  async function callWS(message) {{
    window.__ws.push(message.type);
    if (message.type === "recorder/statistics_during_period") {{
      const out = {{}};
      for (const id of message.statistic_ids) {{
        const rows = (scenario.statistics[id] || []).filter((r) => inRange(r.start, message.start_time, message.end_time));
        if (!rows.length) continue;
        const keep = message.types ? ["start", "end", ...message.types] : null;
        out[id] = keep ? rows.map((r) => Object.fromEntries(keep.filter((k) => k in r).map((k) => [k, r[k]]))) : rows;
      }}
      return out;
    }}
    if (message.type === "history/history_during_period") {{
      const out = {{}};
      for (const id of message.entity_ids) {{
        const rows = (scenario.history[id] || []).filter((r) => inRange(r.lu * 1000, message.start_time, message.end_time));
        out[id] = rows.map((r) => (message.no_attributes ? r : {{ ...r, a: {{}} }}));
      }}
      return out;
    }}
    window.__problems.push("unknown websocket call " + message.type);
    throw new Error("unknown websocket call " + message.type);
  }}
  window.__makeHass = (states) => ({{
    states,
    language: "de",
    locale: {{ language: "de", number_format: "language", time_format: "24", time_zone: "server", first_weekday: "monday" }},
    config: {{ time_zone: scenario.time_zone, unit_system: {{}}, location_name: "Home" }},
    themes: {{ darkMode: {dark}, theme: "default", themes: {{}} }},
    user: {{ id: "stub", name: "Stub", is_admin: true }},
    connection,
    callWS,
    callService: async (...args) => {{ window.__problems.push("callService " + JSON.stringify(args)); }},
  }});
  // A new state object for the price sensor, as Home Assistant hands the card on an update.
  window.__refreshHass = () => {{
    const states = {{ ...window.__card.hass.states }};
    for (const id of Object.keys(states)) states[id] = {{ ...states[id] }};
    window.__card.hass = window.__makeHass(states);
  }};
</script>
<script type="module">
  await import("/card/plotly-graph-card.js");
  await customElements.whenDefined("plotly-graph");
  const card = document.createElement("plotly-graph");
  window.__card = card;
  card.setConfig({config_json});
  card.hass = window.__makeHass(scenario.states);
  document.getElementById("mount").appendChild(card);
</script>
</body>
</html>
""")


def write_html(path: Path, config: dict, scenario: dict, theme: str, width: int) -> None:
    theme_vars = "\n".join(f"    --{k}: {v};" for k, v in THEMES[theme].items())
    path.write_text(
        HTML_TEMPLATE.format(
            title=path.stem,
            theme_vars=theme_vars,
            margin=MARGIN,
            width=width,
            scenario_json=json.dumps(scenario),
            config_json=json.dumps(config),
            dark="true" if theme == "dark" else "false",
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Playwright
# ---------------------------------------------------------------------------

PLAYWRIGHT_SCRIPT = textwrap.dedent("""\
const {{ chromium }} = require({playwright_json});
const http = require('http');
const fs = require('fs');
const path = require('path');

const roots = {{ '/card/': {bundle_json}, '/': {work_json} }};
const jobs = {jobs_json};

const srv = http.createServer((req, res) => {{
  const url = decodeURIComponent(req.url.split('?')[0]);
  const prefix = url.startsWith('/card/') ? '/card/' : '/';
  const file = path.join(roots[prefix], url.slice(prefix.length));
  try {{
    const data = fs.readFileSync(file);
    res.writeHead(200, {{ 'Content-Type': file.endsWith('.js') ? 'text/javascript' : 'text/html; charset=utf-8' }});
    res.end(data);
  }} catch (e) {{
    res.writeHead(404);
    res.end();
  }}
}});

async function ready(page) {{
  // drawn: the plot exists and the loading state is over
  await page.waitForFunction(() => {{
    const root = window.__card?.shadowRoot;
    return root && root.querySelector('.main-svg') && !root.querySelector('ha-card.loading');
  }}, null, {{ timeout: 20000 }});
  await page.waitForTimeout(700);
  const error = await page.evaluate(() => {{
    const el = window.__card.shadowRoot.querySelector('#error-msg');
    return getComputedStyle(el).display !== 'none' ? el.innerText : null;
  }});
  if (error) throw new Error('card error: ' + error);
  // count the renders from here on, to show that a refresh really drew the card again
  await page.evaluate(() => {{
    const gd = window.__card.shadowRoot.querySelector('#plotly');
    if (!window.__renders) {{ window.__renders = 0; gd.on('plotly_afterplot', () => window.__renders++); }}
  }});
}}

// Every line trace must run forwards on the wall clock: a repeated hour (autumn clock
// change) drawn twice would make a step line jump back.
async function checkTimeOrder(page) {{
  return page.evaluate(() => {{
    const fmt = new Intl.DateTimeFormat('en-CA', {{ year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }});
    const wall = (x) => x instanceof Date ? fmt.format(x).replace(',', '') : String(x).slice(0, 19).replace('T', ' ');
    const gd = window.__card.shadowRoot.querySelector('#plotly');
    const bad = [];
    gd.data.forEach((t, i) => {{
      if (!String(t.mode || 'lines').includes('lines') || !t.x || t.visible === false) return;
      const xs = Array.from(t.x).filter((x) => x != null).map(wall);
      for (let k = 1; k < xs.length; k++) {{
        if (xs[k] <= xs[k - 1]) {{ bad.push(`trace ${{i}} (${{t.name}}): ${{xs[k - 1]}} then ${{xs[k]}}`); break; }}
      }}
    }});
    return bad;
  }});
}}

async function clickButton(page, label) {{
  const button = page.locator('.updatemenu-button').filter({{ hasText: new RegExp('^' + label + '$') }});
  await button.first().click();
  await page.waitForTimeout(900);
}}

(async () => {{
  await new Promise(r => srv.listen({port}, '127.0.0.1', r));
  const browser = await chromium.launch();
  let failed = false;
  for (const job of jobs) {{
    const context = await browser.newContext({{
      deviceScaleFactor: 2, timezoneId: job.timeZone, locale: 'de-DE',
      viewport: {{ width: job.width + 2 * {margin} + 40, height: 1400 }},
    }});
    const page = await context.newPage();
    const problems = [];
    page.on('pageerror', e => problems.push('pageerror: ' + e.message));
    page.on('console', m => {{ if (m.type() === 'error') problems.push('console: ' + m.text()); }});
    await page.clock.setFixedTime(new Date(job.now));
    try {{
      await page.goto(`http://127.0.0.1:{port}/${{job.html}}`);
      await ready(page);
      for (const step of job.steps || []) {{
        if (step.click) await clickButton(page, step.click);
        if (step.range) {{
          await page.locator('.rangeselector .button').filter({{ hasText: step.range }}).first().click();
          await page.waitForTimeout(1500);
        }}
        if (step.legend) {{
          await page.locator('.legend .traces').filter({{ hasText: step.legend }}).locator('.legendtoggle').first().click();
          await page.waitForTimeout(900);
        }}
        if (step.refresh) {{
          // a sensor update, then the hourly statistics event: the two card refreshes
          await page.evaluate(() => window.__refreshHass());
          await page.evaluate(() => window.__fireEvent('recorder_hourly_statistics_generated'));
          await page.waitForTimeout(1200);
        }}
        if (step.shot) {{
          await ready(page);
          await page.locator('.frame').screenshot({{ path: step.shot }});
          console.log('saved ' + step.shot + '  renders so far: ' + await page.evaluate(() => window.__renders));
        }}
      }}
      await ready(page);
      await page.locator('.frame').screenshot({{ path: job.png }});
      problems.push(...await page.evaluate(() => window.__problems));
      problems.push(...(await checkTimeOrder(page)).map((b) => 'time order: ' + b));
      console.log('saved ' + job.png + '  ws: ' + [...new Set(await page.evaluate(() => window.__ws))].join(', '));
    }} catch (e) {{
      problems.push(String(e));
    }}
    if (problems.length) {{
      failed = true;
      console.error(job.html + ':\\n  ' + problems.join('\\n  '));
    }}
    await context.close();
  }}
  await browser.close();
  srv.close();
  process.exit(failed ? 1 : 0);
}})().catch(e => {{ console.error(e); process.exit(1); }});
""")


def render(jobs: list[dict]) -> bool:
    if not Path(PLAYWRIGHT_DIR).exists():
        print(
            f"playwright not found at {PLAYWRIGHT_DIR}\n"
            "  npm install --prefix /tmp playwright@1.63.0   (or set PLAYWRIGHT_DIR)",
            file=sys.stderr,
        )
        return False
    script = WORK / "_shoot.js"
    script.write_text(
        PLAYWRIGHT_SCRIPT.format(
            playwright_json=json.dumps(PLAYWRIGHT_DIR),
            bundle_json=json.dumps(str(BUNDLE)),
            work_json=json.dumps(str(WORK)),
            jobs_json=json.dumps(jobs, indent=2),
            port=PORT,
            margin=MARGIN,
        )
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=900)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS = ["default", "dst", "dst_forecast", "stale", "fresh_install"]
# scenarios whose look-back is worth a look (the 2025 ones have no stored forecasts)
LOOKBACK_SCENARIOS = {"default", "fresh_install"}


def _job(card: str, scenario_name: str, scenario: dict, theme: str, width: int, png: Path, **extra):
    config = yaml.safe_load(CARDS[card].read_text(encoding="utf-8"))
    html = WORK / f"{png.stem}.html"
    write_html(html, config, scenario, theme, width)
    return {
        "html": html.name,
        "png": str(png),
        "width": width,
        "now": scenario["now"],
        "timeZone": scenario["time_zone"],
        **extra,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--theme", default="both", choices=["light", "dark", "both"])
    parser.add_argument("--scenario", default="default", choices=[*SCENARIOS, "all"])
    parser.add_argument("--sizes", action="store_true", help="render at 360/600/1000 px")
    parser.add_argument(
        "--switch-test",
        action="store_true",
        help="click a look-back horizon button, refresh the card, and shoot before and after",
    )
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cached scenarios")
    args = parser.parse_args()

    ensure_bundle()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    names = SCENARIOS if args.scenario == "all" else [args.scenario]

    jobs = []
    for name in names:
        data = load_scenario(name, args.rebuild)
        for card in CARDS:
            if card == "lookback" and name not in LOOKBACK_SCENARIOS:
                continue
            for theme in themes:
                if args.switch_test:
                    if card != "lookback":
                        continue
                    stem = WORK / f"switch_{name}_{theme}"
                    steps = [
                        {"click": "D\\+4", "shot": f"{stem}_1_clicked.png"},
                        {"refresh": True, "shot": f"{stem}_2_refreshed.png"},
                        {"click": "alle", "refresh": True},
                    ]
                    png = Path(f"{stem}_3_all_refreshed.png")
                    jobs.append(_job(card, name, data, theme, CARD_WIDTH, png, steps=steps))
                elif args.sizes:
                    for width in SIZE_WIDTHS:
                        png = WORK / f"{card}_{name}_{width}_{theme}.png"
                        jobs.append(_job(card, name, data, theme, width, png))
                else:
                    out = OUT_DIR if name == "default" else WORK
                    stem = f"{card}_{theme}" if name == "default" else f"{card}_{name}_{theme}"
                    jobs.append(_job(card, name, data, theme, CARD_WIDTH, out / f"{stem}.png"))
                    if card == "lookback" and name == "fresh_install":
                        # the longest range, where the window before the backfill shows
                        png = WORK / f"{card}_{name}_6m_{theme}.png"
                        jobs.append(
                            _job(card, name, data, theme, CARD_WIDTH, png, steps=[{"range": "6 M"}])
                        )
    return 0 if render(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
