# Spot Forecast, plan 4 of 4: cards, brand, README, docs and website — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The two Plotly Graph Card dashboards (forecast and look-back, variant A), verified by rendering them in the real `custom:plotly-graph` card with real model output. Plus a brand icon and logo, an SBP-style README, MkDocs documentation, and a bilingual project website on GitHub Pages.

**Architecture:**
- **Cards** are YAML in `examples/`. A screenshot tool (`tools/gen_card_screenshots.py`) renders each card in headless Chromium:
  - it loads the pinned plotly-graph-card bundle into a page with a stub `hass` object;
  - the stub serves states and statistics computed by the real `core` code from research data for a fixed past date.
  The screenshots are therefore a test of the YAML.
- **Site:** `tools/build_site.py` + `site/index.html.j2` + `site/i18n/{en,de}.json`, modelled on SBP.
- **Docs:** MkDocs Material under `/docs/`.

**Tech Stack:** Plotly Graph Card (dbuezas/lovelace-plotly-graph-card, pinned release), Playwright (Node, browsers from the local cache), Jinja2, MkDocs 1.6.1 + Material 9.7.7, `rsvg-convert`.

**Spec:** `docs/superpowers/specs/2026-10-22-spot-forecast-design.md`, sections "Dashboard cards (Plotly Graph Card, variant A)" and "README, website, discoverability". The approved mock-up's source is the reference for the look: `research/mockup/` (copied in Task 1 from the session scratchpad: `template.html`, `main.js`, `build.py`, `export.py`, `shot_a.png`).

## Global Constraints

- **Colours:**
  - Horizons D+1…D+6: `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300`.
  - Risk `#d03b3b`, cheap `#1baf7a`, known price in ink (HA `--primary-text-color`).
  - Both cards must be readable in HA light and dark themes.
- **Card units:** ct/kWh on the axis (EUR/kWh × 100) with German number separators. Labels: German in the shipped YAML, English in comments.
- **Entity and statistic ids:** `sensor.spot_forecast_de_lu_price` (attribute `prices`: `time`, `price`, `source`, and `p90`/`p10`/`h` on forecast entries). Statistics `spot_forecast:de_lu_price` and `spot_forecast:de_lu_forecast_h{1..6}`.
- **Repository URLs:**
  - Repo: `https://github.com/cygnusb/ha-spot-forecast`.
  - Site: `https://cygnusb.github.io/ha-spot-forecast/`.
  - Docs: `https://cygnusb.github.io/ha-spot-forecast/docs/`.
- **README badge row** as in SBP, with HACS **Custom** (not Default): release, downloads, HACS Custom, HA ≥ 2024.11, license MIT, commit activity, validate, stars.
- **Website languages:** en (root) and de (`/de/`). A key missing from a translation fails the build.
- **Data rule:** only open data. The site's accuracy chart uses the shipped backtest (`dist`/`models` backfill + actual prices from research data), embedded as JSON.
- **Brand files:** `custom_components/spot_forecast/brand/{icon.png 256×256, icon@2x.png 512×512, logo.png, logo@2x.png}`, and `assets/{icon.svg, logo.svg, icon.png, logo.png}`. The icon is a generic signifier: no third-party logos.
- **Commands:**
  - `.venv/bin/python` for tools.
  - Playwright via `npm install --prefix /tmp playwright@<version matching a cached chromium>` (SBP's `tools/gen_card_screenshots.py` header explains how).
  - Docs: `pip install -r tools/docs-requirements.txt` into `.venv-docs`.

## Review Focus

1. **The card in HA dark mode:** every text, band and marker stays visible (no ink-on-dark). Pinned in Task 2 (dark screenshots inspected; `test_cards_use_theme_variables_for_ink`).
2. **The forecast card on the DST fall-back day:** no duplicated or missing hour and no zig-zag in the step line. Pinned in Task 2 (a scenario date 2026-10-25 in `--sizes`/scenarios; screenshot inspected).
3. **The prices attribute holds only market data** (model stale, forecast dropped): the card still renders known prices with no JS error and no empty-band artefacts. Pinned in Task 2 (scenario `stale`).
4. **A phone-width dashboard (≈ 360 px):** day headers do not collide and markers stay legible. Pinned in Task 2 (`--sizes` at 360/600/1000 px).
5. **The look-back card in the first days after install,** with the backfill window only: there are no gaps that look like errors, and the error figure is computed over the loaded window. Pinned in Task 2 (scenario `fresh_install`).

---

### Task 1: Brand icon and logo; mock-up reference into the repo

**Files:**
- Create:
  - `research/mockup/` (copy `template.html`, `main.js`, `build.py`, `export.py`, `shot_a.png` from `/private/tmp/claude-501/-Users-valerius-git-ha-spot-forecast/98c22759-0116-43f8-a4a6-6045476799d0/scratchpad/mock/`; not `data.json` or the previews, which are large and regenerable)
  - `research/mockup/README.md` (what it is and how to rebuild it)
  - `assets/icon.svg`, `assets/logo.svg`
  - `assets/BRANDS.md` (how to render the PNGs, as in SBP)
  - the rendered PNGs listed in Global Constraints
- Test: `tests/test_brand.py` checks PNG sizes (256/512) and that the brand and assets files exist.

The design is the implementer's call, within these limits:
- **Icon:** a stepped price line crossing into a forecast band, readable at 32 px, flat with no gradients, using the horizon blue `#2a78d6` and the risk red `#d03b3b` on transparent.
- **Logo:** the icon plus the wordmark "Spot Forecast".
- Render with `rsvg-convert`.

- [ ] **Step 1: Write the failing test** `test_brand_files_exist_with_sizes`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_brand.py -q` and confirm it FAILS.
- [ ] **Step 3: Draw the SVGs, render the PNGs and copy the mock-up reference.**
- [ ] **Step 4: Run** the test, confirm it PASSES, and look at `icon.png` at 32 px and 256 px.
- [ ] **Step 5: Commit** with `git commit -m "Add icon, logo and the approved card mock-up"`.

---

### Task 2: Plotly cards and the screenshot harness

**Files:**
- Create: `examples/plotly_forecast.yaml`, `examples/plotly_lookback.yaml`, `tools/gen_card_screenshots.py`, `tools/card_scenarios.py`, `assets/screenshots/{forecast,lookback}_{light,dark}.png`
- Test: `tests/test_cards.py`

**Interfaces:**
- `tools/card_scenarios.py`: `scenario(name: str) -> dict` returns `{"now": iso, "states": {...}, "statistics": {statistic_id: [{"start", "mean", "state"}]}, "sun": {...}}`.
  - It is built with the real `core` code: `run_forecast` with the bundled model on research data for a fixed past anchor, `Ledger.finalize` for the statistics, and `prices_attribute` for the attribute.
  - Scenarios:
    - `default` (anchor 2026-09-16, 14:30 local, after publication);
    - `dst` (2026-10-25);
    - `stale` (market only);
    - `fresh_install` (look-back with 60 days).
  - It reads research data via `SPOT_RESEARCH_DIR` (default `research/`).
- `tools/gen_card_screenshots.py [--theme light|dark|both] [--scenario NAME|all] [--sizes]`:
  - It writes a page that loads the pinned plotly-graph-card bundle, downloaded once to `tools/.cache/` (git-ignored) from its GitHub release. The version is pinned in the script.
  - It mounts the card with the YAML (parsed with PyYAML) and passes a stub `hass` with `states`, `callWS` (answers `recorder/statistics_during_period` and `history/history_during_period` from the scenario), `themes`, `locale` (de), and HA-like CSS variables for light and dark.
  - It screenshots the card frame to `assets/screenshots/`.
- **Card behaviour:** per spec "Forecast card" and "Look-back card".
  - The forecast card reads the `prices` attribute via `$fn` / `$ex`. It shows yesterday, today and the forecast days, the hatched P90 and P10 bands, ▲/▼ top-4/bottom-4 per day, day headers with "max NN ct" and "⚠ bis NN" when the P90 exceeds the threshold, nights from `sun.sun`, and a now line.
  - The look-back card shows the actual price against `forecast_h{h}`.
    - Horizon switching uses whichever of `updatemenus` or legend toggling survives a card refresh; test both and say which in the YAML comment.
    - Range buttons are 7 T / 30 T / 3 M / 6 M.
    - The MAE is shown over the loaded window.

- [ ] **Step 1: Write the failing tests:**
  - `test_card_yaml_parses_and_references_real_ids`: the entity and statistic ids match the Global Constraints.
  - `test_cards_use_theme_variables_for_ink`: no hard-coded black or white for text and lines.
  - `test_scenarios_build_from_core`: the `default` scenario has market and forecast entries and finalized statistics; skip without research data.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_cards.py -q` and confirm it FAILS.
- [ ] **Step 3: Implement** the scenarios, the harness and the YAML. Iterate on the screenshots until the forecast card matches `research/mockup/shot_a.png` in structure, and check every Review Focus item visually.
- [ ] **Step 4: Run** the tests and `python tools/gen_card_screenshots.py --theme both --scenario all` plus `--sizes`, and look at every PNG.
- [ ] **Step 5: Commit** with `git commit -m "Add forecast and look-back Plotly cards with rendered screenshots"`.

---

### Task 3: README, MkDocs documentation, project website, Pages workflow

**Files:**
- Create:
  - `README.md` (replace the stub)
  - `docs/README.md` (overview), `docs/configuration.md`, `docs/entities.md`, `docs/cards.md`, `docs/model.md`, `docs/accuracy.md`, `docs/smart-battery-pilot.md`
  - `mkdocs.yml`, `tools/mkdocs_hooks.py` (changelog from GitHub releases, as SBP)
  - `tools/docs-requirements.txt`
  - `site/index.html.j2`, `site/i18n/en.json`, `site/i18n/de.json`
  - `tools/build_site.py`, `tools/site_data.py` (backtest JSON for the interactive look-back chart)
  - `.github/workflows/pages.yml`
- Modify: `.github/workflows/validate.yml` (add the `docs` job as SBP), `custom_components/spot_forecast/manifest.json` (`documentation` → the docs URL)
- Test: `tests/test_site.py`: the build succeeds, each language has every key, and the backtest JSON is under 1.5 MB.

**Content** (from the spec, section "README, website, discoverability"):
- **README:**
  - logo, the badge row, the one-line pitch;
  - links to website, docs and releases;
  - the forecast card screenshot as a light/dark `<picture>`;
  - "How it works";
  - an accuracy table with the full-period and trailing-3-month MAE per horizon, skill against naive, and the peak caveat (numbers from the shipped model's `metrics`);
  - HACS custom-repository install with the my.home-assistant badge;
  - configuration, entities, both cards;
  - "Using it with Smart Battery Pilot";
  - data sources and licences (Energy-Charts / SMARD CC BY 4.0, Open-Meteo CC BY 4.0), languages, licence.
- **Website:**
  - the forecast card screenshot as the hero;
  - "How good is it?" with an interactive Plotly look-back chart (variant A) from the backtest JSON, with horizon buttons and range buttons;
  - install steps and data sources;
  - the same visual language as the SBP site, but its own colours from the brand.
- **Docs:** one page per topic listed above; `exclude_docs: superpowers/`; `--strict` must pass.

- [ ] **Step 1: Write the failing tests** `test_site_builds_all_languages`, `test_translations_complete`, `test_backtest_json_size`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_site.py -q` and confirm it FAILS.
- [ ] **Step 3: Implement** the files above. Render the site locally and take a Playwright screenshot of the hero and the accuracy chart at 390 px and 1280 px.
- [ ] **Step 4: Run** `python tools/build_site.py _site && mkdocs build --strict` (in `.venv-docs`) plus the tests, and confirm they PASS.
- [ ] **Step 5: Commit** with `git commit -m "Add README, documentation and bilingual project website"`.

## After the plan (controller)

- Merge, push, and enable Pages (Settings → Pages → GitHub Actions) via `gh api`.
- Set the repo homepage to the site URL.
- Publish release `v0.1.0`. `release.yml` attaches `spot_forecast.zip`.
- Check that the validate, pages and release runs are green.
