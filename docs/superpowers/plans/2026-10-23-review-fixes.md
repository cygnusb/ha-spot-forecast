# Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects and risks found in the full review of v0.1.1 (2026-10-23).

**Architecture:** Four independent work packages on disjoint files (A: HA runtime, B: total
price/options/docs, C: model channel client + HTTP, D: training/CI/tools/site), each committed
separately on branch `review-fixes`, merged and tested together at the end.

**Tech Stack:** Python 3.12+, Home Assistant custom integration, numpy, scikit-learn 1.9.1
(training only), GitHub Actions.

**Spec:** the review report (session scratchpad `review.md`); the findings are restated per task.

## Global Constraints

- Minimum Home Assistant version stays the one in `hacs.json` (2024.11): no API newer than that
  without a version guard (so no `AddConfigEntryEntitiesCallback`).
- `core/` stays numpy-only; inference must stay light enough for a Raspberry Pi.
- `FEATURE_SCHEMA` is NOT bumped on this branch (no retraining, the bundled model stays).
- Tests: pure suite `/Users/valerius/git/ha-spot-forecast/.venv/bin/python -m pytest tests`;
  HA suite `cd tests_ha && /Users/valerius/git/ha-spot-forecast/.venv-ha/bin/python -m pytest`;
  lint `/Users/valerius/git/ha-spot-forecast/.venv-ha/bin/ruff check . && ... ruff format --check .`
  (ruff 0.16.5 = CI). Packages run targeted subsets; both full suites run once after the merge.
- Code style: match the surrounding code (comment density, docstrings, naming). Commit messages
  in the repo's style (imperative sentence, no prefix), with the session's attribution lines.
- No push, no PR, no GitHub settings changes.

## Decisions

- **VAT:** new option `vat` (percent, default 19, range 0–30, step 0.1). The surcharge keeps its
  documented meaning (gross amount per kWh, VAT included). Total = `spot × (1 + vat/100) +
  surcharge`, applied to `price`, `p90`, `p10`. No migration: existing entries get 19 %, which
  corrects their total; the surcharge they entered stays valid. Surcharge step 0.0001, 4 decimals.
- **Revoked models:** optional top-level manifest key `"revoked": ["de_lu-2026-11", ...]`. Old
  clients ignore it (parser reads only `format`/`zones`). New clients: never download a revoked
  version, delete it from disk if present, skip a zone entry whose version is revoked.
  `training/manifest.merge_manifest` preserves the list.
- **Future cutoff:** client and CI reject a model whose `train_cutoff` is later than today + 7 days
  (UTC). Negative model age counts as a problem in `issues.py` and `tools/check_model_age.py`.
- **train.yml split:** job `train` (`contents: read`) installs hash-locked deps, refreshes data,
  trains, uploads `out/` and the raw archive as workflow artifacts; job `publish`
  (`contents: write`, `issues: write`) installs nothing, downloads the artifacts and runs the
  `gh release` commands. Issue creation for a failed gate moves to `publish` (or a third job).
  Lock file: `training/requirements.lock` from `uv pip compile --generate-hashes
  --python-platform linux --python-version 3.12 training/requirements.txt`; `train.yml` and
  `validate.yml` (tests-training) install with `--require-hashes`.
- **Not on this branch (needs the user / is a model change):** manifest signature (needs a key
  secret), tag ruleset + immutable releases (repo settings), regional holidays and DST price-lag
  alignment (schema bump; research backtest only), auto-update-off → bundled (behaviour change).
- **Weather-run skew:** the stored run per anchor is the latest live one (~08:15 UTC on the anchor
  day), close to training's "available at 06:00 on day D". Document in `docs/accuracy.md` which
  run the look-back measures; no code change.

## Review Focus

- A source outage at the moment of a scheduled run → a retry within 30 min (Task A1).
- The 2026-10-25 fall-back day end to end in HA: 25 forecast hours, 25 statistic rows, the tick
  across 02:00 twice (Task A4).
- A corrupt downloaded backfill file → setup still succeeds on the bundled backfill (Task C1).
- A gate run whose previous model cannot be loaded → gate fails (Task D1).
- A surcharge entered by an existing user → same surcharge, total now includes VAT on spot (B1).

---

## Package A — HA runtime (coordinator, schedule, binary sensors, statistics, tests_ha)

Files: `coordinator.py`, `core/schedule.py`, `binary_sensor.py`, `entity.py`, `backfill.py`,
`long_term_statistics.py`, `core/ledger.py`, `__init__.py`, `tests_ha/conftest.py`,
`tests_ha/test_setup.py`, `tests_ha/test_backfill.py`, `tests_ha/test_long_term_statistics.py`,
new `tests_ha/fixtures/model/`, `tests/core/test_schedule.py`, `tests/core/test_ledger.py`.

### Task A0: Decouple tests_ha from the committed model (first)
- [ ] Copy `models/de_lu.json.gz` and `models/de_lu.backfill.json.gz` to
  `tests_ha/fixtures/model/`; in `tests_ha/conftest.py` patch the bundled-model directory
  (wherever `model_loader` resolves it) to that folder for every test.
- [ ] Full `tests_ha` passes unchanged. Commit.

### Task A1: Retry after a failed run
- [ ] Test (`tests/core/test_schedule.py`): `next_run(now, anchor_is_tomorrow, tz, retry=True)`
  returns `min(regular slot, now + RETRY_AFTER)` with `RETRY_AFTER = timedelta(minutes=20)`.
- [ ] Test (`tests_ha/test_setup.py`): source fails at a scheduled run → a new fetch happens 20 min
  later; after a success the regular slot is used again.
- [ ] Implement: `_async_scheduled_run` passes `retry=self._last_attempt_failed` (or the run's
  `ran=False`). Commit.

### Task A2: Binary sensors unknown without data
- [ ] Test: no entries in the 48 h window (or forecast expired and market ended) → state
  `unknown`; partial window keeps on/off from what is there.
- [ ] Implement `is_on -> bool | None` in `_FlagSensor`. Commit.

### Task A3: Run lifecycle and error isolation
- [ ] Scheduled run via `config_entry.async_create_background_task` (cancelled on unload); `_adopt`
  and Store saves are skipped when `_started` is False.
- [ ] Test: run stalled > UNLOAD_WAIT, entry removed → `.storage/spot_forecast.<entry_id>` does not
  reappear.
- [ ] Unexpected exceptions in `_take_prices`/`_settle` are logged and the update keeps the last
  data (entities stay available, `_last_attempt_failed = True`). Test with a patched `_settle`
  raising `RuntimeError`. Commit.

### Task A4: HA-level DST test
- [ ] Test in `tests_ha`: frozen time across 2026-10-25 (prices with 100 quarter-hours for that day,
  synthetic fixtures derived from existing ones): forecast day has 25 hours, statistics rows for
  that day = 25, tick at both 02:00s does not crash and `current_price` changes slot. Commit.

### Task A5: Statistics reconciliation and ledger nits
- [ ] On startup (after the first successful run, in the recorder executor) read
  `get_last_statistics(hass, 1, <actual price statistic id>, True, {"start"})`; if the recorder
  holds nothing, or its last row is older than the newest finalized day − 1 day, clear the
  ledger's finalized marks so the days are written again (upserts). Test: finalized marks present,
  empty recorder → rows written.
- [ ] `core/ledger.py`: `_PREFIX` imported from `const.DOMAIN` (keep core importable without HA:
  if `core` cannot import `const`, pass the prefix in or move `DOMAIN` into `core/zones.py` —
  pick the variant that keeps `core/` free of HA imports); `prune` uses the local date of the
  zone. `backfill.py:94`: count only runs that were actually new or replaced a different run.
  Commit.

### Task A6: Options change without full reload where possible
- [ ] The update listener compares old/new options: if only `surcharge`, `vat` or the peak
  threshold changed, update the coordinator in place (`coordinator.apply_options(options)` →
  rebuild data, notify listeners; add/remove the total-price entity via reload only when the
  surcharge switches between 0 and non-0). Otherwise reload as today. Test both paths.
  Coordinate with Package B: B owns the option keys (`CONF_VAT = "vat"` in `const.py`).
  Commit.

## Package B — Total price, options, translations, docs

Files: `core/output.py`, `sensor.py`, `config_flow.py`, `const.py`, `translations/*.json`,
`docs/configuration.md`, `docs/entities.md`, `docs/accuracy.md`, `README.md` (wording only),
`tests/core/test_output.py`, `tests_ha/test_config_flow.py`, `tests/test_translations.py`.

### Task B1: VAT on the spot part
- [ ] Test (`tests/core/test_output.py`): `with_surcharge(entries, surcharge=0.15, vat=19.0)` on
  price 0.10 → 0.269; on −0.05 → 0.0905; p90/p10 transformed the same way.
- [ ] Implement `with_surcharge(entries: list[dict], surcharge: float, vat: float) -> list[dict]`
  and the total-price sensor state with the same formula. `CONF_VAT = "vat"`, `DEFAULT_VAT = 19.0`.
- [ ] Options flow: field `vat` (0–30, step 0.1, unit %), surcharge step 0.0001; tests in
  `tests_ha/test_config_flow.py`. Translations en/de (en must not contain German; the card view
  name is quoted as the card shows it). Commit.

### Task B2: Display and docs
- [ ] `suggested_display_precision` = 4 on EUR/kWh sensors (price, total, p90/p10 if sensors).
- [ ] Docs: configuration.md (VAT option, formula, LU hint), entities.md (`source: "mixed"`,
  LTS rows leave when the day's prices are published, total-price `attribution`), "at or above"
  wording in README/docs, README "about one hour in ten" → the measured 12–13 %, accuracy.md: the
  look-back measures the last live run per anchor (~10:15 local on the anchor day). Commit.

## Package C — Model channel client and HTTP

Files: `model_loader.py`, `model_update.py`, `core/model.py`, `core/manifest.py`, `issues.py`,
`sources/http.py`, `sources/energy_charts.py`, `sources/open_meteo.py`, `tests/core/test_model.py`,
`tests/core/test_core_manifest.py`, `tests_ha/test_model_update.py`, `tests_ha/test_sources.py`,
`tests_ha/test_diagnostics_repairs.py`.

### Task C1: Backfill file robustness
- [ ] Tests: downloaded backfill with a flipped deflate byte (`zlib.error`), deep-nested JSON
  (`RecursionError`), wrong format → `load_active` returns the downloaded model with the bundled
  backfill; `install()` with an unparsable backfill raises `Rejected` and writes nothing.
- [ ] `_read_backfill` catches `Exception`; `install()` parses the backfill
  (`runs_from_backfill_file` or the shared reader) before writing. Commit.

### Task C2: Decompression and tree limits
- [ ] Shared helper `core/model.py: gunzip_limited(data: bytes, max_size: int) -> bytes` using
  `zlib.decompressobj(wbits=31)` with `max_length`; raises `ModelCorrupt` beyond `max_size`.
  Limits: model 64 MiB, backfill 32 MiB. Used by model load, `_read_backfill` and the ledger's
  backfill parser.
- [ ] `_check_trees`: leaf feature indices may be anything but are clamped/ignored, or checked —
  an out-of-range index anywhere raises `ModelCorrupt`, not `IndexError`; cap total nodes per
  ensemble (2,000,000) and depth (64). Tests for each. Commit.

### Task C3: Future cutoff, revoked list, negative age
- [ ] `core/manifest.py`: `parse_manifest` returns entries plus `revoked: frozenset[str]`
  (new return type `Manifest(entries: dict[str, ManifestEntry], revoked: frozenset[str])` — update
  callers); `wants_download` is False for revoked versions and for `train_cutoff > today + 7 d`.
- [ ] `model_update.py`: after reading the manifest, delete downloaded files of revoked versions
  and reload the active model if it was revoked. `load_checked`/`install` reject a cutoff in the
  future (> today + 7 d).
- [ ] `issues.py`: model age < −7 days is reported like an old model (or its own issue).
- [ ] Tests for each path. Commit.

### Task C4: HTTP limits
- [ ] Every source passes `max_bytes` (Energy-Charts 16 MiB, Open-Meteo 16 MiB, manifest 1 MiB if
  not already capped); `Retry-After` (seconds form) honoured on 429/503, capped at 120 s; the
  Open-Meteo 400 `reason` text included in the `SourceError` message. Tests in
  `tests_ha/test_sources.py`. Commit.

## Package D — Training, CI, tools, site, repo hygiene

Files: `training/*`, `tools/*`, `.github/workflows/*`, `site/*`, `README.md` (badge only),
`.gitignore`/`uv.lock`, `manifest.json` (`loggers`), `tests/training/*`, `tests/test_*.py`
(except `test_translations.py`), `tests/core/sbp_adapter_copy.py` (header only).

### Task D1: Gate fail-closed
- [ ] Tests (`tests/training/test_gate.py`, `test_ci.py`): previous model in the manifest but not
  comparable (schema mismatch, load error, row mismatch, other zone) → gate FAILS with that reason;
  only "no previous model in the manifest" allows skill-only. NaN/inf in candidate or previous MAE
  or skill → FAIL.
- [ ] Best of N: the gate compares against the lowest MAE among up to 3 previous published models
  that are loadable and out of sample (train.yml downloads them); at least the newest must be
  comparable. Commit.

### Task D2: Verify what is published
- [ ] `training/ci.py`: before writing the manifest, load `out/*.json.gz` through the client path
  (`core.model.load_checked` + sample check + backfill parser); fail on error. Reject a cutoff
  later than today + 7 d. `merge_manifest` preserves `revoked`. Tests. Commit.

### Task D3: Early stopping
- [ ] `training/train.py`: `early_stopping=False` (the shuffled 10 % split never stopped; all
  ensembles ran to 600). Fix the docstring of `build_training_data` (row order no longer matters
  for a split). Update tests that pin numbers if they break, and say which. Commit.

### Task D4: Workflows
- [ ] `train.yml` split as in Decisions; `training/requirements.lock` with hashes; `validate.yml`
  tests-training uses the lock; `timeout-minutes` on every job (train 120, others 30).
- [ ] `release.yml`: fail if the tag (without `v`) ≠ `manifest.json` version; ZIP includes the
  repository `LICENSE`.
- [ ] `tools/fetch_published_model.py`: https-only URLs under the release prefix, the model goes
  through `load_checked` + sample check, version/cutoff match the manifest entry, backfill parsed,
  backfill written before the model; skips revoked versions.
- [ ] `tools/check_model_age.py`: check manifest `format`; negative age fails; a 404 fails once
  the repository has published a manifest before (flag file `tools/.manifest-published` or
  existence of the `models` release via the API — choose the API check).
- [ ] `tests/test_workflows.py` updated for the new structure. Commit.

### Task D5: Site, tools, hygiene
- [ ] `tools/site_data.py`: emit timestamps as ISO with UTC offset (or UTC) so the 2026-10-25 hour
  is not duplicated; `site/index.html.j2` parses them accordingly; guard division by zero (`:94`).
  Test in `tests/test_site.py` with a fall-back day.
- [ ] README downloads badge → `github/downloads/cygnusb/ha-spot-forecast/spot_forecast.zip`
  (total) — check the shields URL form renders (`curl -sI`).
- [ ] Remove the tracked `uv.lock` stub (`git rm --cached`); `manifest.json` drops `loggers` entry
  for its own package (keep the key only if hassfest requires it); `sbp_adapter_copy.py` header
  names the upstream repo URL + commit instead of a local path. Commit.

## Package E — Research only (no repo changes on this branch)

- [ ] Backtest regional holidays (population-weighted, as `research/backtest.py` had) and the
  DST-aligned price lags against the current features; report MAE per horizon. Decide with the
  user whether to schedule a schema bump.
