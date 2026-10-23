# Spot Forecast, plan 3 of 4: model channel and CI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New models reach users without integration releases. GitHub Actions trains monthly behind a quality gate and publishes to a rolling `models` release with a manifest. The integration checks the manifest daily, then downloads, verifies and switches models. Repository CI (hassfest, HACS, tests, HA matrix, lint) and ZIP releases work as in SBP.

**Architecture:** Gate and manifest logic live in `training/` as pure functions that `training/ci.py` orchestrates; the workflow only moves files. The integration's manifest parsing and model choice live in `core/` (pure). `model_update.py` is the HA glue: timer, download, `.storage` files, swapping the coordinator's model.

**Tech Stack:** GitHub Actions, `gh` CLI, Python 3.12 (training), numpy/pandas/scikit-learn; HA `async_track_time_change`, aiohttp.

**Spec:** `docs/superpowers/specs/2026-10-22-spot-forecast-design.md`, sections "Releases and the model" (Model channel, In the integration, Training workflow, Integration releases). Plan 1's carry-over items for CI are in scope here: `load_model` must wrap `OverflowError`, and the `capacity_at` lag must be surfaced as a clear CI message.

## Global Constraints

- **Release URLs.**
  - Rolling model release: tag `models`. Manifest URL: `https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json`.
  - Training-data archive: release tag `training-data`, asset `raw.tar.gz`.
- **Manifest** `format: 1` (exact JSON shape in Task 1). Model version string `"{slug}-{YYYY-MM}"` of the cutoff, e.g. `de_lu-2026-11`. Asset names `{version}.json.gz`, `{version}.backfill.json.gz`, `{version}.report.md`.
- **Quality gate.** Publish only if both of these hold, otherwise publish nothing and open an issue:
  - Candidate MAE on the last full month ≤ previous published model's MAE on that month + 1.0 EUR/MWh (0.1 ct/kWh).
  - Skill against the naive baseline on that month > 0.
  - The first run with no previous model checks skill only.
- **Training schedule:** cron `0 2 3 * *`, plus `workflow_dispatch`. Cutoff = the first day of the current month (UTC date of the run).
- **Integration update rules.**
  - The update check runs daily at a random minute chosen once per entry.
  - A model is downloaded only if all of these hold: `feature_schema` equals the integration's `FEATURE_SCHEMA`, its `train_cutoff` is newer than the active model's, the SHA-256 matches, `load_model` succeeds, and the sample check passes.
  - Downloads go to `<config>/.storage/spot_forecast/`; nothing is ever written to `custom_components/`.
  - Fallback order: the newest valid downloaded model for the zone and schema, then the bundled model.
  - A failed check keeps the current model and logs one warning per version.
- **Option** `auto_update` ("Update models automatically"), default true.
- **CI.**
  - Ruff `0.16.5`, with `research/` excluded from lint and format.
  - HA test matrix:
    - floor: `pytest-homeassistant-custom-component==0.13.181` + `josepy<2` on Python 3.13;
    - latest: unpinned PHCC on Python 3.14.
  - Daily validate cron `0 4 * * *`.
  - `hacs.json`: `{"name": "Spot Forecast", "render_readme": true, "homeassistant": "2024.11.0", "zip_release": true, "filename": "spot_forecast.zip"}`.
- **Workflows** set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` like SBP, and use `actions/checkout@v4` / `actions/setup-python@v5`.
- **Commands and files.**
  - Pure tests: `SPOT_RESEARCH_DIR=… .venv/bin/python -m pytest tests -q`.
  - HA tests: `.venv-ha/bin/python -m pytest tests_ha` (`tests_ha/pytest.ini` exists).
  - Never commit raw data or `dist/`.

## Review Focus

1. **A model with a different `feature_schema` in the manifest** (published after a features change, user on an old integration): it is never downloaded, there is no warning spam, and the old model stays active. Pinned in Task 3 (`test_schema_mismatch_is_ignored_quietly`).
2. **A truncated or tampered download** (SHA mismatch, gzip error, NaN predictions): the file is deleted, the current model is kept, and one warning is logged. Pinned in Task 3 (`test_sha_mismatch_keeps_current_model`, `test_sample_check_rejects_nan_model`).
3. **GitHub unreachable or the manifest 404** (no `models` release yet): no error state, entities unaffected, retry next day. Pinned in Task 3 (`test_manifest_404_is_quiet`).
4. **A gate failure:** nothing is uploaded, the manifest is unchanged, and an issue is opened with the report. Pinned in Task 1 (`test_gate_fails_when_worse_than_previous`) and Task 2 (the workflow's `if:` on the gate output, checked by `test_train_workflow_publishes_only_on_pass`).
5. **Restart after a model update:** the downloaded model is chosen on load, without a network call, and stays chosen when it is newer than the bundled one. When a later integration release bundles a newer model, the bundled one wins. Pinned in Task 3 (`test_load_active_prefers_newest_by_cutoff`).

---

### Task 1: Gate, manifest and CI driver in `training/`; plan 1 carry-over

**Files:**
- Create: `training/gate.py`, `training/manifest.py`, `training/ci.py`
- Modify: `custom_components/spot_forecast/core/model.py` (`load_model` wraps `OverflowError` into `ModelCorrupt`), `training/train.py` (`capacity_at` raises `DataLag` with message `"installed_power has no value for YYYY-MM; Energy-Charts publishes monthly with a lag — rerun after it appears or pass --capacity-month"`; add the `--capacity-month YYYY-MM` override), `ruff.toml` (exclude `research`)
- Test: `tests/training/test_gate.py`, `tests/training/test_manifest.py`, `tests/training/test_ci.py`, `tests/core/test_model.py` (overflow)

**Interfaces:**
- `training/gate.py`:
  - `month_metrics(p: Predictions, month_start: date) -> dict`: `{"mae", "naive_mae", "skill", "rows"}` in EUR/MWh over target dates in that calendar month. It uses the same naive baseline as `report.evaluate`, so reuse its helper and do not duplicate it.
  - `previous_model_predictions(model: ModelFile, ds: Dataset, month_start: date) -> Predictions`: predicts the dataset rows whose target date is in that month with the previous model's trees (numpy `Trees.predict` + decode as in `core.features.decode`).
  - `GateResult` (frozen dataclass): `passed: bool`, `candidate: dict`, `previous: dict | None`, `reason: str`.
  - `evaluate_gate(candidate: dict, previous: dict | None, tolerance: float = 1.0) -> GateResult`.
- `training/manifest.py`:
  - `manifest_entry(model_bytes: bytes, backfill_bytes: bytes, model: ModelFile, base_url: str) -> dict`.
  - `merge_manifest(old: dict | None, slug: str, entry: dict) -> dict`.
  - Manifest JSON shape:
    ```json
    {"format": 1, "updated": "<iso>", "zones": {"de_lu": {
      "version": "de_lu-2026-11", "feature_schema": 1, "train_cutoff": "2026-11-01", "created": "<iso>",
      "model": {"name": "de_lu-2026-11.json.gz", "url": "<base>/de_lu-2026-11.json.gz", "sha256": "<hex>", "size": 123},
      "backfill": {"name": "...backfill.json.gz", "url": "...", "sha256": "...", "size": 0},
      "metrics": {"mae_h1": 17.9, "mae": 19.2, "skill": 0.45, "p90_exceedance": 0.12}}}}
    ```
- `training/ci.py`: `python -m training.ci --raw DIR --out DIR --cutoff YYYY-MM-DD [--previous-manifest FILE] [--previous-model FILE]`.
  - It runs `train.run` and computes the gate for the month before the cutoff.
  - It writes `out/gate.json` (`GateResult` as JSON), `out/manifest.json` (merged, only if passed), `out/release-notes.md` (report + gate table), and `out/issue.md` (only if failed).
  - It exits 0 in both cases; the workflow reads `gate.json`.
  - A data lag exits 2 with the `DataLag` message.

- [ ] **Step 1: Write the failing tests:**
  - `test_month_metrics_matches_evaluate_for_full_window`
  - `test_gate_passes_within_tolerance`
  - `test_gate_fails_when_worse_than_previous`: candidate 20.0, previous 18.5, so it fails with a reason naming both values.
  - `test_gate_fails_without_skill`
  - `test_gate_first_run_checks_skill_only`
  - `test_manifest_entry_sha_and_urls`
  - `test_merge_manifest_keeps_other_zones`
  - `test_ci_writes_manifest_only_on_pass`: use the research-data fixture with `max_iter=20` and skip without data. Monkeypatch the gate to fail and pass.
  - `test_capacity_lag_message`
  - `test_load_model_overflow_is_corrupt`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/training/test_gate.py tests/training/test_manifest.py tests/training/test_ci.py tests/core/test_model.py -q` and confirm it FAILS.
- [ ] **Step 3: Implement** the functions above.
- [ ] **Step 4: Run** the full pure suite and `ruff check . && ruff format --check .`, which must now be clean repo-wide. Confirm it PASSES.
- [ ] **Step 5: Commit** with `git commit -m "Add quality gate, manifest builder and CI driver for monthly training"`.

---

### Task 2: Workflows: `train.yml`, `validate.yml`, `release.yml`; `hacs.json`, CODEOWNERS

**Files:**
- Create: `.github/workflows/train.yml`, `.github/workflows/validate.yml`, `.github/workflows/release.yml`, `.github/CODEOWNERS` (`* @cygnusb`), `hacs.json`, `tools/fetch_published_model.py`
- Test: `tests/test_workflows.py` (YAML structure; skip if `yaml` is not importable), plus `actionlint` if installed locally

**Interfaces:**
- **`train.yml`.**
  - Permissions: `contents: write`, `issues: write`.
  - Steps:
    1. Checkout, set up Python 3.12 and install `training/requirements.txt`.
    2. `gh release download training-data -p raw.tar.gz` and untar into `training/data/raw` (tolerate a missing release on the first run).
    3. `python -m training.fetch --raw training/data/raw` (incremental, yesterday's end).
    4. Re-tar **only** `raw/prices/`, `raw/weather/` and `raw/installed_power.json` (a whitelist: the public archive must hold only CC BY data from Energy-Charts and Open-Meteo, never research-only files such as Yahoo gas prices) and `gh release upload training-data raw.tar.gz --clobber`. A test asserts the tar command lists exactly these paths.
    5. Download the current manifest and previous model (tolerate a 404).
    6. Run `python -m training.ci …` with the cutoff = the first of the current month.
    7. **If** `gate.json.passed`: `gh release upload models <model> <backfill> <report> manifest.json --clobber`, then `gh release edit models --notes-file out/release-notes.md`.
    8. **Else:** `gh issue create --title "Model training gate failed for <version>" --body-file out/issue.md`.
  - A data-lag exit 2 opens an issue titled "Model training waiting for data" with the message.
  - Concurrency group `train`.
- **`validate.yml`:** hassfest, `hacs/action@main` (category integration), pure tests (Python 3.13, `pip install -r tests/requirements.txt`), the HA test matrix with the frontend-package step and latest-version check copied from `~/git/ha-smart-battery-pilot/.github/workflows/validate.yml`, and lint (`ruff==0.16.5`).
- **`release.yml`** on `release: published`:
  1. Checkout.
  2. Run `python tools/fetch_published_model.py custom_components/spot_forecast/models`. It downloads the manifest, the model and the backfill per zone, verifies the SHA-256, and writes `models/{slug}.json.gz` / `models/{slug}.backfill.json.gz`. If the manifest is absent, it keeps the committed files.
  3. Zip `custom_components/spot_forecast` to `spot_forecast.zip`, excluding `__pycache__`.
  4. `gh release upload <tag> spot_forecast.zip --clobber`.
- `tools/fetch_published_model.py` uses only the standard library and is unit-tested with a local HTTP fixture or monkeypatched `urlopen`.

- [ ] **Step 1: Write the failing tests:**
  - `test_train_workflow_publishes_only_on_pass`: the upload step's `if:` references the gate output, and the issue step its negation.
  - `test_validate_workflow_has_matrix_floor_and_latest`
  - `test_release_workflow_zips_after_model_fetch`
  - `test_hacs_json_values`
  - `test_fetch_published_model_verifies_sha`
  - `test_fetch_published_model_keeps_files_without_manifest`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_workflows.py tests/test_fetch_published_model.py -q` and confirm it FAILS.
- [ ] **Step 3: Implement** the files above. Run `actionlint` if available.
- [ ] **Step 4: Run** the full pure suite and ruff, and confirm they PASS.
- [ ] **Step 5: Commit** with `git commit -m "Add monthly training, validation and release workflows"`.

---

### Task 3: Automatic model update in the integration

**Files:**
- Create: `custom_components/spot_forecast/core/manifest.py`, `custom_components/spot_forecast/model_update.py`
- Modify:
  - `core/model.py`: `sample_check`.
  - `model_loader.py`: `load_active`.
  - `__init__.py`: use `load_active`, start the updater.
  - `coordinator.py`: add `set_model(model, source)`. The next run uses it, and the repairs and diagnostics read the active model.
  - `const.py`: add `CONF_AUTO_UPDATE`, `MANIFEST_URL`, `MODEL_DIR = "spot_forecast"` under `.storage`.
  - `config_flow.py`: add the options field.
  - `sensor.py`: the `model_version` attributes `source` and `train_cutoff`.
  - `translations/*.json`.
- Test: `tests/core/test_manifest.py`, `tests/core/test_model.py`, `tests_ha/test_model_update.py`

**Interfaces:**
- `core/manifest.py`:
  - `ManifestEntry` (frozen): `version`, `feature_schema`, `train_cutoff: date`, `model_url`, `model_sha256`, `backfill_url`, `backfill_sha256`.
  - `parse_manifest(doc: dict) -> dict[str, ManifestEntry]` raises `ValueError` on a wrong `format`.
  - `wants_download(entry, active_cutoff: date, feature_schema: int) -> bool`.
- `core/model.py`: `sample_check(model: ModelFile) -> None` raises `ModelCorrupt` unless the median, p90 and p10 predictions on a fixed built-in sample are finite and below 1e3 in absolute value.
  - The sample is two rows with every feature NaN and every feature 0.0. It must be deterministic and must not need data files.
- `model_loader.py`: `load_active(zone_code: str, storage_dir: Path) -> tuple[ModelFile, bytes, str]`. It returns the model, the backfill bytes and the source `"bundled"` or `"downloaded"`.
  - It considers `storage_dir/{slug}-*.json.gz` with the matching feature_schema, passes `load_model` and `sample_check`, and picks the max `train_cutoff` including the bundled model.
  - Ties go to the bundled model.
  - It runs in the executor.
- `model_update.py`: `ModelUpdater(hass, entry, coordinator)` with `async_start()` (daily `async_track_time_change` at a random hour/minute stored in entry data or derived from `entry.entry_id` hash) and `async_check() -> str`. The returned outcome is one of `"updated"`, `"current"`, `"skipped_schema"`, `"failed"` or `"unavailable"`.
  - It downloads via `async_get_clientsession` and verifies SHA-256.
  - It writes atomically (temp file plus rename in the executor), then loads, checks and calls `coordinator.set_model`.
  - It keeps at most 3 downloaded versions per zone and deletes older ones.

- [ ] **Step 1: Write the failing tests:**
  - `test_parse_manifest_and_wants_download`
  - `test_schema_mismatch_is_ignored_quietly`: no warning in the log, outcome `"skipped_schema"`.
  - `test_sample_check_rejects_nan_model`: a model whose leaf values are NaN.
  - `test_sha_mismatch_keeps_current_model`
  - `test_manifest_404_is_quiet`
  - `test_update_switches_model_and_next_run_uses_it`: the diagnostic `model_version` changes and `source` is `downloaded`.
  - `test_load_active_prefers_newest_by_cutoff`: the bundled cutoff 2026-10-01 against a downloaded 2026-11-01 picks downloaded; a downloaded 2026-09-01 picks bundled.
  - `test_auto_update_off_never_fetches`
- [ ] **Step 2: Run** the new tests and confirm they FAIL.
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run** both suites and ruff, and confirm they PASS.
- [ ] **Step 5: Commit** with `git commit -m "Download newer models from the models release, verified"`.

---

## After the plan (controller, not a subagent)

- Merge to main and push.
- Bootstrap the releases:
  - `training-data` with `raw.tar.gz` built from `research/data` (the same file layout as the training raw dir; check this first).
  - `models` with the 2026-10 model, backfill, report and a manifest built by `training.manifest`.
- Trigger `train.yml` once via `workflow_dispatch` as a smoke test, and confirm validate is green.
