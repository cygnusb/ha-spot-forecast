"""CI driver of the monthly training, and the installed-capacity data lag."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from core.model import dump_model
from training import ci, train
from training.dataset import RawData
from training.gate import GateResult
from training.train import DataLag, capacity_at

RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)
RESEARCH_RAW = RESEARCH / "data" / "raw"
CUTOFF = "2025-10-01"

needs_raw = pytest.mark.skipif(not RESEARCH_RAW.exists(), reason="research raw data missing")


def test_capacity_lag_message():
    caps = {"2026-09": {"cap_solar": 1.0}}
    raw = RawData(prices=None, weather={}, capacity=caps)
    with pytest.raises(DataLag) as err:
        capacity_at(raw, date(2026, 11, 1))
    assert str(err.value) == (
        "installed_power has no value for 2026-10; Energy-Charts publishes monthly with a lag"
        " — rerun after it appears or pass --capacity-month"
    )
    assert isinstance(err.value, ValueError)
    assert capacity_at(raw, date(2026, 11, 1), "2026-09") == {"cap_solar": 1.0}


@pytest.mark.parametrize("value", ["2026-09-01", "09-2026", "2026-13"])
def test_capacity_month_option_is_a_month(value):
    argv = ["--raw", "r", "--out", "o", "--cutoff", CUTOFF, "--capacity-month", value]
    with pytest.raises(SystemExit):
        train._parse(argv)


def test_gate_month_is_the_last_full_month_before_the_cutoff():
    assert ci.gate_month(date(2026, 11, 1)) == date(2026, 10, 1)
    assert ci.gate_month(date(2026, 1, 1)) == date(2025, 12, 1)
    assert ci.gate_month(date(2026, 11, 15)) == date(2026, 10, 1)


def test_ci_exits_2_on_data_lag(monkeypatch, tmp_path, capsys):
    calls = []

    def lagging(argv, max_iter=600):
        calls.append(argv)
        raise DataLag("installed_power has no value for 2025-09; …")

    monkeypatch.setattr(train, "run", lagging)
    out = tmp_path / "out"
    argv = ["--raw", "raw", "--out", str(out), "--cutoff", CUTOFF, "--capacity-month", "2025-08"]
    assert ci.run(argv) == 2
    assert "installed_power has no value for 2025-09" in capsys.readouterr().err
    assert calls == [
        ["--raw", "raw", "--out", str(out), "--cutoff", CUTOFF, "--capacity-month", "2025-08"]
    ]
    assert not (out / "gate.json").exists()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One small training pass shared by the ci tests."""
    out = tmp_path_factory.mktemp("train")
    argv = ["--raw", RESEARCH_RAW, "--out", out, "--cutoff", CUTOFF, "--backfill-days", "31"]
    return train.run(argv, max_iter=20)


@pytest.fixture
def run_ci(trained, monkeypatch, tmp_path):
    """``training.ci`` on the shared training result; returns (out dir, exit code)."""
    out = tmp_path / "out"
    expected_argv = ["--raw", str(RESEARCH_RAW), "--out", str(out), "--cutoff", CUTOFF]

    def cached(argv, max_iter=600):
        assert argv == expected_argv
        return trained

    monkeypatch.setattr(train, "run", cached)

    def go(*extra: str) -> tuple[Path, int]:
        return out, ci.run([*expected_argv, *extra])

    return go


@needs_raw
def test_train_run_returns_its_outputs(trained):
    assert trained.version == "de_lu-2025-10"
    assert trained.model.version == trained.version
    assert trained.model_path.name == "de_lu-2025-10.json.gz" and trained.model_path.exists()
    assert trained.backfill_path.name == "de_lu-2025-10.backfill.json.gz"
    assert trained.report_path.read_text().startswith("# Spot Forecast model de_lu-2025-10")
    assert max(trained.predictions.target_date) < date(2025, 10, 1)
    assert max(trained.dataset.target_date) < date(2025, 10, 1)


@needs_raw
@pytest.mark.parametrize("passed", [False, True])
def test_ci_writes_manifest_only_on_pass(run_ci, trained, monkeypatch, tmp_path, passed):
    previous = replace(trained.model, version="de_lu-2025-09", train_cutoff=date(2025, 9, 1))
    previous_model = tmp_path / "previous.json.gz"
    previous_model.write_bytes(dump_model(previous))
    previous_manifest = tmp_path / "manifest.json"
    other_zone = {"version": "at-2025-09"}
    old_zones = {"de_lu": {"version": "de_lu-2025-09"}, "at": other_zone}
    previous_manifest.write_text(json.dumps({"format": 1, "updated": "x", "zones": old_zones}))
    monkeypatch.setattr(
        ci,
        "evaluate_gate",
        lambda candidate, previous, tolerance=1.0: GateResult(
            passed, candidate, previous, "forced by the test"
        ),
    )

    out, code = run_ci(
        "--previous-manifest", str(previous_manifest), "--previous-model", str(previous_model)
    )

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["passed"] is passed
    assert (gate["version"], gate["month"]) == ("de_lu-2025-10", "2025-09")
    assert gate["already_published"] is False
    # both models scored on the same September rows
    assert gate["candidate"]["rows"] == gate["previous"]["rows"] > 0
    assert gate["candidate"]["mae"] > 0 and gate["previous"]["mae"] > 0
    assert "forced by the test" in gate["reason"]
    notes = (out / "release-notes.md").read_text()
    assert "Quality gate" in notes and "D+6" in notes
    if passed:
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["zones"]["at"] == other_zone
        entry = manifest["zones"]["de_lu"]
        assert entry["version"] == "de_lu-2025-10"
        assert entry["model"]["url"] == f"{ci.MODELS_BASE_URL}/de_lu-2025-10.json.gz"
        sha = hashlib.sha256(trained.model_path.read_bytes()).hexdigest()
        assert entry["model"]["sha256"] == sha
        assert not (out / "issue.md").exists()
    else:
        assert not (out / "manifest.json").exists()
        issue = (out / "issue.md").read_text()
        assert "forced by the test" in issue and "de_lu-2025-10" in issue


@needs_raw
@pytest.mark.parametrize("passed", [False, True])
def test_ci_never_republishes_a_published_version(run_ci, monkeypatch, tmp_path, capsys, passed):
    # A rerun in the same month: the manifest already lists this version.
    previous_manifest = tmp_path / "manifest.json"
    zones = {"de_lu": {"version": "de_lu-2025-10"}, "at": {"version": "at-2025-10"}}
    previous_manifest.write_text(json.dumps({"format": 1, "updated": "x", "zones": zones}))
    monkeypatch.setattr(
        ci,
        "evaluate_gate",
        lambda candidate, previous, tolerance=1.0: GateResult(
            passed, candidate, previous, "forced by the test"
        ),
    )
    out = tmp_path / "out"
    out.mkdir()
    for stale in ("manifest.json", "release-notes.md", "issue.md"):
        (out / stale).write_text("from an earlier run")

    out, code = run_ci("--previous-manifest", str(previous_manifest))

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["already_published"] is True
    assert gate["passed"] is passed
    assert gate["version"] == "de_lu-2025-10"
    for name in ("manifest.json", "release-notes.md", "issue.md"):
        assert not (out / name).exists(), name
    assert "de_lu-2025-10 is already published" in capsys.readouterr().out


def _manifest(path: Path, **zones: str) -> Path:
    doc = {"format": 1, "updated": "x", "zones": {z: {"version": v} for z, v in zones.items()}}
    path.write_text(json.dumps(doc))
    return path


def _previous_file(kind: str, trained, path: Path) -> None:
    """A previous model file for de_lu-2025-09 that cannot be compared, of ``kind``."""
    sept = replace(trained.model, version="de_lu-2025-09", train_cutoff=date(2025, 9, 1))
    if kind == "trained":  # trained up to 2025-10-01, after the gate month start
        path.write_bytes(dump_model(replace(sept, train_cutoff=date(2025, 10, 1))))
    elif kind == "other zone":
        path.write_bytes(dump_model(replace(sept, zone="at")))
    elif kind == "corrupt":
        path.write_bytes(b"not a model")
    elif kind == "schema":
        doc = json.loads(gzip.decompress(dump_model(sept)))
        doc["feature_schema"] += 1
        path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    elif kind in ("rows", "nan"):
        path.write_bytes(dump_model(sept))


@needs_raw
@pytest.mark.parametrize(
    ("previous", "why"),
    [
        ("trained", "not out of sample"),
        ("other zone", "is for zone at, not de_lu"),
        ("corrupt", "cannot be used"),
        ("schema", "feature schema"),
        ("missing", "not found"),
        ("rows", "not comparable"),
        ("nan", "not finite"),
        ("not given", "it was not given"),
    ],
)
def test_ci_fails_when_the_published_model_cannot_be_compared(
    run_ci, trained, monkeypatch, tmp_path, previous, why
):
    # Fail closed: the manifest lists a model for the zone, so skill alone is not enough.
    path = tmp_path / "de_lu-2025-09.json.gz"
    _previous_file(previous, trained, path)
    if previous in ("rows", "nan"):
        real = ci.previous_model_predictions

        def tampered(model, ds, month):
            p = real(model, ds, month)
            if previous == "nan":
                return replace(p, median=np.full(len(p.median), np.nan))
            return p.subset(np.arange(len(p.h)) > 0)  # one row fewer than the candidate

        monkeypatch.setattr(ci, "previous_model_predictions", tampered)
    manifest = _manifest(tmp_path / "manifest.json", de_lu="de_lu-2025-09", at="at-2025-09")
    extra = [] if previous == "not given" else ["--previous-model", str(path)]

    out, code = run_ci("--previous-manifest", str(manifest), *extra)

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["passed"] is False
    assert gate["previous"] is None
    assert "the published model de_lu-2025-09 cannot be compared" in gate["reason"]
    assert why in gate["reason"]
    assert not (out / "manifest.json").exists()
    assert why in (out / "issue.md").read_text()


@needs_raw
@pytest.mark.parametrize("manifest", [False, True])
def test_ci_without_a_published_model_checks_skill_only(run_ci, trained, tmp_path, manifest):
    # No manifest at all, or one without a de_lu model: the first run for the zone. A
    # previous model given anyway that cannot be compared is skipped, not a failure.
    corrupt = tmp_path / "de_lu-2025-08.json.gz"
    corrupt.write_bytes(b"not a model")
    path = _manifest(tmp_path / "manifest.json", at="at-2025-09") if manifest else tmp_path / "no"

    out, code = run_ci("--previous-manifest", str(path), "--previous-model", str(corrupt))

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["previous"] is None
    assert "no previous model in the manifest; skipped: de_lu-2025-08.json.gz" in gate["reason"]
    assert gate["passed"] is (gate["candidate"]["skill"] > 0)
    if gate["passed"]:
        zones = json.loads((out / "manifest.json").read_text())["zones"]
        assert sorted(zones) == (["at", "de_lu"] if manifest else ["de_lu"])


@needs_raw
def test_ci_compares_with_the_best_previous_model(run_ci, trained, tiny_model_file, tmp_path):
    # The bar is the lowest MAE of the comparable previous models, revoked ones excluded:
    # tolerance cannot add up over a run of models each a little worse than the one before.
    sept = replace(tiny_model_file, version="de_lu-2025-09", train_cutoff=date(2025, 9, 1))
    aug = replace(trained.model, version="de_lu-2025-08", train_cutoff=date(2025, 8, 1))
    july = replace(trained.model, version="de_lu-2025-07", train_cutoff=date(2025, 7, 1))
    paths = []
    for model in (sept, aug, july):
        paths += ["--previous-model", str(tmp_path / f"{model.version}.json.gz")]
        (tmp_path / f"{model.version}.json.gz").write_bytes(dump_model(model))
    manifest = _manifest(tmp_path / "manifest.json", de_lu="de_lu-2025-09")
    doc = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**doc, "revoked": ["de_lu-2025-07"]}))

    out, code = run_ci("--previous-manifest", str(manifest), *paths)

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    best = "compared with de_lu-2025-08, the best of de_lu-2025-08, de_lu-2025-09"
    assert best in gate["reason"]
    assert "skipped: de_lu-2025-07 is revoked" in gate["reason"]
    # the August trees are the trained ones, the September model has random trees
    month = date(2025, 9, 1)
    sept_m = ci.month_metrics(ci.previous_model_predictions(sept, trained.dataset, month), month)
    aug_m = ci.month_metrics(ci.previous_model_predictions(aug, trained.dataset, month), month)
    assert gate["previous"] == aug_m
    assert aug_m["mae"] < sept_m["mae"]
