"""CI driver of the monthly training, and the installed-capacity data lag."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import hashlib
import json
import os
from pathlib import Path

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
@pytest.mark.parametrize(
    ("previous", "why"),
    [
        ("trained", "not out of sample"),  # cutoff 2025-10-01 is after the gate month start
        ("other zone", "is for zone at, not de_lu"),
        ("corrupt", "cannot be used"),
        ("missing", "not found"),
    ],
)
def test_ci_without_a_comparable_previous_model_checks_skill_only(
    run_ci, trained, tmp_path, previous, why
):
    path = tmp_path / "previous.json.gz"
    if previous == "trained":
        path.write_bytes(trained.model_path.read_bytes())
    elif previous == "other zone":
        other = replace(trained.model, zone="at", train_cutoff=date(2025, 9, 1))
        path.write_bytes(dump_model(other))
    elif previous == "corrupt":
        path.write_bytes(b"not a model")

    out, code = run_ci("--previous-model", str(path), "--previous-manifest", str(tmp_path / "no"))

    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["previous"] is None
    assert why in gate["reason"]
    assert gate["passed"] is (gate["candidate"]["skill"] > 0)
    if gate["passed"]:
        manifest = json.loads((out / "manifest.json").read_text())
        assert list(manifest["zones"]) == ["de_lu"]
