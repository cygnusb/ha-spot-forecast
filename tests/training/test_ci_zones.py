"""Every zone in one CI run: the manifests chain from zone to zone (fake ``training.ci``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from training import ci, ci_zones
from training.manifest import merge_manifest

CUTOFF = "2026-11-01"
DOWNLOADED = {
    "format": 1,
    "updated": "x",
    "zones": {"de_lu": {"version": "de_lu-2026-10"}, "nl": {"version": "nl-2026-10"}},
    "revoked": ["de_lu-2026-08"],
}


def _arg(argv: list, name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


@pytest.fixture
def fake_ci(monkeypatch):
    """``training.ci.run`` with a scripted outcome per zone code (if not given
    ``"published"``: nothing to upload or report); records each zone's argv and the manifest
    it was handed. A run that gets as far as the gate writes the candidate model and
    predictions, as ``training.train`` does."""
    outcomes: dict[str, str] = {}
    seen: dict[str, dict] = {}

    def run(argv, max_iter=600, today=None):
        code, out = _arg(argv, "--zone"), Path(_arg(argv, "--out"))
        slug = code.lower().replace("-", "_")
        prev = _arg(argv, "--previous-manifest")
        old = json.loads(Path(prev).read_text()) if prev else None
        seen[code] = {"argv": argv, "manifest": old}
        outcome = outcomes.get(code, "published")
        if outcome == "error":
            raise ValueError("weather nrw__wind ends before the cutoff")
        if outcome == "lag":
            return 2
        version = f"{slug}-2026-11"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{version}.json.gz").write_bytes(b"model")
        (out / f"{version}.predictions.json.gz").write_bytes(b"predictions")
        gate = {
            "version": version,
            "already_published": outcome == "published",
            "passed": outcome in ("pass", "published"),
        }
        (out / "gate.json").write_text(json.dumps(gate))
        if outcome == "pass":
            manifest = merge_manifest(old, slug, {"version": version})
            (out / "manifest.json").write_text(json.dumps(manifest))
        return 0

    monkeypatch.setattr(ci, "run", run)
    return outcomes, seen


def _go(tmp_path: Path, downloaded: dict | None = DOWNLOADED) -> tuple[int, dict, Path]:
    previous = tmp_path / "previous"
    (previous / "nl").mkdir(parents=True)
    (previous / "nl" / "nl-2026-10.json.gz").write_bytes(b"model")
    if downloaded is not None:
        (previous / "manifest.json").write_text(json.dumps(downloaded))
    out = tmp_path / "out"
    argv = ["--raw", "raw", "--out", str(out), "--cutoff", CUTOFF, "--previous", str(previous)]
    rc = ci_zones.run(argv)
    return rc, json.loads((out / "summary.json").read_text()), out


def _final(out: Path) -> dict | None:
    path = out / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_both_pass_the_second_zone_gets_the_first_zones_manifest(tmp_path, fake_ci):
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "pass", "NL": "pass"})
    rc, summary, out = _go(tmp_path)
    assert rc == 0 and summary["passed"] == ["de_lu-2026-11", "nl-2026-11"]
    assert seen["DE-LU"]["manifest"] == DOWNLOADED
    assert seen["NL"]["manifest"]["zones"]["de_lu"] == {"version": "de_lu-2026-11"}
    final = _final(out)
    assert final["zones"] == {
        "de_lu": {"version": "de_lu-2026-11"},
        "nl": {"version": "nl-2026-11"},
    }
    assert final["revoked"] == ["de_lu-2026-08"]
    # each zone into its own directory, with only its own previous models
    assert _arg(seen["NL"]["argv"], "--out") == str(out / "nl")
    assert _arg(seen["NL"]["argv"], "--previous-model").endswith("nl/nl-2026-10.json.gz")
    assert "--previous-model" not in seen["DE-LU"]["argv"]


def test_only_nl_passes_de_lu_keeps_its_entry(tmp_path, fake_ci):
    """(a) DE-LU's gate fails: NL gets the downloaded manifest, DE-LU's entry is untouched,
    and there is one manifest to upload."""
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "fail", "NL": "pass"})
    rc, summary, out = _go(tmp_path)
    assert rc == 0
    assert (summary["passed"], summary["failed"]) == (["nl-2026-11"], ["de_lu-2026-11"])
    assert seen["NL"]["manifest"] == DOWNLOADED
    final = _final(out)
    assert final["zones"]["de_lu"] == DOWNLOADED["zones"]["de_lu"]
    assert final["zones"]["nl"] == {"version": "nl-2026-11"}
    assert sorted(p.name for p in out.glob("**/manifest.json")) == ["manifest.json"] * 2


def test_only_de_lu_passes_nl_keeps_its_entry(tmp_path, fake_ci):
    outcomes, _ = fake_ci
    outcomes.update({"DE-LU": "pass", "NL": "fail"})
    _, summary, out = _go(tmp_path)
    assert (summary["passed"], summary["failed"]) == (["de_lu-2026-11"], ["nl-2026-11"])
    final = _final(out)
    assert final["zones"]["nl"] == DOWNLOADED["zones"]["nl"]
    assert final["zones"]["de_lu"] == {"version": "de_lu-2026-11"}


@pytest.mark.parametrize("outcomes", [("fail", "fail"), ("published", "fail"), ("lag", "lag")])
def test_no_zone_passes_no_manifest(tmp_path, fake_ci, outcomes):
    """(b) Nothing passed: no manifest at all, so the publish job uploads none."""
    fake_ci[0].update(dict(zip(("DE-LU", "NL"), outcomes, strict=True)))
    rc, summary, out = _go(tmp_path)
    assert rc == 0 and summary["passed"] == [] and _final(out) is None


def test_one_already_published_the_other_passes(tmp_path, fake_ci):
    """(c) DE-LU reruns a published month: its entry stays, NL's lands."""
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "published", "NL": "pass"})
    rc, summary, out = _go(tmp_path)
    assert rc == 0 and summary["passed"] == ["nl-2026-11"] and summary["failed"] == []
    assert summary["zones"]["de_lu"]["already_published"] is True
    assert seen["NL"]["manifest"] == DOWNLOADED
    final = _final(out)
    assert final["zones"]["de_lu"] == DOWNLOADED["zones"]["de_lu"]
    assert final["zones"]["nl"] == {"version": "nl-2026-11"}


def test_an_erroring_or_lagging_zone_does_not_block_the_next(tmp_path, fake_ci, capsys):
    outcomes, _ = fake_ci
    outcomes.update({"DE-LU": "error", "NL": "pass"})
    rc, summary, out = _go(tmp_path)
    assert rc == 1  # reported, after NL ran and the files are written
    # DK1 and DK2 blend with DE-LU: skipped, as errors
    assert summary["errors"] == ["DE-LU", "DK1", "DK2"] and summary["passed"] == ["nl-2026-11"]
    assert "weather nrw__wind ends before the cutoff" in capsys.readouterr().err
    assert _final(out)["zones"]["de_lu"] == DOWNLOADED["zones"]["de_lu"]

    outcomes.update({"DE-LU": "lag"})
    rc, summary, _ = _go(tmp_path / "again")
    assert rc == 0 and summary["lagged"] == ["DE-LU", "DK1", "DK2"]
    assert summary["passed"] == ["nl-2026-11"]


def test_first_run_without_a_manifest(tmp_path, fake_ci):
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "pass", "NL": "pass"})
    _, _, out = _go(tmp_path, downloaded=None)
    assert seen["DE-LU"]["manifest"] is None and "--previous-manifest" not in seen["DE-LU"]["argv"]
    assert sorted(_final(out)["zones"]) == ["de_lu", "nl"]


def test_options_reach_every_zone(tmp_path, fake_ci):
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "fail", "NL": "fail"})
    argv = ["--raw", "r", "--out", str(tmp_path / "o"), "--cutoff", CUTOFF,
            "--accept-incomparable", "--capacity-month", "2026-09"]  # fmt: skip
    ci_zones.run(argv)
    for code in ("DE-LU", "NL"):
        assert "--accept-incomparable" in seen[code]["argv"]
        assert _arg(seen[code]["argv"], "--capacity-month") == "2026-09"


def test_zones_train_in_order_parent_first(tmp_path, fake_ci):
    _, seen = fake_ci
    _go(tmp_path)
    assert list(seen) == ["DE-LU", "NL", "DK1", "DK2"]


@pytest.mark.parametrize("de_lu", ["pass", "fail", "published"])
def test_dk_zones_get_the_de_lu_candidate_of_the_run(tmp_path, fake_ci, de_lu):
    """Whether DE-LU's gate passed, failed or its version is published already: DK1 and DK2
    train with the DE-LU model this run trained, which is what their files embed."""
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": de_lu, "DK1": "pass", "DK2": "pass"})
    rc, summary, out = _go(tmp_path)
    assert rc == 0
    for code in ("DK1", "DK2"):
        argv = seen[code]["argv"]
        assert _arg(argv, "--parent-model") == str(out / "de_lu" / "de_lu-2026-11.json.gz")
        assert _arg(argv, "--parent-predictions") == str(
            out / "de_lu" / "de_lu-2026-11.predictions.json.gz"
        )
    for code in ("DE-LU", "NL"):
        assert "--parent-model" not in seen[code]["argv"]
    assert summary["passed"][-2:] == ["dk1-2026-11", "dk2-2026-11"]
    final = _final(out)
    assert final["zones"]["dk1"] == {"version": "dk1-2026-11"}
    expected = {"version": "de_lu-2026-11"} if de_lu == "pass" else DOWNLOADED["zones"]["de_lu"]
    assert final["zones"]["de_lu"] == expected


def test_de_lu_fails_its_gate_dk_still_published(tmp_path, fake_ci):
    outcomes, _ = fake_ci
    outcomes.update({"DE-LU": "fail", "NL": "fail", "DK1": "pass", "DK2": "fail"})
    rc, summary, out = _go(tmp_path)
    assert rc == 0
    assert summary["passed"] == ["dk1-2026-11"]
    assert summary["failed"] == ["de_lu-2026-11", "nl-2026-11", "dk2-2026-11"]
    final = _final(out)
    assert final["zones"]["de_lu"] == DOWNLOADED["zones"]["de_lu"]
    assert final["zones"]["dk1"] == {"version": "dk1-2026-11"} and "dk2" not in final["zones"]


@pytest.mark.parametrize(("de_lu", "listed"), [("error", "errors"), ("lag", "lagged")])
def test_de_lu_without_a_candidate_skips_dk_and_keeps_their_entries(
    tmp_path, fake_ci, capsys, de_lu, listed
):
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": de_lu, "NL": "pass", "DK1": "pass", "DK2": "pass"})
    downloaded = {**DOWNLOADED, "zones": {**DOWNLOADED["zones"], "dk1": {"version": "dk1-2026-10"}}}
    rc, summary, out = _go(tmp_path, downloaded)
    assert "DK1" not in seen and "DK2" not in seen  # never trained
    assert summary[listed] == ["DE-LU", "DK1", "DK2"]
    assert summary["zones"]["dk1"] == {"rc": 1 if de_lu == "error" else 2,
                                       "skipped": "parent DE-LU"}  # fmt: skip
    assert rc == (1 if de_lu == "error" else 0)
    assert "DK1: skipped, the training of its parent zone DE-LU" in capsys.readouterr().err
    final = _final(out)  # NL passed: the manifest keeps the DK1 entry and has no DK2 one
    assert final["zones"]["dk1"] == {"version": "dk1-2026-10"} and "dk2" not in final["zones"]
    assert final["zones"]["nl"] == {"version": "nl-2026-11"}


def test_de_lu_without_its_files_skips_dk(tmp_path, fake_ci, monkeypatch):
    """A DE-LU run that ended without writing the files (a bug) is no parent either."""
    outcomes, seen = fake_ci
    outcomes.update({"DE-LU": "pass", "DK1": "pass", "DK2": "pass"})
    real = ci.run

    def run(argv, max_iter=600, today=None):
        rc = real(argv, max_iter, today)
        if _arg(argv, "--zone") == "DE-LU":
            (Path(_arg(argv, "--out")) / "de_lu-2026-11.predictions.json.gz").unlink()
        return rc

    monkeypatch.setattr(ci, "run", run)
    rc, summary, _ = _go(tmp_path)
    assert rc == 1 and "DK1" not in seen and summary["errors"] == ["DK1", "DK2"]
