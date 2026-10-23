"""Structure of the GitHub workflows, hacs.json and CODEOWNERS."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _on(wf: dict) -> dict:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return wf.get("on", wf.get(True))


def _steps(wf: dict, job: str) -> list[dict]:
    return wf["jobs"][job]["steps"]


def _step(steps: list[dict], needle: str) -> dict:
    """The only step whose ``run`` contains ``needle``."""
    found = [s for s in steps if needle in s.get("run", "")]
    assert len(found) == 1, f"{len(found)} steps run {needle!r}"
    return found[0]


def _index(steps: list[dict], needle: str) -> int:
    return steps.index(_step(steps, needle))


def _commands(run: str, program: str) -> list[list[str]]:
    """Every command line in ``run`` that starts with ``program``, split into words."""
    out = []
    for line in run.replace("\\\n", " ").splitlines():
        words = shlex.split(line.strip(), comments=True)
        if words and words[0] == program:
            out.append(words)
    return out


def test_train_workflow_triggers_and_permissions():
    wf = _load("train.yml")
    on = _on(wf)
    assert on["schedule"] == [{"cron": "0 2 3 * *"}]
    assert "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "write", "issues": "write"}
    assert wf["concurrency"]["group"] == "train"
    assert wf["env"]["FORCE_JAVASCRIPT_ACTIONS_TO_NODE24"] is True
    assert "permissions" not in wf["jobs"]["train"]  # no actions: write for the training


@pytest.mark.parametrize(
    ("workflow", "needs"),
    [
        ("train.yml", "train"),
        (
            "validate.yml",
            [
                "hassfest",
                "hacs",
                "tests",
                "tests-training",
                "tests-homeassistant",
                "model-age",
                "lint",
            ],
        ),
    ],
)
def test_workflows_keep_their_schedule_enabled(workflow, needs):
    wf = _load(workflow)
    keepalive = wf["jobs"]["keepalive"]
    assert keepalive["permissions"] == {"actions": "write"}
    assert keepalive["needs"] == needs
    assert keepalive["if"].startswith("always()")
    (step,) = keepalive["steps"]
    assert step["run"] == (
        f'gh api -X PUT "repos/$GITHUB_REPOSITORY/actions/workflows/{workflow}/enable"'
    )
    # every other job of the workflow runs before it
    assert set(wf["jobs"]) - {"keepalive"} == set([needs] if isinstance(needs, str) else needs)


def test_train_workflow_keeps_data_releases_out_of_hacs():
    steps = _steps(_load("train.yml"), "train")
    for tag in ("models", "training-data"):
        create = [
            c
            for s in steps
            for c in _commands(s.get("run", ""), "gh")
            if c[:4] == ["gh", "release", "create", tag]
        ]
        assert len(create) == 1 and {"--prerelease", "--latest=false"} <= set(create[0])
        # every run, also for a release created by hand
        edit = _step(steps, f"gh release edit {tag} --prerelease --latest=false")
        assert "if" not in edit


def test_train_workflow_publishes_only_on_pass():
    steps = _steps(_load("train.yml"), "train")
    ci = _step(steps, "python -m training.ci")
    gate = _step(steps, "out/gate.json")
    publish = _step(steps, "gh release upload models")
    issue = _step(steps, "Model training gate failed")
    assert ci["id"] == "ci"
    assert gate["id"] == "gate"
    assert "steps.ci.outputs.rc == '0'" in gate["if"]
    assert "steps.gate.outputs.passed == 'true'" in publish["if"]
    assert "steps.gate.outputs.manifest == 'true'" in publish["if"]
    assert "steps.gate.outputs.already_published != 'true'" in publish["if"]
    assert "steps.gate.outputs.passed != 'true'" in issue["if"]
    assert "steps.gate.outputs.already_published != 'true'" in issue["if"]
    assert "steps.ci.outputs.rc == '0'" in issue["if"]
    assert "--body-file out/issue.md" in issue["run"]
    assert "already_published" in gate["run"]
    assert _index(steps, "python -m training.ci") < _index(steps, "gh release upload models")


def test_train_workflow_never_replaces_a_published_version():
    publish = _step(_steps(_load("train.yml"), "train"), "gh release upload models")
    commands = _commands(publish["run"], "gh")
    uploads = [c for c in commands if c[:3] == ["gh", "release", "upload"]]
    assert len(uploads) == 2
    # the versioned assets: an existing name fails the step instead of being replaced
    assert uploads[0][:4] == ["gh", "release", "upload", "models"]
    assert [u.removeprefix("out/$VERSION") for u in uploads[0][4:]] == [
        ".json.gz",
        ".backfill.json.gz",
        ".report.md",
    ]
    # the manifest goes up last: it must never name assets that are not there yet
    assert uploads[1] == ["gh", "release", "upload", "models", "out/manifest.json", "--clobber"]
    assert commands.index(uploads[1]) == len(commands) - 3  # then only the notes


def test_train_workflow_prepends_the_report_to_the_release_notes():
    publish = _step(_steps(_load("train.yml"), "train"), "gh release upload models")
    run = publish["run"]
    earlier = run.index("gh release view models --json body -q .body")
    assert run.index("gh release upload models out/manifest.json") < earlier
    assert earlier < run.index("cat out/release-notes.md") < run.index('cat "$RUNNER_TEMP/earlier')
    assert run.rstrip().endswith('gh release edit models --notes-file "$RUNNER_TEMP/notes.md"')


@pytest.mark.parametrize("title", ["Model training gate failed", "Model training waiting for data"])
def test_train_workflow_opens_an_issue_only_once(title):
    run = _step(_steps(_load("train.yml"), "train"), title)["run"]
    assert run.index("gh issue list --state open --search") < run.index("gh issue create")
    assert "select(.title == $t)" in run


def test_train_workflow_opens_data_lag_issue():
    steps = _steps(_load("train.yml"), "train")
    lag = _step(steps, "Model training waiting for data")
    assert lag["if"] == "steps.ci.outputs.rc == '2'"
    assert "out/ci-stderr.txt" in lag["run"]
    ci = _step(steps, "python -m training.ci")
    # Exit 2 is the data lag; every other non-zero code fails the job.
    assert "2>out/ci-stderr.txt" in ci["run"].replace(" ", "")
    assert "--cutoff" in ci["run"]


def test_train_workflow_archives_only_open_data():
    steps = _steps(_load("train.yml"), "train")
    archive = _step(steps, "gh release upload training-data")
    (tar,) = _commands(archive["run"], "tar")
    assert tar[:4] == ["tar", "-czf", "raw.tar.gz", "-C"]
    assert tar[4] == "training/data"
    assert [p.rstrip("/") for p in tar[5:]] == [
        "raw/prices",
        "raw/weather",
        "raw/installed_power.json",
    ]
    fetch = _step(steps, "python -m training.fetch --raw training/data/raw")
    assert steps.index(fetch) < steps.index(archive)


def test_validate_workflow_has_matrix_floor_and_latest():
    wf = _load("validate.yml")
    assert _on(wf)["schedule"] == [{"cron": "0 4 * * *"}]
    jobs = wf["jobs"]
    assert {"hassfest", "hacs", "tests", "tests-homeassistant", "lint"} <= set(jobs)
    hacs = next(s for s in jobs["hacs"]["steps"] if s.get("uses") == "hacs/action@main")
    assert hacs["with"]["category"] == "integration"
    include = jobs["tests-homeassistant"]["strategy"]["matrix"]["include"]
    floor = next(e for e in include if "==" in e["phcc"])
    latest = next(e for e in include if e.get("check_latest"))
    assert floor["phcc"] == "pytest-homeassistant-custom-component==0.13.181"
    assert floor["python"] == "3.13"
    assert "josepy<2" in floor["extra"].split()
    assert latest["phcc"] == "pytest-homeassistant-custom-component"
    assert latest["python"] == "3.14"
    ha_steps = jobs["tests-homeassistant"]["steps"]
    _step(ha_steps, "pytest tests_ha")
    assert _step(ha_steps, "pypi.org/pypi/homeassistant")["if"] == "matrix.check_latest"
    _step(jobs["tests"]["steps"], "pip install -r tests/requirements.txt")
    _step(jobs["lint"]["steps"], "pip install ruff==0.16.5")
    _step(jobs["model-age"]["steps"], "python tools/check_model_age.py")


def test_release_workflow_zips_after_model_fetch():
    wf = _load("release.yml")
    assert _on(wf)["release"]["types"] == ["published"]
    steps = _steps(wf, "zip")
    fetch = _index(steps, "tools/fetch_published_model.py custom_components/spot_forecast/models")
    zip_ = _index(steps, "zip -r")
    upload = _index(steps, "gh release upload")
    assert fetch < zip_ < upload
    assert "__pycache__" in steps[zip_]["run"]
    assert "spot_forecast.zip" in steps[upload]["run"]


def test_hacs_json_values():
    assert json.loads((REPO / "hacs.json").read_text()) == {
        "name": "Spot Forecast",
        "render_readme": True,
        "homeassistant": "2024.11.0",
        "zip_release": True,
        "filename": "spot_forecast.zip",
    }


def test_codeowners():
    assert (REPO / ".github" / "CODEOWNERS").read_text().strip() == "* @cygnusb"
