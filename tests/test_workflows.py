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
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "train"
    assert wf["env"]["FORCE_JAVASCRIPT_ACTIONS_TO_NODE24"] is True
    jobs = wf["jobs"]
    # the training code and its third-party packages only ever see a read-only token
    assert jobs["train"]["permissions"] == {"contents": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "write", "issues": "write"}
    assert jobs["publish"]["needs"] == "train"


def test_train_workflow_publish_job_runs_no_repository_code():
    publish = _load("train.yml")["jobs"]["publish"]
    uses = [s["uses"] for s in publish["steps"] if "uses" in s]
    assert uses and all(u.startswith("actions/download-artifact@") for u in uses)
    for step in publish["steps"]:
        run = step.get("run", "")
        for program in ("pip", "python", "python3", "uv", "npm"):
            assert not _commands(run, program), (program, step.get("name"))
    # without a checkout, gh needs to be told the repository
    assert publish["env"]["GH_REPO"] == "${{ github.repository }}"


def test_training_installs_the_hash_locked_requirements():
    lock = (REPO / "training" / "requirements.lock").read_text()
    assert "scikit-learn==1.9.1 \\\n    --hash=sha256:" in lock
    pinned = [line for line in lock.splitlines() if line and not line.startswith((" ", "#"))]
    assert pinned and all("==" in line and line.endswith("\\") for line in pinned)
    train = _steps(_load("train.yml"), "train")
    tests = _steps(_load("validate.yml"), "tests-training")
    for steps in (train, tests):
        step = _step(steps, "training/requirements.lock")
        assert step["run"] == "pip install --require-hashes -r training/requirements.lock"
        setup = next(s for s in steps if s.get("uses", "").startswith("actions/setup-python"))
        assert setup["with"]["python-version"] == "3.12"  # what the lock is compiled for
        assert setup["with"]["cache-dependency-path"] == "training/requirements.lock"
    for wf in ("train.yml", "validate.yml", "release.yml"):
        assert "-r training/requirements.txt" not in (WORKFLOWS / wf).read_text(), wf


@pytest.mark.parametrize("workflow", ["train.yml", "validate.yml", "release.yml", "pages.yml"])
def test_every_job_has_a_timeout(workflow):
    for name, job in _load(workflow)["jobs"].items():
        expected = 360 if (workflow, name) == ("train.yml", "train") else 30
        assert job.get("timeout-minutes") == expected, name


@pytest.mark.parametrize(
    ("workflow", "needs"),
    [
        ("train.yml", ["train", "publish"]),
        (
            "validate.yml",
            [
                "hassfest",
                "hacs",
                "tests",
                "tests-training",
                "tests-homeassistant",
                "docs",
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
    steps = _steps(_load("train.yml"), "publish")
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
    wf = _load("train.yml")
    train, publish = _steps(wf, "train"), _steps(wf, "publish")
    ci = _step(train, "python -m training.ci_zones")
    gate = _step(train, "jq -r '.passed | join")
    upload = _step(publish, "gh release upload models")
    issue = _step(publish, "Model training gate failed")
    assert ci["id"] == "ci"
    assert gate["id"] == "gate" and "if" not in gate  # runs only after the ci step succeeded
    assert "already_published" in gate["run"]
    outputs = wf["jobs"]["train"]["outputs"]
    for name in ("read", "passed", "failed", "lagged", "manifest"):
        assert outputs[name] == f"${{{{ steps.gate.outputs.{name} }}}}"
    assert upload["if"] == (
        "needs.train.outputs.passed != '' && needs.train.outputs.manifest == 'true'"
    )
    assert upload["env"]["PASSED"] == "${{ needs.train.outputs.passed }}"
    assert issue["if"] == "needs.train.outputs.failed != ''"
    assert issue["env"]["FAILED"] == "${{ needs.train.outputs.failed }}"
    assert '--body-file "out/$zone/issue.md"' in issue["run"]
    # the manifest goes up only if it names the very files about to be uploaded
    assert upload["run"].index("sha256sum -c") < upload["run"].index("gh release upload")
    assert "select(.version == $v)" in upload["run"]


def test_train_workflow_fails_the_run_for_an_erroring_zone_after_the_hand_over():
    wf = _load("train.yml")
    train = _steps(wf, "train")
    ci = _step(train, "python -m training.ci_zones")
    # a zone's error does not stop the step before the results are read and handed on
    assert 'python -m training.ci_zones "${args[@]}" 2>out/ci-stderr.txt || rc=$?' in ci["run"]
    assert ci["run"].rstrip().endswith("fi") and "out/summary.json" in ci["run"]
    last = train[-1]
    assert last["if"] == "steps.gate.outputs.errors != ''"
    assert "exit 1" in last["run"]
    uploads = [s for s in train if "upload-artifact" in s.get("uses", "")]
    assert train.index(uploads[-1]) < train.index(last)


def test_train_workflow_override_only_by_hand():
    wf = _load("train.yml")
    (name,) = _on(wf)["workflow_dispatch"]["inputs"]
    assert name == "accept_incomparable"
    field = _on(wf)["workflow_dispatch"]["inputs"][name]
    assert field["type"] == "boolean" and field["default"] is False
    assert "feature-schema change" in field["description"]
    ci = _step(_steps(wf, "train"), "python -m training.ci_zones")
    assert ci["env"]["ACCEPT_INCOMPARABLE"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.accept_incomparable == true }}"
    )
    run = ci["run"]
    assert 'if [ "$ACCEPT_INCOMPARABLE" = true ]; then' in run
    assert run.index("args+=(--accept-incomparable)") < run.index("python -m training.ci_zones")
    # the input reaches nothing else
    text = (WORKFLOWS / "train.yml").read_text()
    assert text.count("inputs.accept_incomparable") == 1


def test_train_workflow_hands_its_files_to_publish():
    wf = _load("train.yml")
    train, publish = _steps(wf, "train"), _steps(wf, "publish")
    uploads = {s["with"]["name"]: s for s in train if "upload-artifact" in s.get("uses", "")}
    assert uploads["out"]["with"]["path"] == "out/"
    assert uploads["out"]["if"] == "steps.gate.outputs.read == 'true'"
    assert uploads["raw-data"]["with"]["path"] == "raw.tar.gz"
    assert train.index(uploads["raw-data"]) < _index(train, "python -m training.ci_zones")
    downloads = {s["with"]["name"]: s for s in publish if "download-artifact" in s.get("uses", "")}
    assert set(downloads) == {"out", "raw-data"}
    assert downloads["out"]["if"] == "needs.train.outputs.read == 'true'"
    # the archive goes up also when the training fails after it was made
    assert wf["jobs"]["publish"]["if"] == "always() && needs.train.outputs.archived == 'true'"


def test_train_workflow_compares_with_up_to_three_published_models():
    train = _steps(_load("train.yml"), "train")
    previous = _step(train, "releases/tags/models")
    run = previous["run"]
    # per zone, into its own directory
    assert "for zone in $(python -m training.zones); do" in run
    assert 'echo "$sha  previous/$zone/$name" | sha256sum -c -' in run
    assert 'test("^" + $z + "-[0-9]{4}-[0-9]{2}' in run
    assert ".[-2:]" in run and "revoked" in run  # two older ones, never a revoked one
    ci = _step(train, "python -m training.ci_zones")
    assert "--previous previous" in ci["run"]
    # writes belong to the publish job
    assert "gh release edit" not in "".join(s.get("run", "") for s in train)


def test_train_workflow_never_replaces_a_published_version():
    publish = _step(_steps(_load("train.yml"), "publish"), "gh release upload models")
    commands = _commands(publish["run"], "gh")
    uploads = [c for c in commands if c[:3] == ["gh", "release", "upload"]]
    assert len(uploads) == 2
    # the versioned assets of every passed zone: an existing name fails the step
    assert uploads[0][:4] == ["gh", "release", "upload", "models"]
    assert [u.removeprefix("out/$zone/$version") for u in uploads[0][4:]] == [
        ".json.gz",
        ".backfill.json.gz",
        ".report.md",
    ]
    # the manifest goes up last and once: it must never name assets that are not there yet
    assert uploads[1] == ["gh", "release", "upload", "models", "out/manifest.json", "--clobber"]
    assert commands.index(uploads[1]) == len(commands) - 3  # then only the notes
    run = publish["run"]
    assert run.index("for version in $PASSED; do", run.index("gh release create")) < run.index(
        'gh release upload models "out/$zone'
    )


def test_train_workflow_prepends_the_reports_to_the_release_notes():
    publish = _step(_steps(_load("train.yml"), "publish"), "gh release upload models")
    run = publish["run"]
    earlier = run.index("gh release view models --json body -q .body")
    assert run.index("gh release upload models out/manifest.json") < earlier
    assert run.index('cat "out/$zone/release-notes.md"') < earlier
    assert 'cat "$RUNNER_TEMP/new-notes.md" "$RUNNER_TEMP/earlier-notes.md"' in run
    assert run.rstrip().endswith('gh release edit models --notes-file "$RUNNER_TEMP/notes.md"')


@pytest.mark.parametrize("title", ["Model training gate failed", "Model training waiting for data"])
def test_train_workflow_opens_an_issue_only_once(title):
    run = _step(_steps(_load("train.yml"), "publish"), title)["run"]
    assert run.index("gh issue list --state open --search") < run.index("gh issue create")
    assert "select(.title == $t)" in run


def test_train_workflow_opens_data_lag_issue():
    wf = _load("train.yml")
    lag = _step(_steps(wf, "publish"), "Model training waiting for data")
    assert lag["if"] == "needs.train.outputs.lagged != ''"
    assert "out/ci-stderr.txt" in lag["run"] and "$LAGGED" in lag["run"]
    ci = _step(_steps(wf, "train"), "python -m training.ci_zones")
    assert "2>out/ci-stderr.txt" in ci["run"].replace(" ", "")
    assert "--cutoff" in ci["run"]


def test_train_workflow_archives_only_open_data():
    wf = _load("train.yml")
    steps = _steps(wf, "train")
    archive = _step(steps, "tar -czf raw.tar.gz")
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
    upload = _step(_steps(wf, "publish"), "gh release upload training-data")
    assert "gh release upload training-data raw.tar.gz --clobber" in upload["run"]


def test_validate_workflow_has_matrix_floor_and_latest():
    wf = _load("validate.yml")
    assert _on(wf)["schedule"] == [{"cron": "0 4 * * 1"}]
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
    age = _step(jobs["model-age"]["steps"], "python tools/check_model_age.py")
    assert age["env"]["GH_TOKEN"] == "${{ github.token }}"  # the API check of the release


def test_validate_workflow_runs_once_per_change():
    """Branches run through their pull request, tags not at all; superseded runs are cancelled."""
    wf = _load("validate.yml")
    on = _on(wf)
    assert on["push"] == {"branches": ["main"]}
    assert "pull_request" in on
    assert wf["concurrency"]["group"] == "validate-${{ github.event_name }}-${{ github.ref }}"
    cancel = wf["concurrency"]["cancel-in-progress"]
    assert "push" in cancel and "pull_request" in cancel and "schedule" not in cancel


def test_release_workflow_zips_after_model_fetch():
    wf = _load("release.yml")
    assert _on(wf)["release"]["types"] == ["published"]
    assert wf["permissions"] == {"contents": "read"}
    assert wf["jobs"]["zip"]["permissions"] == {"contents": "read"}
    steps = _steps(wf, "zip")
    check = _index(steps, 'if [ "${TAG#v}" != "$version" ]')
    install = _index(steps, "pip install --require-hashes -r training/requirements.lock")
    fetch = _index(steps, "tools/fetch_published_model.py custom_components/spot_forecast/models")
    zip_ = _index(steps, "zip -r")
    assert check < install < fetch < zip_
    assert "__pycache__" in steps[zip_]["run"]
    assert (
        'zip -j "$GITHUB_WORKSPACE/spot_forecast.zip" "$GITHUB_WORKSPACE/LICENSE"'
        in (steps[zip_]["run"])
    )
    assert "jq -r .version custom_components/spot_forecast/manifest.json" in steps[check]["run"]
    # the token that can write the release only attaches the ZIP the read-only job built
    upload = wf["jobs"]["upload"]
    assert upload["needs"] == "zip"
    assert upload["permissions"] == {"contents": "write"}
    (attach,) = [s for s in upload["steps"] if "run" in s]
    assert attach["run"] == 'gh release upload "$TAG" spot_forecast.zip --clobber'
    assert [s["uses"].split("@")[0] for s in upload["steps"] if "uses" in s] == [
        "actions/download-artifact"
    ]


def test_hacs_json_values():
    assert json.loads((REPO / "hacs.json").read_text()) == {
        "name": "Spot Forecast",
        "render_readme": True,
        "homeassistant": "2024.11.0",
        "zip_release": True,
        "filename": "spot_forecast.zip",
        # HACS filters by region: the integration covers the DE-LU bidding zone only
        "country": ["DE", "LU"],
    }


def test_codeowners():
    assert (REPO / ".github" / "CODEOWNERS").read_text().strip() == "* @cygnusb"


def test_pages_workflow_builds_site_then_strict_docs():
    wf = _load("pages.yml")
    steps = _steps(wf, "deploy")
    assert _index(steps, "tools/build_site.py _site") < _index(steps, "mkdocs build --strict")
    assert wf["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    # the monthly training edits the rolling `models` release: no redeploy for that
    assert (
        "startsWith(github.event.release.tag_name, 'v')" in (wf["jobs"]["redeploy-from-main"]["if"])
    )


def test_validate_workflow_checks_the_docs():
    steps = _steps(_load("validate.yml"), "docs")
    assert _index(steps, "tools/build_site.py _site") < _index(steps, "mkdocs build --strict")
