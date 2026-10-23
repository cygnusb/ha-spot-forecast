"""Monthly CI training: train, run the quality gate, build the release manifest.

python -m training.ci --raw DIR --out DIR --cutoff YYYY-MM-DD
    [--previous-manifest FILE] [--previous-model FILE ...] [--capacity-month YYYY-MM]

Besides the training outputs (see ``training.train``) it writes to ``--out``:

- ``gate.json``: the gate result, plus the candidate ``version``, the gate ``month`` and
  ``already_published``;
- ``release-notes.md``: the accuracy report and the gate table;
- ``manifest.json``: the previous manifest with this zone's new entry, only if the gate passed;
- ``issue.md``: the issue body, only if the gate failed.

A published version is never replaced: if the previous manifest already lists this zone's
version (a rerun in the same month), ``gate.json`` says ``already_published`` and none of the
other three files is written.

The gate month is the last full month before the cutoff. The candidate is compared with
the best of the previous models (``--previous-model``, repeatable): the model the manifest
lists for the zone must be one of them and comparable, or the gate fails; only a manifest
without a model for the zone (the first run) lets the candidate pass on skill alone.

Exit 0 whether or not the gate passed, 2 on a data lag.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import date, timedelta
import json
from pathlib import Path
import sys

from core.model import ModelMismatch, load_model

from . import gate, train
from .dataset import Dataset
from .gate import GateResult, evaluate_gate, month_metrics, previous_model_predictions
from .manifest import manifest_entry, merge_manifest, revoked_versions
from .train import DataLag

MODELS_BASE_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models"


def gate_month(cutoff: date) -> date:
    """First day of the last full calendar month before ``cutoff``."""
    return (cutoff.replace(day=1) - timedelta(days=1)).replace(day=1)


def _score_previous(
    path: Path, zone: str, ds: Dataset, month: date, candidate_rows: int
) -> tuple[str, dict | None, str]:
    """The version of the previous model at ``path`` (its file name if it does not load),
    its month metrics, or ``None`` with the reason it cannot be compared."""
    name = path.name.removesuffix(".json.gz")
    if not path.exists():
        return name, None, f"{path.name} not found"
    try:
        model = load_model(path.read_bytes())
    except ModelMismatch as e:
        return name, None, f"{path.name} cannot be used: {e}"
    v = model.version
    if model.zone != zone:
        return v, None, f"{v} is for zone {model.zone}, not {zone}"
    if model.train_cutoff > month:
        why = f"trained up to {model.train_cutoff}, after the gate month start {month}"
        return v, None, f"{v} is not out of sample: {why}"
    metrics = month_metrics(previous_model_predictions(model, ds, month), month)
    if metrics["rows"] != candidate_rows:
        why = f"{metrics['rows']} rows, the candidate {candidate_rows}"
        return v, None, f"{v} scored {why}: not comparable"
    if not gate._finite(metrics["mae"]):
        return v, None, f"{v} has MAE {metrics['mae']}, not finite"
    return v, metrics, ""


def _previous_metrics(
    paths: list[Path],
    published: str | None,
    revoked: frozenset[str],
    zone: str,
    ds: Dataset,
    month: date,
    candidate_rows: int,
) -> tuple[dict | None, str, bool]:
    """The month metrics of the best (lowest MAE) previous model, a note for the gate reason
    and whether the gate may go on.

    ``published`` is the manifest's version for ``zone``. Unless it is ``None`` or revoked,
    it must be among ``paths`` and comparable, or the gate fails (False): only a manifest
    without a model for the zone lets the candidate pass on skill alone. The other previous
    models (train.yml downloads up to three) count when they can be compared and are
    skipped otherwise, as are revoked ones.
    """
    scored: dict[str, dict] = {}
    skipped: list[str] = []
    for path in paths:
        version, metrics, why = _score_previous(path, zone, ds, month, candidate_rows)
        if metrics is None:
            skipped.append(why)
        elif version in revoked:
            skipped.append(f"{version} is revoked")
        else:
            scored[version] = metrics
    required = published if published is not None and published not in revoked else None
    if required is not None and required not in scored:
        why = "; ".join(skipped) or "it was not given"
        return None, f"the published model {required} cannot be compared ({why})", False
    tail = f"; skipped: {'; '.join(skipped)}" if skipped else ""
    if not scored:
        if published is None:
            return None, f"no previous model in the manifest{tail}", True
        return None, f"the published model {published} is revoked{tail}", True
    best = min(scored, key=lambda v: scored[v]["mae"])
    note = f"compared with {best}, the best of {', '.join(sorted(scored))}{tail}"
    return scored[best], note, True


def _previous_manifest(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        print("no previous manifest: starting a new one")
        return None
    return json.loads(path.read_text())


def _published_version(manifest: dict | None, zone: str) -> str | None:
    """The version the manifest lists for ``zone``, if any."""
    if manifest is None:
        return None
    return manifest.get("zones", {}).get(zone, {}).get("version")


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m training.ci", description=__doc__)
    ap.add_argument("--raw", required=True, help="raw-data directory (fetch output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", type=date.fromisoformat, required=True)
    ap.add_argument("--previous-manifest", type=Path, default=None)
    ap.add_argument(
        "--previous-model",
        type=Path,
        action="append",
        default=[],
        help="a previously published model; repeatable, the best of them is the bar",
    )
    ap.add_argument("--capacity-month", default=None, help="passed on to training.train")
    return ap.parse_args(argv)


def run(argv: list[str], max_iter: int = 600) -> int:
    """Exit code: 0 whether or not the gate passed, 2 on ``DataLag``."""
    args = _parse(argv)
    train_argv = ["--raw", args.raw, "--out", str(args.out), "--cutoff", args.cutoff.isoformat()]
    if args.capacity_month:
        train_argv += ["--capacity-month", args.capacity_month]
    try:
        result = train.run(train_argv, max_iter)
    except DataLag as e:
        print(e, file=sys.stderr)
        return 2

    old_manifest = _previous_manifest(args.previous_manifest)
    published = _published_version(old_manifest, result.model.zone)
    month = gate_month(args.cutoff)
    candidate = month_metrics(result.predictions, month)
    already_published = published == result.version
    previous, note, comparable = _previous_metrics(
        args.previous_model,
        None if already_published else published,  # a rerun: nothing to compare, see below
        frozenset(revoked_versions(old_manifest)),
        result.model.zone,
        result.dataset,
        month,
        candidate["rows"],
    )
    if comparable:
        verdict = evaluate_gate(candidate, previous)
        verdict = replace(verdict, reason=f"{verdict.reason} ({note})")
    else:
        verdict = GateResult(False, candidate, None, note)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    gate_doc = {
        "version": result.version,
        "month": f"{month:%Y-%m}",
        "already_published": already_published,
        **asdict(verdict),
    }
    (out / "gate.json").write_text(json.dumps(gate_doc, indent=2) + "\n")
    notes_path, manifest_path, issue_path = (
        out / "release-notes.md",
        out / "manifest.json",
        out / "issue.md",
    )
    if already_published:
        for path in (notes_path, manifest_path, issue_path):
            path.unlink(missing_ok=True)
        print(
            f"{result.version} is already published: a published version is never replaced, "
            f"nothing to upload (gate {'passed' if verdict.passed else 'FAILED'}: "
            f"{verdict.reason})"
        )
        return 0
    notes = result.report_path.read_text() + "\n" + gate.render_markdown(verdict, month)
    notes_path.write_text(notes)
    if verdict.passed:
        entry = manifest_entry(
            result.model_path.read_bytes(),
            result.backfill_path.read_bytes(),
            result.model,
            MODELS_BASE_URL,
        )
        manifest = merge_manifest(old_manifest, result.model.zone, entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        issue_path.unlink(missing_ok=True)
    else:
        manifest_path.unlink(missing_ok=True)
        issue_path.write_text(
            f"Model training gate failed for {result.version}: nothing was published and "
            "the previous model stays active.\n\n" + notes
        )
    print(f"gate {'passed' if verdict.passed else 'FAILED'}: {verdict.reason}")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
