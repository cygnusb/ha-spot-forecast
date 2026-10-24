"""Monthly CI training: train, run the quality gate, build the release manifest.

python -m training.ci --raw DIR --out DIR --cutoff YYYY-MM-DD [--zone CODE]
    [--previous-manifest FILE] [--previous-model FILE ...] [--capacity-month YYYY-MM]
    [--parent-model FILE --parent-predictions FILE]

One zone per run; ``training.ci_zones`` runs every zone and chains the manifests.

Besides the training outputs (see ``training.train``) it writes to ``--out``:

- ``gate.json``: the gate result, plus the candidate ``version``, the gate ``month``,
  ``already_published`` and ``override`` (whether ``--accept-incomparable`` was used);
- ``release-notes.md``: the accuracy report and the gate table (and the override, if used);
- ``manifest.json``: the previous manifest with this zone's new entry, only if the gate passed;
- ``issue.md``: the issue body, only if the gate failed.

A published version is never replaced: if the previous manifest already lists this zone's
version (a rerun in the same month), ``gate.json`` says ``already_published`` and none of the
other three files is written.

The gate month is the last full month before the cutoff. The candidate is compared with
the best of the previous models (``--previous-model``, repeatable): the model the manifest
lists for the zone must be one of them and comparable, or the gate fails; only a manifest
without a model for the zone (the first run) lets the candidate pass on skill alone. The
one exception is ``--accept-incomparable``, set by hand for the first run after a
feature-schema change (the published model cannot load then): see ``_previous_metrics``.

Before the gate, the model file and backfill written to ``--out`` are loaded the way the
integration loads them (``load_model``, zone, sample check, backfill parser); a file that
does not load, or a cutoff more than ``FUTURE_CUTOFF_DAYS`` in the future, exits 1 and
writes nothing else. Exit 0 whether or not the gate passed, 2 on a data lag.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import sys
import zlib

from core.ledger import runs_from_backfill_file
from core.manifest import MAX_CUTOFF_AHEAD
from core.model import ModelFile, ModelMismatch, load_model, sample_check
from core.zones import ZONES, zone_by_slug

from . import gate, train
from .dataset import Dataset
from .gate import GateResult, evaluate_gate, month_metrics, previous_model_predictions
from .manifest import manifest_entry, merge_manifest, revoked_versions
from .train import DataLag

MODELS_BASE_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models"
OVERRIDE = "OVERRIDE --accept-incomparable (set by hand):"
FUTURE_CUTOFF_DAYS = MAX_CUTOFF_AHEAD.days  # the integration rejects a later cutoff too


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
    if model.parent is not None and ds.X_parent is None:
        return v, None, f"{v} blends with {model.parent.zone}, the dataset has no rows of it"
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
    accept_incomparable: bool = False,
) -> tuple[dict | None, str, bool]:
    """The month metrics of the best (lowest MAE) previous model, a note for the gate reason
    and whether the gate may go on.

    ``published`` is the manifest's version for ``zone``. Unless it is ``None`` or revoked,
    it must be among ``paths`` and comparable, or the gate fails (False): only a manifest
    without a model for the zone lets the candidate pass on skill alone. The other previous
    models (train.yml downloads up to three) count when they can be compared and are
    skipped otherwise, as are revoked ones.

    ``accept_incomparable`` is the manual override for the first run after a feature-schema
    change, when the published model cannot load: an incomparable published model then does
    not fail the gate, the note says so, and the comparable others (if any) still set the bar.
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
    head = ""
    if required is not None and required not in scored:
        why = "; ".join(skipped) or "it was not given"
        problem = f"the published model {required} cannot be compared ({why})"
        if not accept_incomparable:
            return None, problem, False
        head = f"{OVERRIDE} {problem}, accepted; "
        skipped = []  # already named
    tail = f"; skipped: {'; '.join(skipped)}" if skipped else ""
    if not scored:
        if head:
            return None, f"{head}no other previous model is comparable: skill only", True
        if published is None:
            return None, f"no previous model in the manifest{tail}", True
        return None, f"the published model {published} is revoked{tail}", True
    best = min(scored, key=lambda v: scored[v]["mae"])
    note = f"{head}compared with {best}, the best of {', '.join(sorted(scored))}{tail}"
    return scored[best], note, True


def _future_cutoff(cutoff: date, today: date) -> str | None:
    limit = today + timedelta(days=FUTURE_CUTOFF_DAYS)
    if cutoff > limit:
        return f"cutoff {cutoff} is more than {FUTURE_CUTOFF_DAYS} days after today ({today})"
    return None


def check_outputs(
    model_bytes: bytes, backfill_bytes: bytes, zone: str, version: str, today: date
) -> ModelFile:
    """Load the files to publish as the integration does; ``ValueError`` if one fails.

    The model: ``load_model``, the zone and the sample check (``model_loader.load_checked``),
    its version and a cutoff at most ``FUTURE_CUTOFF_DAYS`` ahead. The backfill: the
    integration's parser (``runs_from_backfill_file``), at least one run, all of ``version``.
    """
    model = load_model(model_bytes)
    if model.zone != zone:
        raise ModelMismatch(f"model {model.version} is for zone {model.zone!r}, not {zone!r}")
    sample_check(model)
    if model.version != version:
        raise ValueError(f"model file says version {model.version}, expected {version}")
    if (why := _future_cutoff(model.train_cutoff, today)) is not None:
        raise ValueError(f"model {model.version}: {why}")
    tz = zone_by_slug(zone).tz
    try:
        runs = runs_from_backfill_file(backfill_bytes, version, tz)
    except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"backfill of {version} does not parse: {type(e).__name__}: {e}") from e
    if not runs:
        raise ValueError(f"backfill of {version} has no rows")
    if {r.model_version for r in runs} != {version}:
        raise ValueError(f"backfill of {version} names another model version")
    return model


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
    ap.add_argument(
        "--zone",
        choices=list(ZONES),
        default=next(iter(ZONES)),
        help="bidding zone (default: %(default)s)",
    )
    ap.add_argument("--previous-manifest", type=Path, default=None)
    ap.add_argument(
        "--previous-model",
        type=Path,
        action="append",
        default=[],
        help="a previously published model; repeatable, the best of them is the bar",
    )
    ap.add_argument(
        "--accept-incomparable",
        action="store_true",
        help="manual override for the first run after a feature-schema change: a published "
        "model that cannot be compared does not fail the gate (the reason says so)",
    )
    ap.add_argument("--capacity-month", default=None, help="passed on to training.train")
    ap.add_argument("--parent-model", default=None, help="passed on to training.train")
    ap.add_argument("--parent-predictions", default=None, help="passed on to training.train")
    return ap.parse_args(argv)


def run(argv: list[str], max_iter: int = 600, today: date | None = None) -> int:
    """Exit code: 0 whether or not the gate passed, 1 on files that would not load in the
    integration, 2 on ``DataLag``. ``today`` (UTC) is for tests."""
    args = _parse(argv)
    today = today or datetime.now(UTC).date()
    if (why := _future_cutoff(args.cutoff, today)) is not None:
        print(f"refusing to train: {why}", file=sys.stderr)
        return 1
    train_argv = ["--raw", args.raw, "--out", str(args.out), "--cutoff", args.cutoff.isoformat()]
    train_argv += ["--zone", args.zone]
    if args.capacity_month:
        train_argv += ["--capacity-month", args.capacity_month]
    if args.parent_model:
        train_argv += ["--parent-model", args.parent_model]
    if args.parent_predictions:
        train_argv += ["--parent-predictions", args.parent_predictions]
    try:
        result = train.run(train_argv, max_iter)
    except DataLag as e:
        print(e, file=sys.stderr)
        return 2
    model_bytes = result.model_path.read_bytes()
    backfill_bytes = result.backfill_path.read_bytes()
    try:
        check_outputs(model_bytes, backfill_bytes, result.model.zone, result.version, today)
    except ValueError as e:
        print(f"the trained files would not load in the integration: {e}", file=sys.stderr)
        for name in ("gate.json", "release-notes.md", "manifest.json", "issue.md"):
            (args.out / name).unlink(missing_ok=True)  # nothing of an earlier run either
        return 1

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
        args.accept_incomparable,
    )
    if comparable:
        verdict = evaluate_gate(candidate, previous)
        verdict = replace(verdict, reason=f"{verdict.reason} ({note})")
    else:
        verdict = GateResult(False, candidate, None, note)

    overridden = note.startswith(OVERRIDE)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    gate_doc = {
        "version": result.version,
        "month": f"{month:%Y-%m}",
        "already_published": already_published,
        "override": overridden,
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
    if overridden:
        notes += (
            "\n**Manual override:** this run was started with --accept-incomparable, so the "
            "published model was not compared with. "
            f"{note.removeprefix(OVERRIDE).strip()}\n"
        )
    notes_path.write_text(notes)
    if verdict.passed:
        entry = manifest_entry(model_bytes, backfill_bytes, result.model, MODELS_BASE_URL)
        manifest = merge_manifest(old_manifest, result.model.zone, entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        issue_path.unlink(missing_ok=True)
    else:
        manifest_path.unlink(missing_ok=True)
        issue_path.write_text(
            f"Model training gate failed for {result.version}: nothing was published and "
            "the previous model stays active.\n\n" + notes
        )
    if overridden:
        print(f"::warning::gate run with --accept-incomparable: {note}")
    print(f"gate {'passed' if verdict.passed else 'FAILED'}: {verdict.reason}")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
