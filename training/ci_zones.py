"""Monthly CI training of every zone: ``training.ci`` per zone, with the manifests chained.

python -m training.ci_zones --raw DIR --out DIR --cutoff YYYY-MM-DD [--previous DIR]
    [--accept-incomparable] [--capacity-month YYYY-MM]

``--previous`` holds what train.yml downloaded: ``manifest.json`` (if published) and per zone
``<slug>/*.json.gz``, its previous models. Each zone runs ``training.ci`` into
``--out``/<slug>/ with, as ``--previous-manifest``, the manifest written by the last zone
that passed, else the downloaded one, so every zone's entry lands and the others' stay as
they were. A zone that fails its gate, waits for data or errors never stops the next one.

Besides the per-zone outputs it writes to ``--out``:

- ``manifest.json``: the last manifest a zone wrote, only if a zone passed;
- ``summary.json``: per zone ``rc``, ``version``, ``passed`` and ``already_published``, and
  the lists ``passed`` (versions to publish), ``failed`` (versions whose gate failed),
  ``lagged`` and ``errors`` (zone codes, exit codes 2 and anything else but 0).

Exit 1 if a zone errored (after all zones ran and the files above are written), else 0.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import sys
import traceback

from core.zones import ZONES

from . import ci


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m training.ci_zones", description=__doc__)
    ap.add_argument("--raw", required=True, help="raw-data directory (fetch output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--previous", type=Path, default=None, help="the downloaded release files")
    ap.add_argument("--accept-incomparable", action="store_true", help="see training.ci")
    ap.add_argument("--capacity-month", default=None, help="passed on to training.ci")
    return ap.parse_args(argv)


def _zone_argv(args: argparse.Namespace, code: str, out: Path, manifest: Path | None) -> list:
    argv = ["--raw", args.raw, "--out", str(out), "--cutoff", args.cutoff, "--zone", code]
    if manifest is not None:
        argv += ["--previous-manifest", str(manifest)]
    if args.previous is not None:
        for model in sorted((args.previous / ZONES[code].slug).glob("*.json.gz")):
            argv += ["--previous-model", str(model)]
    if args.accept_incomparable:
        argv.append("--accept-incomparable")
    if args.capacity_month:
        argv += ["--capacity-month", args.capacity_month]
    return argv


def run(argv: list[str], max_iter: int = 600, today: date | None = None) -> int:
    args = _parse(argv)
    downloaded = args.previous / "manifest.json" if args.previous is not None else None
    manifest = downloaded if downloaded is not None and downloaded.exists() else None
    zones: dict[str, dict] = {}
    summary: dict = {"zones": zones, "passed": [], "failed": [], "lagged": [], "errors": []}
    for code, zone in ZONES.items():
        out = args.out / zone.slug
        print(f"::group::{code}", flush=True)
        try:
            rc = ci.run(_zone_argv(args, code, out, manifest), max_iter, today)
        except Exception:  # noqa: BLE001 - one zone's error never stops the others
            traceback.print_exc()
            rc = 1
        print("::endgroup::", flush=True)
        entry: dict = {"rc": rc}
        if rc == 0:
            gate = json.loads((out / "gate.json").read_text())
            entry.update(
                version=gate["version"],
                passed=gate["passed"],
                already_published=gate["already_published"],
            )
            if gate["already_published"]:
                pass  # a published version is never replaced: nothing to upload or report
            elif not gate["passed"]:
                summary["failed"].append(gate["version"])
            elif (out / "manifest.json").exists():
                manifest = out / "manifest.json"
                summary["passed"].append(gate["version"])
            else:
                print(f"{code}: the gate passed but training.ci wrote no manifest", file=sys.stderr)
                summary["errors"].append(code)
        else:
            summary["lagged" if rc == 2 else "errors"].append(code)
        zones[zone.slug] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "manifest.json"
    if summary["passed"]:
        shutil.copyfile(manifest, final)
    else:
        final.unlink(missing_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    if summary["errors"]:
        print(f"training errored for {', '.join(summary['errors'])}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
