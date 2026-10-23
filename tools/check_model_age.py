"""Fail when a published model is older than the monthly training should allow.

python tools/check_model_age.py [--manifest-url URL] [--max-age-days N]

Run daily by the Validate workflow. Training runs monthly, so a zone whose ``train_cutoff`` is
more than N days (default 45) old means it stopped: the schedule was switched off, or data-lag
or gate-failure issues went unanswered. Exits 1 then, which turns the run red.

No manifest yet (HTTP 404: nothing published so far) exits 0 with a notice. Any other download
error exits 1. Standard library only.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import json
import sys
import urllib.error
import urllib.request

MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
MAX_AGE_DAYS = 45
TIMEOUT = 60


def stale_zones(manifest: dict, today: date, max_age_days: int) -> list[str]:
    """One message per zone whose model is more than ``max_age_days`` old on ``today``."""
    stale = []
    for slug, entry in sorted(manifest["zones"].items()):
        age = (today - date.fromisoformat(entry["train_cutoff"])).days
        if age > max_age_days:
            stale.append(
                f"{slug}: published model {entry['version']} has train_cutoff "
                f"{entry['train_cutoff']}, {age} days old (limit {max_age_days})"
            )
    return stale


def main(argv: list[str] | None = None, today: date | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-url", default=MANIFEST_URL)
    ap.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    args = ap.parse_args(argv)
    today = today or datetime.now(UTC).date()
    try:
        with urllib.request.urlopen(args.manifest_url, timeout=TIMEOUT) as response:
            manifest = json.loads(response.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("::notice::No published manifest yet: nothing to check")
            return 0
        print(f"::error::Manifest download failed: HTTP {e.code}")
        return 1
    except urllib.error.URLError as e:
        print(f"::error::Manifest download failed: {e.reason}")
        return 1
    stale = stale_zones(manifest, today, args.max_age_days)
    for message in stale:
        print(f"::error::{message}. Is the Train workflow still enabled and succeeding?")
    if not stale:
        count = len(manifest["zones"])
        print(f"All {count} published models are at most {args.max_age_days} days old")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
