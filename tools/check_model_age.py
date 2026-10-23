"""Fail when a published model is older than the monthly training should allow.

python tools/check_model_age.py [--manifest-url URL] [--release-api-url URL] [--max-age-days N]

Run daily by the Validate workflow. Training runs monthly, so a zone whose ``train_cutoff`` is
more than N days (default 45) old means it stopped: the schedule was switched off, or data-lag
or gate-failure issues went unanswered. Exits 1 then, which turns the run red. A cutoff more
than ``FUTURE_DAYS`` in the future fails as well (the integration does not use such a model),
and so does a manifest in another format.

No manifest (HTTP 404) is fine only while the ``models`` release does not exist either, asked
through the GitHub API (with ``$GH_TOKEN`` when set): nothing published so far, exit 0 with a
notice. Once the release exists a missing manifest fails, as does any other download error.
Standard library only.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import json
import os
import sys
import urllib.error
import urllib.request

MANIFEST_URL = "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
RELEASE_API_URL = "https://api.github.com/repos/cygnusb/ha-spot-forecast/releases/tags/models"
MANIFEST_FORMAT = 1
MAX_AGE_DAYS = 45
FUTURE_DAYS = 7  # as training/ci.py and the integration
TIMEOUT = 60


def stale_zones(manifest: dict, today: date, max_age_days: int) -> list[str]:
    """One message per zone whose model is more than ``max_age_days`` old on ``today``, or
    trained up to a day more than ``FUTURE_DAYS`` ahead."""
    stale = []
    for slug, entry in sorted(manifest["zones"].items()):
        age = (today - date.fromisoformat(entry["train_cutoff"])).days
        if age > max_age_days:
            stale.append(
                f"{slug}: published model {entry['version']} has train_cutoff "
                f"{entry['train_cutoff']}, {age} days old (limit {max_age_days})"
            )
        elif age < -FUTURE_DAYS:
            stale.append(
                f"{slug}: published model {entry['version']} has train_cutoff "
                f"{entry['train_cutoff']}, {-age} days in the future (limit {FUTURE_DAYS})"
            )
    return stale


def _release_exists(url: str) -> bool:
    """Whether the GitHub API knows the release (``HTTPError`` other than 404 propagates)."""
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def _missing_manifest(release_api_url: str) -> int:
    try:
        exists = _release_exists(release_api_url)
    except urllib.error.HTTPError as e:
        print(f"::error::No manifest, and the release check failed: HTTP {e.code}")
        return 1
    except urllib.error.URLError as e:
        print(f"::error::No manifest, and the release check failed: {e.reason}")
        return 1
    if exists:
        print("::error::The models release exists but has no manifest.json")
        return 1
    print("::notice::No models release yet: nothing to check")
    return 0


def main(argv: list[str] | None = None, today: date | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-url", default=MANIFEST_URL)
    ap.add_argument("--release-api-url", default=RELEASE_API_URL)
    ap.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    args = ap.parse_args(argv)
    today = today or datetime.now(UTC).date()
    try:
        with urllib.request.urlopen(args.manifest_url, timeout=TIMEOUT) as response:
            manifest = json.loads(response.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return _missing_manifest(args.release_api_url)
        print(f"::error::Manifest download failed: HTTP {e.code}")
        return 1
    except urllib.error.URLError as e:
        print(f"::error::Manifest download failed: {e.reason}")
        return 1
    except ValueError as e:
        print(f"::error::The manifest is not JSON: {e}")
        return 1
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        found = manifest.get("format") if isinstance(manifest, dict) else type(manifest).__name__
        print(f"::error::Unsupported manifest format {found!r}")
        return 1
    try:
        stale = stale_zones(manifest, today, args.max_age_days)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        print(f"::error::Malformed manifest: {type(e).__name__}: {e}")
        return 1
    for message in stale:
        print(f"::error::{message}. Is the Train workflow still enabled and publishing?")
    if not stale:
        count = len(manifest["zones"])
        print(f"All {count} published models are at most {args.max_age_days} days old")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
