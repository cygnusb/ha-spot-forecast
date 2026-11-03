#!/usr/bin/env bash
# Monthly raw-data refresh on the data host, then the training.
#
# GitHub-hosted runners time out at Open-Meteo: on 2026-11-03 two Train runs failed in the
# fetch while the same requests answered in under a second from elsewhere. So the Train
# workflow fetches nothing; this script does, from cron on a host of ours
# (proxy.valerius.email), and hands the result over through the training-data release:
#
# 1. fetch the new raw data into a directory that persists between runs, so a fetch that
#    breaks off resumes next time (the first run starts from the published archive);
# 2. archive the openly licensed inputs (the same whitelist as before in train.yml) and
#    upload them to the training-data release;
# 3. start the Train workflow, which trains on that archive (unless --no-train).
#
# A failure opens (or keeps) one GitHub issue and uploads nothing.
#
# Setup on the host, as a user without sudo:
#   curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
#   ~/.local/bin/uv python install 3.12            # what training/requirements.lock is for
#   git clone https://github.com/cygnusb/ha-spot-forecast.git ~/spot-forecast/repo
#   ~/.local/bin/uv venv -p 3.12 ~/spot-forecast/venv
#   gh in ~/.local/bin (the release tarball of cli/cli, checksum verified)
#   ~/.config/spot-forecast/token, mode 600: a fine-grained token for this repository only,
#     with Contents, Actions and Issues: read and write
# Crontab (the host runs on Europe/Berlin; the 3rd: the month before is complete by then):
#   17 3 3 * * cd ~/spot-forecast/repo && git pull -q --ff-only && training/refresh_data.sh >> ~/spot-forecast/refresh.log 2>&1
# The pull runs before this script starts: bash reads a script while running it.
#
# Usage: training/refresh_data.sh [--no-train]

set -euo pipefail

BASE="${SPOT_FORECAST_HOME:-$HOME/spot-forecast}"
VENV="$BASE/venv"
DATA="$BASE/data"
TOKEN_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/spot-forecast/token"
export GH_REPO=cygnusb/ha-spot-forecast
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

train=true
case "${1:-}" in
  "") ;;
  --no-train) train=false ;;
  *) echo "usage: $0 [--no-train]" >&2; exit 2 ;;
esac

mkdir -p "$BASE"
exec 9>"$BASE/refresh.lock"
if ! flock -n 9; then
  echo "$(date -Is) another refresh is running" >&2
  exit 0
fi

if [ "$(stat -c %a "$TOKEN_FILE")" != 600 ]; then
  echo "$TOKEN_FILE must exist with mode 600" >&2
  exit 1
fi
GH_TOKEN="$(< "$TOKEN_FILE")"
export GH_TOKEN

log="$BASE/fetch.log"
step=""

fail() {
  local title="Training data refresh failed"
  local body="$BASE/issue.md"
  {
    echo "The monthly raw-data refresh on $(hostname -f) failed in: $step."
    echo "Nothing was uploaded and the Train workflow was not started. Rerun"
    echo "\`training/refresh_data.sh\` on the host once the cause is gone: the fetch resumes."
    echo
    echo '```'
    tail -n 60 "$log" 2>/dev/null || true
    echo '```'
  } > "$body"
  local open
  # gh's own --jq: the host has no jq
  open=$(gh issue list --state open --search "in:title \"$title\"" --json number,title \
    --jq "[.[] | select(.title == \"$title\") | .number] | first // empty") || true
  if [ -n "$open" ]; then
    echo "issue #$open is still open: not opening another" >&2
  else
    gh issue create --title "$title" --body-file "$body" >&2 || true
  fi
}
trap 'echo "$(date -Is) failed: $step" >&2; fail' ERR

echo "$(date -Is) refresh at $(git -C "$REPO" rev-parse --short HEAD)"

step="installing the hash-locked training requirements"
VIRTUAL_ENV="$VENV" uv pip install -q --require-hashes -r "$REPO/training/requirements.lock" > "$log" 2>&1

if [ ! -d "$DATA/raw" ]; then
  step="restoring the published archive (first run)"
  mkdir -p "$DATA"
  gh release download training-data -p raw.tar.gz -D "$BASE" --clobber > "$log" 2>&1
  tar -xzf "$BASE/raw.tar.gz" -C "$DATA"
fi

step="fetching the new raw data"
(cd "$REPO" && PYTHONUNBUFFERED=1 "$VENV/bin/python" -m training.fetch --raw "$DATA/raw") \
  > "$log" 2>&1
tail -n 3 "$log"

step="archiving the raw data"
# A whitelist: the public archive holds only the openly licensed (CC BY) inputs from
# Energy-Charts and Open-Meteo, never research-only files that may not be shared.
tar -czf "$BASE/raw.tar.gz" -C "$DATA" raw/prices/ raw/weather/ raw/installed_power.json

step="uploading the archive to the training-data release"
if ! gh release view training-data > /dev/null 2>&1; then
  gh release create training-data --title "Training data" --latest=false --prerelease \
    --notes "Raw inputs of the monthly model training, refreshed every month: day-ahead prices and installed capacity from Energy-Charts (price data: Bundesnetzagentur | SMARD.de, CC BY 4.0) and weather forecasts from Open-Meteo (CC BY 4.0)."
fi
# Also a release created by hand: HACS must never offer it.
gh release edit training-data --prerelease --latest=false > /dev/null
gh release upload training-data "$BASE/raw.tar.gz" --clobber
echo "$(date -Is) uploaded raw.tar.gz ($(du -h "$BASE/raw.tar.gz" | cut -f1))"

if [ "$train" = true ]; then
  step="starting the Train workflow"
  gh workflow run train.yml --ref main
  echo "$(date -Is) started the Train workflow"
fi
