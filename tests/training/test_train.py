"""Training CLI end to end on a small range (needs the research raw data)."""

from __future__ import annotations

from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.model import load_model

RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)
RESEARCH_RAW = RESEARCH / "data" / "raw"
BERLIN = ZoneInfo("Europe/Berlin")

pytestmark = pytest.mark.skipif(not RESEARCH_RAW.exists(), reason="research raw data missing")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from training.train import run

    out = tmp_path_factory.mktemp("train")
    run(
        [
            "--raw",
            str(RESEARCH_RAW),
            "--out",
            str(out),
            "--cutoff",
            "2025-10-01",
            "--backfill-days",
            "31",
        ],
        max_iter=30,
    )
    return out


def _local_date(iso: str) -> date:
    return datetime.fromisoformat(iso).astimezone(BERLIN).date()


def test_train_end_to_end_small(trained):
    m = load_model((trained / "de_lu-2025-10.json.gz").read_bytes())
    assert m.version == "de_lu-2025-10"
    assert m.train_cutoff == date(2025, 10, 1)
    assert set(m.p90_offsets) == set(m.p10_offsets) == {1, 2, 3, 4, 5, 6}
    # 31 backfill days give > 200 residuals per horizon, so no offset is the 0.0 fallback
    assert all(v != 0.0 for v in [*m.p90_offsets.values(), *m.p10_offsets.values()])
    assert m.capacity["cap_solar"] > 0 and m.peak_threshold > 0
    assert m.metrics["horizons"]["1"]["mae"] > 0

    bf = json.loads(gzip.decompress((trained / "de_lu-2025-10.backfill.json.gz").read_bytes()))
    assert bf["format"] == 1 and bf["model_version"] == "de_lu-2025-10"
    days = {_local_date(r[0]) for r in bf["rows"]}
    assert min(days) == date(2025, 8, 31) and max(days) == date(2025, 9, 30)
    assert {r[1] for r in bf["rows"]} == {1, 2, 3, 4, 5, 6}
    # every target day in the window is covered for every horizon
    assert len({(_local_date(r[0]), r[1]) for r in bf["rows"]}) == 31 * 6

    metrics = json.loads((trained / "de_lu-2025-10.metrics.json").read_text())
    assert metrics == m.metrics
    md = (trained / "de_lu-2025-10.report.md").read_text()
    assert "MAE" in md and "D+6" in md


def test_backfill_rows_are_out_of_sample():
    from training.dataset import load_raw
    from training.train import build_training_data, walk_forward

    ds = build_training_data(load_raw(RESEARCH_RAW), date(2025, 10, 1))
    wf = walk_forward(ds, date(2025, 8, 25), date(2025, 10, 1), max_iter=5)
    assert len(wf.anchor) > 0
    rows = zip(wf.train_cutoff, wf.anchor, wf.target_date, strict=True)
    for train_cutoff, anchor, target in rows:
        # the month's model was fitted only on targets before its month start, which is
        # on or before the issue day, and the prediction is for a later day
        assert train_cutoff <= anchor < target < date(2025, 10, 1)
        assert train_cutoff == anchor.replace(day=1)
