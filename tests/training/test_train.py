"""Training CLI end to end on a small range (needs the research raw data)."""

from __future__ import annotations

from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from core.model import load_model

RESEARCH = Path(
    os.environ.get("SPOT_RESEARCH_DIR", Path(__file__).resolve().parents[2] / "research")
)
RESEARCH_RAW = RESEARCH / "data" / "raw"
BERLIN = ZoneInfo("Europe/Berlin")

needs_raw = pytest.mark.skipif(not RESEARCH_RAW.exists(), reason="research raw data missing")


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


@needs_raw
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


@needs_raw
def test_backfill_rows_are_out_of_sample():
    from training.dataset import load_raw
    from training.train import build_training_data, walk_forward

    ds = build_training_data(load_raw(RESEARCH_RAW), date(2025, 10, 1))
    wf = walk_forward(ds, date(2025, 8, 25), date(2025, 10, 1), max_iter=5)
    assert len(wf.anchor) > 0
    rows = zip(wf.train_last_target, wf.anchor, wf.target_date, strict=True)
    for last_fitted, anchor, target in rows:
        # the predicting model saw no target day from its issue day on
        assert last_fitted < anchor < target < date(2025, 10, 1)
    # monthly refit: the September model saw August, the August model only July
    assert max(wf.train_last_target) == date(2025, 8, 31)
    assert min(wf.train_last_target) == date(2025, 7, 31)


def test_capacity_is_the_last_training_anchor_month():
    from training.dataset import RawData
    from training.train import capacity_at

    caps = {m: {"cap_solar": float(i)} for i, m in enumerate(["2026-08", "2026-09", "2026-10"])}
    raw = RawData(prices=None, weather={}, capacity=caps)
    assert capacity_at(raw, date(2026, 10, 1)) == {"cap_solar": 1.0}
    assert capacity_at(raw, date(2026, 10, 15)) == {"cap_solar": 2.0}
    with pytest.raises(ValueError, match="2026-11"):
        capacity_at(raw, date(2026, 11, 2))
