"""Training a zone with a parent zone (DK1, blended with DE-LU) on a small synthetic raw dir."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from core.features import FEATURE_NAMES, feature_names
from core.model import dump_model, load_model, sample_check
from core.zones import DE_LU, DK1
from training import ci, gate, train
from training.dataset import build_dataset, load_raw
from training.fetch import month_chunks, price_zones

FIRST, LAST = date(2026, 6, 1), date(2026, 9, 30)
CUTOFF = "2026-10-01"


def _hours(a: date, b: date) -> list[datetime]:
    t = datetime(a.year, a.month, a.day, tzinfo=UTC)
    return [t + timedelta(hours=i) for i in range(((b - a).days + 1) * 24)]


def _raw_dir(root: Path) -> Path:
    """DE-LU and DK1 prices, the weather of every DK1 point (the DE-LU ones included) and
    the installed capacity, June to September 2026, in the layout of ``training.fetch``."""
    rng = np.random.default_rng(0)
    hours = _hours(FIRST - timedelta(days=1), LAST + timedelta(days=1))
    daily = 60 + 25 * np.sin(np.arange(len(hours)) / 24 * 2 * np.pi)
    for code, shift in (("DE-LU", 0.0), ("DK1", 8.0)):
        noisy = daily + shift + rng.normal(0, 5, len(hours))
        price = {"unix_seconds": [int(t.timestamp()) for t in hours],
                 "price": [round(float(v), 2) for v in noisy]}  # fmt: skip
        (root / "prices" / code).mkdir(parents=True)
        (root / "prices" / code / "2026-06_6m.json").write_text(json.dumps(price))
    (root / "weather").mkdir()
    for points, variables in DK1.point_sets():
        for name in points:
            for a, b in month_chunks(FIRST, LAST):
                times = _hours(a, b)
                hourly = {"time": [f"{t:%Y-%m-%dT%H:%M}" for t in times]}
                for v in variables:
                    for offset in range(2, 8):
                        hourly[f"{v}_previous_day{offset}"] = list(
                            np.round(rng.uniform(0, 30, len(times)), 1)
                        )
                doc = {"utc_offset_seconds": 0, "hourly": hourly}
                (root / "weather" / f"{name}_{a:%Y-%m}.json").write_text(json.dumps(doc))
    months = [f"{m:02d}.2026" for m in range(1, 11)]
    installed = {
        "time": months,
        "production_types": [
            {"name": "Solar AC", "data": [100.0 + i for i in range(len(months))]},
            {"name": "Wind onshore", "data": [70.0] * len(months)},
            {"name": "Wind offshore", "data": [9.0] * len(months)},
        ],
    }
    (root / "installed_power.json").write_text(json.dumps(installed))
    return root


def test_fetch_gets_the_price_inputs_too():
    assert [z.code for z in price_zones([DK1])] == ["DK1", "DE-LU"]
    assert [z.code for z in price_zones([DE_LU, DK1])] == ["DE-LU", "DK1"]


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    return _raw_dir(tmp_path_factory.mktemp("raw"))


def test_dk1_raw_has_de_lu_prices_weather_and_capacity(raw_dir):
    raw = load_raw(raw_dir, DK1)
    assert set(raw.extra_prices) == {"DE-LU"} and raw.capacity  # the parent's capacity
    keys = set(raw.weather[2])
    assert {"bayern__cloud_cover", "nb_fi__precipitation", "no2_sirdal__precipitation"} <= keys
    ds = build_dataset(raw, date(2026, 7, 10), date(2026, 7, 11))
    assert ds.X.shape == (2 * 6 * 24, len(feature_names(DK1)))
    assert ds.X_parent.shape == (len(ds.X), len(FEATURE_NAMES))
    # the parent rows are DE-LU's own rows of the same (anchor, time)
    de = build_dataset(raw.parent(), date(2026, 7, 10), date(2026, 7, 11))
    assert raw.parent().zone is DE_LU
    np.testing.assert_array_equal(ds.X_parent, de.X)
    np.testing.assert_array_equal(ds.parent_mean7, de.mean7)
    np.testing.assert_array_equal(ds.parent_scale, de.scale)
    assert not np.isnan(ds.X[:, feature_names(DK1).index("de_D_mean")]).any()
    sub = ds.subset(ds.h == 3)
    assert sub.X_parent.shape == (2 * 24, len(FEATURE_NAMES))


def test_rows_the_parent_lacks_are_dropped(raw_dir, capsys):
    raw = load_raw(raw_dir, DK1)
    de = raw.extra_prices["DE-LU"]
    values = de.values.copy()
    hole = int((datetime(2026, 7, 13, 10, tzinfo=UTC) - de.start) / timedelta(hours=1))
    values[hole] = np.nan  # no DE-LU actual price: DE-LU has no row for that hour
    raw = replace(raw, extra_prices={"DE-LU": replace(de, values=values)})
    ds = build_dataset(raw, date(2026, 7, 10), date(2026, 7, 11))
    assert len(ds.X) == 2 * 6 * 24 - 2  # the hour is D+3 of 07-10 and D+2 of 07-11
    assert "2 of 288 rows dropped: DE-LU has no row for them" in capsys.readouterr().out
    assert len(ds.X_parent) == len(ds.X)


@pytest.fixture(scope="module")
def trained(raw_dir, tmp_path_factory):
    de_out = tmp_path_factory.mktemp("de_lu")
    dk_out = tmp_path_factory.mktemp("dk1")
    common = ["--raw", raw_dir, "--cutoff", CUTOFF, "--backfill-days", "31"]
    de = train.run(["--zone", "DE-LU", "--out", de_out, *common], max_iter=5)
    dk = train.run(
        ["--zone", "DK1", "--out", dk_out, *common, "--parent-model", de.model_path,
         "--parent-predictions", de.predictions_path],
        max_iter=5,
    )  # fmt: skip
    return de, dk


def test_every_zone_writes_its_predictions(trained):
    de, _ = trained
    version, rows = train.read_predictions(de.predictions_path)
    assert version == "de_lu-2026-10" and len(rows) == len(de.predictions.times)
    p = de.predictions
    assert rows[(p.times[0], int(p.h[0]))] == (p.median[0], p.p90_raw[0], p.p10_raw[0])


def test_dk1_walk_forward_is_the_blend(trained):
    de, dk = trained
    _, parent = train.read_predictions(de.predictions_path)
    own, wf = dk.own_predictions, dk.predictions
    assert len(wf.times) == len(own.times) > 0
    from_parent = np.array([parent[(t, int(h))] for t, h in zip(wf.times, wf.h, strict=True)])
    for i, name in enumerate(("median", "p90_raw", "p10_raw")):
        expected = 0.5 * getattr(own, name) + 0.5 * from_parent[:, i]
        np.testing.assert_array_equal(getattr(wf, name), expected)
    assert not np.array_equal(wf.median, own.median)
    # offsets, metrics and backfill come from the blend
    m = dk.model
    offsets = train.window_offsets(
        wf.target_date, wf.h, wf.actual - wf.p90_raw, date.fromisoformat(CUTOFF), 0.9
    )
    assert m.p90_offsets == offsets
    assert m.metrics == train.evaluate(
        wf, date(2026, 8, 31), date.fromisoformat(CUTOFF), m.peak_threshold
    )
    bf = json.loads(gzip.decompress(dk.backfill_path.read_bytes()))
    first = bf["rows"][0]
    i = next(
        k for k, (t, h) in enumerate(zip(wf.times, wf.h, strict=True))
        if t.isoformat() == first[0] and int(h) == first[1]
    )  # fmt: skip
    assert first[2] == round(float(wf.median[i]), 2)
    report = dk.report_path.read_text()
    assert "Blend: 0.5 x own model + 0.5 x DE-LU model de_lu-2026-10." in report


def test_dk1_model_file_embeds_the_de_lu_candidate(trained):
    de, dk = trained
    m = load_model(dk.model_path.read_bytes())
    sample_check(m)
    assert (m.zone, m.version, m.blend, m.capacity) == ("dk1", "dk1-2026-10", 0.5, {})
    doc = json.loads(gzip.decompress(dk.model_path.read_bytes()))
    assert doc["parent"] == json.loads(gzip.decompress(de.model_path.read_bytes()))
    ci.check_outputs(
        dk.model_path.read_bytes(), dk.backfill_path.read_bytes(), "dk1", "dk1-2026-10",
        date(2026, 10, 3),
    )  # fmt: skip


def test_gate_scores_a_previous_composite_model_on_the_blend(trained, tmp_path):
    _, dk = trained
    month = date(2026, 9, 1)
    m = dk.model
    got = gate.previous_model_predictions(m, dk.dataset, month)
    rows = dk.dataset.subset(gate._in_month(dk.dataset.target_date, month))
    own = m.median.predict(rows.X) * rows.scale + rows.mean7
    parent = m.parent.median.predict(rows.X_parent) * rows.parent_scale + rows.parent_mean7
    np.testing.assert_allclose(got.median, 0.5 * own + 0.5 * parent, rtol=1e-12)
    # the candidate's walk-forward and the previous model score the same rows
    assert (
        gate.month_metrics(got, month)["rows"] == gate.month_metrics(dk.predictions, month)["rows"]
    )
    previous = tmp_path / "dk1-2026-09.json.gz"  # as if trained a month earlier
    previous.write_bytes(dump_model(replace(m, version="dk1-2026-09", train_cutoff=month)))
    rows = gate.month_metrics(dk.predictions, month)["rows"]
    v, metrics, why = ci._score_previous(previous, "dk1", dk.dataset, month, rows)
    assert (v, why) == ("dk1-2026-09", "") and metrics["rows"] == rows
    without = replace(dk.dataset, X_parent=None, parent_mean7=None, parent_scale=None)
    assert "the dataset has no rows of it" in ci._score_previous(previous, "dk1", without, month,
                                                                 rows)[2]  # fmt: skip


@pytest.mark.parametrize("missing", ["--parent-model", "--parent-predictions", "both"])
def test_dk1_needs_both_parent_files(raw_dir, tmp_path, trained, missing):
    de, _ = trained
    argv = ["--zone", "DK1", "--raw", raw_dir, "--out", tmp_path, "--cutoff", CUTOFF]
    if missing != "--parent-model" and missing != "both":
        argv += ["--parent-model", de.model_path]
    if missing != "--parent-predictions" and missing != "both":
        argv += ["--parent-predictions", de.predictions_path]
    with pytest.raises(ValueError, match="--parent-model and --parent-predictions are required"):
        train.run(argv, max_iter=5)


def test_parent_must_match(trained, tmp_path, tiny_nl_model_file):
    de, _ = trained
    other = tmp_path / "nl.json.gz"
    other.write_bytes(dump_model(tiny_nl_model_file))
    cutoff = date.fromisoformat(CUTOFF)
    with pytest.raises(ValueError, match="not a model of DE-LU"):
        train.load_parent(DK1, other, de.predictions_path, cutoff)
    with pytest.raises(ValueError, match="cutoff 2026-10-01, not 2026-11-01"):
        train.load_parent(DK1, de.model_path, de.predictions_path, date(2026, 11, 1))
    wrong = tmp_path / "p.json.gz"
    train.write_predictions(wrong, de.predictions, "de_lu-2026-09")
    with pytest.raises(ValueError, match="are of de_lu-2026-09, not of de_lu-2026-10"):
        train.load_parent(DK1, de.model_path, wrong, cutoff)
    with pytest.raises(ValueError, match="has no parent zone"):
        train.load_parent(DE_LU, de.model_path, None, cutoff)
