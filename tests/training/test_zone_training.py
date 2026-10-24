"""Training a zone of the generic layout (NL, NO2) end to end on a small synthetic raw dir."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from core.features import feature_names
from core.model import load_model
from core.zones import DE_LU, NL, NO2
from training.dataset import build_dataset, load_raw
from training.fetch import month_chunks
from training.train import build_training_data

FIRST, LAST = date(2026, 6, 1), date(2026, 9, 30)


def _hours(a: date, b: date) -> list[datetime]:
    t = datetime(a.year, a.month, a.day, tzinfo=UTC)
    return [t + timedelta(hours=i) for i in range(((b - a).days + 1) * 24)]


def _raw_dir(root: Path, zone=NL) -> Path:
    """Prices of ``zone`` and the weather of its points, June to September 2026, in the
    layout of ``training.fetch``; plus files of a point NL does not use."""
    rng = np.random.default_rng(0)
    hours = _hours(FIRST - timedelta(days=1), LAST + timedelta(days=1))
    daily = 60 + 25 * np.sin(np.arange(len(hours)) / 24 * 2 * np.pi)
    noisy = daily + rng.normal(0, 5, len(hours))
    price = {"unix_seconds": [int(t.timestamp()) for t in hours],
             "price": [round(float(v), 2) for v in noisy]}  # fmt: skip
    (root / "prices" / zone.code).mkdir(parents=True)
    (root / "prices" / zone.code / "2026-06_6m.json").write_text(json.dumps(price))
    (root / "weather").mkdir()
    sets = [*zone.point_sets(), ({"bayern": (48.4, 11.8)}, DE_LU.weather_vars)]
    for points, variables in sets:
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
    return root


def test_nl_raw_loads_only_its_points_and_no_capacity(tmp_path):
    raw = load_raw(_raw_dir(tmp_path), NL)
    assert raw.zone is NL and raw.capacity == {}
    keys = set(raw.weather[2])
    assert "nl_borssele__cloud_cover" in keys and "nb_no_south__precipitation" in keys
    assert not any(k.startswith("bayern__") for k in keys)
    ds = build_dataset(raw, date(2026, 7, 10), date(2026, 7, 11))
    assert ds.X.shape == (2 * 6 * 24, len(feature_names(NL)))
    assert not np.isnan(ds.y_price).any()


@pytest.fixture(scope="module")
def trained_nl(tmp_path_factory):
    from training import train

    raw = _raw_dir(tmp_path_factory.mktemp("raw"))
    out = tmp_path_factory.mktemp("out")
    argv = ["--zone", "NL", "--raw", raw, "--out", out, "--cutoff", "2026-10-01",
            "--backfill-days", "31"]  # fmt: skip
    return train.run(argv, max_iter=5), out


def test_train_nl_end_to_end(trained_nl):
    result, out = trained_nl
    assert result.version == "nl-2026-10"
    m = load_model((out / "nl-2026-10.json.gz").read_bytes())
    assert (m.zone, m.version, m.capacity) == ("nl", "nl-2026-10", {})
    assert m.features == feature_names(NL)
    assert m.peak_threshold > 0 and m.metrics["horizons"]["1"]["mae"] > 0
    bf = json.loads(gzip.decompress((out / "nl-2026-10.backfill.json.gz").read_bytes()))
    assert bf["model_version"] == "nl-2026-10" and bf["rows"]
    assert (out / "nl-2026-10.report.md").read_text().startswith("# Spot Forecast model nl-2026-10")


def test_ci_checks_the_nl_files_like_the_integration(trained_nl):
    from training import ci

    result, _ = trained_nl
    model = ci.check_outputs(
        result.model_path.read_bytes(),
        result.backfill_path.read_bytes(),
        "nl",
        "nl-2026-10",
        date(2026, 10, 3),
    )
    assert model.zone == "nl"


def test_no2_dataset_takes_the_hydro_history_from_the_offset_2_weather(tmp_path):
    """``hydro_history`` = ``weather[1]`` (offset 2, research ``s1``); the 90-day sums skip the
    hours before the data, as research does."""
    raw = load_raw(_raw_dir(tmp_path, NO2), NO2)
    ds = build_training_data(raw, date(2026, 10, 1))
    names = feature_names(NO2)
    assert ds.X.shape[1] == len(names) == 51 and "doy" not in names
    c = {n: i for i, n in enumerate(names)}
    anchor = date(2026, 9, 20)
    row = ds.X[np.flatnonzero(ds.anchor == anchor)[0]]
    end = datetime(2026, 9, 20, 22, tzinfo=UTC)  # end of the anchor day in Europe/Oslo
    for days in (14, 90):
        expected = sum(
            np.nansum(raw.weather[2][f"{p}__precipitation"].window(end - timedelta(days=days),
                                                                    days * 24))
            for p in NO2.hydro_points
        )  # fmt: skip
        assert row[c[f"hyd_precip_past{days}"]] == expected
    assert row[c["hyd_precip_past90"]] > row[c["hyd_precip_past30"]] > 0
    # an early anchor: its 90 days start before June, the sums cover the data there is
    early = ds.X[np.flatnonzero(ds.anchor == date(2026, 7, 5))[0]]
    assert 0 < early[c["hyd_precip_past90"]] < row[c["hyd_precip_past90"]]
