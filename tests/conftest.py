"""Fixtures shared across tests/core and tests/training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN_FILE = Path(__file__).parent / "fixtures" / "golden_features.json.gz"


@dataclass(frozen=True, eq=False)
class GoldenCase:
    """Inputs and research-produced features for one anchor (see training/make_golden.py)."""

    anchor: date
    prices: object  # core.features.Hourly
    weather: dict[int, dict[str, object]]  # h -> "{point}__{var}" -> Hourly
    capacity: dict[str, float]
    times: list[str]  # UTC ISO timestamps of the rows research kept
    X: np.ndarray
    scale: np.ndarray
    mean7: np.ndarray


def _array(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.fixture(scope="session")
def golden_feature_names() -> tuple[str, ...]:
    return tuple(json.loads(gzip.decompress(GOLDEN_FILE.read_bytes()))["feature_names"])


@pytest.fixture(scope="session")
def golden() -> dict[date, GoldenCase]:
    """Golden cases keyed by anchor date."""
    from core.features import Hourly

    def hourly(d: dict) -> Hourly:
        return Hourly(datetime.fromisoformat(d["start"]), _array(d["values"]))

    payload = json.loads(gzip.decompress(GOLDEN_FILE.read_bytes()))
    cases = {}
    for c in payload["cases"]:
        anchor = date.fromisoformat(c["anchor"])
        cases[anchor] = GoldenCase(
            anchor=anchor,
            prices=hourly(c["prices"]),
            weather={
                int(h): {k: hourly(s) for k, s in series.items()}
                for h, series in c["weather"].items()
            },
            capacity={k: np.nan if v is None else v for k, v in c["capacity"].items()},
            times=c["times"],
            X=np.array([_array(r) for r in c["X"]]),
            scale=_array(c["scale"]),
            mean7=_array(c["mean7"]),
        )
    return cases


def make_random_trees(n_trees: int, n_leaves: int, n_features: int, seed: int):
    """Synthetic trees: each grows by splitting a random leaf until it has ``n_leaves`` leaves."""
    from core.model import Trees

    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {
        k: [] for k in ("feature", "threshold", "left", "right", "missing_left", "is_leaf", "value")
    }
    roots = []
    for _ in range(n_trees):
        base = len(cols["feature"])
        roots.append(base)
        feature, threshold, left, right = [0], [0.0], [0], [0]
        missing_left, is_leaf, value = [False], [True], [float(rng.normal(0, 0.05))]
        leaves = [0]
        while len(leaves) < n_leaves:
            n = leaves.pop(int(rng.integers(len(leaves))))
            for _child in range(2):
                feature.append(0)
                threshold.append(0.0)
                left.append(0)
                right.append(0)
                missing_left.append(False)
                is_leaf.append(True)
                value.append(float(rng.normal(0, 0.05)))
            lo, hi = len(feature) - 2, len(feature) - 1
            feature[n] = int(rng.integers(n_features))
            threshold[n] = float(rng.normal())
            left[n], right[n] = base + lo, base + hi
            missing_left[n] = bool(rng.integers(2))
            is_leaf[n] = False
            leaves += [lo, hi]
        for k, v in zip(
            cols, (feature, threshold, left, right, missing_left, is_leaf, value), strict=True
        ):
            cols[k].extend(v)
    return Trees(
        baseline=float(rng.normal()),
        roots=np.array(roots, dtype=np.int64),
        feature=np.array(cols["feature"], dtype=np.int64),
        threshold=np.array(cols["threshold"]),
        left=np.array(cols["left"], dtype=np.int64),
        right=np.array(cols["right"], dtype=np.int64),
        missing_left=np.array(cols["missing_left"], dtype=bool),
        is_leaf=np.array(cols["is_leaf"], dtype=bool),
        value=np.array(cols["value"]),
    )


@pytest.fixture(scope="session")
def real_sized_trees():
    """600 trees x 31 leaves over the 82 model features (synthetic)."""
    from core.features import FEATURE_NAMES

    return make_random_trees(600, 31, len(FEATURE_NAMES), seed=1)


@pytest.fixture(scope="session")
def tiny_model_file():
    """A small valid ModelFile with hand-made trees (no scikit-learn needed)."""
    from datetime import UTC

    from core.features import FEATURE_NAMES, FEATURE_SCHEMA
    from core.model import ModelFile

    n = len(FEATURE_NAMES)
    return ModelFile(
        format=1,
        zone="de_lu",
        version="de_lu-2026-11",
        feature_schema=FEATURE_SCHEMA,
        features=FEATURE_NAMES,
        train_cutoff=date(2026, 11, 1),
        created=datetime(2026, 11, 2, 3, 4, 5, tzinfo=UTC),
        median=make_random_trees(5, 4, n, seed=10),
        p90=make_random_trees(5, 4, n, seed=11),
        p10=make_random_trees(5, 4, n, seed=12),
        p90_offsets={h: 1.5 * h for h in range(1, 7)},
        p10_offsets={h: -0.5 * h for h in range(1, 7)},
        peak_threshold=185.25,
        capacity={"cap_solar": 100.5, "cap_wind_on": 70.0, "cap_wind_off": 9.0},
        metrics={"mae_ct": 2.14, "note": "tiny"},
    )


@pytest.fixture(scope="session")
def tiny_nl_model_file(tiny_model_file):
    """``tiny_model_file`` for zone NL: its feature layout, trees over it, no capacity."""
    from dataclasses import replace

    from core.features import feature_names
    from core.zones import NL

    n = len(feature_names(NL))
    return replace(
        tiny_model_file,
        zone="nl",
        version="nl-2026-11",
        features=feature_names(NL),
        median=make_random_trees(5, 4, n, seed=20),
        p90=make_random_trees(5, 4, n, seed=21),
        p10=make_random_trees(5, 4, n, seed=22),
        capacity={},
    )


@pytest.fixture(scope="session")
def tiny_dk1_model_file(tiny_model_file):
    """A composite DK1 model: trees over the DK1 layout, ``tiny_model_file`` as its DE-LU
    parent, blended 50/50."""
    from dataclasses import replace

    from core.features import feature_names
    from core.zones import DK1

    n = len(feature_names(DK1))
    return replace(
        tiny_model_file,
        zone="dk1",
        version="dk1-2026-11",
        features=feature_names(DK1),
        median=make_random_trees(5, 4, n, seed=30),
        p90=make_random_trees(5, 4, n, seed=31),
        p10=make_random_trees(5, 4, n, seed=32),
        capacity={},
        parent=tiny_model_file,
        blend=0.5,
    )
