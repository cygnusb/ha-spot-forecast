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
