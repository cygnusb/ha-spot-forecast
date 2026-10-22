from dataclasses import replace
import time

import numpy as np
import pytest

from core.features import FEATURE_NAMES
from core.model import ModelMismatch, Trees, dump_model, load_model


def _X(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


def test_roundtrip(tiny_model_file):
    m = tiny_model_file
    back = load_model(dump_model(m))
    assert back.p90_offsets == m.p90_offsets and back.p10_offsets == m.p10_offsets
    assert all(isinstance(h, int) for h in back.p10_offsets)
    assert (back.version, back.train_cutoff, back.created) == (m.version, m.train_cutoff, m.created)
    assert back.features == m.features and back.capacity == m.capacity and back.metrics == m.metrics
    X = _X(50)
    for name in ("median", "p90", "p10"):
        np.testing.assert_array_equal(getattr(back, name).predict(X), getattr(m, name).predict(X))


def test_rejects_wrong_schema(tiny_model_file):
    with pytest.raises(ModelMismatch):
        load_model(dump_model(replace(tiny_model_file, feature_schema=2)))


def test_rejects_wrong_format(tiny_model_file):
    with pytest.raises(ModelMismatch):
        load_model(dump_model(replace(tiny_model_file, format=2)))


def test_rejects_reordered_features(tiny_model_file):
    feats = list(tiny_model_file.features)
    feats[0], feats[1] = feats[1], feats[0]
    with pytest.raises(ModelMismatch):
        load_model(dump_model(replace(tiny_model_file, features=tuple(feats))))


def test_split_semantics_and_nan_routing():
    # one stump: x0 <= 1.0 -> left (value 10) else right (20); NaN goes left or right
    def stump(missing_left: bool) -> Trees:
        return Trees(
            baseline=1.0,
            feature=np.array([0, 0, 0]),
            threshold=np.array([1.0, 0.0, 0.0]),
            left=np.array([1, 0, 0]),
            right=np.array([2, 0, 0]),
            missing_left=np.array([missing_left, False, False]),
            is_leaf=np.array([False, True, True]),
            value=np.array([0.0, 10.0, 20.0]),
            roots=np.array([0]),
        )

    X = np.array([[1.0], [1.0000001], [np.nan]])
    np.testing.assert_array_equal(stump(True).predict(X), [11.0, 21.0, 11.0])
    np.testing.assert_array_equal(stump(False).predict(X), [11.0, 21.0, 21.0])


def test_predict_is_chunk_independent(real_sized_trees):
    X = _X(5000, seed=3)
    whole = real_sized_trees.predict(X)
    np.testing.assert_array_equal(whole[:7], real_sized_trees.predict(X[:7]))


def test_predict_144_rows_under_1s(real_sized_trees):
    X = _X(144)
    start = time.perf_counter()
    out = real_sized_trees.predict(X)
    assert time.perf_counter() - start < 1.0
    assert out.shape == (144,) and np.isfinite(out).all()
