from dataclasses import replace
import gzip
import json
import threading
import time

import numpy as np
import pytest

from core.features import FEATURE_NAMES
from core.model import ModelCorrupt, ModelMismatch, Trees, dump_model, load_model


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


def _mutated(m, change) -> bytes:
    """``m`` dumped, decoded, changed in place by ``change(doc)`` and re-encoded."""
    doc = json.loads(gzip.decompress(dump_model(m)))
    change(doc)
    return gzip.compress(json.dumps(doc).encode())


def _first_internal(trees: dict) -> int:
    return trees["is_leaf"].index(0)


def _load_within(data: bytes, seconds: float = 5.0) -> BaseException | None:
    """Run ``load_model`` plus a prediction in a daemon thread so a hang fails the test."""
    result: list[BaseException | None] = []

    def target() -> None:
        try:
            m = load_model(data)
            for trees in (m.median, m.p90, m.p10):
                trees.predict(_X(3))
            result.append(None)
        except BaseException as e:  # noqa: BLE001 - handed to the test thread
            result.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert result, "load_model/predict did not finish (hang)"
    return result[0]


def test_corrupt_model_is_model_mismatch():
    assert issubclass(ModelCorrupt, ModelMismatch)


def test_rejects_truncated_gzip(tiny_model_file):
    data = dump_model(tiny_model_file)
    with pytest.raises(ModelCorrupt):
        load_model(data[: len(data) // 2])


def test_rejects_not_gzip():
    with pytest.raises(ModelCorrupt):
        load_model(b"not a model")


def test_rejects_invalid_json():
    with pytest.raises(ModelCorrupt):
        load_model(gzip.compress(b'{"format": 1, '))


def test_rejects_json_that_is_not_an_object():
    with pytest.raises(ModelCorrupt):
        load_model(gzip.compress(b"[1, 2, 3]"))


def test_rejects_missing_field(tiny_model_file):
    with pytest.raises(ModelCorrupt):
        load_model(_mutated(tiny_model_file, lambda d: d.pop("p90_offsets")))


def test_rejects_mismatched_array_lengths(tiny_model_file):
    with pytest.raises(ModelCorrupt, match="length"):
        load_model(_mutated(tiny_model_file, lambda d: d["median"]["value"].pop()))


def test_rejects_root_out_of_bounds(tiny_model_file):
    def change(d):
        d["p10"]["roots"][-1] = len(d["p10"]["feature"])

    with pytest.raises(ModelCorrupt, match="root"):
        load_model(_mutated(tiny_model_file, change))


@pytest.mark.parametrize("child", [-1, 10_000])
def test_rejects_child_out_of_bounds(tiny_model_file, child):
    def change(d):
        d["median"]["right"][_first_internal(d["median"])] = child

    with pytest.raises(ModelCorrupt, match="child"):
        load_model(_mutated(tiny_model_file, change))


def test_rejects_cyclic_child_index_without_hanging(tiny_model_file):
    def change(d):
        t = d["p90"]
        i = _first_internal(t)
        t["left"][i] = t["right"][i] = i  # points back at itself: predict would spin forever

    err = _load_within(_mutated(tiny_model_file, change))
    assert isinstance(err, ModelCorrupt) and "child" in str(err)


def test_rejects_feature_out_of_range(tiny_model_file):
    def change(d):
        d["median"]["feature"][_first_internal(d["median"])] = len(FEATURE_NAMES)

    with pytest.raises(ModelCorrupt, match="feature"):
        load_model(_mutated(tiny_model_file, change))


def test_rejects_non_finite_leaf_value(tiny_model_file):
    def change(d):
        d["p10"]["value"][d["p10"]["is_leaf"].index(1)] = 1e400  # parses as inf

    with pytest.raises(ModelCorrupt, match="leaf"):
        load_model(_mutated(tiny_model_file, change))


def test_rejects_missing_horizon_offset(tiny_model_file):
    with pytest.raises(ModelCorrupt, match="offset"):
        load_model(_mutated(tiny_model_file, lambda d: d["p90_offsets"].pop("6")))


@pytest.mark.parametrize(
    ("trees", "field", "value"),
    [("median", "threshold", 10**400), ("p90", "roots", 2**70)],
    ids=["float", "int"],
)
def test_load_model_overflow_is_corrupt(tiny_model_file, trees, field, value):
    # JSON integers have no size limit; numpy cannot hold them and raises OverflowError
    def change(d):
        d[trees][field][0] = value

    with pytest.raises(ModelCorrupt, match="OverflowError"):
        load_model(_mutated(tiny_model_file, change))
