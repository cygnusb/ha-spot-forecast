from dataclasses import replace
import gzip
import json
import threading
import time

import numpy as np
import pytest

from core import ledger, model
from core.features import FEATURE_NAMES, FEATURE_SCHEMA
from core.model import (
    MAX_DEPTH,
    ModelCorrupt,
    ModelMismatch,
    Trees,
    dump_model,
    gunzip_limited,
    load_model,
    sample_check,
)


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
        load_model(dump_model(replace(tiny_model_file, feature_schema=FEATURE_SCHEMA + 1)))


def test_rejects_wrong_format(tiny_model_file):
    with pytest.raises(ModelMismatch):
        load_model(dump_model(replace(tiny_model_file, format=2)))


def test_rejects_reordered_features(tiny_model_file):
    feats = list(tiny_model_file.features)
    feats[0], feats[1] = feats[1], feats[0]
    with pytest.raises(ModelMismatch):
        load_model(dump_model(replace(tiny_model_file, features=tuple(feats))))


def test_nl_model_roundtrip(tiny_nl_model_file):
    from core.features import feature_names
    from core.zones import NL

    m = load_model(dump_model(tiny_nl_model_file))
    assert m.zone == "nl" and m.features == feature_names(NL) and m.capacity == {}
    sample_check(m)


@pytest.mark.parametrize(
    ("zone", "features"),
    [("nl", "de_lu"), ("de_lu", "nl"), ("at", "de_lu")],
    ids=["nl-with-de_lu-features", "de_lu-with-nl-features", "unknown-zone"],
)
def test_rejects_features_of_another_zone(tiny_model_file, tiny_nl_model_file, zone, features):
    """A model file is checked against its own zone's layout, never mis-applied."""
    source = tiny_nl_model_file if features == "nl" else tiny_model_file
    with pytest.raises(ModelMismatch, match="zone") as err:
        load_model(dump_model(replace(source, zone=zone)))
    assert not isinstance(err.value, ModelCorrupt)


def test_rejects_nl_tree_feature_past_its_layout(tiny_nl_model_file):
    """The feature-index bound is the zone's feature count (NL has fewer than DE-LU)."""
    from core.features import feature_names
    from core.zones import NL

    def change(d):
        d["median"]["feature"][_first_internal(d["median"])] = len(feature_names(NL))

    with pytest.raises(ModelCorrupt, match="feature index out of range"):
        load_model(_mutated(tiny_nl_model_file, change))


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


def test_rejects_leaf_feature_out_of_range(tiny_model_file):
    """``predict`` indexes the feature of every current node, leaves included."""

    def change(d):
        d["p90"]["feature"][d["p90"]["is_leaf"].index(1)] = len(FEATURE_NAMES) + 5

    err = _load_within(_mutated(tiny_model_file, change))
    assert isinstance(err, ModelCorrupt) and "feature" in str(err)


def test_rejects_too_many_nodes(tiny_model_file, monkeypatch):
    n = len(tiny_model_file.median.feature)
    monkeypatch.setattr(model, "MAX_NODES", n - 1)
    with pytest.raises(ModelCorrupt, match="nodes"):
        load_model(dump_model(tiny_model_file))
    monkeypatch.setattr(model, "MAX_NODES", n)
    load_model(dump_model(tiny_model_file))


def _chain(depth: int) -> dict:
    """One tree whose left branch is ``depth`` splits deep (its deepest leaf at ``depth``)."""
    n = 2 * depth + 1  # inner node at 2k, its right leaf at 2k + 1, the last left leaf at n - 1
    inner = [i % 2 == 0 and i < n - 1 for i in range(n)]
    return {
        "baseline": 0.0,
        "feature": [0] * n,
        "threshold": [0.0] * n,
        "left": [i + 2 if inner[i] else 0 for i in range(n)],
        "right": [i + 1 if inner[i] else 0 for i in range(n)],
        "missing_left": [1] * n,
        "is_leaf": [0 if x else 1 for x in inner],
        "value": [0.0] * n,
        "roots": [0],
    }


def test_depth_limit(tiny_model_file):
    def deep(depth):
        return _mutated(tiny_model_file, lambda d: d.update(p10=_chain(depth)))

    m = load_model(deep(MAX_DEPTH))
    assert m.p10.predict(np.zeros((1, len(FEATURE_NAMES)))).tolist() == [0.0]
    err = _load_within(deep(MAX_DEPTH + 1))
    assert isinstance(err, ModelCorrupt) and "deeper" in str(err)


def test_rejects_deeply_nested_json():
    with pytest.raises(ModelCorrupt, match="RecursionError"):
        load_model(gzip.compress(b"[" * 100_000 + b"]" * 100_000))


def test_rejects_a_model_above_the_decompressed_limit(tiny_model_file, monkeypatch):
    data = dump_model(tiny_model_file)
    monkeypatch.setattr(model, "MAX_MODEL_JSON_BYTES", len(gzip.decompress(data)) - 1)
    with pytest.raises(ModelCorrupt, match="limit"):
        load_model(data)


def test_gunzip_limited():
    payload = b"x" * 1000
    data = gzip.compress(payload)
    assert gunzip_limited(data, 1000) == payload
    with pytest.raises(ModelCorrupt, match="limit"):
        gunzip_limited(data, 999)
    with pytest.raises(ModelCorrupt, match="limit"):
        gunzip_limited(gzip.compress(b"\0" * 10**8), 10**6)  # a bomb stops at the limit
    with pytest.raises(ModelCorrupt, match="truncated"):
        gunzip_limited(data[:-3], 1000)
    with pytest.raises(ModelCorrupt, match="trailing"):
        gunzip_limited(data + b"junk", 1000)
    with pytest.raises(ModelCorrupt):
        gunzip_limited(b"not gzip", 1000)
    flipped = bytearray(gzip.compress(json.dumps(list(range(500))).encode()))
    flipped[20] ^= 0xFF
    with pytest.raises(ModelCorrupt):
        gunzip_limited(bytes(flipped), 10**6)


def test_backfill_parser_has_a_decompressed_limit(monkeypatch):
    data = gzip.compress(json.dumps({"format": 1, "rows": []}).encode())
    assert ledger.runs_from_backfill_file(data, "v", "UTC") == []
    monkeypatch.setattr(ledger, "MAX_BACKFILL_JSON_BYTES", 10)
    with pytest.raises(ModelCorrupt, match="limit"):
        ledger.runs_from_backfill_file(data, "v", "UTC")


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


def test_sample_check_passes_a_sound_model(tiny_model_file):
    sample_check(tiny_model_file)
    sample_check(load_model(dump_model(tiny_model_file)))


def test_sample_check_rejects_nan_model(tiny_model_file):
    nan_leaves = replace(tiny_model_file.p90, value=np.full_like(tiny_model_file.p90.value, np.nan))
    with pytest.raises(ModelCorrupt, match="p90"):
        sample_check(replace(tiny_model_file, p90=nan_leaves))


def test_sample_check_rejects_huge_predictions(tiny_model_file):
    huge = replace(tiny_model_file.p10, baseline=1e3)
    with pytest.raises(ModelCorrupt, match="p10"):
        sample_check(replace(tiny_model_file, p10=huge))


# ---------------------------------------------------------------- composite models (DK1, DK2)


def _doc(m) -> dict:
    return json.loads(gzip.decompress(dump_model(m)))


def _encoded(doc: dict) -> bytes:
    return gzip.compress(json.dumps(doc).encode())


def test_composite_roundtrip(tiny_dk1_model_file, tiny_model_file):
    m = load_model(dump_model(tiny_dk1_model_file))
    assert (m.zone, m.blend, m.capacity) == ("dk1", 0.5, {})
    assert m.parent is not None and m.parent.parent is None
    assert (m.parent.zone, m.parent.version, m.parent.capacity) == (
        "de_lu", tiny_model_file.version, tiny_model_file.capacity,
    )  # fmt: skip
    assert m.parent.p90_offsets == tiny_model_file.p90_offsets
    X = _X(20)
    np.testing.assert_array_equal(m.parent.median.predict(X), tiny_model_file.median.predict(X))
    sample_check(m)
    # the parent is the complete DE-LU document, which loads on its own as well
    doc = _doc(tiny_dk1_model_file)
    assert doc["parent"] == _doc(tiny_model_file) and doc["blend"] == 0.5
    assert load_model(_encoded(doc["parent"])).version == tiny_model_file.version


def test_models_without_parent_have_no_parent_fields(tiny_model_file, tiny_nl_model_file):
    for m in (tiny_model_file, tiny_nl_model_file):
        doc = _doc(m)
        assert "parent" not in doc and "blend" not in doc
        back = load_model(dump_model(m))
        assert back.parent is None and back.blend == 1.0


@pytest.mark.parametrize("slug", ["de_lu", "nl"])
def test_bundled_models_dump_to_the_same_bytes(slug):
    """Loading and dumping a model file of a zone without parent gives its bytes back."""
    from pathlib import Path

    path = Path(model.__file__).parent.parent / "models" / f"{slug}.json.gz"
    data = path.read_bytes()
    assert dump_model(load_model(data)) == data


def _mismatch(data: bytes, match: str) -> None:
    with pytest.raises(ModelMismatch, match=match) as err:
        load_model(data)
    assert not isinstance(err.value, ModelCorrupt)


def test_composite_without_its_parent_is_a_mismatch(tiny_dk1_model_file):
    doc = _doc(tiny_dk1_model_file)
    del doc["parent"]
    _mismatch(_encoded(doc), "lacks its DE-LU model")
    _mismatch(dump_model(replace(tiny_dk1_model_file, parent=None)), "lacks its DE-LU model")


def test_composite_with_a_parent_of_another_zone_is_a_mismatch(
    tiny_dk1_model_file, tiny_nl_model_file
):
    _mismatch(dump_model(replace(tiny_dk1_model_file, parent=tiny_nl_model_file)), "'nl'")


def test_composite_with_a_parent_of_another_schema_is_a_mismatch(
    tiny_dk1_model_file, tiny_model_file
):
    parent = replace(tiny_model_file, feature_schema=FEATURE_SCHEMA + 1)
    _mismatch(
        dump_model(replace(tiny_dk1_model_file, parent=parent)), "parent model: feature schema"
    )


def test_composite_with_a_parent_of_other_features_is_a_mismatch(tiny_dk1_model_file):
    doc = _doc(tiny_dk1_model_file)
    doc["parent"]["features"] = doc["parent"]["features"][::-1]
    _mismatch(_encoded(doc), "parent model: model feature list")


def test_a_parent_on_a_zone_without_one_is_a_mismatch(tiny_model_file, tiny_nl_model_file):
    _mismatch(dump_model(replace(tiny_nl_model_file, parent=tiny_model_file)), "no parent zone")
    _mismatch(dump_model(replace(tiny_model_file, parent=tiny_model_file)), "no parent zone")


def test_composite_parent_trees_are_checked(tiny_dk1_model_file):
    def change(d):
        p = d["parent"]["p90"]
        p["left"][_first_internal(p)] = len(p["left"])

    with pytest.raises(ModelCorrupt, match="parent model: p90: child index"):
        load_model(_mutated(tiny_dk1_model_file, change))


def test_composite_parent_node_cap_is_per_tree_set(tiny_dk1_model_file, monkeypatch):
    n = max(len(t.feature) for m in (tiny_dk1_model_file, tiny_dk1_model_file.parent)
            for t in (m.median, m.p90, m.p10))  # fmt: skip
    monkeypatch.setattr(model, "MAX_NODES", n)
    load_model(dump_model(tiny_dk1_model_file))  # each set within the cap, together above it
    monkeypatch.setattr(model, "MAX_NODES", n - 1)
    with pytest.raises(ModelCorrupt, match="nodes"):
        load_model(dump_model(tiny_dk1_model_file))


@pytest.mark.parametrize("blend", [-0.1, 1.5, "x", None], ids=str)
def test_composite_blend_must_be_a_weight(tiny_dk1_model_file, blend):
    doc = _doc(tiny_dk1_model_file)
    doc["blend"] = blend
    with pytest.raises(ModelCorrupt):
        load_model(_encoded(doc))


def test_composite_without_blend_is_corrupt(tiny_dk1_model_file):
    doc = _doc(tiny_dk1_model_file)
    del doc["blend"]
    with pytest.raises(ModelCorrupt):
        load_model(_encoded(doc))


def test_sample_check_checks_the_parent(tiny_dk1_model_file):
    parent = tiny_dk1_model_file.parent
    bad = replace(parent, p10=replace(parent.p10, baseline=5e3))
    m = load_model(dump_model(replace(tiny_dk1_model_file, parent=bad)))
    with pytest.raises(ModelCorrupt, match="parent model: p10"):
        sample_check(m)
