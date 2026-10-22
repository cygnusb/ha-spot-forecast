import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.ensemble import HistGradientBoostingRegressor

from training.export import trees_from_sklearn

LOSSES = ({"loss": "absolute_error"}, {"loss": "quantile", "quantile": 0.9},
          {"loss": "quantile", "quantile": 0.1})  # fmt: skip


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 82))
    y = X[:, 0] * 2 + np.sin(X[:, 1] * 3) + X[:, 2] * X[:, 3] + rng.normal(scale=0.5, size=5000)
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y


@pytest.mark.parametrize("kw", LOSSES)
def test_export_reproduces_sklearn(kw):
    X, y = _data()
    m = HistGradientBoostingRegressor(max_iter=50, **kw, random_state=0).fit(X, y)
    np.testing.assert_allclose(trees_from_sklearn(m).predict(X), m.predict(X), atol=1e-6)


@pytest.mark.parametrize("kw", LOSSES)
def test_nan_routing_matches_sklearn(kw):
    X, y = _data()
    m = HistGradientBoostingRegressor(max_iter=50, **kw, random_state=0).fit(X, y)
    probe = X[:200].copy()
    probe[:50] = np.nan  # all-NaN rows
    for j in range(50, 200):  # one NaN column per row, cycling over the used features
        probe[j, j % 82] = np.nan
    np.testing.assert_allclose(trees_from_sklearn(m).predict(probe), m.predict(probe), atol=1e-6)


def test_rejects_categorical_splits():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(0, 5, 2000), rng.normal(size=2000)]).astype(float)
    y = (X[:, 0] == 2) * 3.0 + rng.normal(size=2000)
    m = HistGradientBoostingRegressor(max_iter=10, categorical_features=[0], random_state=0).fit(
        X, y
    )
    with pytest.raises(ValueError, match="categorical"):
        trees_from_sklearn(m)


def _no_constants(name: str) -> float:
    raise ValueError(f"non-standard JSON constant {name}")


def test_inf_threshold_exports_as_strict_json(tiny_model_file):
    """sklearn stores +inf as the threshold of a split that sends every known value left and
    only missing values right; the model file must still be strict JSON (orjson in HA)."""
    from dataclasses import replace
    import gzip
    import json

    from core.model import dump_model, load_model

    rng = np.random.default_rng(2)
    X = rng.normal(size=(4000, 82))
    X[rng.random(4000) < 0.5, 0] = np.nan
    y = 10.0 * np.isnan(X[:, 0]) + X[:, 1] + rng.normal(scale=0.3, size=4000)
    m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)
    thresholds = np.concatenate([p[0].nodes["num_threshold"] for p in m._predictors])
    assert np.isinf(thresholds).any()  # the case under test really occurs

    trees = trees_from_sklearn(m)
    assert np.isfinite(trees.threshold).all()
    data = dump_model(replace(tiny_model_file, median=trees, p90=trees, p10=trees))
    json.loads(gzip.decompress(data), parse_constant=_no_constants)
    back = load_model(data)
    probe = np.vstack([X[:500], np.full((5, 82), np.nan)])
    np.testing.assert_array_equal(back.median.predict(probe), trees.predict(probe))
    np.testing.assert_allclose(back.median.predict(probe), m.predict(probe), atol=1e-6)


def test_dump_refuses_non_finite_values(tiny_model_file):
    from dataclasses import replace

    from core.model import dump_model

    with pytest.raises(ValueError):
        dump_model(replace(tiny_model_file, peak_threshold=float("inf")))
