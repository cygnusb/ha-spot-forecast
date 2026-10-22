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
