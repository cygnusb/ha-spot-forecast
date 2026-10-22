"""Export fitted scikit-learn HistGradientBoostingRegressor trees to ``core.model.Trees``."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from core.model import Trees

# sklearn uses +inf as the threshold of a split that sends every known value left and only
# missing values right. JSON has no infinity, so it is stored as the largest finite double:
# for every finite x, ``x <= max`` equals ``x <= inf`` (sklearn rejects infinite inputs, and
# NaN never reaches the comparison). Both signs are clamped so the file is always finite.
_FLOAT_MAX = np.finfo(np.float64).max


def trees_from_sklearn(model: HistGradientBoostingRegressor) -> Trees:
    """Flatten all trees of a fitted single-output model (numeric splits only)."""
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("feature", "threshold", "left", "right", "missing_left", "is_leaf", "value")
    }
    roots: list[int] = []
    offset = 0
    for i, per_iteration in enumerate(model._predictors):
        nodes = per_iteration[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError(f"tree {i} has categorical splits; only numeric splits are supported")
        roots.append(offset)
        is_leaf = nodes["is_leaf"].astype(bool)
        parts["feature"].append(nodes["feature_idx"].astype(np.int64))
        parts["threshold"].append(
            np.clip(nodes["num_threshold"].astype(np.float64), -_FLOAT_MAX, _FLOAT_MAX)
        )
        # child indices of leaves are never followed, so their value after offsetting is irrelevant
        parts["left"].append(nodes["left"].astype(np.int64) + offset)
        parts["right"].append(nodes["right"].astype(np.int64) + offset)
        parts["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
        parts["is_leaf"].append(is_leaf)
        parts["value"].append(nodes["value"].astype(np.float64))
        offset += len(nodes)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return Trees(
        baseline=float(np.ravel(model._baseline_prediction)[0]),
        roots=np.array(roots, dtype=np.int64),
        **cat,
    )
