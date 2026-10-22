"""Model file format and numpy-only gradient-boosted tree inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gzip
import json
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, FEATURE_SCHEMA

FORMAT = 1
_ROW_CHUNK = 4096


class ModelMismatch(ValueError):
    """The model file does not fit this version of the code."""


@dataclass(frozen=True, eq=False)
class Trees:
    """Flat arrays over all nodes of all trees of one boosted ensemble."""

    baseline: float
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    missing_left: np.ndarray
    is_leaf: np.ndarray
    value: np.ndarray
    roots: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Baseline plus the sum of leaf values; NaN follows ``missing_left``."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0])
        if len(self.roots) == 0:
            out.fill(self.baseline)
            return out
        for start in range(0, X.shape[0], _ROW_CHUNK):
            out[start : start + _ROW_CHUNK] = self._predict_chunk(X[start : start + _ROW_CHUNK])
        return out

    def _predict_chunk(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        # node[t, r]: current node of tree t for row r; all trees advance one level per step
        node = np.repeat(self.roots[:, None], n_rows, axis=1)
        cols = np.arange(n_rows)[None, :]
        while True:
            leaf = self.is_leaf[node]
            if leaf.all():
                break
            x = X[cols, self.feature[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(leaf, node, nxt)
        return self.baseline + self.value[node].sum(axis=0)


@dataclass(frozen=True)
class ModelFile:
    """Everything the integration needs from one training run."""

    format: int
    zone: str
    version: str
    feature_schema: int
    features: tuple[str, ...]
    train_cutoff: date
    created: datetime
    median: Trees
    p90: Trees
    p10: Trees
    p90_offsets: dict[int, float]
    p10_offsets: dict[int, float]
    peak_threshold: float
    capacity: dict[str, float]
    metrics: dict


_TREE_FIELDS = (
    ("feature", np.int64),
    ("threshold", np.float64),
    ("left", np.int64),
    ("right", np.int64),
    ("missing_left", np.bool_),
    ("is_leaf", np.bool_),
    ("value", np.float64),
    ("roots", np.int64),
)


def _trees_to_json(t: Trees) -> dict[str, Any]:
    d: dict[str, Any] = {"baseline": float(t.baseline)}
    for name, _ in _TREE_FIELDS:
        arr = getattr(t, name)
        d[name] = arr.astype(int).tolist() if arr.dtype == np.bool_ else arr.tolist()
    return d


def _trees_from_json(d: dict[str, Any]) -> Trees:
    return Trees(
        baseline=float(d["baseline"]),
        **{name: np.asarray(d[name], dtype=dtype) for name, dtype in _TREE_FIELDS},
    )


def dump_model(m: ModelFile) -> bytes:
    """Serialise to gzip JSON; strict JSON, so non-finite numbers raise ``ValueError``."""
    doc = {
        "format": m.format,
        "zone": m.zone,
        "version": m.version,
        "feature_schema": m.feature_schema,
        "features": list(m.features),
        "train_cutoff": m.train_cutoff.isoformat(),
        "created": m.created.isoformat(),
        "median": _trees_to_json(m.median),
        "p90": _trees_to_json(m.p90),
        "p10": _trees_to_json(m.p10),
        "p90_offsets": {str(h): v for h, v in m.p90_offsets.items()},
        "p10_offsets": {str(h): v for h, v in m.p10_offsets.items()},
        "peak_threshold": m.peak_threshold,
        "capacity": m.capacity,
        "metrics": m.metrics,
    }
    return gzip.compress(json.dumps(doc, allow_nan=False).encode(), mtime=0)


def load_model(data: bytes) -> ModelFile:
    """Parse a model file; raise ``ModelMismatch`` if it does not fit the code."""
    doc = json.loads(gzip.decompress(data))
    if doc.get("format") != FORMAT:
        raise ModelMismatch(f"unsupported model format {doc.get('format')!r}")
    if doc.get("feature_schema") != FEATURE_SCHEMA:
        raise ModelMismatch(f"feature schema {doc.get('feature_schema')!r} != {FEATURE_SCHEMA}")
    if tuple(doc.get("features", ())) != tuple(FEATURE_NAMES):
        raise ModelMismatch("model feature list differs from the code")
    return ModelFile(
        format=doc["format"],
        zone=doc["zone"],
        version=doc["version"],
        feature_schema=doc["feature_schema"],
        features=tuple(doc["features"]),
        train_cutoff=date.fromisoformat(doc["train_cutoff"]),
        created=datetime.fromisoformat(doc["created"]),
        median=_trees_from_json(doc["median"]),
        p90=_trees_from_json(doc["p90"]),
        p10=_trees_from_json(doc["p10"]),
        p90_offsets={int(h): float(v) for h, v in doc["p90_offsets"].items()},
        p10_offsets={int(h): float(v) for h, v in doc["p10_offsets"].items()},
        peak_threshold=float(doc["peak_threshold"]),
        capacity={k: float(v) for k, v in doc["capacity"].items()},
        metrics=doc["metrics"],
    )
