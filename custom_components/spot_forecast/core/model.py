"""Model file format and numpy-only gradient-boosted tree inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gzip
import json
from typing import Any
import zlib

import numpy as np

from .features import FEATURE_NAMES, FEATURE_SCHEMA, HORIZONS

FORMAT = 1
_ROW_CHUNK = 4096
SAMPLE_LIMIT = 1e3  # |prediction| bound of the sample check (scaled deviations, not EUR/MWh)
# decompressed sizes (bytes); the published files hold ~10 MB and ~2 MB of JSON
MAX_MODEL_JSON_BYTES = 64 * 1024 * 1024
MAX_BACKFILL_JSON_BYTES = 32 * 1024 * 1024
MAX_NODES = 2_000_000  # per ensemble; the published ones have ~37,000
MAX_DEPTH = 64  # splits from a root to its deepest leaf


class ModelMismatch(ValueError):
    """The model file does not fit this version of the code."""


class ModelCorrupt(ModelMismatch):
    """The model file cannot be decoded or its trees are not well-formed."""


def gunzip_limited(data: bytes, max_size: int) -> bytes:
    """``data`` (one gzip member) decompressed; ``ModelCorrupt`` past ``max_size`` bytes.

    Stops decompressing at the limit, so a small file cannot expand into all memory. Also
    ``ModelCorrupt`` if ``data`` is not gzip, is truncated or has bytes after the member.
    """
    d = zlib.decompressobj(wbits=31)  # gzip header and trailer
    try:
        out = d.decompress(data, max_size + 1)
    except zlib.error as e:
        raise ModelCorrupt(f"not valid gzip: {e}") from e
    if len(out) > max_size:
        raise ModelCorrupt(f"more than the limit of {max_size} bytes decompressed")
    if not d.eof:
        raise ModelCorrupt("gzip data is truncated")
    if d.unused_data:
        raise ModelCorrupt("trailing data after the gzip member")
    return out


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


def _trees_from_json(d: dict[str, Any], name: str) -> Trees:
    t = Trees(
        baseline=float(d["baseline"]),
        **{field: np.asarray(d[field], dtype=dtype) for field, dtype in _TREE_FIELDS},
    )
    _check_trees(t, name)
    return t


def _check_trees(t: Trees, name: str) -> None:
    """Structural checks that make ``predict`` safe: in bounds, finite, and terminating."""
    n = len(t.feature)
    if n > MAX_NODES:
        raise ModelCorrupt(f"{name}: {n} nodes, more than {MAX_NODES}")
    nodes = [getattr(t, field) for field, _ in _TREE_FIELDS if field != "roots"]
    if any(a.ndim != 1 or len(a) != n for a in nodes):
        raise ModelCorrupt(f"{name}: node arrays differ in length")
    if t.roots.ndim != 1 or ((t.roots < 0) | (t.roots >= n)).any():
        raise ModelCorrupt(f"{name}: root index out of bounds")
    if not np.isfinite(t.baseline) or not np.isfinite(t.value[t.is_leaf]).all():
        raise ModelCorrupt(f"{name}: non-finite baseline or leaf value")
    inner = np.flatnonzero(~t.is_leaf)
    # sklearn numbers every child after its parent; requiring this proves the trees acyclic,
    # so the descent in ``_predict_chunk`` ends after at most n steps
    for child in (t.left[inner], t.right[inner]):
        if ((child <= inner) | (child >= n)).any():
            raise ModelCorrupt(f"{name}: child index out of bounds or not after its parent")
    # leaves too: ``_predict_chunk`` looks up the feature of every current node
    if ((t.feature < 0) | (t.feature >= len(FEATURE_NAMES))).any():
        raise ModelCorrupt(f"{name}: feature index out of range")
    if not np.isfinite(t.threshold[inner]).all():
        raise ModelCorrupt(f"{name}: non-finite split threshold")
    # level by level from the roots (after the acyclicity check, so the levels stay small)
    level = t.roots
    for _ in range(MAX_DEPTH):
        level = level[~t.is_leaf[level]]
        if len(level) == 0:
            break
        level = np.unique(np.concatenate((t.left[level], t.right[level])))
    else:
        if not t.is_leaf[level].all():
            raise ModelCorrupt(f"{name}: a tree is deeper than {MAX_DEPTH}")


def _offsets(d: dict[str, Any], name: str) -> dict[int, float]:
    offsets = {int(h): float(v) for h, v in d.items()}
    if set(offsets) != set(HORIZONS):
        raise ModelCorrupt(f"{name}: horizons {sorted(offsets)} != {list(HORIZONS)}")
    return offsets


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
    """Parse and check a model file.

    Raises ``ModelMismatch`` if it does not fit the code, and its subclass ``ModelCorrupt``
    if it cannot be decoded or is not well-formed.
    """
    try:
        raw = gunzip_limited(data, MAX_MODEL_JSON_BYTES)
    except ModelCorrupt as e:
        raise ModelCorrupt(f"model file is corrupt: {e}") from e
    try:
        return _model_from_json(json.loads(raw))
    except ModelMismatch:
        raise
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,  # JSON integers too large for numpy
        RecursionError,  # JSON nested too deeply
    ) as e:
        raise ModelCorrupt(f"model file is corrupt: {type(e).__name__}: {e}") from e


def _model_from_json(doc: Any) -> ModelFile:
    if not isinstance(doc, dict):
        raise ModelCorrupt("model file is not a JSON object")
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
        median=_trees_from_json(doc["median"], "median"),
        p90=_trees_from_json(doc["p90"], "p90"),
        p10=_trees_from_json(doc["p10"], "p10"),
        p90_offsets=_offsets(doc["p90_offsets"], "p90_offsets"),
        p10_offsets=_offsets(doc["p10_offsets"], "p10_offsets"),
        peak_threshold=float(doc["peak_threshold"]),
        capacity={k: float(v) for k, v in doc["capacity"].items()},
        metrics=doc["metrics"],
    )


def _sample() -> np.ndarray:
    """Two rows: every feature missing, and every feature 0.0."""
    sample = np.vstack([np.full(len(FEATURE_NAMES), np.nan), np.zeros(len(FEATURE_NAMES))])
    sample.flags.writeable = False
    return sample


_SAMPLE = _sample()


def sample_check(model: ModelFile) -> None:
    """Raise ``ModelCorrupt`` unless the median, P90 and P10 predictions of a fixed sample
    are finite and below ``SAMPLE_LIMIT`` in absolute value.

    For a model from ``load_model`` (whose trees are checked to be well-formed).
    """
    for name in ("median", "p90", "p10"):
        y = getattr(model, name).predict(_SAMPLE)
        if not (np.isfinite(y).all() and (np.abs(y) < SAMPLE_LIMIT).all()):
            raise ModelCorrupt(f"{name}: sample predictions {y.tolist()} are not plausible")
