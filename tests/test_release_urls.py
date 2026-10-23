"""Every copy of the models release URL agrees.

The integration only accepts asset URLs under its own MANIFEST_URL's release path, so a
drifting copy in training would make every published model unusable. Read from the source
(no imports: training.ci needs pandas).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _constant(path: str, name: str) -> str:
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_release_urls_agree():
    manifest_url = _constant("custom_components/spot_forecast/const.py", "MANIFEST_URL")
    assert manifest_url == (
        "https://github.com/cygnusb/ha-spot-forecast/releases/download/models/manifest.json"
    )
    assert _constant("training/ci.py", "MODELS_BASE_URL") + "/manifest.json" == manifest_url
    assert _constant("tools/fetch_published_model.py", "MANIFEST_URL") == manifest_url
    assert _constant("tools/check_model_age.py", "MANIFEST_URL") == manifest_url
