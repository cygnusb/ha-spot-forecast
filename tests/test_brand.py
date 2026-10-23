"""The brand images HA loads from the integration, and their sources, are in place.

Home Assistant (2026.3+) reads custom_components/<domain>/brand/ instead of the brands
CDN and expects fixed sizes. PNG sizes come from the IHDR chunk, so no imaging library.
"""

from __future__ import annotations

from pathlib import Path
import struct

REPO = Path(__file__).resolve().parents[1]
BRAND = REPO / "custom_components" / "spot_forecast" / "brand"
ASSETS = REPO / "assets"

BRAND_SIZES = {
    "icon.png": (256, 256),
    "icon@2x.png": (512, 512),
    "logo.png": (256, 80),
    "logo@2x.png": (512, 160),
}


def _png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    assert head[12:16] == b"IHDR"
    return struct.unpack(">II", head[16:24])


def test_brand_files_exist_with_sizes():
    for name, size in BRAND_SIZES.items():
        assert _png_size(BRAND / name) == size, name
    for name in ("icon.svg", "logo.svg", "BRANDS.md"):
        assert (ASSETS / name).is_file(), name
    # the PNGs next to the sources are the same renders as the shipped ones
    for name in ("icon.png", "logo.png"):
        assert (ASSETS / name).read_bytes() == (BRAND / name).read_bytes(), name
