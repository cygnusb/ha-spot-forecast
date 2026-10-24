#!/usr/bin/env python3
"""Build site/data/zone_map.json, the map of the bidding zones on the project page.

Run by hand; the output is committed. Inputs are Natural Earth 1:10m (public domain),
release v5.1.2 of github.com/nvkelso/natural-earth-vector:

    geojson/ne_10m_admin_0_countries.geojson
    geojson/ne_10m_admin_1_states_provinces.geojson

    python -I tools/zone_map.py ADMIN0.geojson ADMIN1.geojson [--out site/data/zone_map.json]

Countries of northern and central Europe are the grey context. A zone is a set of countries
(ADM0_A3) or of first-level regions (``iso_3166_2``, never names: v5.1.2 spells Sjælland
"Sjaælland" and has the Norwegian counties of before 2020). Rings far from Europe are
dropped (NE's Netherlands includes the Caribbean municipalities), the rest projected
(Lambert azimuthal equal-area around 10°E 55°N), clipped to the view box, simplified and
written as SVG path strings in a 1000-wide view box. Which zones are highlighted is up to
the page: those with a bundled model (tools/build_site.py). Standard library only.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "site" / "data" / "zone_map.json"
SOURCE = "Natural Earth 1:10m Cultural Vectors, v5.1.2 (public domain)"

BOX = (-7.0, 45.5, 22.0, 62.5)  # lon/lat: west, south, east, north
LON0, LAT0 = 10.0, 55.0
WIDTH = 1000
# Douglas-Peucker tolerance (projected km), smallest island kept (km²), decimals of a pixel
ZONE_DETAIL = (1.5, 100.0, 1)
CONTEXT_DETAIL = (3.0, 200.0, 0)

CONTEXT = ("GBR", "IRL", "FRA", "BEL", "NLD", "LUX", "DEU", "DNK", "NOR", "SWE", "FIN", "EST",
           "LVA", "LTU", "POL", "CZE", "SVK", "AUT", "CHE", "LIE", "HUN", "SVN", "HRV", "ITA",
           "ESP", "AND", "MCO", "BLR", "UKR", "RUS", "ROU", "SRB", "BIH", "SMR", "VAT", "ALD",
           "FRO", "IMN", "GGY", "JEY", "MDA")  # fmt: skip
# code -> (countries, regions, approximate)
ZONES: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool]] = {
    "DE-LU": (("DEU", "LUX"), (), False),
    "NL": (("NLD",), (), False),
    "DK1": ((), ("DK-81", "DK-82", "DK-83"), False),
    "DK2": ((), ("DK-84", "DK-85"), False),
    # the NO2 price area follows grid lines, not county borders: Agder, Rogaland, Telemark
    "NO2": ((), ("NO-08", "NO-09", "NO-10", "NO-11"), True),
}


def _clip(ring: list, box: tuple[float, float, float, float]) -> list:
    """Sutherland-Hodgman: the ring clipped to the box (x0, y0, x1, y1); may come back empty."""
    west, south, east, north = box
    edges = (
        (lambda p: p[0] >= west, lambda a, b: _cut_x(a, b, west)),
        (lambda p: p[0] <= east, lambda a, b: _cut_x(a, b, east)),
        (lambda p: p[1] >= south, lambda a, b: _cut_y(a, b, south)),
        (lambda p: p[1] <= north, lambda a, b: _cut_y(a, b, north)),
    )
    out = ring
    for inside, cut in edges:
        pts, out = out, []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
        if not out:
            return []
    return out


def _near(lon: float, lat: float) -> bool:
    """Whether a point is anywhere near the map (far away rings are not projected)."""
    west, south, east, north = BOX
    return west - 15 <= lon <= east + 15 and south - 10 <= lat <= north + 10


def _cut_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return [x, a[1] + t * (b[1] - a[1])]


def _cut_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return [a[0] + t * (b[0] - a[0]), y]


def _project(lon: float, lat: float) -> tuple[float, float]:
    """Lambert azimuthal equal-area on a sphere of 6371 km: x east, y north, km."""
    lam, phi = math.radians(lon - LON0), math.radians(lat)
    phi0 = math.radians(LAT0)
    k = math.sqrt(
        2 / (1 + math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(lam))
    )
    x = k * math.cos(phi) * math.sin(lam)
    y = k * (math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(lam))
    return 6371 * x, 6371 * y


def _simplify(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Douglas-Peucker (iterative) of an open polyline."""
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (ax, ay), (bx, by) = pts[i], pts[j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best, index = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            d = (
                abs(dy * (px - ax) - dx * (py - ay)) / norm
                if norm
                else math.hypot(px - ax, py - ay)
            )
            if d > best:
                best, index = d, k
        if index >= 0 and best > tol:
            keep[index] = True
            stack += [(i, index), (index, j)]
    return [p for p, k in zip(pts, keep, strict=True) if k]


def _area(pts: list[tuple[float, float]]) -> float:
    pairs = zip(pts, pts[1:] + pts[:1], strict=True)
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairs)) / 2


def _rings(geometry: dict) -> list[list[list[float]]]:
    polys = (
        geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    )
    return [ring for poly in polys for ring in poly]


class Canvas:
    """Projected km to view-box pixels: the box's projected extent, y flipped."""

    def __init__(self) -> None:
        """The view box: the projected extent of ``BOX``, so what is clipped to it ends at
        the frame of the SVG."""
        west, south, east, north = BOX
        edge = [(lon, lat) for lon in (west, east) for lat in (south, north)]
        edge += [(west + (east - west) * i / 40, lat) for i in range(41) for lat in (south, north)]
        edge += [(lon, south + (north - south) * i / 40) for i in range(41) for lon in (west, east)]
        xy = [_project(lon, lat) for lon, lat in edge]
        self.x0, self.x1 = min(x for x, _ in xy), max(x for x, _ in xy)
        self.y0, self.y1 = min(y for _, y in xy), max(y for _, y in xy)
        self.scale = WIDTH / (self.x1 - self.x0)
        self.height = round((self.y1 - self.y0) * self.scale)

    def path(self, geometries: list[dict], detail: tuple[float, float, int]) -> str:
        """One SVG path of the outer and inner rings, in relative coordinates."""
        tolerance, min_area, digits = detail
        parts = []
        for geometry in geometries:
            for ring in _rings(geometry):
                ring = ring[:-1] if ring[0] == ring[-1] else ring
                if not any(_near(lon, lat) for lon, lat in ring):
                    continue  # e.g. the Caribbean Netherlands
                projected = [_project(lon, lat) for lon, lat in ring]
                pts = _clip(projected, (self.x0, self.y0, self.x1, self.y1))
                if len(pts) < 3 or _area(pts) < min_area:
                    continue
                simple = _simplify(pts + pts[:1], tolerance)[:-1]
                if len(simple) < 3:
                    continue
                unit = 10**digits  # integer steps of 1/unit pixel, so the sums stay exact
                px = self.scale * unit
                coords = [(round((x - self.x0) * px), round((self.y1 - y) * px)) for x, y in simple]
                steps = [(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(coords)]
                steps = [d for d in steps if d != (0, 0)]
                if len(steps) < 2:
                    continue
                parts.append(
                    f"M{_n(coords[0][0], unit)} {_n(coords[0][1], unit)}l"
                    + " ".join(f"{_n(dx, unit)} {_n(dy, unit)}" for dx, dy in steps)
                    + "z"
                )
        return "".join(parts)


def _n(v: int, unit: int) -> str:
    """``v`` / ``unit`` as short as possible: 12, 1.5, -0.3."""
    return str(v) if unit == 1 else f"{v / unit:g}"


def build(admin0: Path, admin1: Path) -> dict:
    countries = {
        f["properties"]["ADM0_A3"]: f["geometry"]
        for f in json.loads(admin0.read_text(encoding="utf-8"))["features"]
    }
    regions = {
        f["properties"]["iso_3166_2"]: f
        for f in json.loads(admin1.read_text(encoding="utf-8"))["features"]
    }
    canvas = Canvas()
    missing = [c for c in CONTEXT if c not in countries]
    missing += [c for zone in ZONES.values() for c in zone[0] if c not in countries]
    missing += [r for zone in ZONES.values() for r in zone[1] if r not in regions]
    if missing:
        raise SystemExit(f"not in the Natural Earth files: {', '.join(missing)}")
    zones = {}
    for code, (adm0, adm1, approximate) in ZONES.items():
        geometries = [countries[c] for c in adm0] + [regions[r]["geometry"] for r in adm1]
        zones[code] = {
            "path": canvas.path(geometries, ZONE_DETAIL),
            "regions": [*adm0, *(f"{r} {regions[r]['properties']['name']}" for r in adm1)],
            "approximate": approximate,
        }
    return {
        "source": SOURCE,
        "projection": f"Lambert azimuthal equal-area, centre {LON0}°E {LAT0}°N",
        "simplified_km": {"zones": ZONE_DETAIL[0], "context": CONTEXT_DETAIL[0]},
        "width": WIDTH,
        "height": canvas.height,
        "context": canvas.path([countries[c] for c in CONTEXT], CONTEXT_DETAIL),
        "zones": zones,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("admin0", type=Path)
    ap.add_argument("admin1", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    doc = build(args.admin0, args.admin1)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {args.out.stat().st_size // 1024} KB, zones {', '.join(doc['zones'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
