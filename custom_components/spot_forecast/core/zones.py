"""Bidding-zone definitions: weather sample points and feature groupings."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]

DE_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
NB_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
S_VARS = ("wind_speed_100m", "temperature_2m", "cloud_cover")

POINTS: dict[str, tuple[Point, tuple[str, ...]]] = {
    # DE-LU
    "north_sea": ((54.2, 6.6), DE_VARS),
    "baltic": ((54.7, 13.8), DE_VARS),
    "schleswig": ((54.5, 9.2), DE_VARS),
    "ostfriesland": ((53.3, 7.6), DE_VARS),
    "uckermark": ((53.2, 13.8), DE_VARS),
    "sachsen_anhalt": ((51.9, 11.6), DE_VARS),
    "nrw": ((51.4, 7.3), DE_VARS),
    "hessen": ((50.4, 9.3), DE_VARS),
    "bawue": ((48.7, 9.2), DE_VARS),
    "bayern": ((48.4, 11.8), DE_VARS),
    # neighbours of DE-LU
    "nb_nl": ((52.6, 4.6), NB_VARS),
    "nb_be": ((50.8, 4.4), NB_VARS),
    "nb_fr_north": ((49.0, 2.4), NB_VARS),
    "nb_fr_south": ((45.5, 4.8), NB_VARS),
    "nb_dk_west": ((56.0, 8.5), NB_VARS),
    "nb_pl_north": ((54.2, 17.5), NB_VARS),
    "nb_alps_ch": ((46.6, 8.0), NB_VARS),
    "nb_alps_at": ((47.3, 12.5), NB_VARS),
    "nb_no_south": ((59.5, 7.5), NB_VARS),
    "nb_se_south": ((56.5, 14.5), NB_VARS),
    "nb_se_north": ((63.5, 16.0), NB_VARS),
    "nb_fi": ((61.0, 25.0), NB_VARS),
    # NL
    "nl_borssele": ((51.70, 3.00), S_VARS),  # offshore wind
    "nl_flevoland": ((52.50, 5.60), S_VARS),  # onshore wind + solar
    "nl_groningen": ((53.30, 6.70), S_VARS),  # wind + solar
    "nl_randstad": ((52.00, 4.40), S_VARS),  # demand + solar
    "nl_brabant": ((51.50, 5.40), S_VARS),  # solar + demand
}
"""Every weather point with its coordinates and its one variable set.

One variable set per point, whatever zones use it, so the raw files and the Open-Meteo
requests of a point are the same for every zone.
"""


def point_vars(name: str) -> tuple[str, ...]:
    """The variables fetched for point ``name``."""
    return POINTS[name][1]


@dataclass(frozen=True)
class Zone:
    """Geography and scaling constants of one bidding zone.

    The order of ``own`` and ``nb`` fixes the feature order. ``layout`` selects the feature
    layout (``core.features.feature_names``): ``"de_lu"`` is the one of the original
    research backtest, ``"generic"`` the zone-generic one of research/zones/genfeat.py.
    The roles name the points their aggregates average over; ``nb_nordic`` is used by the
    ``"de_lu"`` layout only, ``hydro`` by the ``"generic"`` one only.
    """

    code: str
    name: str  # English label, e.g. "Netherlands"
    tz: str
    country: str  # holiday calendar, see core.calendar.holidays
    layout: str  # "de_lu" | "generic"
    default_vat: float  # percent, the VAT option's default for a new entry
    own: tuple[str, ...]
    nb: tuple[str, ...]
    wind: tuple[str, ...]
    solar: tuple[str, ...]
    nb_wind: tuple[str, ...]
    nb_hydro: tuple[str, ...]
    price_attribution: str  # who publishes the zone's day-ahead prices, and the licence
    hydro: tuple[str, ...] = ()
    nb_nordic: tuple[str, ...] = ()
    evening_hours: tuple[int, int] = (18, 21)
    scale_floor: float = 50.0
    scale_divisor: float = 100.0

    @property
    def slug(self) -> str:
        """Lower-case identifier for file names, statistic ids and storage keys."""
        return self.code.lower().replace("-", "_")

    @property
    def points(self) -> dict[str, Point]:
        """The zone's own points with their coordinates."""
        return {p: POINTS[p][0] for p in self.own}

    @property
    def neighbor_points(self) -> dict[str, Point]:
        """The neighbour points with their coordinates."""
        return {p: POINTS[p][0] for p in self.nb}

    @property
    def weather_vars(self) -> tuple[str, ...]:
        """The variables of the own points (all of them, in first-appearance order)."""
        return tuple(dict.fromkeys(v for p in self.own for v in point_vars(p)))

    @property
    def neighbor_vars(self) -> tuple[str, ...]:
        """The variables of the neighbour points (all of them, in first-appearance order)."""
        return tuple(dict.fromkeys(v for p in self.nb for v in point_vars(p)))

    def point_sets(self) -> list[tuple[dict[str, Point], tuple[str, ...]]]:
        """Own and neighbour points grouped by variable set, in first-appearance order.

        One Open-Meteo request each. For DE-LU: the own points, then the neighbours.
        """
        groups: dict[tuple[str, ...], dict[str, Point]] = {}
        for p in dict.fromkeys(self.own + self.nb):
            groups.setdefault(point_vars(p), {})[p] = POINTS[p][0]
        return [(points, variables) for variables, points in groups.items()]


_SMARD = "Bundesnetzagentur | SMARD.de via Energy-Charts (CC BY 4.0)"

DE_LU = Zone(
    code="DE-LU",
    name="Germany and Luxembourg",
    tz="Europe/Berlin",
    country="DE",
    layout="de_lu",
    default_vat=19.0,
    own=("north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt",
         "nrw", "hessen", "bawue", "bayern"),
    nb=("nb_nl", "nb_be", "nb_fr_north", "nb_fr_south", "nb_dk_west", "nb_pl_north",
        "nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_south", "nb_se_north", "nb_fi"),
    wind=("north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt"),
    solar=("sachsen_anhalt", "nrw", "hessen", "bawue", "bayern", "uckermark"),
    nb_wind=("nb_nl", "nb_be", "nb_dk_west", "nb_pl_north", "nb_se_south"),
    nb_hydro=("nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_north"),
    nb_nordic=("nb_no_south", "nb_se_north", "nb_fi"),
    price_attribution=_SMARD,
)  # fmt: skip

# research/zones/zonecfg.py ZONES["NL"]
NL = Zone(
    code="NL",
    name="Netherlands",
    tz="Europe/Amsterdam",
    country="NL",
    layout="generic",
    default_vat=21.0,
    own=("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen", "nl_randstad",
         "nl_brabant"),
    nb=("ostfriesland", "nrw", "nb_be", "nb_fr_north", "nb_dk_west", "schleswig", "nb_no_south"),
    wind=("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen"),
    solar=("nl_flevoland", "nl_groningen", "nl_randstad", "nl_brabant"),
    nb_wind=("ostfriesland", "nb_be", "nb_dk_west", "schleswig"),
    nb_hydro=("nb_no_south",),
    price_attribution=_SMARD,
)  # fmt: skip

ZONES: dict[str, Zone] = {z.code: z for z in (DE_LU, NL)}
"""Supported zones by code; the first is the config flow's default."""


def zone_by_slug(slug: str) -> Zone:
    """The zone whose ``slug`` is ``slug``; ``KeyError`` if there is none."""
    for zone in ZONES.values():
        if zone.slug == slug:
            return zone
    raise KeyError(slug)
