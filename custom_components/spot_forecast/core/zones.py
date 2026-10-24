"""Bidding-zone definitions: weather sample points and feature groupings."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]

DE_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
NB_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
S_VARS = ("wind_speed_100m", "temperature_2m", "cloud_cover")
N_VARS = ("wind_speed_100m", "temperature_2m", "precipitation")

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
    # DK1
    "dk1_horns_rev": ((55.50, 7.80), S_VARS),  # offshore
    "dk1_north_jutland": ((57.20, 9.90), S_VARS),
    "dk1_aarhus": ((56.20, 10.10), S_VARS),
    "dk1_south_jutland": ((55.30, 9.20), S_VARS),
    # DK2
    "dk2_copenhagen": ((55.70, 12.40), S_VARS),
    "dk2_kriegers_flak": ((55.00, 12.90), S_VARS),  # offshore
    "dk2_lolland": ((54.80, 11.40), S_VARS),
    "dk2_bornholm": ((55.10, 14.90), S_VARS),
    # neighbours of DK1 and DK2
    "no2_sirdal": ((58.90, 6.90), N_VARS),  # hydro
    "se3_gothenburg": ((57.70, 12.00), N_VARS),
    "se4_malmo": ((55.60, 13.10), S_VARS),
    # NO2
    "no2_kristiansand": ((58.20, 8.00), N_VARS),
    "no2_stavanger": ((58.90, 5.80), N_VARS),
    # neighbours of NO2 (NO5 and NO1 hydro)
    "no5_sogn": ((61.20, 7.00), N_VARS),
    "no1_hallingdal": ((60.60, 9.00), N_VARS),
    "no5_bergen": ((60.40, 5.30), N_VARS),
    "no1_oslo": ((59.90, 10.80), N_VARS),
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

    The generic layout's research options: ``raw_extra`` adds the raw columns of more points
    after the own points' ones, ``de_extra`` the DE-LU wind, solar and temperature aggregates
    (over DE-LU's own points, which must be among ``own`` + ``raw_extra``), and
    ``price_inputs`` names zones whose prices are features too (``("DE-LU",)`` only).

    ``hydro_extra`` adds the hydro features of research/zones/genfeat.py ``_hydro_extra``:
    trailing precipitation and temperature of ``hydro_points`` from the archived offset-2
    forecasts, and their precipitation forecast over the six target days. ``doy=False``
    drops the ``doy`` column (research/zones/bt.py variant ``hydnd``).

    A zone with a ``parent`` forecasts the blend ``blend`` x own + (1 - ``blend``) x parent:
    its model file embeds the parent zone's model (``core.model``).
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
    raw_extra: tuple[str, ...] = ()
    de_extra: bool = False
    price_inputs: tuple[str, ...] = ()  # zone codes
    parent: str | None = None  # zone code
    blend: float = 1.0  # weight of the own model, with a parent
    hydro_extra: bool = False
    doy: bool = True
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
    def hydro_points(self) -> tuple[str, ...]:
        """The points the ``hydro_extra`` features sum over: ``hydro``, then ``nb_hydro``."""
        return tuple(dict.fromkeys(self.hydro + self.nb_hydro))

    @property
    def weather_vars(self) -> tuple[str, ...]:
        """The variables of the own points (all of them, in first-appearance order)."""
        return tuple(dict.fromkeys(v for p in self.own for v in point_vars(p)))

    @property
    def neighbor_vars(self) -> tuple[str, ...]:
        """The variables of the neighbour points (all of them, in first-appearance order)."""
        return tuple(dict.fromkeys(v for p in self.nb for v in point_vars(p)))

    def point_sets(self) -> list[tuple[dict[str, Point], tuple[str, ...]]]:
        """Every point a forecast of the zone reads, grouped by variable set, in
        first-appearance order: own, neighbour and ``raw_extra`` points, then the parent's.

        One Open-Meteo request each. For DE-LU: the own points, then the neighbours.
        """
        names = self.own + self.nb + self.raw_extra
        if self.parent is not None:
            parent = ZONES[self.parent]
            names += parent.own + parent.nb
        groups: dict[tuple[str, ...], dict[str, Point]] = {}
        for p in dict.fromkeys(names):
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

# research/zones/zonecfg.py ZONES["DK1"] and ZONES["DK2"], with the inputs of research/zones/bt.py
# variant "desetprel" (DE-LU weather points and prices) and blended 50/50 with DE-LU. The raw
# extra points are the DE-LU points that are not own points: none is for DK1 or DK2.
_DE_LU_RAW = DE_LU.own

DK1 = Zone(
    code="DK1",
    name="Denmark West",
    tz="Europe/Copenhagen",
    country="DK",
    layout="generic",
    default_vat=25.0,
    own=("nb_dk_west", "dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
    nb=("schleswig", "north_sea", "ostfriesland", "nb_nl", "nb_no_south", "no2_sirdal",
        "se3_gothenburg", "dk2_copenhagen"),
    wind=("nb_dk_west", "dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
    solar=("dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
    nb_wind=("schleswig", "north_sea", "ostfriesland", "nb_nl", "dk2_copenhagen"),
    nb_hydro=("nb_no_south", "no2_sirdal"),
    raw_extra=_DE_LU_RAW,
    de_extra=True,
    price_inputs=("DE-LU",),
    parent="DE-LU",
    blend=0.5,
    price_attribution=_SMARD,
)  # fmt: skip

DK2 = Zone(
    code="DK2",
    name="Denmark East",
    tz="Europe/Copenhagen",
    country="DK",
    layout="generic",
    default_vat=25.0,
    own=("dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm"),
    nb=("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus", "se3_gothenburg", "nb_no_south",
        "nb_se_north"),
    wind=("dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm"),
    solar=("dk2_copenhagen", "dk2_lolland"),
    nb_wind=("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus"),
    nb_hydro=("nb_no_south", "nb_se_north"),
    raw_extra=_DE_LU_RAW,
    de_extra=True,
    price_inputs=("DE-LU",),
    parent="DE-LU",
    blend=0.5,
    price_attribution=_SMARD,
)  # fmt: skip

# research/zones/zonecfg.py ZONES["NO2"], with research/zones/bt.py variant "hydrel": the hydro
# features, no ``doy``
NO2 = Zone(
    code="NO2",
    name="Southern Norway",
    tz="Europe/Oslo",
    country="NO",
    layout="generic",
    default_vat=25.0,
    own=("nb_no_south", "no2_kristiansand", "no2_stavanger", "no2_sirdal"),
    nb=("nb_dk_west", "north_sea", "schleswig", "nb_nl", "no5_bergen", "no5_sogn", "no1_oslo",
        "no1_hallingdal"),
    wind=("no2_kristiansand", "no2_stavanger"),
    solar=(),
    nb_wind=("nb_dk_west", "north_sea", "schleswig", "nb_nl"),
    nb_hydro=("no5_sogn", "no1_hallingdal", "no5_bergen"),
    hydro=("nb_no_south", "no2_sirdal", "no2_stavanger"),
    hydro_extra=True,
    doy=False,
    price_attribution=_SMARD,
)  # fmt: skip

ZONES: dict[str, Zone] = {z.code: z for z in (DE_LU, NL, DK1, DK2, NO2)}
"""Supported zones by code; the first is the config flow's default."""


def zone_by_slug(slug: str) -> Zone:
    """The zone whose ``slug`` is ``slug``; ``KeyError`` if there is none."""
    for zone in ZONES.values():
        if zone.slug == slug:
            return zone
    raise KeyError(slug)
