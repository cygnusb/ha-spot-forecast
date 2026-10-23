"""Bidding-zone definitions: weather sample points and feature groupings."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """Geography and scaling constants of one bidding zone.

    The order of ``points`` and ``weather_vars`` fixes the feature order.
    """

    code: str
    tz: str
    points: dict[str, Point]
    neighbor_points: dict[str, Point]
    weather_vars: tuple[str, ...]
    neighbor_vars: tuple[str, ...]
    wind_points: tuple[str, ...]
    solar_points: tuple[str, ...]
    nb_wind: tuple[str, ...]
    nb_hydro: tuple[str, ...]
    nb_nordic: tuple[str, ...]
    price_attribution: str  # who publishes the zone's day-ahead prices, and the licence
    evening_hours: tuple[int, int] = (18, 21)
    scale_floor: float = 50.0
    scale_divisor: float = 100.0

    @property
    def slug(self) -> str:
        """Lower-case identifier for file names, statistic ids and storage keys."""
        return self.code.lower().replace("-", "_")


DE_LU = Zone(
    code="DE-LU",
    tz="Europe/Berlin",
    points={
        "north_sea": (54.2, 6.6),
        "baltic": (54.7, 13.8),
        "schleswig": (54.5, 9.2),
        "ostfriesland": (53.3, 7.6),
        "uckermark": (53.2, 13.8),
        "sachsen_anhalt": (51.9, 11.6),
        "nrw": (51.4, 7.3),
        "hessen": (50.4, 9.3),
        "bawue": (48.7, 9.2),
        "bayern": (48.4, 11.8),
    },
    neighbor_points={
        "nb_nl": (52.6, 4.6),
        "nb_be": (50.8, 4.4),
        "nb_fr_north": (49.0, 2.4),
        "nb_fr_south": (45.5, 4.8),
        "nb_dk_west": (56.0, 8.5),
        "nb_pl_north": (54.2, 17.5),
        "nb_alps_ch": (46.6, 8.0),
        "nb_alps_at": (47.3, 12.5),
        "nb_no_south": (59.5, 7.5),
        "nb_se_south": (56.5, 14.5),
        "nb_se_north": (63.5, 16.0),
        "nb_fi": (61.0, 25.0),
    },
    weather_vars=("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"),
    neighbor_vars=("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation"),
    wind_points=("north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt"),
    solar_points=("sachsen_anhalt", "nrw", "hessen", "bawue", "bayern", "uckermark"),
    nb_wind=("nb_nl", "nb_be", "nb_dk_west", "nb_pl_north", "nb_se_south"),
    nb_hydro=("nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_north"),
    nb_nordic=("nb_no_south", "nb_se_north", "nb_fi"),
    price_attribution="Bundesnetzagentur | SMARD.de via Energy-Charts (CC BY 4.0)",
)

ZONES: dict[str, Zone] = {z.code: z for z in (DE_LU,)}
"""Supported zones by code."""
