"""Zone catalogue for the multi-zone study: weather points, roles, timezones, holiday calendars."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent  # research/ of the repo
RESEARCH_RAW = RESEARCH / "data" / "raw"  # DE-LU raw data of the training pipeline
DATA = RESEARCH / "data" / "zones"  # other zones and points (fetch_prices.py, fetch_weather.py)

# Variable sets
DE_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
NB_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
S_VARS = ("wind_speed_100m", "temperature_2m", "cloud_cover")  # south / solar-relevant
N_VARS = ("wind_speed_100m", "temperature_2m", "precipitation")  # nordic / hydro

# Existing research cache points (read-only reuse)
RESEARCH_POINTS = {
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
}

# New points, in fetch-priority order (tier 1: shippable zones; tier 2: other priority; tier 3)
NEW_POINTS: dict[str, tuple[tuple[float, float], tuple[str, ...], int]] = {
    # NL
    "nl_borssele": ((51.70, 3.00), S_VARS, 1),  # offshore wind
    "nl_flevoland": ((52.50, 5.60), S_VARS, 1),  # onshore wind + solar
    "nl_groningen": ((53.30, 6.70), S_VARS, 1),  # wind + solar
    "nl_randstad": ((52.00, 4.40), S_VARS, 1),  # demand + solar
    "nl_brabant": ((51.50, 5.40), S_VARS, 1),  # solar + demand
    # DK1
    "dk1_horns_rev": ((55.50, 7.80), S_VARS, 1),  # offshore
    "dk1_north_jutland": ((57.20, 9.90), S_VARS, 1),
    "dk1_aarhus": ((56.20, 10.10), S_VARS, 1),
    "dk1_south_jutland": ((55.30, 9.20), S_VARS, 1),
    # DK2
    "dk2_copenhagen": ((55.70, 12.40), S_VARS, 1),
    "dk2_kriegers_flak": ((55.00, 12.90), S_VARS, 1),  # offshore
    "dk2_lolland": ((54.80, 11.40), S_VARS, 1),
    "dk2_bornholm": ((55.10, 14.90), S_VARS, 1),
    # SE4
    "se4_malmo": ((55.60, 13.10), S_VARS, 1),
    "se4_kalmar": ((56.70, 16.20), S_VARS, 1),
    "se4_halmstad": ((56.70, 12.90), S_VARS, 1),
    # NO2 (also neighbours of DK1)
    "no2_kristiansand": ((58.20, 8.00), N_VARS, 1),
    "no2_stavanger": ((58.90, 5.80), N_VARS, 1),
    "no2_sirdal": ((58.90, 6.90), N_VARS, 1),  # hydro
    # SE3 (also neighbours of DK1/SE4)
    "se3_gothenburg": ((57.70, 12.00), N_VARS, 1),
    "se3_stockholm": ((59.30, 18.00), N_VARS, 1),
    # NO5 / NO1 hydro used as NO2 neighbours
    "no5_sogn": ((61.20, 7.00), N_VARS, 1),
    "no1_hallingdal": ((60.60, 9.00), N_VARS, 1),
    "no5_bergen": ((60.40, 5.30), N_VARS, 1),
    "no1_oslo": ((59.90, 10.80), N_VARS, 1),
    # tier 2
    "se3_vasteras": ((59.60, 16.50), N_VARS, 2),
    "se3_varmland": ((59.60, 13.40), N_VARS, 2),
    "se3_ostergotland": ((58.40, 15.60), N_VARS, 2),
    "no1_innlandet": ((61.10, 10.50), N_VARS, 2),  # hydro
    "no1_ostfold": ((59.30, 11.10), N_VARS, 2),
    "fi_helsinki": ((60.20, 24.90), N_VARS, 2),
    "fi_oulu": ((65.00, 25.50), N_VARS, 2),  # wind
    "fi_pori": ((61.50, 21.80), N_VARS, 2),  # wind
    "fi_vaasa": ((63.10, 21.60), N_VARS, 2),  # wind
    "fi_kemi": ((66.50, 26.00), N_VARS, 2),  # hydro (Kemijoki)
    "ee_tallinn": ((59.40, 24.70), N_VARS, 2),
    "se1_hydro": ((66.80, 19.00), N_VARS, 2),  # Lule river
    "se1_lulea": ((65.60, 22.10), N_VARS, 2),
    "se2_sundsvall": ((62.40, 17.30), N_VARS, 2),
    "no3_trondheim": ((63.40, 10.40), N_VARS, 2),
    # tier 3
    "se1_kiruna": ((67.80, 20.20), N_VARS, 3),
    "se2_umea": ((63.80, 20.30), N_VARS, 3),
    "se2_storuman": ((65.10, 17.10), N_VARS, 3),  # hydro (Ume river)
    "no3_fosen": ((63.90, 10.00), N_VARS, 3),  # wind
    "no3_more": ((62.50, 7.00), N_VARS, 3),
    "no4_tromso": ((69.60, 19.00), N_VARS, 3),
    "no4_bodo": ((67.30, 14.40), N_VARS, 3),
    "no4_narvik": ((68.40, 17.40), N_VARS, 3),  # hydro
    "no5_hardanger": ((60.20, 6.60), N_VARS, 3),  # hydro
}


def point_vars(name: str) -> tuple[str, ...]:
    if name in RESEARCH_POINTS:
        return RESEARCH_POINTS[name][1]
    return NEW_POINTS[name][1]


@dataclass(frozen=True)
class ZoneCfg:
    code: str
    tz: str
    country: str  # holidays package code
    points: tuple[str, ...]
    wind: tuple[str, ...]
    solar: tuple[str, ...]  # points with cloud_cover -> clear-sky estimate
    hydro: tuple[str, ...]  # own points with precipitation
    nb: tuple[str, ...]
    nb_wind: tuple[str, ...]
    nb_hydro: tuple[str, ...]
    licence: str = "CC BY 4.0 (SMARD)"
    tier: int = 1
    extra: dict = field(default_factory=dict)


ZONES = {
    "DE-LU": ZoneCfg(
        "DE-LU", "Europe/Berlin", "DE",
        points=("north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt",
                "nrw", "hessen", "bawue", "bayern"),
        wind=("north_sea", "baltic", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt"),
        solar=("sachsen_anhalt", "nrw", "hessen", "bawue", "bayern", "uckermark"),
        hydro=(),
        nb=("nb_nl", "nb_be", "nb_fr_north", "nb_fr_south", "nb_dk_west", "nb_pl_north",
            "nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_south", "nb_se_north", "nb_fi"),
        nb_wind=("nb_nl", "nb_be", "nb_dk_west", "nb_pl_north", "nb_se_south"),
        nb_hydro=("nb_alps_ch", "nb_alps_at", "nb_no_south", "nb_se_north"),
    ),
    "NL": ZoneCfg(
        "NL", "Europe/Amsterdam", "NL",
        points=("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen", "nl_randstad",
                "nl_brabant"),
        wind=("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen"),
        solar=("nl_flevoland", "nl_groningen", "nl_randstad", "nl_brabant"),
        hydro=(),
        nb=("ostfriesland", "nrw", "nb_be", "nb_fr_north", "nb_dk_west", "schleswig", "nb_no_south"),
        nb_wind=("ostfriesland", "nb_be", "nb_dk_west", "schleswig"),
        nb_hydro=("nb_no_south",),
    ),
    "DK1": ZoneCfg(
        "DK1", "Europe/Copenhagen", "DK",
        points=("nb_dk_west", "dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
        wind=("nb_dk_west", "dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
        solar=("dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland"),
        hydro=(),
        nb=("schleswig", "north_sea", "ostfriesland", "nb_nl", "nb_no_south", "no2_sirdal",
            "se3_gothenburg", "dk2_copenhagen"),
        nb_wind=("schleswig", "north_sea", "ostfriesland", "nb_nl", "dk2_copenhagen"),
        nb_hydro=("nb_no_south", "no2_sirdal"),
    ),
    "DK2": ZoneCfg(
        "DK2", "Europe/Copenhagen", "DK",
        points=("dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm"),
        wind=("dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm"),
        solar=("dk2_copenhagen", "dk2_lolland"),
        hydro=(),
        nb=("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus", "se3_gothenburg", "nb_no_south",
            "nb_se_north"),
        nb_wind=("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus"),
        nb_hydro=("nb_no_south", "nb_se_north"),
    ),
    "SE4": ZoneCfg(
        "SE4", "Europe/Stockholm", "SE",
        points=("se4_malmo", "nb_se_south", "se4_kalmar", "se4_halmstad"),
        wind=("se4_malmo", "nb_se_south", "se4_kalmar", "se4_halmstad"),
        solar=("se4_malmo", "se4_kalmar"),
        hydro=(),
        nb=("dk2_copenhagen", "baltic", "nb_pl_north", "se3_gothenburg", "se3_stockholm",
            "nb_se_north", "dk1_aarhus"),
        nb_wind=("dk2_copenhagen", "baltic", "nb_pl_north", "se3_gothenburg", "dk1_aarhus"),
        nb_hydro=("nb_se_north",),
    ),
    "NO2": ZoneCfg(
        "NO2", "Europe/Oslo", "NO",
        points=("nb_no_south", "no2_kristiansand", "no2_stavanger", "no2_sirdal"),
        wind=("no2_kristiansand", "no2_stavanger"),
        solar=(),
        hydro=("nb_no_south", "no2_sirdal", "no2_stavanger"),
        nb=("nb_dk_west", "north_sea", "schleswig", "nb_nl", "no5_bergen", "no5_sogn", "no1_oslo",
            "no1_hallingdal"),
        nb_wind=("nb_dk_west", "north_sea", "schleswig", "nb_nl"),
        nb_hydro=("no5_sogn", "no1_hallingdal", "no5_bergen"),
    ),
    "SE3": ZoneCfg(
        "SE3", "Europe/Stockholm", "SE",
        points=("se3_stockholm", "se3_gothenburg", "se3_vasteras", "se3_varmland", "se3_ostergotland"),
        wind=("se3_gothenburg", "se3_vasteras", "se3_varmland", "se3_ostergotland"),
        solar=(),
        hydro=("se3_varmland",),
        nb=("nb_se_north", "se2_sundsvall", "se1_hydro", "no1_oslo", "no1_hallingdal",
            "fi_helsinki", "se4_malmo", "dk1_aarhus", "nb_fi"),
        nb_wind=("se2_sundsvall", "se4_malmo", "dk1_aarhus", "nb_fi"),
        nb_hydro=("nb_se_north", "se1_hydro", "no1_hallingdal"),
        licence="private/internal only", tier=2,
    ),
    "NO1": ZoneCfg(
        "NO1", "Europe/Oslo", "NO",
        points=("no1_oslo", "no1_innlandet", "no1_hallingdal", "no1_ostfold"),
        wind=("no1_ostfold", "no1_innlandet"),
        solar=(),
        hydro=("no1_innlandet", "no1_hallingdal"),
        nb=("se3_varmland", "se3_gothenburg", "nb_no_south", "no5_sogn", "no3_trondheim",
            "se3_stockholm"),
        nb_wind=("se3_varmland", "se3_gothenburg"),
        nb_hydro=("nb_no_south", "no5_sogn", "no3_trondheim"),
        licence="private/internal only", tier=2,
    ),
    "FI": ZoneCfg(
        "FI", "Europe/Helsinki", "FI",
        points=("fi_helsinki", "nb_fi", "fi_oulu", "fi_pori", "fi_vaasa", "fi_kemi"),
        wind=("fi_oulu", "fi_pori", "fi_vaasa", "fi_kemi"),
        solar=(),
        hydro=("fi_kemi", "nb_fi"),
        nb=("se1_lulea", "se1_hydro", "nb_se_north", "se3_stockholm", "ee_tallinn"),
        nb_wind=("se1_lulea", "nb_se_north", "ee_tallinn"),
        nb_hydro=("se1_hydro", "nb_se_north"),
        licence="private/internal only", tier=2,
    ),
    "SE1": ZoneCfg(
        "SE1", "Europe/Stockholm", "SE",
        points=("se1_lulea", "se1_hydro", "se1_kiruna"),
        wind=("se1_lulea", "se1_kiruna"),
        solar=(),
        hydro=("se1_hydro", "se1_kiruna"),
        nb=("fi_oulu", "fi_kemi", "nb_se_north", "no4_narvik", "se2_sundsvall"),
        nb_wind=("fi_oulu", "se2_sundsvall"),
        nb_hydro=("fi_kemi", "nb_se_north", "no4_narvik"),
        licence="private/internal only", tier=3,
    ),
    "SE2": ZoneCfg(
        "SE2", "Europe/Stockholm", "SE",
        points=("nb_se_north", "se2_sundsvall", "se2_umea", "se2_storuman"),
        wind=("nb_se_north", "se2_sundsvall", "se2_umea"),
        solar=(),
        hydro=("nb_se_north", "se2_storuman"),
        nb=("se1_lulea", "se1_hydro", "se3_vasteras", "se3_stockholm", "no3_trondheim", "fi_vaasa"),
        nb_wind=("se1_lulea", "fi_vaasa"),
        nb_hydro=("se1_hydro", "no3_trondheim"),
        licence="private/internal only", tier=3,
    ),
    "NO3": ZoneCfg(
        "NO3", "Europe/Oslo", "NO",
        points=("no3_trondheim", "no3_fosen", "no3_more"),
        wind=("no3_fosen", "no3_trondheim"),
        solar=(),
        hydro=("no3_more", "no3_trondheim"),
        nb=("nb_se_north", "no5_sogn", "no1_innlandet", "no4_bodo"),
        nb_wind=("nb_se_north",),
        nb_hydro=("no5_sogn", "no1_innlandet", "no4_bodo"),
        licence="private/internal only", tier=3,
    ),
    "NO4": ZoneCfg(
        "NO4", "Europe/Oslo", "NO",
        points=("no4_tromso", "no4_bodo", "no4_narvik"),
        wind=("no4_tromso", "no4_bodo"),
        solar=(),
        hydro=("no4_narvik", "no4_bodo"),
        nb=("se1_hydro", "se1_kiruna", "no3_fosen", "fi_kemi"),
        nb_wind=("no3_fosen",),
        nb_hydro=("se1_hydro", "se1_kiruna", "fi_kemi"),
        licence="private/internal only", tier=3,
    ),
    "NO5": ZoneCfg(
        "NO5", "Europe/Oslo", "NO",
        points=("no5_bergen", "no5_sogn", "no5_hardanger"),
        wind=("no5_bergen",),
        solar=(),
        hydro=("no5_sogn", "no5_hardanger", "no5_bergen"),
        nb=("nb_no_south", "no2_stavanger", "no1_hallingdal", "no3_more"),
        nb_wind=("no2_stavanger",),
        nb_hydro=("nb_no_south", "no1_hallingdal", "no3_more"),
        licence="private/internal only", tier=3,
    ),
}


def price_dir(zone: str) -> Path:
    p = RESEARCH_RAW / "prices" / zone
    return p if p.exists() else DATA / "prices" / zone


def weather_dir(point: str) -> Path:
    return RESEARCH_RAW / "weather" if point in RESEARCH_POINTS else DATA / "weather"
