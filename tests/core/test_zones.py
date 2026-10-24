import pytest

from core.zones import DE_LU, DK1, DK2, NL, NO2, POINTS, ZONES, point_vars, zone_by_slug

DE_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
NB_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
S_VARS = ("wind_speed_100m", "temperature_2m", "cloud_cover")
N_VARS = ("wind_speed_100m", "temperature_2m", "precipitation")


def test_de_lu_matches_research():
    assert list(DE_LU.points) == [
        "north_sea", "baltic", "schleswig", "ostfriesland", "uckermark",
        "sachsen_anhalt", "nrw", "hessen", "bawue", "bayern",
    ]  # fmt: skip
    assert DE_LU.points["bayern"] == (48.4, 11.8)
    assert len(DE_LU.neighbor_points) == 12
    assert DE_LU.weather_vars == DE_VARS
    assert DE_LU.neighbor_vars == NB_VARS


def test_de_lu_point_sets_are_its_points_then_its_neighbours():
    """The two Open-Meteo requests of DE-LU, as before the point catalogue."""
    assert DE_LU.point_sets() == [
        (DE_LU.points, DE_VARS),
        (DE_LU.neighbor_points, NB_VARS),
    ]
    first, second = DE_LU.point_sets()
    assert list(first[0]) == list(DE_LU.own) and list(second[0]) == list(DE_LU.nb)
    assert second[0]["nb_fi"] == (61.0, 25.0)


def test_nl_point_sets_group_by_variable_set():
    """NL mixes points of three variable sets: three requests, first-appearance order."""
    sets = NL.point_sets()
    assert [(list(points), variables) for points, variables in sets] == [
        (["nb_nl", "nb_be", "nb_fr_north", "nb_dk_west", "nb_no_south"], NB_VARS),
        (["nl_borssele", "nl_flevoland", "nl_groningen", "nl_randstad", "nl_brabant"], S_VARS),
        (["north_sea", "ostfriesland", "nrw", "schleswig"], DE_VARS),
    ]
    assert sets[1][0]["nl_borssele"] == (51.70, 3.00)


def test_every_point_has_one_variable_set():
    # a dict cannot hold a point twice; every point's set is one of the four known ones
    assert {v for _, v in POINTS.values()} == {DE_VARS, NB_VARS, S_VARS, N_VARS}
    assert len({coords for coords, _ in POINTS.values()}) == len(POINTS)
    for name in DE_LU.own:
        assert point_vars(name) == DE_VARS
    for name in DE_LU.nb:
        assert point_vars(name) == NB_VARS


@pytest.mark.parametrize("zone", list(ZONES.values()), ids=lambda z: z.code)
def test_roles_reference_catalogue_points(zone):
    for role in ("own", "nb", "wind", "solar", "hydro", "nb_wind", "nb_hydro", "nb_nordic"):
        assert set(getattr(zone, role)) <= set(POINTS), role
    assert set(zone.wind) | set(zone.solar) | set(zone.hydro) <= set(zone.own)
    assert set(zone.nb_wind) | set(zone.nb_hydro) | set(zone.nb_nordic) <= set(zone.nb)
    assert not set(zone.own) & set(zone.nb)
    assert set(zone.raw_extra) <= set(POINTS) and not set(zone.raw_extra) & set(zone.own)
    if zone.de_extra:  # its aggregates read the raw series of every DE-LU own point
        assert set(DE_LU.own) <= set(zone.own + zone.raw_extra)
    assert set(zone.price_inputs) <= set(ZONES) - {zone.code}
    assert zone.parent is None or zone.parent in zone.price_inputs


def test_nl_matches_research():
    """research/zones/zonecfg.py ZONES["NL"], verbatim."""
    assert NL.own == ("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen",
                      "nl_randstad", "nl_brabant")  # fmt: skip
    assert NL.wind == ("nb_nl", "nl_borssele", "north_sea", "nl_flevoland", "nl_groningen")
    assert NL.solar == ("nl_flevoland", "nl_groningen", "nl_randstad", "nl_brabant")
    assert NL.hydro == ()
    assert NL.nb == ("ostfriesland", "nrw", "nb_be", "nb_fr_north", "nb_dk_west", "schleswig",
                     "nb_no_south")  # fmt: skip
    assert NL.nb_wind == ("ostfriesland", "nb_be", "nb_dk_west", "schleswig")
    assert NL.nb_hydro == ("nb_no_south",)
    assert (NL.tz, NL.country, NL.layout) == ("Europe/Amsterdam", "NL", "generic")


def test_zone_registry_and_slug():
    assert ZONES == {"DE-LU": DE_LU, "NL": NL, "DK1": DK1, "DK2": DK2, "NO2": NO2}
    assert next(iter(ZONES)) == "DE-LU"  # the config flow's default
    assert (DE_LU.slug, NL.slug, DK1.slug, DK2.slug) == ("de_lu", "nl", "dk1", "dk2")
    assert zone_by_slug("nl") is NL and zone_by_slug("de_lu") is DE_LU
    assert zone_by_slug("dk1") is DK1 and zone_by_slug("dk2") is DK2
    assert NO2.slug == "no2" and zone_by_slug("no2") is NO2
    with pytest.raises(KeyError):
        zone_by_slug("at")


def test_names_and_vat_defaults():
    assert (DE_LU.name, DE_LU.default_vat, DE_LU.country, DE_LU.layout) == (
        "Germany and Luxembourg", 19.0, "DE", "de_lu",
    )  # fmt: skip
    assert (NL.name, NL.default_vat) == ("Netherlands", 21.0)
    assert (DK1.name, DK1.default_vat, DK1.country) == ("Denmark West", 25.0, "DK")
    assert (DK2.name, DK2.default_vat, DK2.country) == ("Denmark East", 25.0, "DK")
    assert (NO2.name, NO2.default_vat, NO2.country) == ("Southern Norway", 25.0, "NO")


def test_dk1_matches_research():
    """research/zones/zonecfg.py ZONES["DK1"] verbatim, and bt.py variant "desetp" inputs."""
    assert DK1.own == ("nb_dk_west", "dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus",
                       "dk1_south_jutland")  # fmt: skip
    assert DK1.wind == DK1.own
    assert DK1.solar == ("dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland")
    assert DK1.hydro == ()
    assert DK1.nb == ("schleswig", "north_sea", "ostfriesland", "nb_nl", "nb_no_south",
                      "no2_sirdal", "se3_gothenburg", "dk2_copenhagen")  # fmt: skip
    assert DK1.nb_wind == ("schleswig", "north_sea", "ostfriesland", "nb_nl", "dk2_copenhagen")
    assert DK1.nb_hydro == ("nb_no_south", "no2_sirdal")
    assert (DK1.tz, DK1.layout) == ("Europe/Copenhagen", "generic")


def test_dk2_matches_research():
    assert DK2.own == ("dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm")
    assert DK2.wind == DK2.own
    assert DK2.solar == ("dk2_copenhagen", "dk2_lolland")
    assert DK2.hydro == ()
    assert DK2.nb == ("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus", "se3_gothenburg",
                      "nb_no_south", "nb_se_north")  # fmt: skip
    assert DK2.nb_wind == ("baltic", "nb_se_south", "se4_malmo", "dk1_aarhus")
    assert DK2.nb_hydro == ("nb_no_south", "nb_se_north")
    assert (DK2.tz, DK2.layout) == ("Europe/Copenhagen", "generic")


@pytest.mark.parametrize("zone", [DK1, DK2], ids=lambda z: z.code)
def test_dk_inputs_and_parent(zone):
    # bt.py: the DE-LU points that are not own points, in DE-LU order
    assert zone.raw_extra == tuple(p for p in DE_LU.own if p not in zone.own)
    assert zone.de_extra
    assert (zone.price_inputs, zone.parent, zone.blend) == (("DE-LU",), "DE-LU", 0.5)
    for z in (DE_LU, NL):
        assert (z.raw_extra, z.de_extra, z.price_inputs, z.parent) == ((), False, (), None)


def test_new_dk_points_match_research():
    """research/zones/zonecfg.py NEW_POINTS: coordinates and variable sets."""
    expected = {
        "dk1_horns_rev": ((55.50, 7.80), S_VARS),
        "dk1_north_jutland": ((57.20, 9.90), S_VARS),
        "dk1_aarhus": ((56.20, 10.10), S_VARS),
        "dk1_south_jutland": ((55.30, 9.20), S_VARS),
        "dk2_copenhagen": ((55.70, 12.40), S_VARS),
        "dk2_kriegers_flak": ((55.00, 12.90), S_VARS),
        "dk2_lolland": ((54.80, 11.40), S_VARS),
        "dk2_bornholm": ((55.10, 14.90), S_VARS),
        "no2_sirdal": ((58.90, 6.90), N_VARS),
        "se3_gothenburg": ((57.70, 12.00), N_VARS),
        "se4_malmo": ((55.60, 13.10), S_VARS),
    }
    assert {p: POINTS[p] for p in expected} == expected


def _groups(zone):
    return [(list(points), variables) for points, variables in zone.point_sets()]


def test_dk1_point_sets_cover_its_inputs_and_the_parent():
    """Own, neighbour and DE-LU raw points, then every DE-LU point (its model runs too)."""
    assert _groups(DK1) == [
        (["nb_dk_west", "nb_nl", "nb_no_south", "nb_be", "nb_fr_north", "nb_fr_south",
          "nb_pl_north", "nb_alps_ch", "nb_alps_at", "nb_se_south", "nb_se_north", "nb_fi"],
         NB_VARS),
        (["dk1_horns_rev", "dk1_north_jutland", "dk1_aarhus", "dk1_south_jutland",
          "dk2_copenhagen"], S_VARS),
        (["schleswig", "north_sea", "ostfriesland", "baltic", "uckermark", "sachsen_anhalt",
          "nrw", "hessen", "bawue", "bayern"], DE_VARS),
        (["no2_sirdal", "se3_gothenburg"], N_VARS),
    ]  # fmt: skip


def test_dk2_point_sets_cover_its_inputs_and_the_parent():
    assert _groups(DK2) == [
        (["dk2_copenhagen", "dk2_kriegers_flak", "dk2_lolland", "dk2_bornholm", "se4_malmo",
          "dk1_aarhus"], S_VARS),
        (["baltic", "north_sea", "schleswig", "ostfriesland", "uckermark", "sachsen_anhalt",
          "nrw", "hessen", "bawue", "bayern"], DE_VARS),
        (["nb_se_south", "nb_no_south", "nb_se_north", "nb_nl", "nb_be", "nb_fr_north",
          "nb_fr_south", "nb_dk_west", "nb_pl_north", "nb_alps_ch", "nb_alps_at", "nb_fi"],
         NB_VARS),
        (["se3_gothenburg"], N_VARS),
    ]  # fmt: skip


@pytest.mark.parametrize("zone", [DK1, DK2], ids=lambda z: z.code)
def test_dk_point_sets_hold_every_parent_point(zone):
    points = {p for group, _ in zone.point_sets() for p in group}
    assert set(DE_LU.own + DE_LU.nb) <= points
    assert points == set(zone.own + zone.nb + zone.raw_extra + DE_LU.own + DE_LU.nb)


@pytest.mark.parametrize("zone", list(ZONES.values()), ids=lambda z: z.code)
def test_price_attribution_names_the_publisher(zone):
    assert "SMARD" in zone.price_attribution and "CC BY 4.0" in zone.price_attribution


def test_no2_matches_research():
    """research/zones/zonecfg.py ZONES["NO2"] verbatim, and bt.py variant "hydrel"."""
    assert NO2.own == ("nb_no_south", "no2_kristiansand", "no2_stavanger", "no2_sirdal")
    assert NO2.wind == ("no2_kristiansand", "no2_stavanger")
    assert NO2.solar == ()
    assert NO2.hydro == ("nb_no_south", "no2_sirdal", "no2_stavanger")
    assert NO2.nb == ("nb_dk_west", "north_sea", "schleswig", "nb_nl", "no5_bergen", "no5_sogn",
                      "no1_oslo", "no1_hallingdal")  # fmt: skip
    assert NO2.nb_wind == ("nb_dk_west", "north_sea", "schleswig", "nb_nl")
    assert NO2.nb_hydro == ("no5_sogn", "no1_hallingdal", "no5_bergen")
    assert (NO2.tz, NO2.layout) == ("Europe/Oslo", "generic")
    assert (NO2.hydro_extra, NO2.doy) == (True, False)
    assert (NO2.raw_extra, NO2.de_extra, NO2.price_inputs, NO2.parent) == ((), False, (), None)


def test_hydro_points_are_the_hydro_then_the_neighbour_hydro_points():
    """genfeat ``_hydro_extra``: ``dict.fromkeys(cfg.hydro + cfg.nb_hydro)``."""
    assert NO2.hydro_points == ("nb_no_south", "no2_sirdal", "no2_stavanger", "no5_sogn",
                                "no1_hallingdal", "no5_bergen")  # fmt: skip
    assert DK1.hydro_points == ("nb_no_south", "no2_sirdal")


@pytest.mark.parametrize("zone", [DE_LU, NL, DK1, DK2], ids=lambda z: z.code)
def test_other_zones_keep_doy_and_no_hydro_features(zone):
    assert (zone.hydro_extra, zone.doy) == (False, True)


def test_new_no2_points_match_research():
    """research/zones/zonecfg.py NEW_POINTS: coordinates and variable sets."""
    expected = {
        "no2_kristiansand": ((58.20, 8.00), N_VARS),
        "no2_stavanger": ((58.90, 5.80), N_VARS),
        "no2_sirdal": ((58.90, 6.90), N_VARS),
        "no5_sogn": ((61.20, 7.00), N_VARS),
        "no1_hallingdal": ((60.60, 9.00), N_VARS),
        "no5_bergen": ((60.40, 5.30), N_VARS),
        "no1_oslo": ((59.90, 10.80), N_VARS),
    }
    assert {p: POINTS[p] for p in expected} == expected


def test_no2_point_sets_group_by_variable_set():
    """Own and neighbour points only: NO2 has no parent and no raw extra points."""
    assert _groups(NO2) == [
        (["nb_no_south", "nb_dk_west", "nb_nl"], NB_VARS),
        (["no2_kristiansand", "no2_stavanger", "no2_sirdal", "no5_bergen", "no5_sogn",
          "no1_oslo", "no1_hallingdal"], N_VARS),
        (["north_sea", "schleswig"], DE_VARS),
    ]  # fmt: skip
