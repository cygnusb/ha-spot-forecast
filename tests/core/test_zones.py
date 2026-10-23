import pytest

from core.zones import DE_LU, NL, POINTS, ZONES, point_vars, zone_by_slug

DE_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover")
NB_VARS = ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
S_VARS = ("wind_speed_100m", "temperature_2m", "cloud_cover")


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
    # a dict cannot hold a point twice; every point's set is one of the three known ones
    assert {v for _, v in POINTS.values()} == {DE_VARS, NB_VARS, S_VARS}
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
    assert ZONES == {"DE-LU": DE_LU, "NL": NL}
    assert next(iter(ZONES)) == "DE-LU"  # the config flow's default
    assert (DE_LU.slug, NL.slug) == ("de_lu", "nl")
    assert zone_by_slug("nl") is NL and zone_by_slug("de_lu") is DE_LU
    with pytest.raises(KeyError):
        zone_by_slug("at")


def test_names_and_vat_defaults():
    assert (DE_LU.name, DE_LU.default_vat, DE_LU.country, DE_LU.layout) == (
        "Germany and Luxembourg", 19.0, "DE", "de_lu",
    )  # fmt: skip
    assert (NL.name, NL.default_vat) == ("Netherlands", 21.0)


@pytest.mark.parametrize("zone", list(ZONES.values()), ids=lambda z: z.code)
def test_price_attribution_names_the_publisher(zone):
    assert "SMARD" in zone.price_attribution and "CC BY 4.0" in zone.price_attribution
