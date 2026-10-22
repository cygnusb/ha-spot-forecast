from core.zones import DE_LU


def test_de_lu_matches_research():
    assert list(DE_LU.points) == [
        "north_sea", "baltic", "schleswig", "ostfriesland", "uckermark",
        "sachsen_anhalt", "nrw", "hessen", "bawue", "bayern",
    ]  # fmt: skip
    assert DE_LU.points["bayern"] == (48.4, 11.8)
    assert len(DE_LU.neighbor_points) == 12
    assert DE_LU.weather_vars == (
        "wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover",
    )  # fmt: skip


def test_zone_registry_and_slug():
    from core.zones import ZONES

    assert ZONES == {"DE-LU": DE_LU}
    assert DE_LU.slug == "de_lu"
