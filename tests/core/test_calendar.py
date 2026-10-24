from datetime import date

import holidays as holidays_package
import pytest

from core.calendar import de_holidays, holidays


@pytest.mark.parametrize("year", range(2020, 2041))
def test_de_matches_holidays_package(year):
    assert de_holidays(year) == frozenset(holidays_package.Germany(years=year).keys())
    assert holidays("DE", year) == de_holidays(year)


@pytest.mark.parametrize("year", range(2020, 2041))
def test_nl_matches_holidays_package(year):
    expected = frozenset(holidays_package.country_holidays("NL", years=year).keys())
    assert holidays("NL", year) == expected


def test_nl_edge_cases():
    assert date(2025, 4, 26) in holidays("NL", 2025)  # the 27th is a Sunday
    assert date(2025, 4, 27) not in holidays("NL", 2025)
    assert date(2026, 4, 27) in holidays("NL", 2026)
    assert date(2025, 5, 5) in holidays("NL", 2025)  # Bevrijdingsdag, lustrum year
    assert date(2026, 5, 5) not in holidays("NL", 2026)
    assert date(2030, 5, 5) in holidays("NL", 2030)
    assert date(2026, 4, 5) in holidays("NL", 2026)  # Eerste paasdag, a Sunday
    assert date(2026, 5, 24) in holidays("NL", 2026)  # Eerste pinksterdag, a Sunday


@pytest.mark.parametrize("year", range(2020, 2041))
def test_dk_matches_holidays_package_plus_the_eves(year):
    """research/zones/genfeat.py ``offdays("DK", year)``."""
    package = set(holidays_package.country_holidays("DK", years=year).keys())
    assert holidays("DK", year) == frozenset(package | {date(year, 12, 24), date(year, 12, 31)})


def test_dk_edge_cases():
    assert date(2023, 5, 5) in holidays("DK", 2023)  # Store bededag, the last one
    assert date(2024, 4, 26) not in holidays("DK", 2024)  # abolished from 2024
    assert date(2026, 4, 2) in holidays("DK", 2026)  # Skærtorsdag
    assert date(2026, 12, 24) in holidays("DK", 2026) and date(2026, 12, 31) in holidays("DK", 2026)
    assert date(2026, 6, 5) not in holidays("DK", 2026)  # Grundlovsdag: not in the package


def test_unknown_country_raises():
    with pytest.raises(KeyError):
        holidays("AT", 2026)
