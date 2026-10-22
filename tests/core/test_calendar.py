import pytest

from core.calendar import de_holidays

holidays = pytest.importorskip("holidays")


@pytest.mark.parametrize("year", range(2023, 2031))
def test_matches_holidays_package(year):
    assert de_holidays(year) == frozenset(holidays.Germany(years=year).keys())
