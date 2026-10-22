"""German nationwide public holidays (no regional ones)."""

from __future__ import annotations

from datetime import date, timedelta


def _easter(year: int) -> date:
    """Easter Sunday, anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def de_holidays(year: int) -> frozenset[date]:
    """Nationwide German holidays of ``year``."""
    easter = _easter(year)
    return frozenset(
        {
            date(year, 1, 1),
            easter - timedelta(days=2),  # Karfreitag
            easter + timedelta(days=1),  # Ostermontag
            date(year, 5, 1),
            easter + timedelta(days=39),  # Christi Himmelfahrt
            easter + timedelta(days=50),  # Pfingstmontag
            date(year, 10, 3),
            date(year, 12, 25),
            date(year, 12, 26),
        }
    )
