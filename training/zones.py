"""The zones the monthly training covers, for the Train workflow.

python -m training.zones  ->  the zone slugs, one per line, in training order
"""

from __future__ import annotations

from core.zones import ZONES


def main() -> None:
    for zone in ZONES.values():
        print(zone.slug)


if __name__ == "__main__":
    main()
