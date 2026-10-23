"""External long-term statistics: actual prices and the forecasts made for them."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import groupby
from operator import attrgetter
import re
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
)
from homeassistant.core import HomeAssistant, callback

from .const import DOMAIN
from .core.ledger import StatRow
from .core.zones import ZONES

try:  # moved between recorder modules over the years (as in ha-netze-bw)
    from homeassistant.components.recorder.statistics import (
        StatisticData,
        StatisticMetaData,
    )
except ImportError:  # pragma: no cover - depends on the HA version
    from homeassistant.components.recorder.models import StatisticData, StatisticMetaData

try:  # HA 2025.4+; older versions only know has_mean
    from homeassistant.components.recorder.models import StatisticMeanType
except ImportError:  # pragma: no cover - depends on the HA version
    StatisticMeanType = None

UNIT = "EUR/kWh"
_HORIZON_SERIES = re.compile(r"(forecast|peak_risk|low)_h(\d+)")
_LABELS = {"forecast": "forecast", "peak_risk": "peak risk", "low": "low"}


def statistic_name(statistic_id: str) -> str:
    """Readable name, e.g. "Spot Forecast DE-LU forecast D+1" for ``…:de_lu_forecast_h1``."""
    object_id = statistic_id.partition(":")[2]
    # longest slug first, so a zone "de" could never claim "de_lu_price"
    for zone in sorted(ZONES.values(), key=lambda z: -len(z.slug)):
        series = object_id.removeprefix(f"{zone.slug}_")
        if series != object_id:
            break
    else:
        return object_id
    if match := _HORIZON_SERIES.fullmatch(series):
        series = f"{_LABELS[match[1]]} D+{match[2]}"
    return f"Spot Forecast {zone.code} {series.replace('_', ' ')}"


def declared_keys(typed_dict: type) -> frozenset[str]:
    """Keys a TypedDict declares, inherited and ``NotRequired`` ones included."""
    keys = getattr(typed_dict, "__required_keys__", frozenset()) | getattr(
        typed_dict, "__optional_keys__", frozenset()
    )
    if not keys:  # not a real TypedDict: walk the annotations of the class and its bases
        for cls in getattr(typed_dict, "__mro__", (typed_dict,)):
            keys |= set(getattr(cls, "__annotations__", {}))
    return frozenset(keys)


def build_metadata(
    statistic_id: str,
    meta_type: type = StatisticMetaData,
    mean_type: Any = StatisticMeanType,
) -> StatisticMetaData:
    """Metadata with exactly the optional keys the running HA declares.

    The recorder builds its ``StatisticsMeta`` row with ``StatisticsMeta(**meta)``, so a key
    the running version does not know (``unit_class`` before 2025.11, ``has_mean`` once it
    is removed) fails every import. ``mean_type`` exists from 2025.4 on.
    """
    declared = declared_keys(meta_type)
    meta: dict[str, Any] = {
        "has_sum": False,
        "name": statistic_name(statistic_id),
        "source": DOMAIN,
        "statistic_id": statistic_id,
        "unit_of_measurement": UNIT,
    }
    if "has_mean" in declared:
        meta["has_mean"] = True
    if "mean_type" in declared and mean_type is not None:
        meta["mean_type"] = mean_type.ARITHMETIC
    if "unit_class" in declared:
        meta["unit_class"] = None
    return meta  # type: ignore[return-value]


Batch = tuple[StatisticMetaData, list[StatisticData]]  # one import of one statistic id


def batches(rows: list[StatRow]) -> list[Batch]:
    """``rows`` grouped into one import per statistic id, sorted by start (blocking)."""
    by_id = attrgetter("statistic_id")
    out: list[Batch] = []
    for statistic_id, group in groupby(sorted(rows, key=by_id), key=by_id):
        stats = [
            StatisticData(start=r.start, mean=r.value, min=r.value, max=r.value, state=r.value)
            for r in sorted(group, key=attrgetter("start"))
        ]
        out.append((build_metadata(statistic_id), stats))
    return out


@callback
def async_write(hass: HomeAssistant, imports: list[Batch]) -> None:
    """Queue the ``batches`` with the recorder."""
    for metadata, stats in imports:
        async_add_external_statistics(hass, metadata, stats)


async def async_last_start(hass: HomeAssistant, statistic_id: str) -> datetime | None:
    """Start (UTC) of the newest row the recorder holds for ``statistic_id``; None if none."""
    stats = await get_instance(hass).async_add_executor_job(
        get_last_statistics, hass, 1, statistic_id, True, {"mean"}
    )
    rows = stats.get(statistic_id)
    return datetime.fromtimestamp(rows[0]["start"], UTC) if rows else None
