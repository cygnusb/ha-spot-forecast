"""External long-term statistics: actual prices and the forecasts made for them."""

from __future__ import annotations

from itertools import groupby
from operator import attrgetter
import re
from typing import Any

from homeassistant.components.recorder.statistics import async_add_external_statistics
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


def _metadata(statistic_id: str) -> StatisticMetaData:
    meta: dict[str, Any] = {
        "has_mean": True,
        "has_sum": False,
        "name": statistic_name(statistic_id),
        "source": DOMAIN,
        "statistic_id": statistic_id,
        "unit_of_measurement": UNIT,
    }
    if StatisticMeanType is not None:
        meta["mean_type"] = StatisticMeanType.ARITHMETIC
        meta["unit_class"] = None
    return meta  # type: ignore[return-value]


@callback
def async_write(hass: HomeAssistant, rows: list[StatRow]) -> None:
    """Queue ``rows`` with the recorder, one import per statistic id."""
    by_id = attrgetter("statistic_id")
    for statistic_id, group in groupby(sorted(rows, key=by_id), key=by_id):
        stats = [
            StatisticData(start=r.start, mean=r.value, min=r.value, max=r.value, state=r.value)
            for r in sorted(group, key=attrgetter("start"))
        ]
        async_add_external_statistics(hass, _metadata(statistic_id), stats)
