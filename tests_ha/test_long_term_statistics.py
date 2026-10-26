"""Statistic metadata carries exactly the keys the running Home Assistant declares."""

from __future__ import annotations

from enum import Enum
from typing import NotRequired, TypedDict

from custom_components.spot_forecast.long_term_statistics import (
    StatisticMetaData,
    build_metadata,
    declared_keys,
)

STAT_ID = "spot_forecast:de_lu_forecast_h1"
BASE = {
    "has_sum": False,
    "name": "Spot Forecast DE-LU forecast D+1",
    "source": "spot_forecast",
    "statistic_id": STAT_ID,
    "unit_of_measurement": "EUR/kWh",
}


class MeanType(Enum):
    NONE = 0
    ARITHMETIC = 1


class _Base(TypedDict):
    has_sum: bool
    name: str | None
    source: str
    statistic_id: str
    unit_of_measurement: str | None


class Meta2024_11(_Base):  # named after the HA release
    has_mean: bool


class Meta2025_4(_Base):
    has_mean: NotRequired[bool]
    mean_type: MeanType


class MetaFuture(_Base):
    mean_type: MeanType
    unit_class: str | None


def test_2024_11_shape_has_mean_only() -> None:
    assert build_metadata(STAT_ID, Meta2024_11, None) == {**BASE, "has_mean": True}


def test_2024_11_shape_ignores_mean_type_enum_it_does_not_declare() -> None:
    assert build_metadata(STAT_ID, Meta2024_11, MeanType) == {**BASE, "has_mean": True}


def test_2025_4_shape_mean_type_replaces_the_deprecated_has_mean() -> None:
    assert build_metadata(STAT_ID, Meta2025_4, MeanType) == {
        **BASE,
        "mean_type": MeanType.ARITHMETIC,
    }


def test_2025_4_shape_without_the_enum_falls_back_to_has_mean() -> None:
    assert build_metadata(STAT_ID, Meta2025_4, None) == {**BASE, "has_mean": True}


def test_future_shape_without_has_mean() -> None:
    assert build_metadata(STAT_ID, MetaFuture, MeanType) == {
        **BASE,
        "mean_type": MeanType.ARITHMETIC,
        "unit_class": None,
    }


def test_inherited_and_not_required_keys_count_as_declared() -> None:
    assert {"has_sum", "has_mean", "mean_type"} <= declared_keys(Meta2025_4)


def test_running_ha_gets_only_declared_keys() -> None:
    meta = build_metadata(STAT_ID)
    assert set(meta) <= declared_keys(StatisticMetaData)
    assert set(BASE) <= set(meta)
