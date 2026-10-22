"""Offline checks of the fetch helpers (no network)."""

from datetime import date

import pytest

pytest.importorskip("requests")
from training.fetch import _is_complete, month_chunks


def test_month_chunks_clip_to_range():
    assert list(month_chunks(date(2024, 3, 5), date(2024, 5, 10))) == [
        (date(2024, 3, 5), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 5, 1), date(2024, 5, 10)),
    ]
    assert list(month_chunks(date(2024, 1, 1), date(2024, 12, 31), months=6))[1] == (
        date(2024, 7, 1),
        date(2024, 12, 31),
    )


def test_recent_chunks_are_refetched(tmp_path):
    f = tmp_path / "x.json"
    assert not _is_complete(f, date(2024, 1, 1), date(2024, 6, 1))
    f.write_text("{}")
    assert _is_complete(f, date(2024, 1, 31), date(2024, 6, 1))
    assert not _is_complete(f, date(2024, 5, 30), date(2024, 6, 1))
