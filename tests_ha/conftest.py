"""Fixtures for the tests that run against a real Home Assistant.

Separate from `tests/`, which is pure Python and runs without Home Assistant. Never
collect both suites in one pytest run.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
import sys
from typing import Any

from homeassistant.core import HomeAssistant
import pytest

# The tests import `custom_components.spot_forecast` the way Home Assistant loads it,
# so the repo root has to be on sys.path; pytest only adds the test directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest_plugins = "pytest_homeassistant_custom_component"

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Silence the SQL echo the plugin turns on.

    It sets `sqlalchemy.engine` to INFO at import; the recorder thread then logs its
    last statements after pytest's capture of a test has ended, straight onto the
    terminal between the progress dots.
    """
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@pytest.fixture
def recorded() -> Callable[[str], Any]:
    """Loader of a recorded API payload from `tests_ha/fixtures` (see its README)."""
    return lambda name: json.loads((FIXTURES / name).read_text())


@pytest.fixture
def hass_with_recorder(recorder_mock, enable_custom_integrations, hass) -> HomeAssistant:
    """A Home Assistant with the recorder that can load this custom integration.

    The parameter order is load-bearing: `recorder_mock` refuses to run once `hass`
    exists, so it must be requested before anything that pulls `hass` in.
    """
    return hass
