"""Training pipeline. Puts the runtime core on sys.path so ``import core`` works in the CLIs."""

from pathlib import Path
import sys

_CORE_PARENT = str(Path(__file__).resolve().parent.parent / "custom_components" / "spot_forecast")
if _CORE_PARENT not in sys.path:
    sys.path.insert(0, _CORE_PARENT)
