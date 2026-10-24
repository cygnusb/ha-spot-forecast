"""Exit 0 when all weather files and prices for a zone are present."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zonecfg import ZONES, weather_dir, price_dir
cfg = ZONES[sys.argv[1]]
months = [f"{y}-{m:02d}" for y in (2024, 2025, 2026) for m in range(1, 13) if "2024-03" <= f"{y}-{m:02d}" <= "2026-10"]
missing = [f"{p}_{mo}" for p in dict.fromkeys(cfg.points + cfg.nb) for mo in months if not (weather_dir(p) / f"{p}_{mo}.json").exists()]
pr = len(list(price_dir(cfg.code).glob("*.json"))) >= 6
sys.exit(0 if not missing and pr else 1)
