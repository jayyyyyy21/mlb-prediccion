"""Configuración central de NFL Predictor Pro (variables de entorno opcionales)."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NFLPP_DATA_DIR", BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("NFLPP_DB_PATH", DATA_DIR / "nfl.db"))

# Primera temporada descargada. Se usan varias temporadas para calibrar el
# modelo con validación walk-forward (sin información posterior al partido).
FIRST_SEASON = int(os.environ.get("NFLPP_FIRST_SEASON", "2016"))


def current_season(today: date | None = None) -> int:
    """La temporada NFL empieza en septiembre; enero-julio pertenece a la anterior."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


CURRENT_SEASON = int(os.environ.get("NFLPP_SEASON", current_season()))

# Simulaciones Monte Carlo por partido (mínimo 10,000).
N_SIMS = max(10_000, int(os.environ.get("NFLPP_N_SIMS", "10000")))

# Escudos: los logos de la NFL son marcas registradas. Sólo se muestran si el
# operador declara tener autorización de uso.
SHOW_LOGOS = os.environ.get("NFLPP_SHOW_LOGOS", "false").lower() in {"1", "true", "yes", "si", "sí"}

# Pronóstico del clima (Open-Meteo, sin API key). Se puede desactivar.
WEATHER_ENABLED = os.environ.get("NFLPP_WEATHER", "true").lower() in {"1", "true", "yes", "si", "sí"}

HTTP_TIMEOUT = float(os.environ.get("NFLPP_HTTP_TIMEOUT", "60"))
USER_AGENT = "NFL-Predictor-Pro/1.0 (uso analitico personal)"

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
