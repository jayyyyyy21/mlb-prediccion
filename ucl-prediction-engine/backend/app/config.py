"""Configuración central. Todas las credenciales se leen del entorno del servidor
y nunca se envían al frontend."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _load_dotenv() -> None:
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL") or f"sqlite:///{DATA_DIR / 'ucl.db'}"
    )
    api_football_key: str = field(default_factory=lambda: os.environ.get("API_FOOTBALL_KEY", ""))
    admin_token: str = field(default_factory=lambda: os.environ.get("ADMIN_TOKEN", ""))
    refresh_interval_min: int = field(
        default_factory=lambda: int(os.environ.get("REFRESH_INTERVAL_MIN", "0") or 0)
    )
    ucl_season_start: int = field(default_factory=lambda: int(os.environ.get("UCL_SEASON_START", "2026")))
    history_from: str = field(default_factory=lambda: os.environ.get("HISTORY_FROM", "2024-08"))
    # Número de simulaciones Monte Carlo por partido
    mc_iterations: int = field(default_factory=lambda: int(os.environ.get("MC_ITERATIONS", "20000")))


settings = Settings()

UCL_SLUG = "uefa.champions"
# Competiciones europeas adicionales: conectan la fuerza relativa entre ligas.
EURO_SLUGS = ["uefa.champions", "uefa.europa", "uefa.europa.conf",
              "uefa.champions_qual", "uefa.europa_qual", "uefa.europa.conf_qual"]
# Competiciones que NO se usan para entrenar (amistosos)
EXCLUDED_FROM_MODEL = ("club.friendly",)
