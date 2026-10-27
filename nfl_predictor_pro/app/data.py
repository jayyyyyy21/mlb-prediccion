"""Carga de tablas desde la base de datos ya combinadas para el modelo."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import db
from .context import add_context
from .features import build_team_games
from .ratings import prepare


@lru_cache(maxsize=1)
def _load(version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = db.read_table("schedule")
    stats = db.read_table("team_stats").drop(columns=["season"], errors="ignore")
    sched = sched[sched["season"] >= int(stats_seasons_min())] if len(stats) else sched
    tg_all = add_context(build_team_games(sched, stats))
    tg = prepare(tg_all)
    return sched, tg_all, tg


def stats_seasons_min() -> int:
    df = db.read_df("SELECT MIN(season) AS s FROM team_stats")
    return int(df["s"].iloc[0])


def data_version() -> str:
    return str(db.kv_get("last_update", "none"))


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(calendario, equipo-partido con contexto [incluye futuros], equipo-partido jugados)."""
    return _load(data_version())


def clear_cache() -> None:
    _load.cache_clear()
