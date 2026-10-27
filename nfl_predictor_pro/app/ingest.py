"""Actualización de datos: descarga, agregación y guardado en la base de datos."""
from __future__ import annotations

import logging

import pandas as pd

from . import db
from .config import CURRENT_SEASON, FIRST_SEASON
from .features import qb_game_stats, team_game_stats
from .sources import espn, nflverse

log = logging.getLogger(__name__)


def _season_stats(season: int, max_age: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = nflverse.pbp(season, max_age)
    ftn = nflverse.ftn_charting(season, max_age)
    return team_game_stats(pbp, ftn), qb_game_stats(pbp)


def update_all(seasons: list[int] | None = None, refresh_hours: float = 3) -> dict:
    """Actualiza todas las fuentes. Temporadas pasadas se descargan una sola vez."""
    report: dict[str, str] = {}
    seasons = seasons or list(range(FIRST_SEASON, CURRENT_SEASON + 1))

    sched = nflverse.schedules(refresh_hours)
    db.write_df(sched, "schedule")
    db.mark_source("nflverse_schedules", len(sched))
    report["calendario"] = f"{len(sched)} partidos"

    teams = nflverse.teams()
    db.write_df(teams, "teams")
    db.mark_source("nflverse_teams", len(teams))

    have = set()
    if db.table_exists("team_stats"):
        have = set(db.read_df("SELECT DISTINCT season FROM team_stats")["season"].astype(int))
    for season in seasons:
        is_current = season == CURRENT_SEASON
        if season in have and not is_current:
            continue
        try:
            stats, qbs = _season_stats(season, refresh_hours if is_current else None)
        except Exception as exc:  # una temporada sin datos no debe tumbar la actualización
            log.warning("pbp %s: %s", season, exc)
            report[f"pbp_{season}"] = f"error: {exc}"
            db.mark_source(f"nflverse_pbp_{season}", 0, "error", str(exc))
            continue
        stats["season"] = season
        db.write_df(stats, "team_stats", replace_where=("season", [season]))
        db.write_df(qbs, "qb_games", replace_where=("season", [season]))
        db.mark_source(f"nflverse_pbp_{season}", len(stats))
        report[f"pbp_{season}"] = f"{len(stats)} filas equipo-partido"

    # Datos de jugadores sólo para la temporada actual (lesiones / snaps / rosters).
    season = CURRENT_SEASON
    inj = nflverse.injuries(season, refresh_hours)
    if inj is not None:
        db.write_df(inj, "injuries_official")
        db.mark_source("nfl_injury_report", len(inj), message=f"Reporte oficial NFL (vía nflverse), hasta semana {int(inj['week'].max())}")
        report["lesiones_oficiales"] = f"{len(inj)} registros"
    else:
        db.mark_source("nfl_injury_report", 0, "error", "No disponible")
    snaps = nflverse.snap_counts(season, refresh_hours)
    if snaps is not None:
        db.write_df(snaps, "snaps")
        db.mark_source("nflverse_snap_counts", len(snaps))
    rost = nflverse.weekly_rosters(season, refresh_hours)
    if rost is not None:
        db.write_df(rost, "rosters")
        db.mark_source("nflverse_rosters", len(rost))
    try:
        ei = espn.injuries()
        db.write_df(ei, "injuries_espn")
        db.mark_source("espn_injuries", len(ei), message="Estado actual (ESPN)")
        report["lesiones_espn"] = f"{len(ei)} registros"
    except Exception as exc:
        db.mark_source("espn_injuries", 0, "error", str(exc))
        report["lesiones_espn"] = f"error: {exc}"
    db.kv_set("last_update", db.utcnow())
    return report
