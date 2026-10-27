"""Descarga de datos abiertos de nflverse (https://github.com/nflverse/nflverse-data).

Licencias: los datos de nflverse se publican bajo CC-BY 4.0; los datos de
charting de FTN bajo CC-BY-SA 4.0. Ver README para atribución.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ..config import HTTP_TIMEOUT, NFLVERSE_RELEASES, RAW_DIR, USER_AGENT

# Abreviaturas históricas -> franquicia actual (para continuidad de ratings).
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}

# Columnas de mercado de apuestas: se eliminan al ingerir para garantizar que
# nunca entren al modelo.
BETTING_COLUMNS = [
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
]


def _download(url: str, dest: Path, max_age_hours: float | None) -> Path:
    """Descarga con caché local. max_age_hours=None: usar la copia local si existe."""
    if dest.exists():
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            return dest
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            return dest
        except requests.RequestException as exc:  # reintento con espera exponencial
            last_exc = exc
            time.sleep(2 ** (attempt + 1))
    if dest.exists():  # se usa la copia previa si la descarga falla
        return dest
    raise RuntimeError(f"No se pudo descargar {url}: {last_exc}")


def fix_team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_FIX)


def schedules(max_age_hours: float | None = 3) -> pd.DataFrame:
    path = _download(f"{NFLVERSE_RELEASES}/schedules/games.csv", RAW_DIR / "games.csv", max_age_hours)
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in BETTING_COLUMNS if c in df.columns])
    for col in ("home_team", "away_team"):
        df[col] = fix_team(df[col])
    return df


def pbp(season: int, max_age_hours: float | None) -> pd.DataFrame:
    path = _download(
        f"{NFLVERSE_RELEASES}/pbp/play_by_play_{season}.csv.gz", RAW_DIR / f"pbp_{season}.csv.gz", max_age_hours
    )
    cols = [
        "game_id", "season", "season_type", "week", "game_date", "posteam", "defteam", "home_team", "away_team",
        "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "sack", "qb_hit",
        "interception", "fumble_lost", "epa", "qb_epa", "success", "yards_gained", "down", "yardline_100",
        "third_down_converted", "third_down_failed", "fixed_drive", "fixed_drive_result", "drive_inside20",
        "passer_player_id", "passer_player_name", "passer_id", "passer", "play_id", "touchdown", "td_team",
        "penalty", "two_point_attempt", "play",
    ]
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in cols)
    for col in ("posteam", "defteam", "home_team", "away_team", "td_team"):
        if col in df.columns:
            df[col] = fix_team(df[col])
    return df


def ftn_charting(season: int, max_age_hours: float | None) -> pd.DataFrame | None:
    if season < 2022:  # FTN publica charting abierto desde 2022
        return None
    try:
        path = _download(
            f"{NFLVERSE_RELEASES}/ftn_charting/ftn_charting_{season}.csv",
            RAW_DIR / f"ftn_{season}.csv",
            max_age_hours,
        )
    except RuntimeError:
        return None
    return pd.read_csv(path, low_memory=False, usecols=["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "is_play_action"])


def injuries(season: int, max_age_hours: float | None) -> pd.DataFrame | None:
    try:
        path = _download(
            f"{NFLVERSE_RELEASES}/injuries/injuries_{season}.csv", RAW_DIR / f"injuries_{season}.csv", max_age_hours
        )
    except RuntimeError:
        return None
    df = pd.read_csv(path, low_memory=False)
    df["team"] = fix_team(df["team"])
    return df


def snap_counts(season: int, max_age_hours: float | None) -> pd.DataFrame | None:
    try:
        path = _download(
            f"{NFLVERSE_RELEASES}/snap_counts/snap_counts_{season}.csv", RAW_DIR / f"snaps_{season}.csv", max_age_hours
        )
    except RuntimeError:
        return None
    df = pd.read_csv(path, low_memory=False)
    df["team"] = fix_team(df["team"])
    df["opponent"] = fix_team(df["opponent"])
    return df


def weekly_rosters(season: int, max_age_hours: float | None) -> pd.DataFrame | None:
    try:
        path = _download(
            f"{NFLVERSE_RELEASES}/weekly_rosters/roster_weekly_{season}.csv",
            RAW_DIR / f"roster_{season}.csv",
            max_age_hours,
        )
    except RuntimeError:
        return None
    cols = ["season", "week", "team", "position", "depth_chart_position", "status", "full_name", "gsis_id",
            "espn_id", "pfr_id", "status_description_abbr", "game_type"]
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in cols)
    df["team"] = fix_team(df["team"])
    return df


def teams(max_age_hours: float | None = 24 * 7) -> pd.DataFrame:
    path = _download(f"{NFLVERSE_RELEASES}/teams/teams_colors_logos.csv", RAW_DIR / "teams.csv", max_age_hours)
    df = pd.read_csv(path)
    df = df[~df["team_abbr"].isin(["OAK", "SD", "STL", "LAR"])]
    return df
