"""Construcción de estadísticas por equipo y partido a partir del play-by-play.

Cada fila de `team_games` describe la ofensiva de `team` contra la defensa de
`opp` en un partido. Las estadísticas defensivas de un equipo son las
ofensivas de sus rivales (se obtienen uniendo la fila espejo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OFF_PLAY_TYPES = {"pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"}


def team_game_stats(pbp: pd.DataFrame, ftn: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agrega el play-by-play a una fila por (partido, equipo ofensivo)."""
    p = pbp[pbp["posteam"].notna()].copy()
    for col in ("pass", "rush", "sack", "qb_hit", "interception", "fumble_lost", "success",
                "third_down_converted", "third_down_failed", "qb_dropback"):
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)

    plays = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["epa"].notna()].copy()
    plays["is_pass"] = (plays["pass"] == 1).astype(int)
    plays["is_rush"] = ((plays["rush"] == 1) & (plays["pass"] == 0)).astype(int)
    plays["explosive"] = (
        ((plays["is_pass"] == 1) & (plays["yards_gained"] >= 20)) | ((plays["is_rush"] == 1) & (plays["yards_gained"] >= 10))
    ).astype(int)
    plays["third_att"] = ((plays["down"] == 3)).astype(int)
    plays["third_conv"] = ((plays["down"] == 3) & (plays["third_down_converted"] == 1)).astype(int)
    plays["pass_epa"] = plays["epa"] * plays["is_pass"]
    plays["rush_epa"] = plays["epa"] * plays["is_rush"]
    plays["pass_succ"] = plays["success"] * plays["is_pass"]
    plays["rush_succ"] = plays["success"] * plays["is_rush"]
    plays["pass_yds"] = plays["yards_gained"].fillna(0) * plays["is_pass"]
    plays["rush_yds"] = plays["yards_gained"].fillna(0) * plays["is_rush"]

    g = plays.groupby(["game_id", "posteam"]).agg(
        plays=("epa", "size"),
        epa_sum=("epa", "sum"),
        succ_sum=("success", "sum"),
        db=("is_pass", "sum"),
        pass_epa_sum=("pass_epa", "sum"),
        pass_succ=("pass_succ", "sum"),
        rushes=("is_rush", "sum"),
        rush_epa_sum=("rush_epa", "sum"),
        rush_succ=("rush_succ", "sum"),
        pass_yds=("pass_yds", "sum"),
        rush_yds=("rush_yds", "sum"),
        sacks=("sack", "sum"),
        qb_hits=("qb_hit", "sum"),
        ints=("interception", "sum"),
        explosive=("explosive", "sum"),
        third_att=("third_att", "sum"),
        third_conv=("third_conv", "sum"),
    )
    # Balones perdidos: fumbles perdidos por la ofensiva en jugadas de scrimmage.
    fl = plays.groupby(["game_id", "posteam"])["fumble_lost"].sum().rename("fum_lost")
    g = g.join(fl)

    # Posesiones (drives) y resultados por drive.
    d = p[p["play_type"].isin(OFF_PLAY_TYPES) & p["fixed_drive"].notna()]
    drives = d.groupby(["game_id", "posteam", "fixed_drive"]).agg(
        result=("fixed_drive_result", "last"),
        inside20=("drive_inside20", "max"),
    ).reset_index()
    drives["inside20"] = pd.to_numeric(drives["inside20"], errors="coerce").fillna(0)
    drives["td"] = (drives["result"] == "Touchdown").astype(int)
    drives["fg"] = (drives["result"] == "Field goal").astype(int)
    drives["rz_td"] = drives["td"] * (drives["inside20"] > 0)
    dg = drives.groupby(["game_id", "posteam"]).agg(
        drives=("fixed_drive", "size"),
        drive_td=("td", "sum"),
        drive_fg=("fg", "sum"),
        rz_trips=("inside20", "sum"),
        rz_td=("rz_td", "sum"),
    )
    g = g.join(dg)

    # Charting FTN: blitz enfrentado y EPA del pase contra blitz.
    if ftn is not None and len(ftn):
        f = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        db_plays = p[(p["qb_dropback"] == 1) & p["epa"].notna()][["game_id", "play_id", "posteam", "epa"]]
        m = db_plays.merge(f[["game_id", "play_id", "n_blitzers"]], on=["game_id", "play_id"], how="inner")
        m["blitz"] = (pd.to_numeric(m["n_blitzers"], errors="coerce").fillna(0) > 0).astype(int)
        m["blitz_epa"] = m["epa"] * m["blitz"]
        bg = m.groupby(["game_id", "posteam"]).agg(
            charted_db=("epa", "size"), blitz_db=("blitz", "sum"), blitz_epa_sum=("blitz_epa", "sum")
        )
        g = g.join(bg)

    for col in ("charted_db", "blitz_db", "blitz_epa_sum"):
        if col not in g.columns:
            g[col] = np.nan
    g = g.reset_index().rename(columns={"posteam": "team"})
    return g


def qb_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """EPA por dropback de cada QB en cada partido (para impacto de lesiones del QB)."""
    p = pbp[(pd.to_numeric(pbp["qb_dropback"], errors="coerce") == 1) & pbp["epa"].notna() & pbp["passer_id"].notna()]
    q = p.groupby(["game_id", "season", "week", "posteam", "passer_id"]).agg(
        passer=("passer", "first"), dropbacks=("epa", "size"), qb_epa_sum=("qb_epa", "sum")
    ).reset_index()
    return q.rename(columns={"posteam": "team", "passer_id": "gsis_id"})


def build_team_games(schedule: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Une el calendario (marcadores, sede, clima, descanso) con las estadísticas."""
    s = schedule.copy()
    s["neutral"] = (s["location"] == "Neutral").astype(int)
    common = ["game_id", "season", "game_type", "week", "gameday", "gametime", "neutral", "roof", "surface",
              "temp", "wind", "div_game", "stadium_id", "overtime"]
    home = s[common + ["home_team", "away_team", "home_score", "away_score", "home_rest", "away_rest"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_score": "pts_for", "away_score": "pts_against",
                 "home_rest": "rest", "away_rest": "opp_rest"})
    home["is_home"] = 1
    away = s[common + ["away_team", "home_team", "away_score", "home_score", "away_rest", "home_rest"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_score": "pts_for", "home_score": "pts_against",
                 "away_rest": "rest", "home_rest": "opp_rest"})
    away["is_home"] = 0
    tg = pd.concat([home, away], ignore_index=True)
    tg["loc"] = np.where(tg["neutral"] == 1, 0, np.where(tg["is_home"] == 1, 1, -1))
    tg = tg.merge(stats, on=["game_id", "team"], how="left")
    return tg.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)
