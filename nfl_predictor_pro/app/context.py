"""Factores de contexto: horario, viaje, zona horaria, descanso y clima."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .stadiums import STADIUMS, haversine_miles, home_stadium

ET = ZoneInfo("America/New_York")


def kickoff_utc(gameday: str, gametime: str | None) -> datetime | None:
    """Los horarios de nflverse están en hora del Este (ET)."""
    if not isinstance(gameday, str):
        return None
    hhmm = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
    try:
        local = datetime.strptime(f"{gameday} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


@lru_cache(maxsize=None)
def _utc_offset_hours(tz: str, day: str) -> float:
    d = datetime.strptime(day, "%Y-%m-%d").replace(hour=12)
    off = d.replace(tzinfo=ZoneInfo(tz)).utcoffset()
    return off.total_seconds() / 3600 if off else 0.0


def travel(team: str, season: int, stadium_id: str | None, day: str) -> dict:
    """Distancia (millas) y cambio horario del estadio local del equipo a la sede."""
    home_id = home_stadium(team, season)
    if not home_id or home_id not in STADIUMS or not stadium_id or stadium_id not in STADIUMS:
        return {"miles": np.nan, "tz_shift": np.nan, "known": False}
    _, la, lo, tz_home = STADIUMS[home_id]
    _, lb, lob, tz_game = STADIUMS[stadium_id]
    miles = haversine_miles((la, lo), (lb, lob))
    shift = _utc_offset_hours(tz_game, day) - _utc_offset_hours(tz_home, day)
    return {"miles": miles, "tz_shift": shift, "known": True}


def add_context(tg: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de contexto a la tabla equipo-partido (vectorizado por filas únicas)."""
    tg = tg.copy()
    keys = tg[["team", "season", "stadium_id", "gameday"]].drop_duplicates()
    rows = []
    for team, season, sid, day in keys.itertuples(index=False):
        tr = travel(team, int(season), sid if isinstance(sid, str) else None, str(day))
        rows.append((team, season, sid, day, tr["miles"], tr["tz_shift"]))
    tr_df = pd.DataFrame(rows, columns=["team", "season", "stadium_id", "gameday", "miles", "tz_shift"])
    tg = tg.merge(tr_df, on=["team", "season", "stadium_id", "gameday"], how="left")
    outdoor = tg["roof"].isin(["outdoors", "open"])
    tg["outdoor"] = outdoor.astype(int)
    tg["wind_eff"] = np.where(outdoor, np.clip(tg["wind"].fillna(0) - 10, 0, None), 0.0)
    tg["cold_eff"] = np.where(outdoor & tg["temp"].notna(), np.clip(40 - tg["temp"].fillna(60), 0, None), 0.0)
    tg["rest_diff"] = np.clip(tg["rest"].fillna(7) - tg["opp_rest"].fillna(7), -7, 7)
    return tg
