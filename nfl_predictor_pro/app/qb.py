"""Calidad del quarterback y cambios de titular.

Los ratings de equipo (ratings.py) reflejan a los QBs que jugaron los
partidos anteriores. Cuando el titular esperado es otro (lesión, regreso de
un lesionado, cambio de titular), la proyección debe corregirse. Para cada
fecha de corte, usando sólo partidos anteriores:

* nivel(QB)   = EPA por dropback de las últimas 3 temporadas, encogido hacia
                el nivel de reemplazo con 250 dropbacks a priori (un QB con
                poca muestra se asume de nivel suplente).
* base(equipo)= promedio del nivel de los QBs que lanzaron por el equipo,
                ponderado por dropbacks y con el mismo decaimiento temporal
                que los ratings (lo que los ratings "ya saben").
* cambio      = (nivel(QB esperado) - base(equipo)) x dropbacks por partido.

`cambio` y `nivel` entran al modelo de puntos como variables; sus
coeficientes se calibran en el backtest (con el titular real de cada partido
histórico), no se asumen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_DB = 250.0          # dropbacks de encogimiento
BACKUP_MAX_DB = 300       # QBs con menos volumen definen el nivel de reemplazo
DEFAULT_DROPBACKS = 36.0


@dataclass
class QBSnapshot:
    per: pd.DataFrame        # índice gsis_id: db, epa, shr (nivel encogido), name
    replacement: float
    mix: dict[str, float]    # nivel base por equipo (lo que reflejan los ratings)
    dbpg: dict[str, float]   # dropbacks por partido por equipo

    def level(self, gsis_id: str | None) -> float:
        if isinstance(gsis_id, str) and gsis_id in self.per.index:
            return float(self.per.at[gsis_id, "shr"])
        return self.replacement

    def volume(self, gsis_id: str | None) -> int:
        if isinstance(gsis_id, str) and gsis_id in self.per.index:
            return int(self.per.at[gsis_id, "db"])
        return 0

    def name(self, gsis_id: str | None) -> str | None:
        if isinstance(gsis_id, str) and gsis_id in self.per.index:
            return str(self.per.at[gsis_id, "name"])
        return None

    def base(self, team: str) -> float:
        return self.mix.get(team, self.replacement)

    def dropbacks(self, team: str) -> float:
        return self.dbpg.get(team, DEFAULT_DROPBACKS)

    def features(self, team: str, gsis_id: str | None) -> dict:
        """Variables de contexto para el modelo. Sin QB conocido: sin cambio."""
        if not isinstance(gsis_id, str):
            return {"qb_delta": 0.0, "qb_level": self.base(team)}
        lvl = self.level(gsis_id)
        return {"qb_delta": (lvl - self.base(team)) * self.dropbacks(team), "qb_level": lvl}


class QBModel:
    def __init__(self, qb_games: pd.DataFrame, half_life: float = 10.0) -> None:
        self.qb = qb_games.copy()
        if len(self.qb):
            self.qb["t"] = self.qb["season"] * 30 + self.qb["week"]
        self.half_life = half_life
        self._cache: dict[tuple[int, int], QBSnapshot] = {}

    def at(self, season: int, week: int) -> QBSnapshot:
        key = (int(season), int(week))
        if key not in self._cache:
            self._cache[key] = self._build(*key)
        return self._cache[key]

    def _build(self, season: int, week: int) -> QBSnapshot:
        if self.qb.empty:
            return QBSnapshot(pd.DataFrame(columns=["db", "epa", "shr", "name"]), -0.15, {}, {})
        now = season * 30 + week
        past = self.qb[(self.qb["t"] < now) & (self.qb["season"] >= season - 2)]
        per = past.groupby("gsis_id").agg(db=("dropbacks", "sum"), epa=("qb_epa_sum", "sum"), name=("passer", "last"))
        backups = per[per["db"] < BACKUP_MAX_DB]
        repl = float(backups["epa"].sum() / backups["db"].sum()) if backups["db"].sum() > 0 else -0.15
        per["shr"] = (per["epa"] + repl * PRIOR_DB) / (per["db"] + PRIOR_DB)
        win = past[past["t"] >= now - 100]
        w = win["dropbacks"] * 0.5 ** ((now - win["t"]) / self.half_life)
        lv = win["gsis_id"].map(per["shr"])
        num = (w * lv).groupby(win["team"]).sum()
        den = w.groupby(win["team"]).sum()
        mix = (num / den).to_dict()
        dbpg = win.groupby(["team", "game_id"])["dropbacks"].sum().groupby("team").mean().to_dict()
        return QBSnapshot(per, repl, {k: float(v) for k, v in mix.items()}, {k: float(v) for k, v in dbpg.items()})


def starters_from_schedule(schedule: pd.DataFrame) -> dict[tuple[str, str], str]:
    """QB titular real de cada partido jugado: (game_id, equipo) -> gsis_id."""
    out: dict[tuple[str, str], str] = {}
    for side in ("home", "away"):
        col = f"{side}_qb_id"
        if col not in schedule.columns:
            continue
        sub = schedule[schedule[col].notna()]
        out.update({(g, t): q for g, t, q in zip(sub["game_id"], sub[f"{side}_team"], sub[col])})
    return out
