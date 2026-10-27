"""Ratings ofensivos y defensivos ajustados por rival (regresión ridge ponderada).

Para cada métrica y se ajusta, con partidos anteriores a la fecha de corte:

    y(ofensiva A vs defensa B) = mu + off[A] + def[B] + h * localía + error

* Los pesos decaen exponencialmente con la antigüedad del partido (vida media
  `half_life` en semanas), por lo que el rendimiento reciente pesa más.
* La penalización ridge `lam` (en "partidos equivalentes") encoge los ratings
  hacia el promedio de la liga cuando la muestra es pequeña.
* Las métricas por jugada se ponderan por número de jugadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}

# métrica: (numerador, denominador o None si es por partido)
METRICS: dict[str, tuple[str, str | None]] = {
    "pts": ("pts_for", None),
    "epa": ("epa_sum", "plays"),
    "sr": ("succ_sum", "plays"),
    "pepa": ("pass_epa_sum", "db"),
    "repa": ("rush_epa_sum", "rushes"),
    "pass_rate": ("db", "plays"),
    "drives": ("drives", None),
    "ppd": ("pts_for", "drives"),
    "sack_rate": ("sacks", "db"),
    "to_rate": ("turnovers", "plays"),
    "expl": ("explosive", "plays"),
    "rz": ("rz_td", "rz_trips"),
    "third": ("third_conv", "third_att"),
    "pass_ypg": ("pass_yds", None),
    "rush_ypg": ("rush_yds", None),
    "blitz_rate": ("blitz_db", "charted_db"),
}


def time_index(season, week):
    return np.asarray(season) * 30 + np.asarray(week)


@dataclass
class Ratings:
    mu: float
    off: dict[str, float]
    deff: dict[str, float]
    home: float
    n_eff: dict[str, float] = field(default_factory=dict)

    def expected(self, off_team: str, def_team: str, loc: int = 0) -> float:
        return self.mu + self.off.get(off_team, 0.0) + self.deff.get(def_team, 0.0) + self.home * loc


def prepare(team_games: pd.DataFrame) -> pd.DataFrame:
    tg = team_games[team_games["pts_for"].notna() & team_games["plays"].notna()].copy()
    tg = tg[tg["team"].isin(TEAM_IDX) & tg["opp"].isin(TEAM_IDX)]
    tg["turnovers"] = tg["ints"].fillna(0) + tg["fum_lost"].fillna(0)
    tg["t"] = time_index(tg["season"], tg["week"])
    tg["oi"] = tg["team"].map(TEAM_IDX)
    tg["di"] = tg["opp"].map(TEAM_IDX)
    return tg.reset_index(drop=True)


def fit_metric(tg: pd.DataFrame, metric: str, now_t: float, half_life: float, lam: float,
               lam_home: float = 30.0) -> Ratings | None:
    num, den = METRICS[metric]
    if num not in tg.columns or (den and den not in tg.columns):
        return None
    # Ventana de ~3 temporadas: más atrás el peso por decaimiento es despreciable.
    sub = tg[(tg["t"] < now_t) & (tg["t"] >= now_t - 100) & tg[num].notna()]
    if den:
        sub = sub[sub[den].notna() & (sub[den] > 0)]
    if len(sub) < 20:
        return None
    if den:
        n = sub[den].to_numpy(float)
        y = sub[num].to_numpy(float) / n
        sw = n / n.mean()
    else:
        y = sub[num].to_numpy(float)
        sw = np.ones(len(sub))
    age = now_t - sub["t"].to_numpy(float)
    w = sw * 0.5 ** (age / half_life)
    mu = float(np.sum(w * y) / np.sum(w))
    k = len(TEAMS)
    oi = sub["oi"].to_numpy()
    di = sub["di"].to_numpy()
    loc = sub["loc"].to_numpy(float)
    yc = y - mu
    # Ecuaciones normales con estructura dispersa (2 equipos + localía por fila).
    A = np.zeros((2 * k + 1, 2 * k + 1))
    b = np.zeros(2 * k + 1)
    np.add.at(A, (oi, oi), w)
    np.add.at(A, (k + di, k + di), w)
    np.add.at(A, (oi, k + di), w)
    np.add.at(A, (k + di, oi), w)
    np.add.at(A, (oi, 2 * k), w * loc)
    np.add.at(A, (k + di, 2 * k), w * loc)
    A[2 * k, :k] = A[:k, 2 * k]
    A[2 * k, k:2 * k] = A[k:2 * k, 2 * k]
    A[2 * k, 2 * k] = np.sum(w * loc * loc)
    np.add.at(b, oi, w * yc)
    np.add.at(b, k + di, w * yc)
    b[2 * k] = np.sum(w * yc * loc)
    reg = np.full(2 * k + 1, lam)
    reg[2 * k] = lam_home
    beta = np.linalg.solve(A + np.diag(reg), b)
    n_off = np.bincount(oi, weights=w, minlength=k)
    return Ratings(
        mu=mu,
        off={t: float(beta[i]) for t, i in TEAM_IDX.items()},
        deff={t: float(beta[k + i]) for t, i in TEAM_IDX.items()},
        home=float(beta[2 * k]),
        n_eff={t: float(n_off[i]) for t, i in TEAM_IDX.items()},
    )


@dataclass
class Snapshot:
    """Todos los ratings disponibles en una fecha de corte."""
    now_t: float
    long: dict[str, Ratings]
    recent: dict[str, Ratings]

    def get(self, metric: str, recent: bool = False) -> Ratings | None:
        return (self.recent if recent else self.long).get(metric)


def snapshot(tg: pd.DataFrame, now_t: float, params: dict) -> Snapshot:
    long, recent = {}, {}
    for m in METRICS:
        r = fit_metric(tg, m, now_t, params["half_life"], params["lam"])
        if r is not None:
            long[m] = r
    for m in ("pts", "epa"):
        r = fit_metric(tg, m, now_t, params["half_life_recent"], params["lam_recent"])
        if r is not None:
            recent[m] = r
    return Snapshot(now_t, long, recent)
