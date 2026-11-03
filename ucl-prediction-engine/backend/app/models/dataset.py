"""Construcción de datasets desde la base de datos (solo datos observados)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select

from ..config import EXCLUDED_FROM_MODEL
from ..db import Match, SessionLocal, Team, TeamMatchStats


@dataclass
class Row:
    match_id: str
    league: str
    kickoff: datetime
    neutral: bool
    home: str
    away: str
    hg: int
    ag: int
    h: dict  # stats local
    a: dict  # stats visitante
    country: str | None = None


STAT_FIELDS = ("shots", "shots_on_target", "corners", "possession", "fouls", "passes",
               "accurate_passes", "crosses", "long_balls", "tackles", "interceptions",
               "clearances", "blocked_shots", "penalty_goals", "red_cards", "formation")


def load_rows() -> list[Row]:
    with SessionLocal() as s:
        matches = s.execute(select(Match).where(Match.completed, Match.home_goals.is_not(None))
                            .order_by(Match.kickoff)).scalars().all()
        stats = defaultdict(dict)
        for st in s.execute(select(TeamMatchStats)).scalars():
            stats[st.match_id][st.team_id] = {f: getattr(st, f) for f in STAT_FIELDS}
    rows = []
    for m in matches:
        if m.league in EXCLUDED_FROM_MODEL:
            continue
        st = stats.get(m.id, {})
        if not stats_valid(st.get(m.home_id, {}), st.get(m.away_id, {})):
            st = {}  # estadísticas ausentes registradas como ceros: se descartan (se conservan los goles)
        rows.append(Row(m.id, m.league, m.kickoff, m.neutral, m.home_id, m.away_id,
                        m.home_goals, m.away_goals, st.get(m.home_id, {}), st.get(m.away_id, {}), m.country))
    return rows


def stats_valid(h: dict, a: dict) -> bool:
    """Control de calidad: la fuente a veces publica estadísticas ausentes como ceros."""
    if not h or not a:
        return False
    ph, pa = h.get("possession"), a.get("possession")
    if not ph or not pa or abs(ph + pa - 100) > 2:
        return False
    if (h.get("shots") or 0) + (a.get("shots") or 0) == 0:
        return False
    return True


def team_leagues(rows: list[Row]) -> dict[str, str]:
    """Liga doméstica de cada equipo: la declarada o la más frecuente fuera de UEFA.

    Equipos cuya liga no cubre la fuente se agrupan por país del estadio en sus partidos
    como local ("otros:<país>"), para no mezclar en un mismo grupo ligas muy distintas.
    """
    with SessionLocal() as s:
        declared = {t.id: t.domestic_league for t in s.execute(select(Team)).scalars() if t.domestic_league}
    c: dict[str, Counter] = defaultdict(Counter)
    home_country: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.country and not r.neutral:
            home_country[r.home][r.country] += 1
        if r.league.startswith("uefa"):
            continue
        c[r.home][r.league] += 1
        c[r.away][r.league] += 1
    out = {}
    for t in set(c) | {r.home for r in rows} | {r.away for r in rows}:
        if t in declared and not declared[t].startswith("club."):
            out[t] = declared[t]
        elif c.get(t):
            out[t] = c[t].most_common(1)[0][0]
        elif home_country.get(t):
            out[t] = "otros:" + home_country[t].most_common(1)[0][0]
        else:
            out[t] = "otros"
    return out


def comp_covariates(league: str) -> tuple[float, float]:
    """(competición europea principal, fase previa europea). Efectos estimados de los datos."""
    if league in ("uefa.champions", "uefa.europa", "uefa.europa.conf"):
        return (1.0, 0.0)
    if league.startswith("uefa.") and league.endswith("_qual"):
        return (0.0, 1.0)
    return (0.0, 0.0)


def rest_days(rows: list[Row]) -> dict[tuple[str, str], float]:
    """Días de descanso de cada equipo antes de cada partido (incluye todas las competiciones)."""
    last: dict[str, datetime] = {}
    out = {}
    for r in sorted(rows, key=lambda r: r.kickoff):
        for t in (r.home, r.away):
            prev = last.get(t)
            out[(r.match_id, t)] = (r.kickoff - prev).total_seconds() / 86400 if prev else 7.0
            last[t] = r.kickoff
    return out


def rest_for_upcoming(rows: list[Row], team: str, kickoff: datetime) -> float:
    prev = [r.kickoff for r in rows if (r.home == team or r.away == team) and r.kickoff < kickoff]
    return (kickoff - max(prev)).total_seconds() / 86400 if prev else 7.0


def short_rest(days: float) -> float:
    """Covariable: 1 si el equipo tuvo ≤3 días de descanso."""
    return 1.0 if days <= 3.0 else 0.0


@dataclass
class XgProxy:
    """xG aproximado a partir de tiros (NO es xG de Opta/Understat).

    goles ≈ b_sot · tiros_a_puerta + b_off · tiros_fuera, ajustado por mínimos cuadrados
    no negativos sobre todo el histórico. Sirve para reducir el ruido de los goles.
    """

    b_sot: float = 0.30
    b_off: float = 0.03

    @classmethod
    def fit(cls, rows: list[Row]) -> "XgProxy":
        X, y = [], []
        for r in rows:
            for st, g in ((r.h, r.hg), (r.a, r.ag)):
                sh, sot = st.get("shots"), st.get("shots_on_target")
                if sh is None or sot is None or sh < sot:
                    continue
                X.append([sot, sh - sot]); y.append(g)
        if len(y) < 200:
            return cls()
        from scipy.optimize import nnls
        coef, _ = nnls(np.array(X, float), np.array(y, float))
        return cls(float(coef[0]), float(coef[1]))

    def value(self, st: dict) -> float | None:
        sh, sot = st.get("shots"), st.get("shots_on_target")
        if sh is None or sot is None or sh < sot:
            return None
        return self.b_sot * sot + self.b_off * (sh - sot)
