"""Alineaciones, bajas y su impacto estimado.

Fuentes:
* Alineación confirmada: rosters de ESPN del partido (titulares) cuando se publican.
* Bajas: API-Football (si hay clave) → 'confirmed' / 'probable'. Los 'rumor'
  se muestran pero NUNCA alteran el modelo.

Impacto (supuesto explícito, no estimado de datos):
  cuota_jugador = 0.5·(goles+asistencias del jugador / del equipo)
                + 0.5·(tiros del jugador / del equipo)       (últimos ≤15 partidos con alineación)
  multiplicador_ataque = 1 − PERDIDA_REEMPLAZO · Σ cuota_jugador · peso_estado
  con PERDIDA_REEMPLAZO = 0.5 (el sustituto recupera la mitad de la producción),
  peso_estado: confirmada 1.0, probable 0.5, ausente de XI confirmado 1.0.
  Acotado a [0.80, 1.00]. Bajas defensivas y de portero no se cuantifican: se
  señalan y reducen la confianza.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from sqlalchemy import select

from ..db import Appearance, Availability, Match, SessionLocal

REPLACEMENT_LOSS = 0.5
STATUS_WEIGHT = {"confirmed": 1.0, "probable": 0.5, "not_in_xi": 1.0, "rumor": 0.0}
DEF_POS = ("G", "CD", "CD-L", "CD-R", "LB", "RB", "D", "LWB", "RWB", "CB")


def _player_shares(team: str, before: datetime, n_matches: int = 15) -> tuple[dict, int]:
    with SessionLocal() as s:
        mids = [m for (m,) in s.execute(
            select(Match.id).join(Appearance, Appearance.match_id == Match.id)
            .where(Appearance.team_id == team, Match.completed, Match.kickoff < before)
            .group_by(Match.id, Match.kickoff).order_by(Match.kickoff.desc()).limit(n_matches))]
        apps = s.execute(select(Appearance).where(Appearance.match_id.in_(mids),
                                                  Appearance.team_id == team)).scalars().all()
    agg = defaultdict(lambda: {"name": "", "pos": None, "starts": 0, "apps": 0, "ga": 0.0, "shots": 0.0})
    tot_ga = tot_sh = 0.0
    for a in apps:
        p = agg[a.player_id]
        p["name"], p["pos"] = a.player_name, a.position or p["pos"]
        p["starts"] += int(a.starter)
        p["apps"] += int(a.starter or a.subbed_in)
        p["ga"] += a.goals + a.assists
        p["shots"] += a.shots
        tot_ga += a.goals + a.assists
        tot_sh += a.shots
    n = len(mids)
    for pid, p in agg.items():
        p["share"] = 0.5 * (p["ga"] / tot_ga if tot_ga else 0) + 0.5 * (p["shots"] / tot_sh if tot_sh else 0)
        p["start_rate"] = p["starts"] / n if n else 0
    return dict(agg), n


def _lineup(match_id: str, team: str) -> list[Appearance]:
    with SessionLocal() as s:
        return list(s.execute(select(Appearance).where(Appearance.match_id == match_id,
                                                       Appearance.team_id == team)).scalars())


def team_availability(match: Match, team: str) -> dict:
    shares, n = _player_shares(team, match.kickoff)
    lineup = [a for a in _lineup(match.id, team) if a.starter]
    lineup_confirmed = len(lineup) >= 11 and not match.completed
    with SessionLocal() as s:
        reported = list(s.execute(select(Availability).where(
            Availability.team_id == team,
            (Availability.match_id == match.id) | (Availability.match_id.is_(None)))).scalars())

    absences = []
    seen = set()
    for r in reported:
        pid = next((k for k, v in shares.items() if v["name"].lower() == r.player_name.lower()), None)
        p = shares.get(pid, {})
        absences.append({"player": r.player_name, "status": r.status, "reason": r.reason, "source": r.source,
                         "share": p.get("share", 0.0), "start_rate": p.get("start_rate", 0.0),
                         "position": p.get("pos"), "used_in_model": r.status in ("confirmed", "probable")})
        seen.add(r.player_name.lower())
    if lineup_confirmed:
        xi = {a.player_id for a in lineup}
        for pid, p in shares.items():
            if p["start_rate"] >= 0.6 and pid not in xi and p["name"].lower() not in seen:
                absences.append({"player": p["name"], "status": "not_in_xi", "reason": "No figura en el XI confirmado",
                                 "source": "ESPN (alineación)", "share": p["share"], "start_rate": p["start_rate"],
                                 "position": p["pos"], "used_in_model": True})

    loss = sum(REPLACEMENT_LOSS * a["share"] * STATUS_WEIGHT.get(a["status"], 0) for a in absences if a["used_in_model"])
    mult = max(0.80, 1.0 - loss)
    key_def = [a for a in absences if a["used_in_model"] and (a.get("position") or "").upper() in DEF_POS
               and a["start_rate"] >= 0.6]
    gk = [a for a in key_def if (a.get("position") or "").upper() == "G"]
    regulars = sorted(({"player": p["name"], "position": p["pos"], "start_rate": p["start_rate"],
                        "share": p["share"], "ga": p["ga"]} for p in shares.values() if p["starts"] > 0),
                      key=lambda x: -x["start_rate"])[:16]
    return {
        "lineup_confirmed": lineup_confirmed,
        "lineup": [{"player": a.player_name, "position": a.position} for a in lineup] if lineup_confirmed else [],
        "absences": absences,
        "attack_multiplier": mult,
        "key_defensive_absences": [a["player"] for a in key_def],
        "goalkeeper_absent": bool(gk),
        "regulars": regulars,
        "sample_matches": n,
    }
