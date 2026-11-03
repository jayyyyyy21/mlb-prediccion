"""Ensamblado del pronóstico completo de un partido y su historial de versiones."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import or_, select

from ..config import UCL_SLUG, settings
from ..db import Availability, Match, Prediction, SessionLocal, Team, Weather, utcnow
from ..models.dataset import rest_for_upcoming
from ..models.engine import MODEL_VERSION
from ..providers import api_football, espn, openmeteo
from ..providers.http import FetchError
from . import modeling, tactics
from .availability import team_availability
from .opportunities import confidence, opportunities
from .team_stats import team_profile

log = logging.getLogger("predictions")


def _season_start(d: datetime) -> datetime:
    return datetime(d.year if d.month >= 7 else d.year - 1, 7, 1)


def standings(before: datetime) -> list[dict]:
    """Tabla de la fase de liga UCL calculada con resultados observados."""
    ss = _season_start(before)
    with SessionLocal() as s:
        ms = s.execute(select(Match).where(Match.league == UCL_SLUG, Match.completed,
                                           Match.kickoff >= ss, Match.kickoff < before,
                                           Match.phase == "league-phase")).scalars().all()
    tab: dict[str, dict] = {}
    for m in ms:
        for t, gf, ga in ((m.home_id, m.home_goals, m.away_goals), (m.away_id, m.away_goals, m.home_goals)):
            r = tab.setdefault(t, {"team_id": t, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0})
            r["p"] += 1; r["gf"] += gf; r["ga"] += ga
            if gf > ga:
                r["w"] += 1; r["pts"] += 3
            elif gf == ga:
                r["d"] += 1; r["pts"] += 1
            else:
                r["l"] += 1
    rows = sorted(tab.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"]))
    for i, r in enumerate(rows):
        r["pos"] = i + 1
    return rows


def _home_city(team: str) -> tuple[str | None, str | None]:
    with SessionLocal() as s:
        rows = s.execute(select(Match.city, Match.country).where(
            Match.home_id == team, Match.neutral.is_(False), Match.city.is_not(None))
            .order_by(Match.kickoff.desc()).limit(10)).all()
    if not rows:
        return None, None
    return Counter(rows).most_common(1)[0][0]


def _haversine(a: dict, b: dict) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def context(m: Match, eng, names: dict) -> dict:
    out: dict = {"phase": m.phase, "season": m.season, "neutral": m.neutral, "venue": m.venue,
                 "city": m.city, "country": m.country}
    # Eliminatoria: ida previa entre los mismos equipos esta temporada
    if m.phase and m.phase != "league-phase":
        with SessionLocal() as s:
            first = s.execute(select(Match).where(
                Match.league == UCL_SLUG, Match.completed, Match.kickoff < m.kickoff,
                Match.kickoff >= _season_start(m.kickoff), Match.phase == m.phase,
                Match.home_id == m.away_id, Match.away_id == m.home_id)).scalar_one_or_none()
        if first:
            out["first_leg"] = {"home_goals": first.home_goals, "away_goals": first.away_goals,
                                "aggregate_text": f"Ida: {names[first.home_id]} {first.home_goals}-{first.away_goals} {names[first.away_id]}"}
    tab = standings(m.kickoff)
    pos = {r["team_id"]: r for r in tab}
    out["standings"] = {"home": pos.get(m.home_id), "away": pos.get(m.away_id), "teams": len(tab)}
    # Clima
    with SessionLocal() as s:
        w = s.get(Weather, m.id)
    out["weather"] = ({"temperature_c": w.temperature_c, "precipitation_mm": w.precipitation_mm,
                       "wind_kmh": w.wind_kmh, "elevation_m": w.elevation_m, "source": w.source,
                       "fetched_at": w.fetched_at.isoformat()} if w else None)
    # Viaje
    try:
        hc, hco = (m.city, m.country) if m.city else _home_city(m.home_id)
        ac, aco = _home_city(m.away_id)
        if hc and ac:
            g1, g2 = openmeteo.geocode(hc, hco), openmeteo.geocode(ac, aco)
            if g1 and g2:
                out["travel_km_away"] = round(_haversine(g1, g2))
                out["away_home_city"] = ac
    except FetchError:
        pass
    out["notes"] = [
        "Descanso corto (≤3 días) se incluye en el modelo con un coeficiente estimado de los datos.",
        "Clima, altitud y distancia de viaje se muestran como contexto: no se aplica ajuste porque no se "
        "estimó un efecto significativo con los datos disponibles.",
        "No se asume que un equipo jugará a la defensiva por la clasificación: no hay ajuste por 'necesidad' sin evidencia.",
    ]
    return out


def _inputs_hash(m: Match, av_h: dict, av_a: dict, weather: dict | None, n_rows: int) -> str:
    key = json.dumps({
        "rows": n_rows, "odds": m.odds,
        "lh": av_h["lineup_confirmed"], "la": av_a["lineup_confirmed"],
        "abs": [(a["player"], a["status"]) for a in av_h["absences"] + av_a["absences"]],
        "w": weather and (weather.get("temperature_c"), weather.get("precipitation_mm")),
        "hyper": modeling.get_hyper().as_dict(),
    }, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()


def _trigger_reason(prev: dict | None, new: dict) -> str:
    if prev is None:
        return "primer cálculo"
    r = []
    pa, na = prev.get("availability", {}), new["availability"]
    for side in ("home", "away"):
        if not pa.get(side, {}).get("lineup_confirmed") and na[side]["lineup_confirmed"]:
            r.append(f"alineación confirmada ({side})")
        if len(pa.get(side, {}).get("absences", [])) != len(na[side]["absences"]):
            r.append(f"cambio de disponibilidad ({side})")
    if (prev.get("context", {}).get("weather") or {}).get("temperature_c") != (new["context"].get("weather") or {}).get("temperature_c"):
        r.append("clima actualizado")
    if prev.get("odds") != new.get("odds"):
        r.append("cuotas actualizadas")
    return ", ".join(r) or "nuevos datos de partidos / reajuste del modelo"


def build_prediction(match_id: str, persist: bool = True) -> dict:
    eng = modeling.get_engine()
    with SessionLocal() as s:
        m = s.get(Match, match_id)
        if m is None:
            raise KeyError(match_id)
        teams = {t.id: t for t in s.execute(select(Team).where(Team.id.in_([m.home_id, m.away_id]))).scalars()}
        ucl_teams = [t.id for t in s.execute(select(Team).where(Team.in_ucl)).scalars()]
    names = {k: v.name for k, v in teams.items()}
    before = m.kickoff if not m.completed else m.kickoff - timedelta(minutes=1)

    av_h, av_a = team_availability(m, m.home_id), team_availability(m, m.away_id)
    rh = rest_for_upcoming(eng.rows, m.home_id, m.kickoff)
    ra = rest_for_upcoming(eng.rows, m.away_id, m.kickoff)
    pred = eng.predict(m.home_id, m.away_id, m.kickoff, neutral=m.neutral,
                       adjust={"home_attack": av_h["attack_multiplier"], "away_attack": av_a["attack_multiplier"]},
                       n_sims=settings.mc_iterations, rest_home=rh, rest_away=ra, league=m.league)
    prof_h = team_profile(eng, m.home_id, before)
    prof_a = team_profile(eng, m.away_id, before)
    pop = tactics.population(eng, ucl_teams, before)
    tac = tactics.matchup(eng, m.home_id, m.away_id, names, pred, prof_h, prof_a, pop)
    bt = modeling.latest_backtest()
    conf = confidence(pred, prof_h, prof_a, av_h, av_a, bt, api_football.available())
    ctx = context(m, eng, names)
    opp = opportunities(pred, (names[m.home_id], names[m.away_id]), m.odds, conf, bt)

    # Nombres de rivales en la forma reciente
    with SessionLocal() as s:
        opp_ids = {f["opponent"] for f in prof_h["form"] + prof_a["form"]}
        opp_names = {t.id: t.name for t in s.execute(select(Team).where(Team.id.in_(opp_ids))).scalars()}
    for f in prof_h["form"] + prof_a["form"]:
        f["opponent_name"] = opp_names.get(f["opponent"], f["opponent"])

    payload = {
        "match": _match_dict(m, teams),
        "prediction": pred,
        "confidence": conf,
        "tactical": tac,
        "teams": {"home": prof_h, "away": prof_a},
        "availability": {"home": av_h, "away": av_a,
                         "injury_source": api_football.SOURCE if api_football.available() else None,
                         "injury_source_missing": not api_football.available()},
        "context": ctx,
        "opportunities": opp,
        "odds": m.odds,
        "transparency": {
            "generated_at": utcnow().isoformat(),
            "model_version": MODEL_VERSION,
            "hyperparameters": modeling.get_hyper().as_dict(),
            "training_matches": len(eng.rows),
            "training_until": eng.as_of.isoformat() if eng.as_of else None,
            "sources": [
                {"name": espn.SOURCE, "used_for": "Calendario, resultados, tiros, tiros a puerta, córners, posesión, boxscore, alineaciones, cuotas", "type": "observado"},
                {"name": openmeteo.SOURCE, "used_for": "Clima y altitud del estadio", "type": "observado / pronóstico meteorológico"},
                {"name": api_football.SOURCE, "used_for": "Lesiones y sanciones",
                 "type": "observado" if api_football.available() else "NO CONECTADO (falta API_FOOTBALL_KEY)"},
            ],
            "labels": {
                "observed": "Estadísticas observadas: windows, splits, forma, boxscore.",
                "model": "Estimaciones del modelo: probabilidades, goles/córners/tiros esperados, posesión estimada.",
                "unconfirmed": "Noticias no confirmadas: bajas con estado 'rumor' (no afectan al modelo).",
            },
            "not_available": [
                "xG real (Opta/Understat/FBref no accesibles desde el servidor) → se usa xG-proxy basado en tiros a puerta.",
                "Balón parado, errores defensivos, PPDA real y ocasiones claras: no publicados por la fuente.",
                "Minutos jugados por jugador: no publicados; la fatiga se aproxima con partidos en 14/30 días.",
            ],
        },
    }
    if persist:
        h = _inputs_hash(m, av_h, av_a, ctx.get("weather"), len(eng.rows))
        with SessionLocal() as s:
            last = s.execute(select(Prediction).where(Prediction.match_id == m.id)
                             .order_by(Prediction.id.desc()).limit(1)).scalar_one_or_none()
            if last is None or last.inputs_hash != h:
                reason = _trigger_reason(last.payload if last else None, payload)
                payload["transparency"]["update_reason"] = reason
                s.add(Prediction(match_id=m.id, model_version=MODEL_VERSION, trigger=reason,
                                 inputs_hash=h, payload=payload))
                s.commit()
            else:
                payload = last.payload
    return payload


def _match_dict(m: Match, teams: dict) -> dict:
    def t(tid):
        x = teams.get(tid)
        return {"id": tid, "name": x.name if x else tid, "short_name": x.short_name if x else None,
                "abbreviation": x.abbreviation if x else None, "logo": x.logo if x else None,
                "color": x.color if x else None}
    return {"id": m.id, "league": m.league, "season": m.season, "phase": m.phase,
            "kickoff": m.kickoff.isoformat() + "Z", "status": m.status, "completed": m.completed,
            "home": t(m.home_id), "away": t(m.away_id), "home_goals": m.home_goals, "away_goals": m.away_goals,
            "venue": m.venue, "city": m.city, "country": m.country, "neutral": m.neutral}


def latest_prediction(match_id: str) -> dict | None:
    with SessionLocal() as s:
        p = s.execute(select(Prediction).where(Prediction.match_id == match_id)
                      .order_by(Prediction.id.desc()).limit(1)).scalar_one_or_none()
        return p.payload if p else None


def prediction_history(match_id: str) -> list[dict]:
    with SessionLocal() as s:
        ps = s.execute(select(Prediction).where(Prediction.match_id == match_id)
                       .order_by(Prediction.id)).scalars().all()
    out = []
    for p in ps:
        g = p.payload["prediction"]["goals"]
        out.append({"created_at": p.created_at.isoformat(), "trigger": p.trigger, "model_version": p.model_version,
                    "home_win": g["home_win"], "draw": g["draw"], "away_win": g["away_win"],
                    "home_xg": g["home_xg"], "away_xg": g["away_xg"],
                    "confidence": p.payload["confidence"]["level"]})
    return out


def refresh_predictions(trigger: str = "manual", days: int = 60) -> dict:
    now = utcnow()
    with SessionLocal() as s:
        ms = s.execute(select(Match.id).where(
            Match.league == UCL_SLUG, Match.completed.is_(False),
            Match.kickoff >= now - timedelta(hours=3), Match.kickoff <= now + timedelta(days=days))).all()
    done, errors = 0, []
    for (mid,) in ms:
        try:
            build_prediction(mid)
            done += 1
        except Exception as e:  # se registra y se sigue con el resto
            log.exception("predicción %s", mid)
            errors.append({"match_id": mid, "error": str(e)})
    return {"trigger": trigger, "computed": done, "errors": errors, "at": utcnow().isoformat()}


def list_matches(date_from: datetime | None, date_to: datetime | None, team: str | None) -> list[dict]:
    with SessionLocal() as s:
        q = select(Match).where(Match.league == UCL_SLUG)
        if date_from:
            q = q.where(Match.kickoff >= date_from)
        if date_to:
            q = q.where(Match.kickoff < date_to)
        if team:
            q = q.where(or_(Match.home_id == team, Match.away_id == team))
        ms = s.execute(q.order_by(Match.kickoff)).scalars().all()
        ids = {m.home_id for m in ms} | {m.away_id for m in ms}
        teams = {t.id: t for t in s.execute(select(Team).where(Team.id.in_(ids))).scalars()}
        preds = {}
        if ms:
            for p in s.execute(select(Prediction).where(Prediction.match_id.in_([m.id for m in ms]))
                               .order_by(Prediction.id)).scalars():
                preds[p.match_id] = p.payload
    out = []
    for m in ms:
        d = _match_dict(m, teams)
        p = preds.get(m.id)
        if p:
            g = p["prediction"]["goals"]
            c = p["prediction"]["corners"]
            so = p["prediction"]["shots_on_target"]
            d["summary"] = {
                "home_win": g["home_win"], "draw": g["draw"], "away_win": g["away_win"],
                "home_xg": g["home_xg"], "away_xg": g["away_xg"], "over_2_5": g["over_2_5"], "btts": g["btts"],
                "most_likely_score": g["most_likely_score"], "corners_total": c["total_exp"],
                "sot_total": so["total_exp"], "confidence": p["confidence"]["level"],
                "decision": p["opportunities"]["recommendation"]["decision"],
                "generated_at": p["transparency"]["generated_at"],
            }
        out.append(d)
    return out
