"""API REST + interfaz web de NFL Predictor Pro."""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import threading
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .config import BASE_DIR, CURRENT_SEASON, N_SIMS, SHOW_LOGOS
from .methodology import METHODOLOGY
from .ratings import TEAMS
from .service import ModelNotReady, engine, reset_engine

log = logging.getLogger(__name__)
app = FastAPI(title="NFL Predictor Pro", version="1.0")
STATIC = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_update_lock = threading.Lock()
_update_state: dict = {"running": False, "last": None, "error": None}


def _clean(obj):
    """Convierte NaN/inf/numpy a JSON válido."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def ok(payload) -> JSONResponse:
    return JSONResponse(_clean(payload))


@app.exception_handler(ModelNotReady)
async def _not_ready(_, exc: ModelNotReady):
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    e = engine()
    updates = db.source_updates()
    alerts = []
    for u in updates:
        if u["status"] != "ok":
            alerts.append({"level": "warn", "text": f"Fuente {u['source']}: {u['message']}"})
    last = db.kv_get("last_update")
    if last:
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 3600
        if age_h > 24:
            alerts.append({"level": "warn", "text": f"Los datos se actualizaron hace {age_h:.0f} horas."})
    if e.model is None:
        alerts.append({"level": "error", "text": "Modelo sin calibrar: ejecute `python -m app.cli init`."})
    season = CURRENT_SEASON
    return ok({
        "season": season, "default_week": e.default_week(season), "seasons": e.seasons(),
        "last_update": last, "sources": updates, "alerts": alerts, "n_sims": N_SIMS, "show_logos": SHOW_LOGOS,
        "model_ready": e.model is not None, "model_created_at": e.backtest.get("created_at"),
        "params": e.params, "update": _update_state,
    })


@app.get("/api/teams")
def teams():
    return ok(engine().team_info())


@app.get("/api/weeks")
def weeks(season: int = CURRENT_SEASON):
    return ok(engine().weeks(season))


@app.get("/api/predictions")
def predictions(season: int = CURRENT_SEASON, week: int | None = None):
    e = engine()
    week = week or e.default_week(season)
    return ok({"season": season, "week": week, "games": e.predict_week(season, week)})


@app.get("/api/game/{game_id}")
def game(game_id: str):
    try:
        return ok(engine().predict_game_id(game_id))
    except KeyError:
        raise HTTPException(404, "Partido no encontrado")


class SimRequest(BaseModel):
    home: str
    away: str
    neutral: bool = False
    out_home: list[str] = Field(default_factory=list)
    out_away: list[str] = Field(default_factory=list)
    include_probable: bool = False
    roof: str | None = None
    temp: float | None = None
    wind: float | None = None
    rest_diff: float | None = None


@app.post("/api/simulate")
def simulate(req: SimRequest):
    if req.home not in TEAMS or req.away not in TEAMS or req.home == req.away:
        raise HTTPException(400, "Equipos inválidos")
    ov: dict = {"out_home": req.out_home, "out_away": req.out_away, "include_probable": req.include_probable}
    if req.roof:
        ov["roof"] = req.roof
    if req.temp is not None or req.wind is not None:
        ov["temp"], ov["wind"] = req.temp, req.wind
    if req.rest_diff is not None:
        ov["rest_diff"] = req.rest_diff
    return ok(engine().simulate_custom(req.home, req.away, req.neutral, ov))


@app.get("/api/team/{team}")
def team(team: str, season: int = CURRENT_SEASON):
    if team not in TEAMS:
        raise HTTPException(404, "Equipo no encontrado")
    return ok(engine().team_stats(team, season))


@app.get("/api/compare")
def compare(a: str, b: str, season: int = CURRENT_SEASON):
    if a not in TEAMS or b not in TEAMS:
        raise HTTPException(404, "Equipo no encontrado")
    e = engine()
    from . import analysis
    from .injuries import evaluate_team
    week = e.default_week(season) if season == CURRENT_SEASON else 30
    snap = e.snap(season, min(week, 23))
    payload = {
        "a": e.team_stats(a, season), "b": e.team_stats(b, season),
        "matchups": {"a_offense": analysis.tactical_matchups(snap, e.tgd, a, b, season, week),
                     "b_offense": analysis.tactical_matchups(snap, e.tgd, b, a, season, week)},
    }
    if season == CURRENT_SEASON:
        payload["injuries"] = {"a": evaluate_team(e.inj, a, season, week), "b": evaluate_team(e.inj, b, season, week)}
    return ok(payload)


@app.get("/api/ratings")
def ratings(season: int = CURRENT_SEASON):
    return ok(engine().ratings_table(season))


@app.get("/api/history")
def history():
    return ok(engine().history())


@app.get("/api/methodology")
def methodology():
    e = engine()
    return ok({"sections": METHODOLOGY, "params": e.params,
               "coefficients": _coefficients(e)})


def _coefficients(e):
    if e.model is None:
        return []
    from .model import FEATURE_LABELS
    return [{"feature": f, "label": FEATURE_LABELS[f], "coef_std": round(float(c), 3),
             "per_unit": round(float(c / s), 4), "mean": round(float(m), 4)}
            for f, c, s, m in zip(e.model.features, e.model.coef, e.model.scale, e.model.mean)]


@app.get("/api/export")
def export(season: int = CURRENT_SEASON, week: int | None = None, fmt: str = Query("csv", pattern="^(csv|json)$")):
    e = engine()
    week = week or e.default_week(season)
    games = e.predict_week(season, week)
    if fmt == "json":
        body = json.dumps(_clean({"season": season, "week": week, "exported_at": db.utcnow(),
                                  "last_data_update": db.kv_get("last_update"), "games": games}), ensure_ascii=False, indent=1)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename=nfl_pronosticos_{season}_s{week}.json"})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["temporada", "semana", "fecha", "local", "visitante", "pts_local", "pts_visitante", "marcador_estimado",
                "prob_local", "prob_visitante", "prob_empate", "diferencia_local", "total", "confianza", "estado",
                "resultado_real", "datos_actualizados"])
    upd = db.kv_get("last_update")
    for g in games:
        gm, p = g["game"], g["projection"]
        real = f"{gm['home_score']}-{gm['away_score']}" if gm.get("played") else ""
        w.writerow([season, week, gm["gameday"], gm["home"], gm["away"], p["home_points"], p["away_points"],
                    f"{p['score_home']}-{p['score_away']}", round(p["p_home"], 4), round(p["p_away"], 4),
                    round(p["p_tie"], 4), p["spread_home"], p["total"], g["confidence"], g["status"], real, upd])
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename=nfl_pronosticos_{season}_s{week}.csv"})


@app.get("/api/export/game/{game_id}")
def export_game(game_id: str):
    try:
        res = engine().predict_game_id(game_id)
    except KeyError:
        raise HTTPException(404, "Partido no encontrado")
    body = json.dumps(_clean({**res, "exported_at": db.utcnow(), "last_data_update": db.kv_get("last_update")}),
                      ensure_ascii=False, indent=1)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=analisis_{game_id}.json"})


def _run_update(recalibrate: bool):
    from . import backtest, ingest
    try:
        _update_state.update(running=True, error=None)
        report = ingest.update_all()
        if recalibrate:
            backtest.run()
        reset_engine()
        engine()
        _update_state.update(last=db.utcnow(), report=report)
    except Exception as exc:  # se reporta en /api/status
        log.exception("update")
        _update_state["error"] = str(exc)
    finally:
        _update_state["running"] = False
        _update_lock.release()


def _auto_update_loop(hours: float):
    """Actualiza datos periódicamente y vuelve a calcular (y registrar) los pronósticos
    de la semana en curso, de modo que nuevas lesiones confirmadas se reflejen."""
    import time
    while True:
        time.sleep(hours * 3600)
        if _update_lock.acquire(blocking=False):
            _run_update(False)
            try:
                e = engine()
                e.predict_week(CURRENT_SEASON, e.default_week(CURRENT_SEASON))
            except Exception:  # se reintenta en el siguiente ciclo
                log.exception("auto-update")


@app.on_event("startup")
def _startup():
    hours = float(os.environ.get("NFLPP_AUTO_UPDATE_HOURS", "3"))
    if hours > 0:
        threading.Thread(target=_auto_update_loop, args=(hours,), daemon=True).start()


@app.post("/api/update")
def update(recalibrate: bool = False):
    """Descarga datos nuevos (calendario, play-by-play, lesiones) y recalcula."""
    if not _update_lock.acquire(blocking=False):
        return ok({"started": False, "detail": "Ya hay una actualización en curso"})
    threading.Thread(target=_run_update, args=(recalibrate,), daemon=True).start()
    return ok({"started": True})
