"""API HTTP (FastAPI). Las credenciales viven solo en el servidor."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select

from .config import BASE_DIR, UCL_SLUG, settings
from .db import Availability, FetchLog, Match, SessionLocal, Team, init_db, utcnow
from .providers import api_football
from .services import modeling, predictions
from .services.ingest import full_refresh

log = logging.getLogger("api")
app = FastAPI(title="UCL Prediction Engine", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_refresh_lock = threading.Lock()
_refresh_state: dict = {"running": False, "last": None}


@app.on_event("startup")
def _startup() -> None:
    init_db()
    if settings.refresh_interval_min > 0:
        threading.Thread(target=_scheduler, daemon=True).start()


def _run_refresh(full: bool, trigger: str) -> None:
    if not _refresh_lock.acquire(blocking=False):
        return
    _refresh_state["running"] = True
    try:
        rep = full_refresh(history=full)
        rep["predictions"] = predictions.refresh_predictions(trigger)
        _refresh_state["last"] = rep
    except Exception as e:
        log.exception("refresh")
        _refresh_state["last"] = {"error": str(e), "at": utcnow().isoformat()}
    finally:
        _refresh_state["running"] = False
        _refresh_lock.release()


def _scheduler() -> None:
    while True:
        time.sleep(settings.refresh_interval_min * 60)
        _run_refresh(False, "actualización programada")


def _check_admin(token: str | None) -> None:
    if not settings.admin_token or token != settings.admin_token:
        raise HTTPException(401, "Token de administración inválido o no configurado")


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    with SessionLocal() as s:
        last_ok = s.execute(select(func.max(FetchLog.fetched_at)).where(FetchLog.ok)).scalar()
        n_err_24h = s.execute(select(func.count()).select_from(FetchLog).where(
            FetchLog.ok.is_(False), FetchLog.fetched_at >= utcnow() - timedelta(hours=24))).scalar()
        n_matches = s.execute(select(func.count()).select_from(Match).where(Match.completed)).scalar()
        n_ucl = s.execute(select(func.count()).select_from(Match).where(Match.league == UCL_SLUG)).scalar()
        leagues = [r for (r,) in s.execute(select(Match.league).distinct())]
    bt = modeling.latest_backtest()
    return {
        "data": {"last_successful_fetch": last_ok and last_ok.isoformat(), "fetch_errors_24h": n_err_24h,
                 "completed_matches": n_matches, "ucl_matches": n_ucl, "competitions": sorted(leagues)},
        "integrations": {
            "espn": {"connected": True, "key_required": False},
            "open_meteo": {"connected": True, "key_required": False},
            "api_football": {"connected": api_football.available(), "key_required": True,
                             "missing_message": None if api_football.available() else
                             "Configura API_FOOTBALL_KEY en el servidor para lesiones y sanciones verificadas."},
            "xg_provider": {"connected": False,
                            "missing_message": "No hay proveedor de xG real conectado (Opta/StatsBomb requieren licencia). Se usa xG-proxy basado en tiros."},
        },
        "model": {"hyperparameters": modeling.get_hyper().as_dict(),
                  "last_backtest": bt and bt.get("created_at")},
        "refresh": {"running": _refresh_state["running"], "last": _refresh_state["last"],
                    "interval_min": settings.refresh_interval_min},
    }


@app.get("/api/matches")
def matches(date_from: str | None = Query(None, alias="from"), date_to: str | None = Query(None, alias="to"),
            team: str | None = None):
    f = datetime.fromisoformat(date_from) if date_from else None
    t = datetime.fromisoformat(date_to) if date_to else None
    return predictions.list_matches(f, t, team)


@app.get("/api/matches/{match_id}")
def match_detail(match_id: str, recompute: bool = False):
    p = None if recompute else predictions.latest_prediction(match_id)
    if p is None:
        try:
            p = predictions.build_prediction(match_id)
        except KeyError:
            raise HTTPException(404, "Partido no encontrado")
    return p


@app.get("/api/matches/{match_id}/history")
def match_history(match_id: str):
    return predictions.prediction_history(match_id)


@app.get("/api/teams")
def teams():
    with SessionLocal() as s:
        ts = s.execute(select(Team).where(Team.in_ucl).order_by(Team.name)).scalars().all()
        return [{"id": t.id, "name": t.name, "logo": t.logo, "abbreviation": t.abbreviation,
                 "domestic_league": t.domestic_league} for t in ts]


@app.get("/api/teams/{team_id}")
def team(team_id: str):
    from .services.team_stats import team_profile
    eng = modeling.get_engine()
    with SessionLocal() as s:
        t = s.get(Team, team_id)
    if t is None:
        raise HTTPException(404, "Equipo no encontrado")
    prof = team_profile(eng, team_id, utcnow())
    prof["team"] = {"id": t.id, "name": t.name, "logo": t.logo}
    return prof


@app.get("/api/standings")
def standings():
    rows = predictions.standings(utcnow())
    with SessionLocal() as s:
        names = {t.id: (t.name, t.logo) for t in s.execute(select(Team)).scalars()}
    for r in rows:
        r["name"], r["logo"] = names.get(r["team_id"], (r["team_id"], None))
    return rows


@app.get("/api/backtest")
def backtest():
    bt = modeling.latest_backtest()
    if bt is None:
        raise HTTPException(404, "Aún no se ha ejecutado el backtest (python -m app.cli backtest)")
    return bt


@app.post("/api/admin/refresh")
def admin_refresh(bg: BackgroundTasks, full: bool = False, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    if _refresh_state["running"]:
        return {"status": "ya en curso"}
    bg.add_task(_run_refresh, full, "actualización manual")
    return {"status": "iniciado", "full": full}


class AvailabilityIn(BaseModel):
    team_id: str
    match_id: str | None = None
    player_name: str
    reason: str | None = None
    status: str  # confirmed | probable | rumor
    source: str


@app.post("/api/admin/availability")
def add_availability(item: AvailabilityIn, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    if item.status not in ("confirmed", "probable", "rumor"):
        raise HTTPException(422, "status debe ser confirmed, probable o rumor")
    with SessionLocal() as s:
        s.add(Availability(**item.model_dump()))
        s.commit()
    if item.match_id:
        predictions.build_prediction(item.match_id)
    return {"ok": True}


# ---------------------------------------------------------------- frontend
DIST = BASE_DIR.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
