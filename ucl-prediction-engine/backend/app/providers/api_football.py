"""Conector opcional a API-Football (api-sports.io) para lesiones y suspensiones.

Requiere API_FOOTBALL_KEY en el entorno del servidor. Sin clave, `available()` es
False y la app declara las bajas como "no verificadas" (reduciendo la confianza).

Clasificación de API-Football -> estado interno:
  - type "Missing Fixture"  -> confirmed (baja confirmada para el partido)
  - type "Questionable"     -> probable (duda)
Los rumores de prensa no se ingieren automáticamente: solo pueden añadirse a mano
vía API con status "rumor" y nunca alteran el modelo.
"""
from __future__ import annotations

import unicodedata
from datetime import datetime

from ..config import settings
from .http import HttpClient

SOURCE = "API-Football (api-sports.io)"
BASE = "https://v3.football.api-sports.io"
UCL_LEAGUE_ID = 2

_client: HttpClient | None = None


def available() -> bool:
    return bool(settings.api_football_key)


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient("api-football", min_interval=0.5,
                             headers={"x-apisports-key": settings.api_football_key})
    return _client


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    for w in (" fc", "fc ", " cf", "cf ", "ac ", " sc", "sc ", "afc ", "1. "):
        s = s.replace(w, " ")
    return " ".join(s.split())


def find_fixture(home_name: str, away_name: str, kickoff: datetime, season: int) -> dict | None:
    js = client().get_json(f"{BASE}/fixtures", params={
        "league": UCL_LEAGUE_ID, "season": season, "date": kickoff.strftime("%Y-%m-%d")})
    h, a = _norm(home_name), _norm(away_name)
    for fx in js.get("response") or []:
        th = _norm(fx["teams"]["home"]["name"])
        ta = _norm(fx["teams"]["away"]["name"])
        if (h in th or th in h) and (a in ta or ta in a):
            return fx
    return None


def injuries_for_fixture(fixture_id: int) -> list[dict]:
    js = client().get_json(f"{BASE}/injuries", params={"fixture": fixture_id})
    out = []
    for it in js.get("response") or []:
        typ = (it.get("player") or {}).get("type") or ""
        status = "confirmed" if typ.lower().startswith("missing") else "probable"
        out.append({
            "team_name": it["team"]["name"],
            "player_name": it["player"]["name"],
            "reason": it["player"].get("reason"),
            "status": status,
        })
    return out
