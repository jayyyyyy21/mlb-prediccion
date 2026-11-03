"""Conector de datos públicos de ESPN (site.api.espn.com).

Provee: calendario y resultados de UEFA Champions League y ligas domésticas,
estadísticas por partido (tiros, tiros a puerta, córners, posesión, faltas),
boxscore detallado (pases, centros, balones largos, entradas, intercepciones),
alineaciones por partido, formación y cuotas publicadas (DraftKings).

No proporciona xG, lesiones ni PPDA real: esas carencias se declaran en la app.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .http import HttpClient

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SOURCE = "ESPN (site.api.espn.com)"

_client: HttpClient | None = None


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient("espn", min_interval=0.2)
    return _client


def parse_dt(s: str) -> datetime:
    s = s.replace("Z", "")
    fmt = "%Y-%m-%dT%H:%M:%S" if s.count(":") == 2 else "%Y-%m-%dT%H:%M"
    return datetime.strptime(s, fmt)


def _num(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def american_to_decimal(ml: Any) -> float | None:
    v = _num(str(ml).replace("+", "")) if ml is not None else None
    if v is None or v == 0:
        return None
    return round(1 + v / 100, 4) if v > 0 else round(1 + 100 / abs(v), 4)


def scoreboard(league: str, yyyymm_or_date: str) -> dict:
    return client().get_json(f"{BASE}/{league}/scoreboard", params={"dates": yyyymm_or_date, "limit": 500})


def team_schedule(team_id: str) -> dict:
    """Calendario del equipo en todas las competiciones de la temporada actual."""
    return client().get_json(f"{BASE}/all/teams/{team_id}/schedule")


def summary(league: str, event_id: str) -> dict:
    return client().get_json(f"{BASE}/{league}/summary", params={"event": event_id})


SCOREBOARD_STATS = {
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "wonCorners": "corners",
    "possessionPct": "possession",
    "foulsCommitted": "fouls",
}

BOXSCORE_STATS = {
    **SCOREBOARD_STATS,
    "totalPasses": "passes",
    "accuratePasses": "accurate_passes",
    "totalCrosses": "crosses",
    "totalLongBalls": "long_balls",
    "totalTackles": "tackles",
    "interceptions": "interceptions",
    "totalClearance": "clearances",
    "blockedShots": "blocked_shots",
    "offsides": "offsides",
    "saves": "saves",
    "penaltyKickGoals": "penalty_goals",
    "redCards": "red_cards",
}


def _team_info(t: dict) -> dict:
    logo = t.get("logo")
    if not logo and t.get("logos"):
        logo = t["logos"][0].get("href")
    return {
        "id": str(t["id"]),
        "name": t.get("displayName") or t.get("name"),
        "short_name": t.get("shortDisplayName"),
        "abbreviation": t.get("abbreviation"),
        "logo": logo,
        "color": t.get("color"),
    }


def parse_odds(comp: dict) -> dict | None:
    odds = [o for o in (comp.get("odds") or []) if isinstance(o, dict)]
    if not odds:
        return None
    o = odds[0]
    home = american_to_decimal((o.get("homeTeamOdds") or {}).get("moneyLine"))
    away = american_to_decimal((o.get("awayTeamOdds") or {}).get("moneyLine"))
    draw = american_to_decimal((o.get("drawOdds") or {}).get("moneyLine"))
    total = _num(o.get("overUnder"))
    over = american_to_decimal(o.get("overOdds"))
    under = american_to_decimal(o.get("underOdds"))
    if not any([home, away, draw]):
        return None
    return {
        "provider": (o.get("provider") or {}).get("name"),
        "home": home, "draw": draw, "away": away,
        "total_line": total, "over": over, "under": under,
        "source": SOURCE,
    }


def parse_event(ev: dict, league: str) -> dict | None:
    """Convierte un evento del scoreboard en un dict normalizado."""
    comps = ev.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return None
    status = (comp.get("status") or ev.get("status") or {}).get("type", {})
    teams = {}
    for c in competitors:
        side = c.get("homeAway")
        stats = {}
        for s in c.get("statistics") or []:
            key = SCOREBOARD_STATS.get(s.get("name"))
            if key:
                stats[key] = _num(s.get("displayValue"))
        score = c.get("score")
        if isinstance(score, dict):
            score = score.get("value")
        teams[side] = {"team": _team_info(c["team"]), "score": _num(score), "stats": stats}
    if "home" not in teams or "away" not in teams:
        return None
    venue = comp.get("venue") or {}
    season = ev.get("season") or {}
    return {
        "id": str(ev["id"]),
        "league": league,
        "season": (f"{season['year']}-{(int(season['year']) + 1) % 100:02d}" if season.get("year") else None),
        "phase": season.get("slug") or (ev.get("seasonType") or {}).get("name"),
        "kickoff": parse_dt(ev["date"]),
        "status": status.get("name", "UNKNOWN"),
        "completed": bool(status.get("completed")),
        "neutral": bool(comp.get("neutralSite")),
        "venue": venue.get("fullName"),
        "city": (venue.get("address") or {}).get("city"),
        "country": (venue.get("address") or {}).get("country"),
        "home": teams["home"],
        "away": teams["away"],
        "odds": parse_odds(comp),
    }


def parse_summary(js: dict) -> dict:
    """Boxscore detallado, formación y alineaciones de un partido."""
    out: dict[str, Any] = {"teams": {}, "rosters": {}, "odds": None}
    for t in (js.get("boxscore") or {}).get("teams") or []:
        tid = str(t["team"]["id"])
        stats = {}
        for s in t.get("statistics") or []:
            key = BOXSCORE_STATS.get(s.get("name"))
            if key:
                stats[key] = _num(s.get("displayValue"))
        out["teams"][tid] = stats
    for r in js.get("rosters") or []:
        tid = str(r["team"]["id"])
        players = []
        for p in r.get("roster") or []:
            st = {s["name"]: s.get("value") or 0 for s in p.get("stats") or []}
            players.append({
                "player_id": str(p["athlete"]["id"]),
                "player_name": p["athlete"].get("displayName"),
                "position": (p.get("position") or {}).get("abbreviation"),
                "starter": bool(p.get("starter")),
                "subbed_in": bool(p.get("subbedIn")),
                "goals": float(st.get("totalGoals", 0) or 0),
                "assists": float(st.get("goalAssists", 0) or 0),
                "shots": float(st.get("totalShots", 0) or 0),
                "shots_on_target": float(st.get("shotsOnTarget", 0) or 0),
            })
        out["rosters"][tid] = {"formation": r.get("formation"), "players": players}
    pick = js.get("pickcenter") or []
    if pick:
        out["odds"] = parse_odds({"odds": pick})
    return out
