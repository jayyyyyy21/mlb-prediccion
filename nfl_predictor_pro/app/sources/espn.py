"""Estado de lesiones en tiempo casi real desde el endpoint público de ESPN.

Es un endpoint no documentado oficialmente; se usa sólo como complemento del
reporte oficial de lesiones de la NFL (publicado por nflverse). Si falla, la
aplicación sigue funcionando y muestra una alerta.
"""
from __future__ import annotations

import pandas as pd
import requests

from ..config import ESPN_INJURIES_URL, HTTP_TIMEOUT, USER_AGENT

ESPN_TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}


def injuries() -> pd.DataFrame:
    r = requests.get(ESPN_INJURIES_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    rows = []
    for team_block in r.json().get("injuries", []):
        for item in team_block.get("injuries", []):
            ath = item.get("athlete", {}) or {}
            team = (ath.get("team") or {}).get("abbreviation")
            pos = (ath.get("position") or {}).get("abbreviation")
            espn_id = None
            for link in ath.get("links", []) or []:
                href = link.get("href", "")
                if "/player/_/id/" in href:
                    espn_id = href.split("/id/")[1].split("/")[0]
                    break
            details = item.get("details") or {}
            rows.append(
                {
                    "espn_id": espn_id,
                    "team_name": team_block.get("displayName"),
                    "team": ESPN_TEAM_FIX.get(team, team),
                    "full_name": ath.get("displayName"),
                    "position": pos,
                    "status": item.get("status"),
                    "comment": item.get("shortComment"),
                    "injury_type": details.get("type"),
                    "return_date": details.get("returnDate"),
                    "date": item.get("date"),
                }
            )
    return pd.DataFrame(rows)


ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_DEPTH_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"


def depth_chart_qbs() -> pd.DataFrame:
    """Orden de quarterbacks de cada equipo según el depth chart de ESPN.

    Sirve para saber quién es el titular esperado cuando cambia sin que haya
    una baja en el reporte de lesiones (regreso de un lesionado, cambio de
    titular). Los equipos cuya consulta falla simplemente no aparecen.
    """
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(ESPN_TEAMS_URL, timeout=HTTP_TIMEOUT, headers=headers)
    r.raise_for_status()
    teams = [t["team"] for t in r.json()["sports"][0]["leagues"][0]["teams"]]
    rows = []
    for t in teams:
        abbr = ESPN_TEAM_FIX.get(t["abbreviation"], t["abbreviation"])
        try:
            d = requests.get(ESPN_DEPTH_URL.format(team_id=t["id"]), timeout=HTTP_TIMEOUT, headers=headers)
            d.raise_for_status()
            charts = d.json().get("depthchart", [])
        except (requests.RequestException, ValueError):
            continue
        qbs = next((c["positions"]["qb"].get("athletes", []) for c in charts if "qb" in c.get("positions", {})), [])
        for rank, ath in enumerate(qbs, start=1):
            rows.append({"team": abbr, "rank": rank, "espn_id": str(ath.get("id")), "full_name": ath.get("displayName")})
    return pd.DataFrame(rows, columns=["team", "rank", "espn_id", "full_name"])
