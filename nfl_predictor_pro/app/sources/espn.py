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
