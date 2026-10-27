"""Estadísticas descriptivas por equipo, splits local/visitante y enfrentamientos tácticos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ratings import TEAMS, Snapshot


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def _side_stats(rows: pd.DataFrame, prefix: str = "") -> dict:
    """Agregados de un conjunto de filas equipo-partido (ofensiva = columnas propias)."""
    n = len(rows)
    if n == 0:
        return {"games": 0}
    s = rows.sum(numeric_only=True)
    return {
        "games": n,
        "ppg": float(rows["pts_for"].mean()),
        "pass_ypg": _safe_div(s["pass_yds"], n),
        "rush_ypg": _safe_div(s["rush_yds"], n),
        "ypg": _safe_div(s["pass_yds"] + s["rush_yds"], n),
        "epa_play": _safe_div(s["epa_sum"], s["plays"]),
        "pass_epa": _safe_div(s["pass_epa_sum"], s["db"]),
        "rush_epa": _safe_div(s["rush_epa_sum"], s["rushes"]),
        "success_rate": _safe_div(s["succ_sum"], s["plays"]),
        "rz_td_pct": _safe_div(s["rz_td"], s["rz_trips"]),
        "turnovers_pg": _safe_div(s["ints"] + s["fum_lost"], n),
        "sacks_pg": _safe_div(s["sacks"], n),
        "qb_hits_pg": _safe_div(s["qb_hits"], n),
        "ints_pg": _safe_div(s["ints"], n),
        "third_pct": _safe_div(s["third_conv"], s["third_att"]),
        "drives_pg": _safe_div(s["drives"], n),
        "plays_pg": _safe_div(s["plays"], n),
        "pass_rate": _safe_div(s["db"], s["plays"]),
        "explosive_rate": _safe_div(s["explosive"], s["plays"]),
        "blitz_rate": _safe_div(s.get("blitz_db", np.nan), s.get("charted_db", np.nan)),
        "epa_vs_blitz": _safe_div(s.get("blitz_epa_sum", np.nan), s.get("blitz_db", np.nan)),
    }


def team_games_with_defense(tg: pd.DataFrame) -> pd.DataFrame:
    """Une a cada fila las estadísticas del rival (= lo que permitió la defensa)."""
    opp = tg.drop(columns=["opp", "pts_against", "loc", "rest", "opp_rest", "is_home"], errors="ignore")
    keep = [c for c in opp.columns if c not in ("game_id", "team")]
    opp = opp[["game_id", "team"] + keep].rename(columns={c: f"d_{c}" for c in keep}).rename(columns={"team": "opp"})
    return tg.merge(opp, on=["game_id", "opp"], how="left")


def _defense_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Convierte filas con prefijo d_ (rival) en columnas estándar para _side_stats."""
    cols = {c: c[2:] for c in rows.columns if c.startswith("d_")}
    return rows[list(cols)].rename(columns=cols)


def team_summary(tgd: pd.DataFrame, team: str, season: int, before_week: int | None = None) -> dict:
    rows = tgd[(tgd["team"] == team) & (tgd["season"] == season)]
    if before_week is not None:
        rows = rows[rows["week"] < before_week]
    rows = rows.sort_values("week")
    out: dict = {"team": team, "season": season, "games": int(len(rows))}
    if rows.empty:
        return out
    off, deff = {}, {}
    for label, sub in [("temporada", rows), ("ult3", rows.tail(3)), ("ult5", rows.tail(5)), ("ult10", rows.tail(10)),
                       ("local", rows[rows["loc"] == 1]), ("visitante", rows[rows["loc"] == -1])]:
        off[label] = _side_stats(sub)
        d = _side_stats(_defense_rows(sub)) if len(sub) else {"games": 0}
        deff[label] = d
    # Últimos 10 partidos incluyendo temporada anterior si hay pocos en la actual.
    last_week = int(rows["week"].max())
    hist = tgd[(tgd["team"] == team) & ((tgd["season"] < season) | ((tgd["season"] == season) & (tgd["week"] <= last_week)))]
    hist = hist.sort_values(["season", "week"]).tail(10)
    off["ult10"] = _side_stats(hist)
    deff["ult10"] = _side_stats(_defense_rows(hist))
    out["offense"] = off
    out["defense"] = deff
    w = (rows["pts_for"] > rows["pts_against"]).sum()
    l = (rows["pts_for"] < rows["pts_against"]).sum()
    t = (rows["pts_for"] == rows["pts_against"]).sum()
    out["record"] = f"{w}-{l}" + (f"-{t}" if t else "")
    out["games_log"] = [
        {"week": int(r.week), "opp": r.opp, "loc": int(r.loc), "pf": int(r.pts_for), "pa": int(r.pts_against),
         "epa": _safe_div(r.epa_sum, r.plays), "d_epa": _safe_div(r.d_epa_sum, r.d_plays)}
        for r in rows.itertuples(index=False)
    ]
    return out


def home_away_profile(tgd: pd.DataFrame, team: str, season: int, snap: Snapshot | None,
                      before_week: int | None = None) -> dict:
    """Rendimiento local/visitante con encogimiento por tamaño de muestra.

    Se usan las últimas 3 temporadas. El efecto local propio del equipo se
    encoge hacia el promedio de la liga con un a priori equivalente a 24
    partidos, porque los splits local/visitante son muy ruidosos.
    """
    rows = tgd[(tgd["team"] == team) & (tgd["season"] >= season - 2) & (tgd["season"] <= season)]
    if before_week is not None:
        rows = rows[~((rows["season"] == season) & (rows["week"] >= before_week))]
    lg = tgd[(tgd["season"] >= season - 2) & (tgd["season"] <= season) & (tgd["loc"] != 0)]
    lg_home_margin = float((lg[lg["loc"] == 1]["pts_for"] - lg[lg["loc"] == 1]["pts_against"]).mean())
    out = {"league_home_margin": lg_home_margin}
    pts_rating = snap.get("pts") if snap else None
    for label, loc in (("local", 1), ("visitante", -1)):
        sub = rows[rows["loc"] == loc]
        margin = sub["pts_for"] - sub["pts_against"]
        opp_q = None
        if pts_rating is not None and len(sub):
            # calidad de rivales: margen neto esperado del rival (ofensiva - defensa)
            opp_q = float(np.mean([pts_rating.off.get(o, 0) - pts_rating.deff.get(o, 0) for o in sub["opp"]]))
        out[label] = {
            "games": int(len(sub)),
            "record": f"{int((margin > 0).sum())}-{int((margin < 0).sum())}" + (f"-{int((margin == 0).sum())}" if (margin == 0).any() else ""),
            "ppg": float(sub["pts_for"].mean()) if len(sub) else None,
            "papg": float(sub["pts_against"].mean()) if len(sub) else None,
            "margin": float(margin.mean()) if len(sub) else None,
            "opp_quality": opp_q,
        }
    h, a = out["local"], out["visitante"]
    if h["games"] and a["games"]:
        raw = (h["margin"] - a["margin"]) / 2
        n_eff = 1 / (1 / h["games"] + 1 / a["games"])
        k = 24.0
        shrunk = (raw * n_eff + (lg_home_margin / 2) * k) / (n_eff + k)
        out["home_edge_raw"] = float(raw)
        out["home_edge_shrunk"] = float(shrunk)
        out["sample_note"] = (f"Muestra: {h['games']} partidos de local y {a['games']} de visitante (3 temporadas). "
                              f"El efecto propio se encoge hacia el promedio de la liga ({lg_home_margin / 2:+.1f} pts por equipo).")
    for label, roofs in (("estadio_cerrado", ["dome", "closed"]), ("aire_libre", ["outdoors", "open"])):
        sub = rows[rows["roof"].isin(roofs)]
        out[label] = {"games": int(len(sub)), "ppg": float(sub["pts_for"].mean()) if len(sub) else None,
                      "papg": float(sub["pts_against"].mean()) if len(sub) else None}
    return out


def _z_table(snap: Snapshot, metric: str) -> tuple[dict, dict] | None:
    r = snap.get(metric)
    if r is None:
        return None
    off = np.array([r.off[t] for t in TEAMS])
    de = np.array([r.deff[t] for t in TEAMS])
    zo = {t: float(v) for t, v in zip(TEAMS, (off - off.mean()) / (off.std() or 1))}
    zd = {t: float(v) for t, v in zip(TEAMS, (de - de.mean()) / (de.std() or 1))}
    return zo, zd


def _rank(snap: Snapshot, metric: str, team: str, side: str, higher_better: bool) -> int | None:
    r = snap.get(metric)
    if r is None:
        return None
    vals = r.off if side == "off" else r.deff
    order = sorted(TEAMS, key=lambda t: vals[t], reverse=higher_better)
    return order.index(team) + 1


MATCHUPS = [
    # métrica, título, mayor valor de la ofensiva es bueno para ella?, texto de la ventaja
    ("repa", "Juego terrestre", True, "EPA por acarreo de la ofensiva vs EPA por acarreo permitido"),
    ("pepa", "Juego aéreo", True, "EPA por dropback de la ofensiva vs EPA por pase permitido"),
    ("sack_rate", "Protección vs presión", False, "Capturas permitidas por dropback vs capturas generadas"),
    ("expl", "Jugadas explosivas", True, "Pases de 20+ y carreras de 10+ yardas"),
    ("rz", "Zona roja", True, "% de visitas a zona roja que terminan en TD"),
    ("to_rate", "Pérdidas de balón", False, "Pérdidas por jugada vs balones recuperados por la defensa"),
    ("third", "Terceros downs", True, "Conversión en tercera oportunidad"),
]


def tactical_matchups(snap: Snapshot, tgd: pd.DataFrame, off_team: str, def_team: str, season: int,
                      before_week: int) -> list[dict]:
    """Ventajas tácticas de la ofensiva `off_team` contra la defensa `def_team`.

    Se comparan ratings ajustados por rival en unidades z (desviaciones
    estándar respecto a la liga). Ventaja clara si la suma supera 1.0.
    """
    items = []
    for metric, title, higher_good, desc in MATCHUPS:
        zt = _z_table(snap, metric)
        if zt is None:
            continue
        zo, zd = zt
        sign = 1 if higher_good else -1
        edge = sign * (zo[off_team] + zd[def_team])  # zd alto = la defensa permite más
        r_off = _rank(snap, metric, off_team, "off", higher_good)
        r_def = _rank(snap, metric, def_team, "def", not higher_good)
        if edge > 1.0:
            adv, txt = off_team, f"Ventaja para la ofensiva de {off_team}"
        elif edge < -1.0:
            adv, txt = def_team, f"Ventaja para la defensa de {def_team}"
        else:
            adv, txt = None, "Enfrentamiento equilibrado"
        items.append({
            "metric": metric, "title": title, "description": desc, "edge": round(edge, 2), "advantage": adv,
            "summary": txt, "off_rank": r_off, "def_rank": r_def,
            "detail": f"Ofensiva {off_team}: puesto {r_off} de 32 · Defensa {def_team}: puesto {r_def} de 32 (1 = mejor)",
        })

    rows = tgd[(tgd["season"] == season) & (tgd["week"] < before_week)]
    # Ritmo: posesiones y jugadas por partido
    pace = snap.get("drives")
    if pace is not None:
        po, pd_ = pace.off[off_team], pace.off[def_team]
        if abs(po - pd_) > 0.6:
            fast, slow = (off_team, def_team) if po > pd_ else (def_team, off_team)
            items.append({"metric": "pace", "title": "Ritmo de juego", "advantage": None, "edge": round(po - pd_, 2),
                          "description": "Posesiones por partido (ajustadas por rival)",
                          "summary": f"{fast} juega a más posesiones; {slow} tiende a controlar el reloj",
                          "detail": "Un choque de ritmos reduce o amplía el total de posesiones y, con ello, el total de puntos."})
    # Blitz (charting FTN, temporada actual)
    d_rows = rows[rows["opp"] == def_team]
    o_rows = rows[rows["team"] == off_team]
    if len(d_rows) and len(o_rows) and d_rows["charted_db"].sum() > 0 and o_rows["blitz_db"].sum() >= 20:
        lg_blitz = rows["blitz_db"].sum() / max(rows["charted_db"].sum(), 1)
        d_blitz = d_rows["blitz_db"].sum() / d_rows["charted_db"].sum()
        qb_vs = o_rows["blitz_epa_sum"].sum() / o_rows["blitz_db"].sum()
        lg_vs = rows["blitz_epa_sum"].sum() / max(rows["blitz_db"].sum(), 1)
        if d_blitz > lg_blitz * 1.15:
            adv = off_team if qb_vs > lg_vs + 0.05 else (def_team if qb_vs < lg_vs - 0.05 else None)
            items.append({"metric": "blitz", "title": "Blitz vs quarterback", "advantage": adv,
                          "edge": round((qb_vs - lg_vs) * 10, 2),
                          "description": "Frecuencia de blitz de la defensa y EPA del QB contra blitz (charting FTN)",
                          "summary": (f"{def_team} manda blitz en {d_blitz:.0%} de dropbacks (liga {lg_blitz:.0%}); "
                                      f"{off_team} promedia {qb_vs:+.2f} EPA contra blitz (liga {lg_vs:+.2f})"),
                          "detail": "Muestra de la temporada actual; usar con cautela."})
    # Defensa que cede yardas pero pocos puntos (bend-don't-break) y ofensiva dependiente de explosivas
    d_all = rows[rows["opp"] == def_team]
    lg_ypp = (rows["pass_yds"].sum() + rows["rush_yds"].sum()) / max(rows["plays"].sum(), 1)
    lg_ppd = rows["pts_for"].sum() / max(rows["drives"].sum(), 1)
    if len(d_all) >= 3:
        ypp = (d_all["pass_yds"].sum() + d_all["rush_yds"].sum()) / max(d_all["plays"].sum(), 1)
        ppd = d_all["pts_for"].sum() / max(d_all["drives"].sum(), 1)
        if ypp > lg_ypp * 1.04 and ppd < lg_ppd * 0.95:
            items.append({"metric": "bend", "title": "Defensa que cede yardas pero no puntos", "advantage": def_team,
                          "edge": 0, "description": "Yardas por jugada permitidas altas, puntos por posesión bajos",
                          "summary": f"{def_team} permite {ypp:.1f} yds/jugada (liga {lg_ypp:.1f}) pero sólo {ppd:.2f} pts/posesión (liga {lg_ppd:.2f})",
                          "detail": "Suele indicar buena defensa en zona roja; parte del efecto puede ser suerte y tiende a regresar a la media."})
    if len(o_rows) >= 3:
        expl = o_rows["explosive"].sum() / max(o_rows["plays"].sum(), 1)
        sr = o_rows["succ_sum"].sum() / max(o_rows["plays"].sum(), 1)
        lg_expl = rows["explosive"].sum() / max(rows["plays"].sum(), 1)
        lg_sr = rows["succ_sum"].sum() / max(rows["plays"].sum(), 1)
        if expl > lg_expl * 1.1 and sr < lg_sr:
            items.append({"metric": "explosive_dep", "title": "Ofensiva dependiente de jugadas explosivas",
                          "advantage": None, "edge": 0, "description": "Mucha explosividad con tasa de éxito baja",
                          "summary": f"{off_team}: {expl:.1%} de jugadas explosivas (liga {lg_expl:.1%}) con success rate {sr:.1%} (liga {lg_sr:.1%})",
                          "detail": "Producción más volátil: aumenta la incertidumbre de sus puntos."})
    return items


def similar_opponents(snap: Snapshot, tgd: pd.DataFrame, def_team: str, upcoming_off: str, season: int,
                      before_week: int, k: int = 3) -> dict | None:
    """Rendimiento de la defensa de `def_team` contra ofensivas parecidas a `upcoming_off`.

    Similitud por estilo: tasa de pase y EPA/jugada ajustados por rival.
    """
    pr, ep = snap.get("pass_rate"), snap.get("epa")
    if pr is None or ep is None:
        return None
    rows = tgd[(tgd["opp"] == def_team) & (tgd["season"] == season) & (tgd["week"] < before_week)]
    if rows.empty:
        return None
    pr_sd = np.std(list(pr.off.values())) or 1
    ep_sd = np.std(list(ep.off.values())) or 1

    def dist(t):
        return np.hypot((pr.off[t] - pr.off[upcoming_off]) / pr_sd, (ep.off[t] - ep.off[upcoming_off]) / ep_sd)

    rows = rows.assign(dist=rows["team"].map(dist)).sort_values("dist").head(k)
    pts = snap.get("pts")
    items = []
    for r in rows.itertuples(index=False):
        exp = pts.mu + pts.off[r.team] if pts else np.nan
        items.append({"week": int(r.week), "opp": r.team, "pts_allowed": int(r.pts_for), "opp_typical": float(exp),
                      "similarity": float(r.dist)})
    if not items:
        return None
    diff = float(np.mean([i["pts_allowed"] - i["opp_typical"] for i in items]))
    return {"games": items, "avg_vs_typical": diff,
            "note": (f"Contra ofensivas de estilo parecido a {upcoming_off}, la defensa de {def_team} permitió "
                     f"{abs(diff):.1f} pts {'más' if diff > 0 else 'menos'} de lo que esas ofensivas anotan normalmente "
                     f"(muestra de {len(items)} partidos: orientativo).")}
