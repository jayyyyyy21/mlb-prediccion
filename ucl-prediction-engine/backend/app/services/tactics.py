"""Análisis táctico: identificación de estilos y cruce de estilos con evidencia cuantitativa.

Todas las etiquetas se derivan de percentiles respecto a los equipos de la UCL actual,
calculados con boxscores observados. Las limitaciones se declaran explícitamente:
- 'Presión' usa un PPDA APROXIMADO de todo el campo (pases rivales / (entradas +
  intercepciones + faltas)); el PPDA real (solo en campo rival) no está en la fuente.
- No hay datos de balón parado, errores defensivos ni mapas de tiro en la fuente.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from ..models.engine import Engine
from .team_stats import team_profile

STYLE_METRICS = {
    "possession": ("Posesión media", 1),
    "passes": ("Pases por partido", 1),
    "pass_acc": ("Precisión de pase", 1),
    "long_ball_share": ("% balones largos", 1),
    "crosses": ("Centros por partido", 1),
    "ppda_approx": ("PPDA aprox. (menor = más presión)", -1),
    "def_actions": ("Acciones defensivas", 1),
    "clearances": ("Despejes", 1),
}

_population_cache: dict = {}


def population(eng: Engine, ucl_teams: list[str], before: datetime) -> dict[str, dict]:
    key = (id(eng), before.date(), tuple(sorted(ucl_teams)))
    if key in _population_cache:
        return _population_cache[key]
    prof = {}
    for t in ucl_teams:
        p = team_profile(eng, t, before)
        st = dict(p["style"])
        w10 = p["windows"]["last15"]
        st["shots"] = w10.get("shots_for")
        st["corners"] = w10.get("corners_for")
        st["xg_proxy_for"] = w10.get("xg_proxy_for")
        prof[t] = st
    _population_cache.clear()
    _population_cache[key] = prof
    return prof


def percentile(pop: dict[str, dict], metric: str, value) -> float | None:
    vals = [v.get(metric) for v in pop.values() if v.get(metric) is not None]
    if value is None or len(vals) < 5:
        return None
    return float((np.array(vals) < value).mean() * 100)


def style_tags(style: dict, pop: dict) -> list[dict]:
    def pc(m):
        return percentile(pop, m, style.get(m))

    tags = []
    poss, pa, lb, cr = pc("possession"), pc("pass_acc"), pc("long_ball_share"), pc("crosses")
    ppda, cl, sh = pc("ppda_approx"), pc("clearances"), pc("shots")

    def add(name, cond, why):
        if cond:
            tags.append({"tag": name, "evidence": why})

    if poss is None:
        return [{"tag": "Sin datos de estilo suficientes", "evidence": "No hay boxscores detallados recientes."}]
    add("Posesión dominante", poss >= 70, f"Posesión en percentil {poss:.0f} de la UCL")
    add("Ataque posicional", poss >= 60 and (pa or 0) >= 65,
        f"Posesión p{poss:.0f} y precisión de pase p{(pa or 0):.0f}")
    add("Presión alta / intensa (aprox.)", ppda is not None and ppda <= 30,
        f"PPDA aproximado en percentil {ppda:.0f} (bajo = presiona más)" if ppda is not None else "")
    add("Bloque bajo / repliegue", poss <= 30 and (cl or 0) >= 60,
        f"Posesión p{poss:.0f} y despejes p{(cl or 0):.0f}")
    add("Juego directo", lb is not None and lb >= 70, f"% de balones largos p{lb:.0f}" if lb is not None else "")
    add("Juego por bandas / centros", cr is not None and cr >= 70, f"Centros por partido p{cr:.0f}" if cr is not None else "")
    add("Transiciones / contraataque", poss <= 40 and (sh or 0) >= 50,
        f"Genera tiros (p{(sh or 0):.0f}) con poca posesión (p{poss:.0f})")
    add("Alto volumen ofensivo", sh is not None and sh >= 75, f"Tiros por partido p{sh:.0f}" if sh is not None else "")
    if not tags:
        tags.append({"tag": "Perfil equilibrado", "evidence": f"Sin métricas extremas (posesión p{poss:.0f})"})
    return tags


def _fmt(x, nd=2):
    return "n/d" if x is None else f"{x:.{nd}f}"


def matchup(eng: Engine, home: str, away: str, names: dict, pred: dict, prof_h: dict, prof_a: dict,
            pop: dict) -> dict:
    nh, na = names[home], names[away]
    sh, sa = prof_h["style"], prof_a["style"]
    tags_h, tags_a = style_tags(sh, pop), style_tags(sa, pop)
    g = pred["goals"]
    findings: list[dict] = []

    # 1. Control del balón vs. generación de ocasiones
    ph = pred["possession"]["home"]
    ctrl = nh if ph >= 52 else (na if ph <= 48 else None)
    sot_h, sot_a = pred["shots_on_target"]["home_exp"], pred["shots_on_target"]["away_exp"]
    chance = nh if g["home_xg"] > g["away_xg"] * 1.1 else (na if g["away_xg"] > g["home_xg"] * 1.1 else None)
    txt = (f"Posesión estimada {ph:.0f}%–{100 - ph:.0f}%. "
           f"{'Control territorial probable de ' + ctrl if ctrl else 'Posesión repartida'}; "
           f"goles esperados {g['home_xg']:.2f}–{g['away_xg']:.2f} y tiros a puerta {sot_h:.1f}–{sot_a:.1f}. ")
    if ctrl and chance and ctrl != chance:
        txt += f"Ojo: {ctrl} tendría más balón pero {chance} genera más peligro esperado — posesión no es igual a ocasiones."
    findings.append({"title": "Posesión vs. ocasiones", "text": txt, "favors": chance})

    # 2. Ataque vs defensa (ratings del modelo, ajustados por rival)
    rh, ra = prof_h["rating"], prof_a["rating"]
    att_edge_h = rh["attack"] - ra["defense"]
    att_edge_a = ra["attack"] - rh["defense"]
    findings.append({
        "title": "Ataque vs. defensa (ajustado por calidad de rivales)",
        "text": (f"Ataque de {nh} frente a la defensa de {na}: índice {att_edge_h:+.2f}; "
                 f"ataque de {na} frente a la defensa de {nh}: {att_edge_a:+.2f} "
                 "(escala log de goles; incluye ventaja de liga estimada con partidos europeos)."),
        "favors": nh if att_edge_h > att_edge_a + 0.05 else (na if att_edge_a > att_edge_h + 0.05 else None),
    })

    # 3. Cruces de estilo con evidencia de splits
    def split_text(team_name, prof, opp_tier_key, label):
        sp = prof["splits"][opp_tier_key]
        base = prof["windows"]["last15"]
        if sp.get("n", 0) < 3:
            return None, f"{team_name}: muestra insuficiente frente a {label} (n={sp.get('n', 0)})."
        d_for = (sp.get("xg_proxy_for") or 0) - (base.get("xg_proxy_for") or 0)
        d_ag = (sp.get("xg_proxy_against") or 0) - (base.get("xg_proxy_against") or 0)
        t = (f"{team_name} frente a {label} (n={sp['n']}): xG-proxy a favor {_fmt(sp.get('xg_proxy_for'))} "
             f"(vs {_fmt(base.get('xg_proxy_for'))} general), en contra {_fmt(sp.get('xg_proxy_against'))} "
             f"(vs {_fmt(base.get('xg_proxy_against'))}). Puntos/partido {sp['ppg']:.2f}.")
        return d_for - d_ag, t

    tier_a, tier_h = eng.poss_tier(away), eng.poss_tier(home)
    key_for = {2: ("vs_possession_teams", "equipos de mucha posesión"),
               0: ("vs_low_possession_teams", "equipos de poca posesión"), 1: (None, None)}
    for team, prof, opp_tier, opp_name in ((nh, prof_h, tier_a, na), (na, prof_a, tier_h, nh)):
        k, label = key_for.get(opp_tier, (None, None)) if opp_tier is not None else (None, None)
        if not k:
            continue
        delta, t = split_text(team, prof, k, label)
        findings.append({"title": f"{team} ante el perfil de {opp_name}", "text": t,
                         "favors": (team if delta and delta > 0.2 else None), "delta": delta})

    # 4. Vulnerabilidades concretas
    vul = []
    for team, prof, opp, oprof in ((nh, prof_h, na, prof_a), (na, prof_a, nh, prof_h)):
        w = prof["windows"]["last10"]
        if w.get("n", 0) < 5:
            continue
        ow = oprof["windows"]["last10"]
        if (w.get("sot_against") or 0) >= 5 and (ow.get("sot_for") or 0) >= 5:
            vul.append(f"{team} concede {w['sot_against']:.1f} tiros a puerta/partido y {opp} genera {ow['sot_for']:.1f}.")
        if (w.get("corners_against") or 0) >= 5.5 and (ow.get("corners_for") or 0) >= 5.5:
            vul.append(f"{team} concede {w['corners_against']:.1f} córners/partido; {opp} fuerza {ow['corners_for']:.1f}.")
        if (prof["style"].get("ppda_approx") or 99) > 0 and (oprof["style"].get("pass_acc") or 0) > 0:
            p_ppda = percentile(pop, "ppda_approx", prof["style"].get("ppda_approx"))
            o_pa = percentile(pop, "pass_acc", oprof["style"].get("pass_acc"))
            if p_ppda is not None and o_pa is not None and p_ppda <= 30 and o_pa <= 35:
                vul.append(f"{team} presiona intensamente (PPDA aprox. p{p_ppda:.0f}) y {opp} tiene precisión de pase baja (p{o_pa:.0f}): posibles recuperaciones altas.")
        if prof["windows"]["last10"].get("clean_sheet_pct", 0) <= 0.1:
            vul.append(f"{team} solo dejó la portería a cero en {w['clean_sheet_pct'] * 100:.0f}% de sus últimos 10.")

    # Conclusión
    votes = [f["favors"] for f in findings if f.get("favors")]
    pw = {nh: g["home_win"], na: g["away_win"]}
    leader = max(pw, key=pw.get)
    margin = abs(g["home_win"] - g["away_win"])
    if margin < 0.08:
        concl = (f"Partido equilibrado según el modelo ({g['home_win'] * 100:.0f}% / {g['draw'] * 100:.0f}% / "
                 f"{g['away_win'] * 100:.0f}%). Ninguna ventaja táctica cuantificable es decisiva.")
    else:
        support = [f["title"] for f in findings if f.get("favors") == leader]
        concl = (f"Ventaja para {leader} ({pw[leader] * 100:.0f}% de victoria). "
                 + (f"La sostienen: {'; '.join(support)}." if support
                    else "Se apoya sobre todo en la diferencia de ratings ajustados por rival."))
    contra = [f["title"] for f in findings if f.get("favors") and f["favors"] != leader]
    if contra and margin >= 0.08:
        other = na if leader == nh else nh
        concl += f" Sin embargo, {other} tiene argumentos: {'; '.join(contra)}."
    return {
        "home_style": {"tags": tags_h, "formation": sh.get("formations"), "metrics": _metrics(sh, pop)},
        "away_style": {"tags": tags_a, "formation": sa.get("formations"), "metrics": _metrics(sa, pop)},
        "findings": findings,
        "vulnerabilities": vul,
        "conclusion": concl,
        "limitations": [
            "PPDA aproximado (todo el campo); el PPDA real requiere datos de eventos por zona.",
            "Sin datos de balón parado, errores defensivos ni xG real (Opta/Understat no accesibles): se usa xG-proxy basado en tiros a puerta.",
            "Estilos calculados con los últimos ≤15 partidos con boxscore detallado.",
        ],
    }


def _metrics(style: dict, pop: dict) -> list[dict]:
    out = []
    for k, (label, _) in STYLE_METRICS.items():
        v = style.get(k)
        out.append({"key": k, "label": label, "value": v, "percentile": percentile(pop, k, v)})
    return out
