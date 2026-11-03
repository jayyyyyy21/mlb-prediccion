"""Detector de oportunidades estadísticas y nivel de confianza.

Principios:
* Alta probabilidad ≠ valor. El 'valor' solo se evalúa cuando hay cuotas verificadas.
* Se usa la banda inferior (p10) de la probabilidad para exigir margen ante la
  incertidumbre del modelo.
* Si ningún mercado supera los filtros, la recomendación es NO BET.
"""
from __future__ import annotations

import math

MIN_EDGE = 0.03  # ventaja mínima sobre la probabilidad implícita (sin margen), con p10


def confidence(pred: dict, prof_h: dict, prof_a: dict, av_h: dict, av_a: dict, backtest: dict | None,
               injuries_source: bool) -> dict:
    reasons, level_points = [], []

    def add(level: str, text: str):
        level_points.append(level)
        reasons.append({"level": level, "text": text})

    for name, prof in (("local", prof_h), ("visitante", prof_a)):
        n = prof["matches_with_stats_365d"]
        if prof["league_group"].startswith("otros"):
            add("low", f"Equipo {name}: su liga doméstica no está cubierta por la fuente; solo partidos europeos ({n} con estadísticas).")
        elif n < 12:
            add("low", f"Equipo {name}: solo {n} partidos con estadísticas en 365 días.")
        elif n < 25:
            add("medium", f"Equipo {name}: {n} partidos con estadísticas en 365 días.")
        else:
            add("high", f"Equipo {name}: {n} partidos con estadísticas en 365 días.")
    sd = max(pred["goals"]["home_xg_sd_log"], pred["goals"]["away_xg_sd_log"])
    if sd > 0.30:
        add("low", f"Incertidumbre de parámetros alta (σ log-goles {sd:.2f}).")
    elif sd > 0.20:
        add("medium", f"Incertidumbre de parámetros moderada (σ log-goles {sd:.2f}).")
    else:
        add("high", f"Incertidumbre de parámetros baja (σ log-goles {sd:.2f}).")
    if av_h["lineup_confirmed"] and av_a["lineup_confirmed"]:
        add("high", "Alineaciones confirmadas.")
    else:
        add("medium", "Alineaciones aún no confirmadas.")
    if not injuries_source:
        add("medium", "Bajas no verificadas: falta API_FOOTBALL_KEY (fuente de lesiones/sanciones).")
    if av_h["goalkeeper_absent"] or av_a["goalkeeper_absent"]:
        add("low", "Ausencia del portero titular: impacto no cuantificado por el modelo.")
    if backtest:
        t = backtest.get("test", {}).get("result_1x2", {})
        m, b = t.get("model", {}).get("log_loss"), t.get("baseline", {}).get("log_loss")
        if m is not None and b is not None:
            if m < b:
                add("high", f"Backtest: log loss 1X2 {m:.3f} mejor que la referencia {b:.3f}.")
            else:
                add("low", f"Backtest: el modelo NO mejora la referencia (log loss {m:.3f} vs {b:.3f}).")
    else:
        add("medium", "Sin backtest ejecutado todavía.")

    lows = level_points.count("low")
    meds = level_points.count("medium")
    level = "Baja" if lows >= 1 else ("Alta" if meds <= 1 else "Media")
    return {"level": level, "reasons": reasons}


def _implied(odds: dict) -> dict | None:
    if not odds or not all(odds.get(k) for k in ("home", "draw", "away")):
        return None
    inv = {k: 1 / odds[k] for k in ("home", "draw", "away")}
    tot = sum(inv.values())
    out = {k: v / tot for k, v in inv.items()}
    out["margin"] = tot - 1
    if odds.get("total_line") and odds.get("over") and odds.get("under"):
        io, iu = 1 / odds["over"], 1 / odds["under"]
        out["total_line"] = odds["total_line"]
        out["over"] = io / (io + iu)
        out["under"] = iu / (io + iu)
    return out


def _family_stability(backtest: dict | None) -> dict:
    """¿El modelo superó a la referencia en el periodo de prueba para cada familia de mercado?"""
    out = {}
    if not backtest:
        return out
    t = backtest.get("test", {})
    r = t.get("result_1x2") or {}
    if r:
        out["1x2"] = r["model"]["log_loss"] < r["baseline"]["log_loss"]
    for fam, key in (("goals_total", "over_2_5"), ("btts", "btts"), ("corners", "corners_over_9_5"),
                     ("sot", "sot_over_8_5")):
        b = t.get(key)
        if b:
            out[fam] = b["model"]["log_loss"] < b["baseline"]["log_loss"]
    return out


def _count_band(p: float, sd_log: float) -> tuple[float, float]:
    # Aproximación de la banda para mercados de conteo a partir de la σ del log-medio.
    w = min(0.25, 0.9 * sd_log * math.sqrt(max(p * (1 - p), 1e-6)) * 2)
    return max(0.0, p - w), min(1.0, p + w)


def opportunities(pred: dict, names: tuple[str, str], odds: dict | None, conf: dict,
                  backtest: dict | None) -> dict:
    nh, na = names
    g, c, s = pred["goals"], pred["corners"], pred["shots_on_target"]
    bands = g["bands_p10_p90"]
    stab = _family_stability(backtest)
    imp = _implied(odds) if odds else None
    sdg = max(g["home_xg_sd_log"], g["away_xg_sd_log"])
    markets = []

    def add(family, label, p, band, implied=None, odds_dec=None):
        m = {"family": family, "market": label, "prob": p, "band": band,
             "stable_in_backtest": stab.get(family), "implied": implied, "odds": odds_dec}
        if implied is not None:
            m["edge"] = p - implied
            m["edge_conservative"] = band[0] - implied
        markets.append(m)

    add("1x2", f"Gana {nh}", g["home_win"], bands["home_win"], imp and imp["home"], odds and odds.get("home"))
    add("1x2", "Empate", g["draw"], bands["draw"], imp and imp["draw"], odds and odds.get("draw"))
    add("1x2", f"Gana {na}", g["away_win"], bands["away_win"], imp and imp["away"], odds and odds.get("away"))
    hb, db, ab = bands["home_win"], bands["draw"], bands["away_win"]
    add("1x2", f"Doble oportunidad 1X ({nh} o empate)", g["home_win"] + g["draw"],
        [hb[0] + db[0], min(1, hb[1] + db[1])], imp and (imp["home"] + imp["draw"]))
    add("1x2", f"Doble oportunidad X2 (empate o {na})", g["away_win"] + g["draw"],
        [ab[0] + db[0], min(1, ab[1] + db[1])], imp and (imp["away"] + imp["draw"]))
    add("1x2", "Doble oportunidad 12 (no empate)", g["home_win"] + g["away_win"],
        [1 - db[1], 1 - db[0]], imp and (imp["home"] + imp["away"]))
    for line, key in ((1.5, "over_1_5"), (2.5, "over_2_5"), (3.5, "over_3_5")):
        i_over = imp["over"] if imp and imp.get("total_line") == line else None
        add("goals_total", f"Más de {line} goles", g[key], bands[key], i_over,
            odds and odds.get("over") if i_over is not None else None)
        add("goals_total", f"Menos de {line} goles", 1 - g[key], [1 - bands[key][1], 1 - bands[key][0]],
            imp["under"] if i_over is not None else None, odds and odds.get("under") if i_over is not None else None)
    add("btts", "Ambos marcan: Sí", g["btts"], bands["btts"])
    add("btts", "Ambos marcan: No", 1 - g["btts"], [1 - bands["btts"][1], 1 - bands["btts"][0]])
    for line, p in c["total_over"].items():
        add("corners", f"Córners totales más de {line}", p, list(_count_band(p, sdg)))
        add("corners", f"Córners totales menos de {line}", 1 - p, list(_count_band(1 - p, sdg)))
    for side, nm in (("home", nh), ("away", na)):
        for line, p in c[f"{side}_over"].items():
            add("corners", f"Córners {nm} más de {line}", p, list(_count_band(p, sdg)))
        for line, p in s[f"{side}_over"].items():
            add("sot", f"Tiros a puerta {nm} más de {line}", p, list(_count_band(p, sdg)))
    add("corners", f"{nh} más córners que {na}", c["home_more"], list(_count_band(c["home_more"], sdg)))
    add("corners", f"{na} más córners que {nh}", c["away_more"], list(_count_band(c["away_more"], sdg)))
    for line, p in s["total_over"].items():
        add("sot", f"Tiros a puerta totales más de {line}", p, list(_count_band(p, sdg)))

    quality = {"Alta": 1.0, "Media": 0.8, "Baja": 0.55}[conf["level"]]
    for m in markets:
        width = m["band"][1] - m["band"][0]
        stable = 1.0 if m["stable_in_backtest"] else (0.7 if m["stable_in_backtest"] is False else 0.85)
        # Respaldo estadístico: probabilidad conservadora (p10) penalizada por calidad y estabilidad
        m["support_score"] = m["band"][0] * quality * stable
        m["uncertainty"] = width
        m["data_quality"] = conf["level"]
        if m.get("implied") is not None:
            ok = (m["edge_conservative"] > MIN_EDGE and conf["level"] != "Baja" and m["stable_in_backtest"])
            m["verdict"] = "VALOR" if ok else "Sin ventaja suficiente"
        else:
            m["verdict"] = "Sin cuota (valor no evaluable)"

    by_support = sorted(markets, key=lambda m: -m["support_score"])
    value = sorted([m for m in markets if m["verdict"] == "VALOR"], key=lambda m: -m["edge_conservative"])
    rec = {
        "decision": "BET" if value else "NO BET",
        "picks": value[:3],
        "explanation": (
            "Hay mercados con cuota verificada donde la probabilidad conservadora (p10) supera la implícita "
            f"sin margen en más de {MIN_EDGE * 100:.0f} pp, con confianza suficiente y familia estable en backtest."
            if value else
            "NO BET: ningún mercado con cuota verificada muestra una ventaja que sobreviva a la incertidumbre del "
            "modelo, a la calidad de datos y a la estabilidad del backtest. Una probabilidad alta no implica valor."),
    }
    return {"markets": by_support, "recommendation": rec, "implied": imp,
            "odds_source": odds.get("source") + f" · {odds.get('provider')}" if odds else None}
