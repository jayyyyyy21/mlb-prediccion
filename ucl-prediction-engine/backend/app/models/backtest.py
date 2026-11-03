"""Backtesting cronológico (walk-forward) y selección de hiperparámetros.

* Para cada semana con partidos europeos, se ajusta el modelo SOLO con partidos
  anteriores al lunes de esa semana y se predicen los partidos de esa semana.
* Métricas: log loss, Brier, RPS (1X2), calibración por tramos, MAE de totales.
* Comparación contra referencias ingenuas (frecuencias históricas) y contra el
  mercado cuando hay cuotas verificadas.
* Los hiperparámetros se eligen en un periodo de VALIDACIÓN y se reportan en un
  periodo de PRUEBA posterior y separado (evita sobreajustar a la prueba).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from .dataset import Row, rest_days
from .engine import Engine, Hyper

EVAL_LEAGUES = ("uefa.champions", "uefa.europa", "uefa.europa.conf")
EPS = 1e-12


def _week_start(d: datetime) -> datetime:
    d0 = datetime(d.year, d.month, d.day)
    return d0 - timedelta(days=d0.weekday())


def _ll(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def walk_forward(rows: list[Row], hyper: Hyper, start: datetime, end: datetime,
                 leagues=EVAL_LEAGUES, step_weeks: int = 1, market: dict | None = None) -> list[dict]:
    rest = rest_days(rows)
    targets = [r for r in rows if r.league in leagues and start <= r.kickoff < end]
    by_week: dict[datetime, list[Row]] = defaultdict(list)
    for r in targets:
        by_week[_week_start(r.kickoff)].append(r)
    weeks = sorted(by_week)[::step_weeks]
    out = []
    for wk in weeks:
        eng = Engine(hyper=hyper).fit(rows, wk, compute_cov=False)
        # Referencia ingenua: frecuencias del entrenamiento (último año, competiciones europeas)
        tr = [r for r in eng.rows if r.league in leagues and r.kickoff >= wk - timedelta(days=365)]
        base = _base_rates(tr)
        for r in by_week[wk]:
            p = eng.quick_predict(r.home, r.away, r.neutral,
                                  rest.get((r.match_id, r.home), 7), rest.get((r.match_id, r.away), 7), r.league)
            res = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            rec = {
                "match_id": r.match_id, "kickoff": r.kickoff.isoformat(), "league": r.league,
                "home": r.home, "away": r.away, "hg": r.hg, "ag": r.ag, "result": res,
                "p": [p["home_win"], p["draw"], p["away_win"]],
                "p_over_2_5": p["over_2_5"], "p_btts": p["btts"],
                "y_over_2_5": int(r.hg + r.ag > 2.5), "y_btts": int(r.hg > 0 and r.ag > 0),
                "base": base,
                "lam": p["lam"], "mu": p["mu"],
            }
            hc, ac = r.h.get("corners"), r.a.get("corners")
            if hc is not None and ac is not None:
                rec.update(corners_total=p["corners_total"], p_corners_over=p["corners_over"],
                           y_corners=hc + ac, corners_home=p["corners_home"], corners_away=p["corners_away"],
                           y_corners_home=hc, y_corners_away=ac)
            hs, as_ = r.h.get("shots_on_target"), r.a.get("shots_on_target")
            if hs is not None and as_ is not None:
                rec.update(sot_total=p["sot_total"], p_sot_over=p["sot_over"], y_sot=hs + as_,
                           sot_home=p["sot_home"], sot_away=p["sot_away"], y_sot_home=hs, y_sot_away=as_)
            if r.h.get("possession") is not None:
                rec.update(poss_pred=p["poss_home"], poss_obs=r.h["possession"])
            if market and r.match_id in market:
                rec["market"] = market[r.match_id]
            out.append(rec)
    return out


def _base_rates(tr: list[Row]) -> dict:
    if not tr:
        return {"p": [0.45, 0.25, 0.30], "over_2_5": 0.55, "btts": 0.52, "corners_over": 0.5,
                "sot_over": 0.5, "corners_mean": 9.5, "sot_mean": 8.5}
    n = len(tr)
    h = sum(r.hg > r.ag for r in tr) / n
    d = sum(r.hg == r.ag for r in tr) / n
    c = [r.h["corners"] + r.a["corners"] for r in tr if r.h.get("corners") is not None and r.a.get("corners") is not None]
    s = [r.h["shots_on_target"] + r.a["shots_on_target"] for r in tr
         if r.h.get("shots_on_target") is not None and r.a.get("shots_on_target") is not None]
    return {
        "p": [h, d, 1 - h - d],
        "over_2_5": sum(r.hg + r.ag > 2.5 for r in tr) / n,
        "btts": sum(r.hg > 0 and r.ag > 0 for r in tr) / n,
        "corners_over": float(np.mean([x > 9.5 for x in c])) if c else 0.5,
        "sot_over": float(np.mean([x > 8.5 for x in s])) if s else 0.5,
        "corners_mean": float(np.mean(c)) if c else 9.5,
        "sot_mean": float(np.mean(s)) if s else 8.5,
    }


def _calibration(probs: list[float], ys: list[int], bins: int = 10) -> list[dict]:
    probs, ys = np.array(probs), np.array(ys)
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1)
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                    "pred": float(probs[m].mean()), "obs": float(ys[m].mean())})
    return out


def _ece(cal: list[dict]) -> float:
    n = sum(c["n"] for c in cal)
    return float(sum(c["n"] * abs(c["pred"] - c["obs"]) for c in cal) / n) if n else float("nan")


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n = len(recs)

    def onexTwo(key_p):
        ll = br = rps = 0.0
        for r in recs:
            p = key_p(r)
            y = [0, 0, 0]; y[r["result"]] = 1
            ll += -math.log(max(p[r["result"]], EPS))
            br += sum((p[k] - y[k]) ** 2 for k in range(3))
            cp, cy = np.cumsum(p)[:2], np.cumsum(y)[:2]
            rps += float(np.sum((cp - cy) ** 2)) / 2
        return {"log_loss": ll / n, "brier": br / n, "rps": rps / n}

    def binary(pkey, ykey, basekey):
        sub = [r for r in recs if pkey in r]
        if not sub:
            return None
        m = len(sub)
        return {
            "n": m,
            "model": {"log_loss": sum(_ll(r[pkey], r[ykey]) for r in sub) / m,
                      "brier": sum((r[pkey] - r[ykey]) ** 2 for r in sub) / m},
            "baseline": {"log_loss": sum(_ll(r["base"][basekey], r[ykey]) for r in sub) / m,
                         "brier": sum((r["base"][basekey] - r[ykey]) ** 2 for r in sub) / m},
            "calibration": _calibration([r[pkey] for r in sub], [r[ykey] for r in sub], 5),
        }

    for r in recs:
        if "y_corners" in r:
            r["y_corners_over"] = int(r["y_corners"] > 9.5)
        if "y_sot" in r:
            r["y_sot_over"] = int(r["y_sot"] > 8.5)

    all_p, all_y = [], []
    for r in recs:
        for k in range(3):
            all_p.append(r["p"][k]); all_y.append(int(r["result"] == k))
    cal = _calibration(all_p, all_y)

    out = {
        "n": n,
        "period": [min(r["kickoff"] for r in recs), max(r["kickoff"] for r in recs)],
        "result_1x2": {"model": onexTwo(lambda r: r["p"]),
                       "baseline": onexTwo(lambda r: r["base"]["p"]),
                       "calibration": cal, "ece": _ece(cal),
                       "accuracy_argmax": sum(int(np.argmax(r["p"]) == r["result"]) for r in recs) / n},
        "over_2_5": binary("p_over_2_5", "y_over_2_5", "over_2_5"),
        "btts": binary("p_btts", "y_btts", "btts"),
        "corners_over_9_5": binary("p_corners_over", "y_corners_over", "corners_over"),
        "sot_over_8_5": binary("p_sot_over", "y_sot_over", "sot_over"),
    }
    cs = [r for r in recs if "y_corners" in r]
    if cs:
        out["corners_total_mae"] = {"model": float(np.mean([abs(r["corners_total"] - r["y_corners"]) for r in cs])),
                                    "baseline": float(np.mean([abs(r["base"]["corners_mean"] - r["y_corners"]) for r in cs])),
                                    "n": len(cs)}
    ss = [r for r in recs if "y_sot" in r]
    if ss:
        out["sot_total_mae"] = {"model": float(np.mean([abs(r["sot_total"] - r["y_sot"]) for r in ss])),
                                "baseline": float(np.mean([abs(r["base"]["sot_mean"] - r["y_sot"]) for r in ss])),
                                "n": len(ss)}
    ps = [r for r in recs if "poss_obs" in r]
    if ps:
        out["possession_mae"] = {"model": float(np.mean([abs(r["poss_pred"] - r["poss_obs"]) for r in ps])),
                                 "baseline": float(np.mean([abs(50 - r["poss_obs"]) for r in ps])), "n": len(ps)}
    mk = [r for r in recs if r.get("market")]
    if mk:
        out["vs_market"] = {"n": len(mk),
                            "model": onexTwo_subset(mk, lambda r: r["p"]),
                            "market": onexTwo_subset(mk, lambda r: r["market"])}
    return out


def onexTwo_subset(recs, key_p):
    n = len(recs)
    ll = br = 0.0
    for r in recs:
        p = key_p(r)
        y = [0, 0, 0]; y[r["result"]] = 1
        ll += -math.log(max(p[r["result"]], EPS))
        br += sum((p[k] - y[k]) ** 2 for k in range(3))
    return {"log_loss": ll / n, "brier": br / n}


def tune(rows: list[Row], val_start: datetime, val_end: datetime, grid: list[Hyper],
         step_weeks: int = 2, log=print) -> tuple[Hyper, list[dict]]:
    """Elige hiperparámetros minimizando log loss 1X2 + O2.5 en el periodo de validación."""
    table = []
    best, best_score = None, float("inf")
    for h in grid:
        recs = walk_forward(rows, h, val_start, val_end, step_weeks=step_weeks)
        s = summarize(recs)
        score = s["result_1x2"]["model"]["log_loss"] + s["over_2_5"]["model"]["log_loss"]
        table.append({"hyper": h.as_dict(), "n": s["n"], "score": score,
                      "log_loss_1x2": s["result_1x2"]["model"]["log_loss"],
                      "log_loss_o25": s["over_2_5"]["model"]["log_loss"]})
        log(f"  {h.as_dict()} -> {score:.4f} (n={s['n']})")
        if score < best_score:
            best, best_score = h, score
    return best, table


DEFAULT_GRID = [
    Hyper(half_life_days=hl, xg_blend=b, lam_team=lt)
    for hl in (120, 240, 400) for b in (0.0, 0.5, 0.8) for lt in (4.0,)
] + [Hyper(half_life_days=240, xg_blend=0.5, lam_team=lt) for lt in (1.0, 2.0, 8.0)]


def with_style(h: Hyper) -> Hyper:
    return replace(h, style_adjust=True)
