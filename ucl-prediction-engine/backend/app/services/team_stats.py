"""Perfiles estadísticos observados por equipo (ofensivo, defensivo, estilo)."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..models.dataset import Row
from ..models.engine import Engine


def _team_rows(rows: list[Row], team: str, before: datetime) -> list[tuple[Row, bool]]:
    out = [(r, r.home == team) for r in rows if (r.home == team or r.away == team) and r.kickoff < before]
    out.sort(key=lambda x: x[0].kickoff, reverse=True)
    return out


def _side(r: Row, is_home: bool):
    if is_home:
        return r.h, r.a, r.hg, r.ag, r.away
    return r.a, r.h, r.ag, r.hg, r.home


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def window_stats(items: list[tuple[Row, bool]], xg) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    gf, ga, xgf, xga, sh, sha, sot, sota, co, coa, poss = ([] for _ in range(11))
    pts = w = d = l = cs = fts = 0
    scored = {1: 0, 2: 0, 3: 0}
    for r, ih in items:
        me, opp, g_for, g_ag, _ = _side(r, ih)
        gf.append(g_for); ga.append(g_ag)
        xgf.append(xg.value(me)); xga.append(xg.value(opp))
        sh.append(me.get("shots")); sha.append(opp.get("shots"))
        sot.append(me.get("shots_on_target")); sota.append(opp.get("shots_on_target"))
        co.append(me.get("corners")); coa.append(opp.get("corners"))
        poss.append(me.get("possession"))
        if g_for > g_ag:
            w += 1; pts += 3
        elif g_for == g_ag:
            d += 1; pts += 1
        else:
            l += 1
        cs += g_ag == 0
        fts += g_for == 0
        for k in scored:
            scored[k] += g_for >= k
    shots_total = sum(x for x in sh if x is not None)
    goals_with_shots = sum(g for g, s in zip(gf, sh) if s is not None)
    sot_total = sum(x for x in sot if x is not None)
    return {
        "n": n, "w": w, "d": d, "l": l, "ppg": pts / n,
        "goals_for": float(np.mean(gf)), "goals_against": float(np.mean(ga)),
        "xg_proxy_for": _mean(xgf), "xg_proxy_against": _mean(xga),
        "shots_for": _mean(sh), "shots_against": _mean(sha),
        "sot_for": _mean(sot), "sot_against": _mean(sota),
        "corners_for": _mean(co), "corners_against": _mean(coa),
        "possession": _mean(poss),
        "conversion": goals_with_shots / shots_total if shots_total else None,
        "sot_accuracy": sot_total / shots_total if shots_total else None,
        "clean_sheet_pct": cs / n, "failed_to_score_pct": fts / n,
        "scored_1plus_pct": scored[1] / n, "scored_2plus_pct": scored[2] / n, "scored_3plus_pct": scored[3] / n,
    }


def detail_stats(items: list[tuple[Row, bool]]) -> dict:
    """Métricas de estilo del boxscore detallado (solo partidos con boxscore)."""
    det = [(r, ih) for r, ih in items if _side(r, ih)[0].get("passes") is not None][:15]
    if not det:
        return {"n": 0}
    acc = {k: [] for k in ("passes", "pass_acc", "crosses", "long_ball_share", "def_actions",
                           "ppda_approx", "clearances", "blocked_shots_against", "fouls", "possession")}
    forms: dict[str, int] = {}
    for r, ih in det:
        me, opp, *_ = _side(r, ih)
        p = me.get("passes") or 0
        acc["passes"].append(p)
        if p:
            acc["pass_acc"].append((me.get("accurate_passes") or 0) / p)
            acc["long_ball_share"].append((me.get("long_balls") or 0) / p)
        acc["crosses"].append(me.get("crosses"))
        da = (me.get("tackles") or 0) + (me.get("interceptions") or 0) + (me.get("fouls") or 0)
        acc["def_actions"].append(da)
        if da and opp.get("passes"):
            acc["ppda_approx"].append(opp["passes"] / da)
        acc["clearances"].append(me.get("clearances"))
        acc["blocked_shots_against"].append(me.get("blocked_shots"))
        acc["fouls"].append(me.get("fouls"))
        acc["possession"].append(me.get("possession"))
        if me.get("formation"):
            forms[me["formation"]] = forms.get(me["formation"], 0) + 1
    out = {k: _mean(v) for k, v in acc.items()}
    out["n"] = len(det)
    out["formations"] = sorted(forms.items(), key=lambda x: -x[1])
    return out


def team_profile(eng: Engine, team: str, before: datetime) -> dict:
    items = _team_rows(eng.rows, team, before)
    season_start = datetime(before.year if before.month >= 7 else before.year - 1, 7, 1)
    home = [x for x in items if x[1] and not x[0].neutral]
    away = [x for x in items if not x[1] and not x[0].neutral]
    europe = [x for x in items if x[0].league.startswith("uefa")]

    # Rivales fuertes / similares según el rating del modelo (ataque + defensa)
    def strength(t):
        rt = eng.goals.ratings(t)
        return rt["attack"] + rt["defense"]

    my_strength = strength(team)
    all_str = np.array([strength(t) for t in eng.goals.teams])
    strong_cut = float(np.percentile(all_str, 85))
    vs_strong = [x for x in items if strength(_side(*x)[4]) >= strong_cut]
    vs_similar = [x for x in items if abs(strength(_side(*x)[4]) - my_strength) <= 0.15]
    vs_possession = [x for x in items if eng.poss_tier(_side(*x)[4]) == 2]
    vs_low_block = [x for x in items if eng.poss_tier(_side(*x)[4]) == 0]

    form = []
    for r, ih in items[:5]:
        me, opp, gf, ga, opp_id = _side(r, ih)
        form.append({"date": r.kickoff.isoformat(), "league": r.league, "opponent": opp_id,
                     "home": ih, "gf": gf, "ga": ga, "res": "W" if gf > ga else ("D" if gf == ga else "L")})

    last14 = [x for x in items if x[0].kickoff >= before - timedelta(days=14)]
    last30 = [x for x in items if x[0].kickoff >= before - timedelta(days=30)]
    rest = (before - items[0][0].kickoff).total_seconds() / 86400 if items else None
    rt = eng.goals.ratings(team)
    pct = float((all_str < my_strength).mean() * 100)
    return {
        "team_id": team,
        "league_group": eng.leagues.get(team, "otros"),
        "rating": {"attack": rt["attack"], "defense": rt["defense"], "overall_percentile": pct,
                   "in_model": rt["known"]},
        "windows": {
            "last5": window_stats(items[:5], eng.xg),
            "last10": window_stats(items[:10], eng.xg),
            "last15": window_stats(items[:15], eng.xg),
            "season": window_stats([x for x in items if x[0].kickoff >= season_start], eng.xg),
            "home10": window_stats(home[:10], eng.xg),
            "away10": window_stats(away[:10], eng.xg),
            "europe": window_stats([x for x in europe if x[0].kickoff >= season_start - timedelta(days=365)][:15], eng.xg),
        },
        "splits": {
            "vs_strong": window_stats(vs_strong[:15], eng.xg),
            "vs_similar": window_stats(vs_similar[:15], eng.xg),
            "vs_possession_teams": window_stats(vs_possession[:15], eng.xg),
            "vs_low_possession_teams": window_stats(vs_low_block[:15], eng.xg),
        },
        "style": detail_stats(items),
        "form": form,
        "schedule": {"rest_days": rest, "matches_last_14d": len(last14), "matches_last_30d": len(last30)},
        "matches_365d": sum(1 for x in items if x[0].kickoff >= before - timedelta(days=365)),
        "matches_with_stats_365d": sum(1 for x in items if x[0].kickoff >= before - timedelta(days=365)
                                       and _side(*x)[0].get("shots_on_target") is not None),
    }
