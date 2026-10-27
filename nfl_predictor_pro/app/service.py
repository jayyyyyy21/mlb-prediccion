"""Motor de pronósticos: combina datos, modelo, lesiones, contexto y simulación."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import analysis, data, db
from .config import CURRENT_SEASON, N_SIMS
from .context import kickoff_utc
from .injuries import InjuryData, evaluate_team, key_players
from .model import DEFAULT_PARAMS, FEATURE_LABELS, PointsModel, matchup_features
from .ratings import TEAMS, Snapshot, snapshot
from .simulate import simulate
from .sources import weather
from .stadiums import STADIUMS, home_stadium

GAME_TYPE_ES = {"REG": "Temporada regular", "WC": "Comodín", "DIV": "Divisional", "CON": "Campeonato de conferencia",
                "SB": "Super Bowl"}


class ModelNotReady(RuntimeError):
    pass


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


class Engine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reload()

    # ------------------------------------------------------------------ carga
    def reload(self) -> None:
        with self._lock:
            data.clear_cache()
            self.schedule, self.tg_all, self.tg = data.load()
            self.tgd = analysis.team_games_with_defense(self.tg)
            self.params = {**DEFAULT_PARAMS, **(db.kv_get("model_params") or {})}
            md = db.kv_get("production_model")
            self.model = PointsModel.from_dict(md) if md else None
            self.backtest = db.kv_get("backtest") or {}
            self.inj = InjuryData()
            self.teams = db.read_table("teams")
            self._snaps: dict[float, Snapshot] = {}

    def require_model(self) -> PointsModel:
        if self.model is None:
            raise ModelNotReady("El modelo no está calibrado. Ejecute: python -m app.cli init")
        return self.model

    def snap(self, season: int, week: int) -> Snapshot:
        t = season * 30 + week
        if t not in self._snaps:
            self._snaps[t] = snapshot(self.tg, t, self.params)
        return self._snaps[t]

    # ------------------------------------------------------------- calendario
    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self.schedule["season"].unique() if s >= self.tg["season"].min())

    def weeks(self, season: int) -> list[dict]:
        s = self.schedule[self.schedule["season"] == season]
        out = []
        for wk, g in s.groupby("week"):
            out.append({"week": int(wk), "games": int(len(g)), "played": int(g["home_score"].notna().sum()),
                        "game_type": str(g["game_type"].iloc[0]), "label": week_label(int(wk), str(g["game_type"].iloc[0]))})
        return out

    def default_week(self, season: int) -> int:
        s = self.schedule[self.schedule["season"] == season]
        pending = s[s["home_score"].isna()]
        return int(pending["week"].min()) if len(pending) else int(s["week"].max())

    def team_info(self) -> dict:
        out = {}
        for r in self.teams.itertuples(index=False):
            out[r.team_abbr] = {"abbr": r.team_abbr, "name": r.team_name, "nick": r.team_nick, "conf": r.team_conf,
                                "division": r.team_division, "color": r.team_color, "color2": r.team_color2,
                                "logo": r.team_logo_espn}
        return out

    # ---------------------------------------------------------- pronósticos
    def _context(self, row: pd.Series | None, team: str, opp: str, loc: int, season: int, day: str,
                 stadium_id: str | None, weather_ctx: dict) -> dict:
        ctx = {"loc": loc}
        if row is not None:
            ctx.update(rest_diff=row.get("rest_diff", 0), tz_shift=row.get("tz_shift", 0), miles=row.get("miles", 0))
        else:  # partido hipotético del simulador
            from .context import travel
            tr = travel(team, season, stadium_id, day)
            ctx.update(rest_diff=0, tz_shift=tr["tz_shift"] if tr["known"] else 0, miles=tr["miles"] if tr["known"] else 0)
        ctx.update(wind_eff=weather_ctx.get("wind_eff", 0), cold_eff=weather_ctx.get("cold_eff", 0))
        for k, v in list(ctx.items()):
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                ctx[k] = 0.0
        return ctx

    def _weather(self, game: dict, overrides: dict) -> dict:
        roof = overrides.get("roof") or game.get("roof")
        outdoor = roof in ("outdoors", "open")
        out = {"roof": roof, "outdoor": outdoor, "temp": None, "wind": None, "source": None, "known": True}
        if "temp" in overrides or "wind" in overrides:
            out.update(temp=overrides.get("temp"), wind=overrides.get("wind"), source="Simulación del usuario")
        elif game.get("played"):
            out.update(temp=game.get("temp"), wind=game.get("wind"), source="Registro del partido (nflverse)")
        elif outdoor and game.get("kickoff") is not None and game.get("stadium_id") in STADIUMS:
            _, lat, lon, _ = STADIUMS[game["stadium_id"]]
            fc = weather.forecast(lat, lon, game["kickoff"])
            if fc:
                out.update(temp=fc["temp_f"], wind=fc["wind_mph"], gust=fc.get("gust_mph"),
                           precip_prob=fc.get("precip_prob"), source=fc["source"], fetched_at=fc["fetched_at"])
            else:
                out["known"] = False
        if not outdoor:
            out["known"] = True
        temp = out["temp"] if out["temp"] is not None and not pd.isna(out["temp"]) else None
        wind = out["wind"] if out["wind"] is not None and not pd.isna(out["wind"]) else None
        out["temp"], out["wind"] = temp, wind
        out["wind_eff"] = max(0.0, (wind or 0) - 10) if outdoor else 0.0
        out["cold_eff"] = max(0.0, 40 - temp) if (outdoor and temp is not None) else 0.0
        if outdoor and out["source"] is None:
            out["known"] = False
        return out

    def _game_dict(self, g: pd.Series) -> dict:
        ko = kickoff_utc(g["gameday"], g.get("gametime"))
        played = not pd.isna(g.get("home_score"))
        return {
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "game_type": g["game_type"], "game_type_es": GAME_TYPE_ES.get(g["game_type"], g["game_type"]),
            "gameday": g["gameday"], "gametime": g.get("gametime") if isinstance(g.get("gametime"), str) else None,
            "kickoff": ko, "kickoff_utc": ko.isoformat() if ko else None,
            "home": g["home_team"], "away": g["away_team"], "neutral": g.get("location") == "Neutral",
            "stadium": g.get("stadium"), "stadium_id": g.get("stadium_id"), "roof": g.get("roof"),
            "surface": g.get("surface"), "temp": g.get("temp"), "wind": g.get("wind"),
            "played": played,
            "home_score": None if not played else int(g["home_score"]),
            "away_score": None if not played else int(g["away_score"]),
            "home_rest": g.get("home_rest"), "away_rest": g.get("away_rest"), "div_game": bool(g.get("div_game", 0)),
        }

    def predict_game_id(self, game_id: str, overrides: dict | None = None, detail: bool = True) -> dict:
        g = self.schedule[self.schedule["game_id"] == game_id]
        if g.empty:
            raise KeyError(game_id)
        return self.predict(self._game_dict(g.iloc[0]), overrides or {}, detail=detail)

    def predict(self, game: dict, overrides: dict, detail: bool = True, log: bool = True) -> dict:
        model = self.require_model()
        season, week = game["season"], game["week"]
        home, away = overrides.get("home") or game["home"], overrides.get("away") or game["away"]
        neutral = bool(overrides.get("neutral", game["neutral"]))
        snap = self.snap(season, week)
        alerts: list[dict] = []

        wx = self._weather(game, overrides)
        rows = self.tg_all[(self.tg_all["game_id"] == game.get("game_id"))] if game.get("game_id") else pd.DataFrame()
        same_teams = not overrides.get("home") and not overrides.get("away")
        row_h = rows[rows["team"] == home].iloc[0] if same_teams and len(rows) == 2 else None
        row_a = rows[rows["team"] == away].iloc[0] if same_teams and len(rows) == 2 else None
        day = str(game.get("gameday") or datetime.now(timezone.utc).date())
        sid = game.get("stadium_id") if same_teams else (None if neutral else home_stadium(home, season))
        loc_h, loc_a = (0, 0) if neutral else (1, -1)
        ctx_h = self._context(row_h, home, away, loc_h, season, day, sid, wx)
        ctx_a = self._context(row_a, away, home, loc_a, season, day, sid, wx)
        if "rest_diff" in overrides:
            ctx_h["rest_diff"], ctx_a["rest_diff"] = overrides["rest_diff"], -overrides["rest_diff"]

        f_h = matchup_features(snap, home, away, ctx_h)
        f_a = matchup_features(snap, away, home, ctx_a)
        missing = [FEATURE_LABELS[k] for k in model.features if not np.isfinite(f_h.get(k, np.nan)) or not np.isfinite(f_a.get(k, np.nan))]
        if missing:
            alerts.append({"level": "warn", "text": "Faltan estadísticas para: " + ", ".join(sorted(set(missing))) +
                           ". Se usa el promedio de la liga en esas variables."})
        base_h = float(model.predict(pd.DataFrame([f_h]))[0])
        base_a = float(model.predict(pd.DataFrame([f_a]))[0])

        # Lesiones: sólo para partidos no jugados (consistente con el backtest).
        use_inj = not game["played"] or overrides.get("force_injuries")
        include_prob = bool(overrides.get("include_probable"))
        out_h = overrides.get("out_home", []) or []
        out_a = overrides.get("out_away", []) or []
        inj_h = evaluate_team(self.inj, home, season, week, out_h, include_prob) if use_inj and season == CURRENT_SEASON else None
        inj_a = evaluate_team(self.inj, away, season, week, out_a, include_prob) if use_inj and season == CURRENT_SEASON else None
        adj_h = (inj_h["off_adj"] if inj_h else 0) + (inj_a["def_adj"] if inj_a else 0)
        adj_a = (inj_a["off_adj"] if inj_a else 0) + (inj_h["def_adj"] if inj_h else 0)
        mu_h = max(base_h + adj_h, 3.0)
        mu_a = max(base_a + adj_a, 3.0)

        sd = model.sd_for_week(week) * self.params.get("sd_scale", 1.0)
        pace = np.nanmean([f_h.get("pace"), f_a.get("pace")])
        seed = int(hashlib.md5(f"{game.get('game_id')}{home}{away}".encode()).hexdigest()[:8], 16)
        sim = simulate(mu_h, mu_a, pace, sd, sd, model.resid_corr, n=N_SIMS,
                       playoff=game["game_type"] != "REG", seed=seed)

        # ---------------- estado, confianza y alertas
        games_h = int(((self.tg["team"] == home) & (self.tg["season"] == season) & (self.tg["week"] < week)).sum())
        games_a = int(((self.tg["team"] == away) & (self.tg["season"] == season) & (self.tg["week"] < week)).sum())
        if min(games_h, games_a) < 4:
            alerts.append({"level": "info", "text": f"Muestra pequeña en la temporada ({games_h} y {games_a} partidos): "
                           "los ratings dependen más de la temporada anterior."})
        if not wx["known"]:
            alerts.append({"level": "warn", "text": "Sin pronóstico del clima disponible para este partido al aire libre "
                           "(se asume clima neutro)."})
        official = bool(inj_h and inj_a and inj_h["official_report"] and inj_a["official_report"])
        unconfirmed_qb = bool((inj_h and inj_h["unconfirmed_qb"]) or (inj_a and inj_a["unconfirmed_qb"]))
        if use_inj and season == CURRENT_SEASON and not official:
            alerts.append({"level": "info", "text": "Aún no se publica el reporte oficial de lesiones de la semana para "
                           "ambos equipos: pronóstico preliminar."})
        if unconfirmed_qb:
            alerts.append({"level": "warn", "text": "El QB titular de al menos un equipo está en duda (no confirmado)."})
        if use_inj and season != CURRENT_SEASON:
            alerts.append({"level": "info", "text": "No hay reportes de lesiones cargados para esa temporada."})

        p_h = sim["p_a"] + sim["p_tie"] / 2
        fav_p = max(p_h, 1 - p_h)
        level = 2 if fav_p >= 0.72 else (1 if fav_p >= 0.60 else 0)
        downgrades = []
        if min(games_h, games_a) < 4:
            downgrades.append("muestra pequeña")
        if unconfirmed_qb:
            downgrades.append("QB titular en duda")
        if not wx["known"]:
            downgrades.append("clima desconocido")
        if missing:
            downgrades.append("estadísticas faltantes")
        level = max(0, level - (1 if downgrades else 0))
        confidence = ["Bajo", "Moderado", "Alto"][level]
        hist = self._historical_bucket(fav_p)

        if game["played"] and not overrides:
            status = "Finalizado (pronóstico retrospectivo)"
        elif overrides:
            status = "Simulación"
        elif official and not unconfirmed_qb and wx["known"]:
            status = "Confirmado"
        else:
            status = "Preliminar"

        r1 = lambda x: round(x, 1)
        proj_h, proj_a = round(mu_h), round(mu_a)
        if proj_h == proj_a and abs(mu_h - mu_a) > 0.05:
            if mu_h > mu_a:
                proj_h += 1
            else:
                proj_a += 1
        result = {
            "game": {k: v for k, v in game.items() if k != "kickoff"} | {"home": home, "away": away, "neutral": neutral},
            "status": status,
            "confidence": confidence,
            "confidence_reasons": downgrades,
            "historical_bucket": hist,
            "projection": {
                "home_points": r1(mu_h), "away_points": r1(mu_a),
                "home_points_base": r1(base_h), "away_points_base": r1(base_a),
                "home_injury_adj": r1(adj_h), "away_injury_adj": r1(adj_a),
                "score_home": proj_h, "score_away": proj_a,
                "p_home": round(sim["p_a"], 4), "p_away": round(sim["p_b"], 4), "p_tie": round(sim["p_tie"], 4),
                "spread_home": r1(mu_h - mu_a), "total": r1(mu_h + mu_a),
            },
            "simulation": sim,
            "weather": {k: v for k, v in wx.items()},
            "alerts": alerts,
            "model_version": self.backtest.get("created_at"),
        }
        if detail:
            result["explanation"] = self._explain(snap, model, season, week, home, away, f_h, f_a, ctx_h, ctx_a,
                                                  base_h, base_a, inj_h, inj_a, wx, sim, neutral)
            result["injuries"] = {"home": inj_h, "away": inj_a}
        if log and not overrides and not game["played"] and game.get("kickoff") and game["kickoff"] > datetime.now(timezone.utc):
            self._log(game, result)
        return result

    def _historical_bucket(self, fav_p: float) -> dict | None:
        for b in self.backtest.get("calibration_test", []):
            lo, hi = [int(x.strip("%")) / 100 for x in b["bin"].split("-")]
            if lo <= fav_p < hi or (hi == 1.0 and fav_p == 1.0):
                return {**b, "text": (f"En la validación fuera de muestra, los favoritos con probabilidad {b['bin']} "
                                      f"ganaron {b['obs']:.0%} de {b['n']} partidos (predicho {b['pred']:.0%}).")}
        return None

    # ------------------------------------------------------------ explicación
    def _explain(self, snap, model, season, week, home, away, f_h, f_a, ctx_h, ctx_a, base_h, base_a,
                 inj_h, inj_a, wx, sim, neutral) -> dict:
        sum_h = analysis.team_summary(self.tgd, home, season, week)
        sum_a = analysis.team_summary(self.tgd, away, season, week)
        if sum_h.get("games", 0) == 0:
            sum_h = analysis.team_summary(self.tgd, home, season - 1)
        if sum_a.get("games", 0) == 0:
            sum_a = analysis.team_summary(self.tgd, away, season - 1)
        pts = snap.get("pts")

        def contrib(f):
            c = model.contributions(f)
            items = [{"feature": k, "label": FEATURE_LABELS[k], "points": round(v, 2)} for k, v in c.items()]
            return sorted(items, key=lambda x: -abs(x["points"]))

        def baseline_text(team, opp, s_off, s_def):
            o = s_off.get("offense", {}).get("temporada", {})
            o3 = s_off.get("offense", {}).get("ult3", {})
            d = s_def.get("defense", {}).get("temporada", {})
            d3 = s_def.get("defense", {}).get("ult3", {})
            exp = pts.expected(team, opp, 0) if pts else float("nan")
            return (f"{team} anota {o.get('ppg', float('nan')):.1f} pts/partido esta temporada "
                    f"({o3.get('ppg', float('nan')):.1f} en los últimos 3). La defensa de {opp} permite "
                    f"{d.get('ppg', float('nan')):.1f} ({d3.get('ppg', float('nan')):.1f} en los últimos 3). "
                    f"Ajustando por la calidad de los rivales enfrentados, la expectativa base es {exp:.1f} pts "
                    f"en campo neutral.")

        ha_h = analysis.home_away_profile(self.tgd, home, season, snap, week)
        ha_a = analysis.home_away_profile(self.tgd, away, season, snap, week)
        loc_coef = model.contributions({**f_h, "f_loc": 1.0})["f_loc"] - model.contributions({**f_h, "f_loc": 0.0})["f_loc"]
        if neutral:
            loc_txt = "Partido en sede neutral: no se aplica ventaja de localía."
        else:
            loc_txt = (f"El modelo estima, con datos históricos, que jugar de local vale {loc_coef:+.1f} pts para el local "
                       f"y {-loc_coef:+.1f} para el visitante (≈{2 * loc_coef:.1f} de diferencia). "
                       f"Efecto propio estimado (encogido por muestra): {home} {ha_h.get('home_edge_shrunk', float('nan')):+.1f} "
                       f"como local; {away} {ha_a.get('home_edge_shrunk', float('nan')):+.1f}. "
                       "El efecto específico por equipo se muestra como contexto; el modelo usa el efecto de liga "
                       "porque los splits individuales no mejoraron la predicción fuera de muestra.")
        uncertainty = []
        if sim["margin_sd"] > 0:
            uncertainty.append(f"Desviación estándar del margen simulado: {sim['margin_sd']:.1f} pts; el 80% de los "
                               f"escenarios cae entre {sim['margin_p10']:+.0f} y {sim['margin_p90']:+.0f} para {home}.")
        for side, inj in (("home", inj_h), ("away", inj_a)):
            if inj and (inj["scenario_off"] or inj["scenario_def"]):
                uncertainty.append(f"Jugadores no confirmados de {inj['team']}: si no juegan, su ofensiva variaría "
                                   f"{inj['scenario_off']:+.1f} pts y su defensa concedería {inj['scenario_def']:+.1f} pts.")
        tm_h = analysis.tactical_matchups(snap, self.tgd, home, away, season, week)
        tm_a = analysis.tactical_matchups(snap, self.tgd, away, home, season, week)
        for t in tm_h + tm_a:
            if t["metric"] == "explosive_dep":
                uncertainty.append(t["summary"] + ".")
        if wx.get("outdoor") and (wx.get("wind") or 0) >= 15:
            uncertainty.append(f"Viento de {wx['wind']:.0f} mph: reduce el juego aéreo y los goles de campo largos.")
        if not wx.get("known"):
            uncertainty.append("Clima no disponible todavía.")

        return {
            "home_baseline": baseline_text(home, away, sum_h, sum_a),
            "away_baseline": baseline_text(away, home, sum_a, sum_h),
            "home_contributions": contrib(f_h),
            "away_contributions": contrib(f_a),
            "home_summary": sum_h, "away_summary": sum_a,
            "home_away": {"home": ha_h, "away": ha_a, "text": loc_txt},
            "matchups": {"home_offense": tm_h, "away_offense": tm_a},
            "similar_opponents": {
                "home_def_vs_away_style": analysis.similar_opponents(snap, self.tgd, home, away, season, week),
                "away_def_vs_home_style": analysis.similar_opponents(snap, self.tgd, away, home, season, week),
            },
            "context": {"home": ctx_h, "away": ctx_a},
            "uncertainty": uncertainty,
            "features": {"home": {k: _f(v, 4) for k, v in f_h.items()}, "away": {k: _f(v, 4) for k, v in f_a.items()}},
        }

    # ---------------------------------------------------------- registro
    def _log(self, game: dict, res: dict) -> None:
        p = res["projection"]
        key = json.dumps([p["home_points"], p["away_points"], round(p["p_home"], 3), res["status"]])
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        with db.session() as con:
            last = con.execute("SELECT inputs_hash FROM prediction_log WHERE game_id=? ORDER BY id DESC LIMIT 1",
                               (game["game_id"],)).fetchone()
            if last and last["inputs_hash"] == h:
                return
            con.execute(
                "INSERT INTO prediction_log(game_id, season, week, kickoff_utc, created_at, home_team, away_team, pred_home,"
                " pred_away, p_home, p_away, p_tie, status, confidence, inputs_hash, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (game["game_id"], game["season"], game["week"], game["kickoff_utc"], db.utcnow(), game["home"], game["away"],
                 p["home_points"], p["away_points"], p["p_home"], p["p_away"], p["p_tie"], res["status"], res["confidence"],
                 h, json.dumps({"projection": p, "alerts": res["alerts"]})),
            )

    def predict_week(self, season: int, week: int) -> list[dict]:
        g = self.schedule[(self.schedule["season"] == season) & (self.schedule["week"] == week)]
        g = g.sort_values(["gameday", "gametime", "game_id"])
        out = []
        for _, row in g.iterrows():
            res = self.predict(self._game_dict(row), {}, detail=False)
            out.append(res)
        return out

    def simulate_custom(self, home: str, away: str, neutral: bool, overrides: dict) -> dict:
        """Simulador: partido hipotético con la información disponible hoy."""
        season = CURRENT_SEASON
        week = self.default_week(season)
        game = {"game_id": None, "season": season, "week": week, "game_type": "REG", "game_type_es": "Simulación",
                "gameday": str(datetime.now(timezone.utc).date()), "gametime": None, "kickoff": None, "kickoff_utc": None,
                "home": home, "away": away, "neutral": neutral, "stadium": None,
                "stadium_id": None if neutral else home_stadium(home, season),
                "roof": overrides.get("roof", "outdoors"), "temp": None, "wind": None, "played": False}
        # Si existe ese partido en el calendario, se usa su contexto real.
        sched = self.schedule[(self.schedule["season"] == season) & (self.schedule["home_score"].isna()) &
                              (self.schedule["home_team"] == home) & (self.schedule["away_team"] == away)]
        if len(sched) and not neutral:
            game = self._game_dict(sched.sort_values("week").iloc[0])
            ov = {k: v for k, v in overrides.items() if k not in ("home", "away")}
        else:
            ov = {**overrides, "home": home, "away": away, "neutral": neutral}
        ov["_sim"] = True
        res = self.predict(game, ov, detail=True, log=False)
        res["key_players"] = {
            "home": key_players(self.inj, home, season, week),
            "away": key_players(self.inj, away, season, week),
        }
        return res

    # --------------------------------------------------------------- equipos
    def team_stats(self, team: str, season: int) -> dict:
        week = 30 if season < CURRENT_SEASON else self.default_week(season)
        snap = self.snap(season, min(week, 23))
        summ = analysis.team_summary(self.tgd, team, season)
        ranks = {}
        for metric, side, hb in [("pts", "off", True), ("pts", "def", False), ("epa", "off", True), ("epa", "def", False),
                                 ("pepa", "off", True), ("pepa", "def", False), ("repa", "off", True), ("repa", "def", False),
                                 ("sr", "off", True), ("sr", "def", False)]:
            r = snap.get(metric)
            if r is None:
                continue
            vals = r.off if side == "off" else r.deff
            order = sorted(TEAMS, key=lambda t: vals[t], reverse=hb)
            ranks[f"{metric}_{side}"] = {"rank": order.index(team) + 1,
                                         "value": round(r.mu + vals[team], 4)}
        return {"summary": summ, "adjusted": ranks,
                "home_away": analysis.home_away_profile(self.tgd, team, season, snap)}

    def ratings_table(self, season: int) -> list[dict]:
        week = 30 if season < CURRENT_SEASON else self.default_week(season)
        snap = self.snap(season, min(week, 23))
        pts, epa = snap.get("pts"), snap.get("epa")
        rows = []
        for t in TEAMS:
            rows.append({"team": t, "off_pts": round(pts.mu + pts.off[t], 1), "def_pts": round(pts.mu + pts.deff[t], 1),
                         "net": round(pts.off[t] - pts.deff[t], 1),
                         "off_epa": round(epa.mu + epa.off[t], 3), "def_epa": round(epa.mu + epa.deff[t], 3)})
        return sorted(rows, key=lambda r: -r["net"])

    # --------------------------------------------------------------- historial
    def history(self) -> dict:
        log = db.read_df("SELECT * FROM prediction_log ORDER BY id")
        sched = self.schedule[["game_id", "home_score", "away_score"]]
        out = {"live": {"games": 0, "rows": []}, "backtest": {k: v for k, v in self.backtest.items() if k != "search"}}
        if not log.empty:
            # Último pronóstico guardado ANTES del inicio de cada partido.
            log = log[log["created_at"] < log["kickoff_utc"]]
            last = log.sort_values("id").groupby("game_id").tail(1)
            m = last.merge(sched, on="game_id", how="left")
            done = m[m["home_score"].notna()].copy()
            rows = []
            for r in m.itertuples(index=False):
                rows.append({"game_id": r.game_id, "week": int(r.week), "home": r.home_team, "away": r.away_team,
                             "pred_home": r.pred_home, "pred_away": r.pred_away, "p_home": r.p_home,
                             "created_at": r.created_at, "status": r.status, "confidence": r.confidence,
                             "home_score": None if pd.isna(r.home_score) else int(r.home_score),
                             "away_score": None if pd.isna(r.away_score) else int(r.away_score)})
            out["live"]["rows"] = rows
            if len(done):
                from .backtest import calibration_table, metrics
                pred = pd.DataFrame({"pred_a": done["pred_home"], "pred_b": done["pred_away"],
                                     "pts_a": done["home_score"], "pts_b": done["away_score"],
                                     "p_a": done["p_home"], "p_tie": done["p_tie"].fillna(0)})
                out["live"] = {**out["live"], **metrics(pred), "calibration": calibration_table(pred)}
        bt = db.read_table("backtest_games")
        if not bt.empty:
            bt["err_a"] = bt["pred_a"] - bt["pts_a"]
            bt["err_b"] = bt["pred_b"] - bt["pts_b"]
            out["backtest_weekly"] = (
                bt.assign(abs_err=(bt["err_a"].abs() + bt["err_b"].abs()) / 2)
                .groupby(["season"])["abs_err"].mean().round(3).to_dict()
            )
        return out


def week_label(week: int, game_type: str) -> str:
    return {"WC": "Comodín", "DIV": "Divisional", "CON": "Final de conferencia", "SB": "Super Bowl"}.get(
        game_type, f"Semana {week}")


_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None
