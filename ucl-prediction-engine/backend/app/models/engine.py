"""Motor de predicción: ajusta todos los submodelos y produce el pronóstico de un partido."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .dataset import Row, XgProxy, comp_covariates, rest_days, short_rest, team_leagues
from .distributions import count_pmf, fit_rho, goal_markets, sample_counts, score_matrix
from .strength import Obs, StrengthModel

MODEL_VERSION = "ucl-engine-1.1"
N_COV = 4  # descanso corto (atacante), descanso corto (defensor), competición europea, fase previa europea
COV_NAMES = ["Descanso corto (≤3 días) del equipo", "Descanso corto del rival",
             "Partido de competición europea", "Fase previa europea"]

CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]
TEAM_CORNER_LINES = [2.5, 3.5, 4.5, 5.5, 6.5]
SOT_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5]
SOT_TEAM_LINES = [2.5, 3.5, 4.5, 5.5]


@dataclass
class Hyper:
    half_life_days: float = 180.0
    xg_blend: float = 0.5  # peso del xG-proxy (tiros) frente a goles reales en el modelo de goles
    lam_team: float = 4.0
    lam_count: float = 20.0  # penalización de equipo para córners / tiros (más ruidosos)
    count_shrink: float = 1.0  # encogimiento de efectos equipo/liga en córners y tiros (validado)
    goal_shrink: float = 1.0
    # Peso del modelo en el TOTAL de córners/tiros frente a la media europea reciente
    # (1 = solo modelo). Se conserva el reparto entre equipos. Elegido por validación.
    count_blend: float = 1.0
    style_adjust: bool = False
    style_k: float = 8.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _logit(p: float) -> float:
    p = min(max(p, 0.05), 0.95)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Engine:
    hyper: Hyper = field(default_factory=Hyper)
    as_of: datetime | None = None
    rows: list[Row] = field(default_factory=list)
    leagues: dict[str, str] = field(default_factory=dict)
    xg: XgProxy = field(default_factory=XgProxy)
    goals: StrengthModel | None = None
    corners: StrengthModel | None = None
    sot: StrengthModel | None = None
    shots: StrengthModel | None = None
    poss: StrengthModel | None = None
    rho: float = 0.0
    rest: dict = field(default_factory=dict)
    euro_means: dict = field(default_factory=dict)
    _team_poss: dict = field(default_factory=dict)
    _poss_terciles: tuple = (0.45, 0.55)

    # ------------------------------------------------------------------ fit
    def fit(self, rows: list[Row], as_of: datetime, compute_cov: bool = True) -> "Engine":
        rows = [r for r in rows if r.kickoff < as_of]
        self.rows, self.as_of = rows, as_of
        self.leagues = team_leagues(rows)
        self.xg = XgProxy.fit(rows)
        self.rest = rest_days(rows)
        hl = self.hyper.half_life_days

        def weight(r: Row) -> float:
            return 0.5 ** ((as_of - r.kickoff).total_seconds() / 86400 / hl)

        def home_flag(r: Row, is_home: bool) -> float:
            return 0.5 if r.neutral else (1.0 if is_home else 0.0)

        g_obs, c_obs, s_obs, sh_obs, p_obs = [], [], [], [], []
        hg, ag, ws, idx_rows = [], [], [], []
        for r in rows:
            w = weight(r)
            if w < 0.02:
                continue
            for att, dfn, st, gl, is_home in ((r.home, r.away, r.h, r.hg, True),
                                              (r.away, r.home, r.a, r.ag, False)):
                hf = home_flag(r, is_home)
                cov = (short_rest(self.rest.get((r.match_id, att), 7)),
                       short_rest(self.rest.get((r.match_id, dfn), 7))) + comp_covariates(r.league)
                xgv = self.xg.value(st)
                y = gl if xgv is None else (1 - self.hyper.xg_blend) * gl + self.hyper.xg_blend * xgv
                g_obs.append(Obs(att, dfn, hf, y, w, cov))
                if st.get("corners") is not None:
                    c_obs.append(Obs(att, dfn, hf, st["corners"], w, cov))
                if st.get("shots_on_target") is not None:
                    s_obs.append(Obs(att, dfn, hf, st["shots_on_target"], w, cov))
                if st.get("shots") is not None:
                    sh_obs.append(Obs(att, dfn, hf, st["shots"], w, cov))
                if st.get("possession") is not None and 5 < st["possession"] < 95:
                    p_obs.append(Obs(att, dfn, hf, _logit(st["possession"] / 100), w))
            hg.append(r.hg); ag.append(r.ag); ws.append(w); idx_rows.append(r)

        lt = self.hyper.lam_team
        cc = compute_cov
        self.goals = StrengthModel(self.leagues, lam_team=lt, n_cov=N_COV).fit(g_obs, cc)
        self.goals.shrink = self.hyper.goal_shrink
        lc = self.hyper.lam_count
        self.corners = StrengthModel(self.leagues, lam_team=lc, n_cov=N_COV).fit(c_obs, cc)
        self.sot = StrengthModel(self.leagues, lam_team=lc, n_cov=N_COV).fit(s_obs, cc)
        self.shots = StrengthModel(self.leagues, lam_team=lc, n_cov=N_COV).fit(sh_obs, cc)
        for mdl in (self.corners, self.sot, self.shots):
            mdl.shrink = self.hyper.count_shrink
        self.poss = StrengthModel(self.leagues, lam_team=lt / 2, gaussian=True).fit(p_obs, cc)

        lams = np.array([self.goals.mean(r.home, r.away, home_flag(r, True),
                                         (short_rest(self.rest.get((r.match_id, r.home), 7)),
                                          short_rest(self.rest.get((r.match_id, r.away), 7))) + comp_covariates(r.league))
                         for r in idx_rows])
        mus = np.array([self.goals.mean(r.away, r.home, home_flag(r, False),
                                        (short_rest(self.rest.get((r.match_id, r.away), 7)),
                                         short_rest(self.rest.get((r.match_id, r.home), 7))) + comp_covariates(r.league))
                        for r in idx_rows])
        self.rho = fit_rho(lams, mus, np.array(hg), np.array(ag), np.array(ws))
        self._style_tables()
        recent = [r for r in rows if r.league in ("uefa.champions", "uefa.europa", "uefa.europa.conf")
                  and (as_of - r.kickoff).days <= 365]
        for key, fld in (("corners", "corners"), ("sot", "shots_on_target"), ("shots", "shots")):
            vals = [r.h[fld] + r.a[fld] for r in recent if r.h.get(fld) is not None and r.a.get(fld) is not None]
            self.euro_means[key] = float(np.mean(vals)) if len(vals) >= 30 else None
        return self

    def _blend_counts(self, key: str, m1: float, m2: float) -> tuple[float, float]:
        """Calibra el total hacia la media europea reciente conservando el reparto entre equipos."""
        w = self.hyper.count_blend
        base = self.euro_means.get(key)
        if w >= 1.0 or not base:
            return m1, m2
        f = (w * (m1 + m2) + (1 - w) * base) / (m1 + m2)
        return m1 * f, m2 * f

    # --------------------------------------------------------- estilo/posesión
    def _style_tables(self) -> None:
        acc: dict[str, list[float]] = {}
        for r in self.rows[-6000:]:
            for t, st in ((r.home, r.h), (r.away, r.a)):
                if st.get("possession") is not None:
                    acc.setdefault(t, []).append(st["possession"])
        self._team_poss = {t: float(np.mean(v[-20:])) for t, v in acc.items() if len(v) >= 3}
        vals = np.array(list(self._team_poss.values())) if self._team_poss else np.array([45, 55])
        self._poss_terciles = (float(np.percentile(vals, 33.3)), float(np.percentile(vals, 66.7)))

    def poss_tier(self, team: str) -> int | None:
        p = self._team_poss.get(team)
        if p is None:
            return None
        lo, hi = self._poss_terciles
        return 0 if p < lo else (1 if p < hi else 2)

    def style_split(self, att: str, dfn: str) -> dict:
        """Rendimiento ofensivo del atacante frente a rivales del mismo perfil de posesión
        que `dfn`, relativo a lo que el modelo esperaba. Encogido hacia 1 (sin efecto)."""
        tier = self.poss_tier(dfn)
        if tier is None:
            return {"tier": None, "n": 0, "ratio": 1.0, "obs": 0.0, "exp": 0.0}
        y_sum = mu_sum = 0.0
        n = 0
        for r in self.rows:
            if r.home == att:
                opp, st, gl, hf = r.away, r.h, r.hg, (0.5 if r.neutral else 1.0)
            elif r.away == att:
                opp, st, gl, hf = r.home, r.a, r.ag, (0.5 if r.neutral else 0.0)
            else:
                continue
            if self.poss_tier(opp) != tier:
                continue
            xgv = self.xg.value(st)
            y = gl if xgv is None else (1 - self.hyper.xg_blend) * gl + self.hyper.xg_blend * xgv
            y_sum += y
            mu_sum += self.goals.mean(att, opp, hf, (0.0, 0.0) + comp_covariates(r.league))
            n += 1
        k = self.hyper.style_k
        ratio = (y_sum + k) / (mu_sum + k) if n else 1.0
        return {"tier": tier, "n": n, "ratio": float(min(max(ratio, 0.85), 1.15)),
                "obs": y_sum, "exp": mu_sum}

    # ------------------------------------------------------------- predicción
    def quick_predict(self, home: str, away: str, neutral: bool = False,
                      rest_home: float = 7.0, rest_away: float = 7.0, league: str = "uefa.champions") -> dict:
        """Predicción puntual (sin Monte Carlo): usada en backtesting."""
        hf_h, hf_a = (0.5, 0.5) if neutral else (1.0, 0.0)
        cc = comp_covariates(league)
        cov_h = (short_rest(rest_home), short_rest(rest_away)) + cc
        cov_a = (short_rest(rest_away), short_rest(rest_home)) + cc
        lam = self.goals.mean(home, away, hf_h, cov_h)
        mu = self.goals.mean(away, home, hf_a, cov_a)
        if self.hyper.style_adjust:
            lam *= self.style_split(home, away)["ratio"]
            mu *= self.style_split(away, home)["ratio"]
        gm = goal_markets(score_matrix(lam, mu, self.rho))
        out = {"lam": lam, "mu": mu, **{k: gm[k] for k in ("home_win", "draw", "away_win", "over_2_5", "btts")}}
        for name, model, line in (("corners", self.corners, 9.5), ("sot", self.sot, 8.5)):
            m1, m2 = self._blend_counts(name, model.mean(home, away, hf_h, cov_h), model.mean(away, home, hf_a, cov_a))
            pmf = np.convolve(count_pmf(m1, model.dispersion), count_pmf(m2, model.dispersion))
            out[f"{name}_total"] = m1 + m2
            out[f"{name}_home"], out[f"{name}_away"] = m1, m2
            out[f"{name}_over"] = float(pmf[int(line) + 1:].sum())
            out[f"{name}_line"] = line
        p1, _ = self.poss.eta(home, away, hf_h)
        p2, _ = self.poss.eta(away, home, hf_a)
        out["poss_home"] = 50 * (_sigmoid(p1) + 1 - _sigmoid(p2))
        return out

    def predict(self, home: str, away: str, kickoff: datetime, neutral: bool = False,
                adjust: dict | None = None, n_sims: int = 20000, seed: int = 7,
                rest_home: float | None = None, rest_away: float | None = None,
                league: str = "uefa.champions") -> dict:
        adjust = adjust or {}
        hf_h, hf_a = (0.5, 0.5) if neutral else (1.0, 0.0)
        rh = rest_home if rest_home is not None else 7.0
        ra = rest_away if rest_away is not None else 7.0
        cc = comp_covariates(league)
        cov_h = (short_rest(rh), short_rest(ra)) + cc
        cov_a = (short_rest(ra), short_rest(rh)) + cc

        style_h = self.style_split(home, away) if self.hyper.style_adjust else None
        style_a = self.style_split(away, home) if self.hyper.style_adjust else None
        lg_adj_h = math.log(adjust.get("home_attack", 1.0)) + (math.log(style_h["ratio"]) if style_h else 0)
        lg_adj_a = math.log(adjust.get("away_attack", 1.0)) + (math.log(style_a["ratio"]) if style_a else 0)

        eh, sh = self.goals.eta(home, away, hf_h, cov_h)
        ea, sa = self.goals.eta(away, home, hf_a, cov_a)
        eh += lg_adj_h; ea += lg_adj_a
        lam, mu = math.exp(eh), math.exp(ea)
        M = score_matrix(lam, mu, self.rho)
        point = goal_markets(M)

        rng = np.random.default_rng(seed)
        K = 300
        zh, za = rng.standard_normal(K), rng.standard_normal(K)
        lam_k = np.exp(eh + sh * zh)
        mu_k = np.exp(ea + sa * za)
        mats = [score_matrix(l, m, self.rho) for l, m in zip(lam_k, mu_k)]
        Mpost = np.mean(mats, axis=0)
        post = goal_markets(Mpost)
        per_draw = [goal_markets(m) for m in mats]

        def band(key):
            v = np.array([d[key] for d in per_draw])
            return [float(np.percentile(v, 10)), float(np.percentile(v, 90))]

        bands = {k: band(k) for k in ("home_win", "draw", "away_win", "over_1_5", "over_2_5",
                                      "over_3_5", "btts")}

        # Conteos: córners, tiros a puerta, tiros
        def count_block(model: StrengthModel, adj_h: float = 0.0, adj_a: float = 0.0):
            e1, s1 = model.eta(home, away, hf_h, cov_h)
            e2, s2 = model.eta(away, home, hf_a, cov_a)
            return (e1 + adj_h, s1), (e2 + adj_a, s2), model.dispersion

        def blended(key, blk):
            (e1, s1), (e2, s2), alpha = blk
            b1, b2 = self._blend_counts(key, math.exp(e1), math.exp(e2))
            return (math.log(b1), s1), (math.log(b2), s2), alpha

        (ce_h, cs_h), (ce_a, cs_a), c_alpha = blended("corners", count_block(self.corners))
        sot_adj_h = math.log(adjust.get("home_attack", 1.0))
        sot_adj_a = math.log(adjust.get("away_attack", 1.0))
        (se_h, ss_h), (se_a, ss_a), s_alpha = blended("sot", count_block(self.sot, sot_adj_h, sot_adj_a))
        (te_h, _), (te_a, _), _ = blended("shots", count_block(self.shots, sot_adj_h, sot_adj_a))

        # Monte Carlo conjunto
        draw_idx = rng.integers(0, K, n_sims)
        flat = Mpost.ravel() / Mpost.sum()
        cells = rng.choice(flat.size, size=n_sims, p=flat)
        g_h, g_a = np.divmod(cells, Mpost.shape[1])
        c_h = sample_counts(rng, np.exp(ce_h + cs_h * rng.standard_normal(n_sims)), c_alpha)
        c_a = sample_counts(rng, np.exp(ce_a + cs_a * rng.standard_normal(n_sims)), c_alpha)
        s_h = sample_counts(rng, np.exp(se_h + ss_h * rng.standard_normal(n_sims)), s_alpha)
        s_a = sample_counts(rng, np.exp(se_a + ss_a * rng.standard_normal(n_sims)), s_alpha)
        del draw_idx

        def p_over(x, line):
            return float(np.mean(x > line))

        def se(p):
            return float(math.sqrt(max(p * (1 - p), 1e-12) / n_sims))

        c_tot = c_h + c_a
        s_tot = s_h + s_a
        corners = {
            "home_exp": float(np.mean(c_h)), "away_exp": float(np.mean(c_a)),
            "total_exp": float(np.mean(c_tot)),
            "dispersion": c_alpha,
            "total_over": {str(l): p_over(c_tot, l) for l in CORNER_LINES},
            "home_over": {str(l): p_over(c_h, l) for l in TEAM_CORNER_LINES},
            "away_over": {str(l): p_over(c_a, l) for l in TEAM_CORNER_LINES},
            "home_more": float(np.mean(c_h > c_a)), "equal": float(np.mean(c_h == c_a)),
            "away_more": float(np.mean(c_a > c_h)),
            "total_dist": [float(np.mean(c_tot == k)) for k in range(0, 21)],
        }
        sot = {
            "home_exp": float(np.mean(s_h)), "away_exp": float(np.mean(s_a)),
            "total_exp": float(np.mean(s_tot)), "dispersion": s_alpha,
            "total_over": {str(l): p_over(s_tot, l) for l in SOT_TOTAL_LINES},
            "home_over": {str(l): p_over(s_h, l) for l in SOT_TEAM_LINES},
            "away_over": {str(l): p_over(s_a, l) for l in SOT_TEAM_LINES},
            "home_shots_exp": math.exp(te_h), "away_shots_exp": math.exp(te_a),
        }
        mc_goals = {
            "home_win": float(np.mean(g_h > g_a)), "draw": float(np.mean(g_h == g_a)),
            "away_win": float(np.mean(g_h < g_a)), "over_2_5": float(np.mean(g_h + g_a > 2.5)),
            "btts": float(np.mean((g_h > 0) & (g_a > 0))),
        }

        # Posesión
        p1, _ = self.poss.eta(home, away, hf_h)
        p2, _ = self.poss.eta(away, home, hf_a)
        ph = 0.5 * (_sigmoid(p1) + 1 - _sigmoid(p2))

        return {
            "model_version": MODEL_VERSION,
            "goals": {
                "home_xg": lam, "away_xg": mu,
                "home_xg_sd_log": sh, "away_xg_sd_log": sa,
                **post,
                "point_estimate": {k: point[k] for k in ("home_win", "draw", "away_win", "over_2_5", "btts")},
                "bands_p10_p90": bands,
                "score_matrix": [[float(x) for x in row[:6]] for row in Mpost[:6]],
                "rho": self.rho,
            },
            "corners": corners,
            "shots_on_target": sot,
            "possession": {"home": ph * 100, "away": (1 - ph) * 100, "resid_sd_logit": self.poss.resid_sd},
            "monte_carlo": {"iterations": n_sims, "goals_check": mc_goals,
                            "std_error_max": se(0.5)},
            "adjustments": {
                "home_attack_mult": adjust.get("home_attack", 1.0),
                "away_attack_mult": adjust.get("away_attack", 1.0),
                "style_home": style_h, "style_away": style_a,
                "rest_days_home": rh, "rest_days_away": ra,
                "covariates": [
                    {"name": n, "goals": g_, "corners": c_, "sot": s_}
                    for n, g_, c_, s_ in zip(COV_NAMES, self.goals.cov_coefs(), self.corners.cov_coefs(),
                                             self.sot.cov_coefs())],
            },
        }
