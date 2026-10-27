"""Simulación Monte Carlo de partidos basada en posesiones.

En cada escenario:
1. Se perturba la expectativa de puntos de cada equipo con ruido normal
   bivariado (incertidumbre del modelo y variación del día de juego) cuya
   varianza y correlación se calibran con los residuos históricos del modelo.
2. Se simula el número de posesiones y, posesión a posesión, si termina en
   touchdown (6 + punto extra / conversión), gol de campo (3) o nada.
   Esto produce marcadores realistas (múltiplos de 3 y 7) en vez de una
   distribución continua.
3. Si hay empate al final del tiempo regular se simula una prórroga
   (en temporada regular puede terminar en empate).
"""
from __future__ import annotations

import numpy as np

TD_VALUE = 6.0 + 0.94 * 1 + 0.025 * 2  # TD + PAT (94%) + 2pts (aprox. 2.5% efectivo)
DRIVE_SD = 1.1


def _drive_probs(ppd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades TD/FG por posesión que reproducen los puntos por posesión dados.

    La proporción TD:FG se basa en promedios de la liga (aprox. 22% TD y 15% FG
    por posesión con ~2.0 puntos por posesión); mejores ofensivas convierten
    una mayor fracción de sus anotaciones en TD.
    """
    ppd = np.clip(ppd, 0.15, 4.5)
    p_fg = np.clip(0.155 * np.sqrt(ppd / 2.0), 0.03, 0.30)
    p_fg = np.minimum(p_fg, 0.6 * ppd / 3.0)
    p_td = np.clip((ppd - 3.0 * p_fg) / TD_VALUE, 0.005, 0.75)
    return p_td, p_fg


def _team_points(rng, lam: np.ndarray, n_drives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ppd = lam / n_drives
    p_td, p_fg = _drive_probs(ppd)
    tds = rng.binomial(n_drives, p_td)
    fgs = rng.binomial(n_drives - tds, np.clip(p_fg / (1 - p_td), 0, 1))
    xp = rng.binomial(tds, 0.94)
    two = rng.binomial(tds - xp, 0.45)
    pts = tds * 6 + xp + 2 * two + fgs * 3
    return pts.astype(float), p_td + p_fg


def simulate(mu_a: float, mu_b: float, drives: float, sd_a: float, sd_b: float, rho: float,
             n: int = 10_000, playoff: bool = False, seed: int | None = 12345) -> dict:
    """Simula `n` partidos. Devuelve probabilidades, distribución y marcadores.

    sd_a / sd_b: desviación estándar total objetivo de los puntos de cada equipo.
    rho: correlación objetivo entre los puntos de ambos equipos.
    """
    rng = np.random.default_rng(seed)
    drives = float(np.clip(drives if np.isfinite(drives) else 11.0, 8, 15))
    # Varianza que ya genera la simulación por posesiones (aprox. analítica).
    def sim_var(mu: float) -> float:
        ppd = max(mu, 1) / drives
        p_td, p_fg = (x.item() for x in _drive_probs(np.array([ppd])))
        e = p_td * TD_VALUE + p_fg * 3
        e2 = p_td * (TD_VALUE ** 2) + p_fg * 9
        return drives * (e2 - e * e) + (ppd ** 2) * DRIVE_SD ** 2

    va, vb = sim_var(mu_a), sim_var(mu_b)
    ex_a = np.sqrt(max(sd_a ** 2 - va, 1.0))
    ex_b = np.sqrt(max(sd_b ** 2 - vb, 1.0))
    cov_sim = (mu_a / drives) * (mu_b / drives) * DRIVE_SD ** 2
    ex_rho = np.clip((rho * sd_a * sd_b - cov_sim) / (ex_a * ex_b), -0.9, 0.9)
    z = rng.multivariate_normal([0, 0], [[1, ex_rho], [ex_rho, 1]], size=n)
    lam_a = np.clip(mu_a + z[:, 0] * ex_a, 1.0, None)
    lam_b = np.clip(mu_b + z[:, 1] * ex_b, 1.0, None)
    nd = np.clip(np.rint(rng.normal(drives, DRIVE_SD, n)), 6, 18).astype(int)
    nd_b = np.clip(nd + rng.choice([-1, 0, 0, 1], n), 6, 18)
    pa, score_p_a = _team_points(rng, lam_a, nd)
    pb, score_p_b = _team_points(rng, lam_b, nd_b)
    reg_a, reg_b = pa.copy(), pb.copy()

    # Prórroga: posesiones alternadas, gana el primero que anota (aprox.).
    tied = np.where(pa == pb)[0]
    ot_ties = 0
    if len(tied):
        max_pos = 30 if playoff else 4
        first_a = rng.random(len(tied)) < 0.5
        resolved = np.zeros(len(tied), bool)
        for k in range(max_pos):
            a_has = first_a ^ (k % 2 == 1)
            p_score = np.where(a_has, score_p_a[tied], score_p_b[tied])
            scores = (~resolved) & (rng.random(len(tied)) < p_score)
            is_td = rng.random(len(tied)) < 0.4
            pts = np.where(is_td, 6, 3)
            pa[tied[scores & a_has]] += pts[scores & a_has]
            pb[tied[scores & ~a_has]] += pts[scores & ~a_has]
            resolved |= scores
        ot_ties = int((~resolved).sum())

    win_a = float(np.mean(pa > pb))
    win_b = float(np.mean(pb > pa))
    tie = float(np.mean(pa == pb))
    margin = pa - pb
    total = pa + pb
    # Marcadores más probables
    pairs, counts = np.unique(np.stack([pa, pb], axis=1), axis=0, return_counts=True)
    top = np.argsort(-counts)[:8]
    hist_edges = np.arange(-45, 46, 3)
    mh, _ = np.histogram(np.clip(margin, -44, 44), bins=hist_edges)
    th_edges = np.arange(0, 96, 4)
    th, _ = np.histogram(np.clip(total, 0, 94), bins=th_edges)
    pct = lambda x, q: float(np.percentile(x, q))
    return {
        "n_sims": n,
        "p_a": win_a, "p_b": win_b, "p_tie": tie,
        "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
        "median_a": pct(pa, 50), "median_b": pct(pb, 50),
        "p10_a": pct(pa, 10), "p90_a": pct(pa, 90), "p10_b": pct(pb, 10), "p90_b": pct(pb, 90),
        "margin_mean": float(margin.mean()), "margin_sd": float(margin.std()),
        "margin_p10": pct(margin, 10), "margin_p90": pct(margin, 90),
        "total_mean": float(total.mean()), "total_sd": float(total.std()),
        "total_p10": pct(total, 10), "total_p90": pct(total, 90),
        "p_ot": float(np.mean(reg_a == reg_b)), "ot_ties": ot_ties,
        "top_scores": [{"a": int(pairs[i, 0]), "b": int(pairs[i, 1]), "p": float(counts[i] / n)} for i in top],
        "margin_hist": {"edges": hist_edges.tolist(), "counts": (mh / n).round(4).tolist()},
        "total_hist": {"edges": th_edges.tolist(), "counts": (th / n).round(4).tolist()},
    }
