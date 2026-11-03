"""Distribuciones de resultado: Dixon-Coles para goles, binomial negativa para conteos."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

MAX_GOALS = 10


def dc_tau(x: np.ndarray, y: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    t = np.ones_like(x, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def score_matrix(lam: float, mu: float, rho: float = 0.0, n: int = MAX_GOALS) -> np.ndarray:
    """Matriz P(local=i, visitante=j) con corrección Dixon-Coles de marcadores bajos."""
    g = np.arange(n + 1)
    m = np.outer(stats.poisson.pmf(g, lam), stats.poisson.pmf(g, mu))
    X, Y = np.meshgrid(g, g, indexing="ij")
    m = m * dc_tau(X, Y, lam, mu, rho)
    m = np.clip(m, 0, None)
    return m / m.sum()


def fit_rho(lams: np.ndarray, mus: np.ndarray, hg: np.ndarray, ag: np.ndarray, w: np.ndarray) -> float:
    """Estima ρ de Dixon-Coles por máxima verosimilitud dados λ, μ ya ajustados."""
    hg = hg.astype(int); ag = ag.astype(int)
    mask = (hg <= 1) & (ag <= 1)
    if mask.sum() < 30:
        return 0.0
    l, m, x, y, ww = lams[mask], mus[mask], hg[mask], ag[mask], w[mask]

    def nll(r):
        t = dc_tau(x, y, l, m, r)
        if np.any(t <= 0):
            return 1e9
        return -np.sum(ww * np.log(t))

    res = optimize.minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


def goal_markets(M: np.ndarray) -> dict:
    n = M.shape[0]
    g = np.arange(n)
    tot = g[:, None] + g[None, :]
    home = float(np.tril(M, -1).sum())
    draw = float(np.trace(M))
    away = float(np.triu(M, 1).sum())
    idx = np.unravel_index(np.argmax(M), M.shape)
    top = sorted(((float(M[i, j]), i, j) for i in range(min(n, 6)) for j in range(min(n, 6))), reverse=True)[:6]
    return {
        "home_win": home, "draw": draw, "away_win": away,
        "over_1_5": float(M[tot > 1.5].sum()),
        "over_2_5": float(M[tot > 2.5].sum()),
        "over_3_5": float(M[tot > 3.5].sum()),
        "btts": float(M[1:, 1:].sum()),
        "home_clean_sheet": float(M[:, 0].sum()),
        "away_clean_sheet": float(M[0, :].sum()),
        "most_likely_score": f"{idx[0]}-{idx[1]}",
        "most_likely_score_prob": float(M[idx]),
        "top_scores": [{"score": f"{i}-{j}", "prob": p} for p, i, j in top],
        "home_goals_dist": [float(x) for x in M.sum(axis=1)[:6]],
        "away_goals_dist": [float(x) for x in M.sum(axis=0)[:6]],
    }


def nb_params(mean: float, alpha: float) -> tuple[float, float] | None:
    """Parámetros (n, p) de scipy para NB con Var = μ + α μ². None -> usar Poisson."""
    if alpha <= 1e-6:
        return None
    n = 1.0 / alpha
    p = n / (n + mean)
    return n, p


def count_pmf(mean: float, alpha: float, kmax: int = 40) -> np.ndarray:
    k = np.arange(kmax + 1)
    nb = nb_params(mean, alpha)
    pmf = stats.poisson.pmf(k, mean) if nb is None else stats.nbinom.pmf(k, *nb)
    return pmf / pmf.sum()


def sample_counts(rng: np.random.Generator, means: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 1e-6:
        return rng.poisson(means)
    # Mezcla Gamma-Poisson
    shape = 1.0 / alpha
    return rng.poisson(rng.gamma(shape, means / shape))
