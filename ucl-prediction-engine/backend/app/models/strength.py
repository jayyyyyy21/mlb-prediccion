"""Modelo jerárquico de fuerza de equipos (GLM de Poisson penalizado).

Para una variable de conteo y (goles, córners, tiros a puerta…) producida por el
equipo i contra el equipo j:

    log E[y] = c + h·local + (A_L(i) + a_i) − (D_L(j) + d_j) + γ·covariables

* A_L / D_L: nivel ofensivo/defensivo de la liga del equipo. Se identifica gracias
  a los partidos de competiciones europeas (UCL, UEL, UECL), que conectan ligas.
* a_i / d_i: desviación del equipo respecto a su liga, con penalización L2
  (encogimiento hacia la media de la liga → evita sobreajuste con pocos datos).
* Ponderación temporal exponencial (vida media configurable): los partidos
  recientes pesan más.
* Covariables con coeficientes estimados de los datos (p. ej. descanso corto).

La incertidumbre de los parámetros se aproxima con la inversa del hessiano
penalizado (aproximación de Laplace), que se usa en la simulación Monte Carlo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse


@dataclass
class Obs:
    """Una observación = un equipo atacando a otro en un partido."""

    att: str
    dfn: str
    home: float  # 1 local, 0 visitante, 0.5 campo neutral
    y: float
    weight: float
    cov: tuple[float, ...] = ()


@dataclass
class StrengthModel:
    team_league: dict[str, str]
    lam_team: float = 4.0
    lam_league: float = 1.0
    n_cov: int = 0
    gaussian: bool = False  # True -> regresión ridge (para posesión en logit)
    # Factor de encogimiento (0–1) de los efectos de equipo/liga en la PREDICCIÓN, elegido por
    # validación cronológica. 1 = sin encoger. Corrige el exceso de dispersión fuera de muestra.
    shrink: float = 1.0

    teams: list[str] = field(default_factory=list)
    leagues: list[str] = field(default_factory=list)
    beta: np.ndarray | None = None
    cov_matrix: np.ndarray | None = None
    dispersion: float = 0.0  # binomial negativa: Var = μ + α μ²
    resid_sd: float = 0.0  # gaussian
    n_obs: int = 0

    # --- diseño -----------------------------------------------------------
    def _index(self, obs: list[Obs]) -> None:
        ts = sorted({o.att for o in obs} | {o.dfn for o in obs})
        self.teams = ts
        self.leagues = sorted({self.team_league.get(t, "otros") for t in ts})
        self._ti = {t: k for k, t in enumerate(ts)}
        self._li = {lg: k for k, lg in enumerate(self.leagues)}

    @property
    def n_params(self) -> int:
        return 2 + self.n_cov + 2 * len(self.leagues) + 2 * len(self.teams)

    def _offsets(self):
        nL, nT = len(self.leagues), len(self.teams)
        o_cov = 2
        o_AL = o_cov + self.n_cov
        o_DL = o_AL + nL
        o_a = o_DL + nL
        o_d = o_a + nT
        return o_cov, o_AL, o_DL, o_a, o_d

    def _row(self, att: str, dfn: str, home: float, cov: tuple[float, ...]):
        """Devuelve (columnas, valores) de la fila de diseño. Equipos desconocidos -> solo liga/media."""
        o_cov, o_AL, o_DL, o_a, o_d = self._offsets()
        cols, vals = [0, 1], [1.0, home]
        for k, v in enumerate(cov):
            cols.append(o_cov + k)
            vals.append(v)
        la = self.team_league.get(att, "otros")
        ld = self.team_league.get(dfn, "otros")
        if la in self._li:
            cols.append(o_AL + self._li[la]); vals.append(1.0)
        if ld in self._li:
            cols.append(o_DL + self._li[ld]); vals.append(-1.0)
        if att in self._ti:
            cols.append(o_a + self._ti[att]); vals.append(1.0)
        if dfn in self._ti:
            cols.append(o_d + self._ti[dfn]); vals.append(-1.0)
        return cols, vals

    def _design(self, obs: list[Obs]):
        rows, cols, vals = [], [], []
        for r, o in enumerate(obs):
            c, v = self._row(o.att, o.dfn, o.home, o.cov)
            rows.extend([r] * len(c)); cols.extend(c); vals.extend(v)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(obs), self.n_params))

    def _penalty_vec(self) -> np.ndarray:
        _, o_AL, _, o_a, _ = self._offsets()
        p = np.zeros(self.n_params)
        p[o_AL:o_a] = self.lam_league
        p[o_a:] = self.lam_team
        if self.n_cov:
            p[2:2 + self.n_cov] = 0.1
        return p

    # --- ajuste -----------------------------------------------------------
    def fit(self, obs: list[Obs], compute_cov: bool = True) -> "StrengthModel":
        if not obs:
            raise ValueError("sin observaciones")
        self._index(obs)
        X = self._design(obs)
        y = np.array([o.y for o in obs], dtype=float)
        w = np.array([o.weight for o in obs], dtype=float)
        pen = self._penalty_vec()
        self.n_obs = len(obs)

        if self.gaussian:
            W = sparse.diags(w)
            A = (X.T @ W @ X).toarray() + np.diag(pen + 1e-9)
            b = X.T @ (w * y)
            self.beta = np.linalg.solve(A, b)
            resid = y - X @ self.beta
            self.resid_sd = float(np.sqrt(np.sum(w * resid ** 2) / max(np.sum(w) - 1, 1)))
            self.cov_matrix = (np.linalg.inv(A) * self.resid_sd ** 2 if compute_cov
                               else np.zeros((self.n_params, self.n_params)))
            return self

        def f(beta):
            eta = X @ beta
            mu = np.exp(eta)
            nll = np.sum(w * (mu - y * eta)) + 0.5 * np.sum(pen * beta ** 2)
            grad = X.T @ (w * (mu - y)) + pen * beta
            return nll, grad

        beta0 = np.zeros(self.n_params)
        beta0[0] = np.log(max(np.average(y, weights=w), 1e-3))
        res = optimize.minimize(f, beta0, jac=True, method="L-BFGS-B",
                                options={"maxiter": 3000, "gtol": 1e-7})
        self.beta = res.x
        mu = np.exp(X @ self.beta)
        if compute_cov:
            H = (X.T @ sparse.diags(w * mu) @ X).toarray() + np.diag(pen + 1e-9)
            self.cov_matrix = np.linalg.inv(H)
        else:
            self.cov_matrix = np.zeros((self.n_params, self.n_params))
        # Sobredispersión (método de momentos, ponderado): Var = μ + α μ²
        num = np.sum(w * ((y - mu) ** 2 - mu))
        den = np.sum(w * mu ** 2)
        self.dispersion = float(max(num / den, 0.0)) if den > 0 else 0.0
        return self

    # --- predicción ---------------------------------------------------------
    def x_vec(self, att: str, dfn: str, home: float, cov: tuple[float, ...] = ()) -> np.ndarray:
        cov = cov or tuple([0.0] * self.n_cov)
        x = np.zeros(self.n_params)
        c, v = self._row(att, dfn, home, cov)
        for ci, vi in zip(c, v):
            x[ci] += vi
        return x

    def eta(self, att: str, dfn: str, home: float, cov: tuple[float, ...] = ()) -> tuple[float, float]:
        """Predictor lineal y su desviación estándar (incertidumbre de parámetros)."""
        cov = cov or tuple([0.0] * self.n_cov)
        c, v = self._row(att, dfn, home, cov)
        v = self._scale(c, v)
        c, v = np.array(c), np.array(v, dtype=float)
        e = float(np.dot(self.beta[c], v))
        var = float(v @ self.cov_matrix[np.ix_(c, c)] @ v)
        return e, float(np.sqrt(max(var, 0.0)))

    def mean(self, att: str, dfn: str, home: float, cov: tuple[float, ...] = ()) -> float:
        """Media esperada (sin calcular la varianza: rápido)."""
        cov = cov or tuple([0.0] * self.n_cov)
        c, v = self._row(att, dfn, home, cov)
        return float(np.exp(np.dot(self.beta[c], self._scale(c, v))))

    def _scale(self, c, v):
        """Aplica el encogimiento a las columnas de liga/equipo (no a intercepto, localía ni covariables)."""
        if self.shrink == 1.0:
            return v
        k0 = 2 + self.n_cov
        return [vi * (self.shrink if ci >= k0 else 1.0) for ci, vi in zip(c, v)]

    def ratings(self, team: str) -> dict:
        _, o_AL, o_DL, o_a, o_d = self._offsets()
        lg = self.team_league.get(team, "otros")
        li = self._li.get(lg)
        ti = self._ti.get(team)
        A = (self.beta[o_AL + li] if li is not None else 0.0) + (self.beta[o_a + ti] if ti is not None else 0.0)
        D = (self.beta[o_DL + li] if li is not None else 0.0) + (self.beta[o_d + ti] if ti is not None else 0.0)
        return {"attack": float(A), "defense": float(D), "known": ti is not None}

    def cov_coefs(self) -> list[tuple[float, float]]:
        return [(float(self.beta[2 + k]), float(np.sqrt(self.cov_matrix[2 + k, 2 + k])))
                for k in range(self.n_cov)]

    @property
    def home_adv(self) -> tuple[float, float]:
        return float(self.beta[1]), float(np.sqrt(self.cov_matrix[1, 1]))
