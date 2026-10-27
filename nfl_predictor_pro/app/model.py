"""Modelo de proyección de puntos.

Para la ofensiva del equipo A contra la defensa del equipo B se calculan
variables explicativas a partir de ratings ajustados por rival (ver
ratings.py) y del contexto del partido. Los puntos de A se estiman con una
regresión lineal regularizada cuyos coeficientes se ajustan con temporadas
anteriores (walk-forward) — nunca con información posterior al partido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ratings import Snapshot

# Hiperparámetros por defecto (se re-optimizan en backtest.py y se guardan en la BD).
DEFAULT_PARAMS = {
    "half_life": 10.0,        # semanas
    "lam": 4.0,               # partidos equivalentes de regresión a la media
    "half_life_recent": 3.0,
    "lam_recent": 3.0,
    "ridge_beta": 25.0,       # regularización de los coeficientes del modelo final
    "sd_scale": 1.0,          # factor de calibración de la dispersión (probabilidades)
}

FEATURES = [
    "f_pts",        # puntos esperados: ofensiva A + defensa B (ajustado por rival)
    "f_pts_recent", # forma reciente (vida media corta)
    "f_epa",        # EPA/jugada esperado combinando pase y carrera según tendencias
    "f_ppd_pace",   # puntos por posesión esperados x posesiones esperadas
    "f_sr",         # tasa de éxito esperada
    "f_sack",       # capturas: permitidas por A vs generadas por B
    "f_to",         # pérdidas de balón esperadas
    "f_expl",       # jugadas explosivas esperadas
    "f_rz",         # TD en zona roja esperado
    "f_loc",        # +1 local, -1 visitante, 0 neutral
    "f_rest",       # diferencia de días de descanso (acotada a ±7)
    "f_tz_own",     # horas de cambio de zona horaria propias (valor absoluto)
    "f_miles_own",  # miles de millas viajadas
    "f_wind",       # viento por encima de 10 mph (exteriores)
    "f_cold",       # grados F por debajo de 40 (exteriores)
]

FEATURE_LABELS = {
    "f_pts": "Producción ofensiva vs puntos permitidos (ajustado por rival)",
    "f_pts_recent": "Rendimiento reciente",
    "f_epa": "Eficiencia EPA/jugada (pase y carrera vs defensa rival)",
    "f_ppd_pace": "Puntos por posesión x ritmo esperado",
    "f_sr": "Success rate esperado",
    "f_sack": "Capturas permitidas vs presión rival",
    "f_to": "Pérdidas de balón esperadas",
    "f_expl": "Jugadas explosivas",
    "f_rz": "Eficiencia en zona roja",
    "f_loc": "Localía",
    "f_rest": "Descanso",
    "f_tz_own": "Cambio de zona horaria",
    "f_miles_own": "Distancia de viaje",
    "f_wind": "Viento",
    "f_cold": "Frío",
}


def _r(snap: Snapshot, metric: str, off: str, deff: str, recent: bool = False) -> float:
    r = snap.get(metric, recent)
    if r is None:
        return np.nan
    return r.expected(off, deff, 0)


def matchup_features(snap: Snapshot, team: str, opp: str, ctx: dict) -> dict:
    """Variables para los puntos de `team` (ofensiva) contra `opp` (defensa)."""
    pr = _r(snap, "pass_rate", team, opp)
    pe = _r(snap, "pepa", team, opp)
    re_ = _r(snap, "repa", team, opp)
    drives_a = _r(snap, "drives", team, opp)
    drives_b = _r(snap, "drives", opp, team)
    pace = np.nanmean([drives_a, drives_b])
    rec = snap.get("pts", recent=True)
    return {
        "f_pts": _r(snap, "pts", team, opp),
        "f_pts_recent": (rec.off.get(team, 0) + rec.deff.get(opp, 0)) if rec else 0.0,
        "f_epa": pr * pe + (1 - pr) * re_,
        "f_ppd_pace": _r(snap, "ppd", team, opp) * pace,
        "f_sr": _r(snap, "sr", team, opp),
        "f_sack": _r(snap, "sack_rate", team, opp),
        "f_to": _r(snap, "to_rate", team, opp),
        "f_expl": _r(snap, "expl", team, opp),
        "f_rz": _r(snap, "rz", team, opp),
        "f_loc": float(ctx.get("loc", 0)),
        "f_rest": float(np.clip(ctx.get("rest_diff", 0) or 0, -7, 7)),
        "f_tz_own": float(abs(ctx.get("tz_shift", 0) or 0)),
        "f_miles_own": float((ctx.get("miles", 0) or 0) / 1000),
        "f_wind": float(ctx.get("wind_eff", 0) or 0),
        "f_cold": float(ctx.get("cold_eff", 0) or 0),
        "pace": pace,
    }


@dataclass
class PointsModel:
    """Regresión ridge sobre variables estandarizadas."""
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    resid_sd: float
    resid_corr: float
    sd_by_week: dict[int, float]
    n_train: int

    @classmethod
    def fit(cls, X: pd.DataFrame, y: np.ndarray, ridge: float, game_ids: np.ndarray | None = None,
            weeks: np.ndarray | None = None, features: list[str] = FEATURES) -> "PointsModel":
        Xf = X[features].to_numpy(float)
        mask = np.isfinite(Xf).all(axis=1) & np.isfinite(y)
        Xf, yv = Xf[mask], y[mask]
        mean = Xf.mean(axis=0)
        scale = Xf.std(axis=0)
        scale[scale == 0] = 1
        Z = (Xf - mean) / scale
        y0 = yv.mean()
        coef = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ (yv - y0))
        resid = yv - (y0 + Z @ coef)
        corr = 0.0
        if game_ids is not None:
            r = pd.DataFrame({"g": game_ids[mask], "r": resid}).groupby("g")["r"].agg(list)
            pairs = np.array([v for v in r if len(v) == 2])
            if len(pairs) > 10:
                corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        sd_by_week: dict[int, float] = {}
        if weeks is not None:
            wk = np.minimum(weeks[mask], 6)  # semanas 1..5 por separado, 6+ juntas
            for w in np.unique(wk):
                sd_by_week[int(w)] = float(np.std(resid[wk == w]))
        return cls(features, mean, scale, coef, float(y0), float(np.std(resid)), corr, sd_by_week, int(mask.sum()))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.features].to_numpy(float)
        Xf = np.where(np.isfinite(Xf), Xf, self.mean)  # faltantes -> promedio (se alerta aparte)
        return self.intercept + ((Xf - self.mean) / self.scale) @ self.coef

    def contributions(self, row: dict) -> dict[str, float]:
        """Aporte (en puntos) de cada variable respecto a un partido promedio."""
        out = {}
        for i, f in enumerate(self.features):
            v = row.get(f, np.nan)
            if not np.isfinite(v):
                v = self.mean[i]
            out[f] = float((v - self.mean[i]) / self.scale[i] * self.coef[i])
        return out

    def sd_for_week(self, week: int) -> float:
        return self.sd_by_week.get(min(int(week), 6), self.resid_sd)

    def to_dict(self) -> dict:
        return {
            "features": self.features, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "coef": self.coef.tolist(), "intercept": self.intercept, "resid_sd": self.resid_sd,
            "resid_corr": self.resid_corr, "sd_by_week": self.sd_by_week, "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PointsModel":
        return cls(d["features"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]),
                   d["intercept"], d["resid_sd"], d["resid_corr"],
                   {int(k): v for k, v in d["sd_by_week"].items()}, d["n_train"])
