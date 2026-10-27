"""Calibración y validación histórica walk-forward (fuera de muestra).

Procedimiento:
1. Para cada semana de cada temporada se calculan los ratings usando SOLO
   partidos anteriores a esa semana, y con ellos las variables del modelo.
2. Para evaluar la temporada S, los coeficientes del modelo se ajustan con
   las temporadas anteriores a S (nunca con la propia S ni posteriores).
3. Los hiperparámetros (vida media, regularización) se eligen con las
   temporadas de VALIDACIÓN; las temporadas de PRUEBA no intervienen en
   ninguna decisión, así que sus métricas son genuinamente fuera de muestra.
4. Las probabilidades de victoria salen de la simulación Monte Carlo con la
   misma configuración que se usa en producción.

Limitación documentada: el backtest no incluye ajustes por lesiones (los
reportes históricos no se reconstruyen tal como se conocían antes de cada
partido) y usa el clima registrado el día del partido como aproximación del
pronóstico disponible antes del juego.
"""
from __future__ import annotations

import itertools
import logging
import math
import time

import numpy as np
import pandas as pd

from . import data, db
from .config import CURRENT_SEASON, FIRST_SEASON, N_SIMS
from .model import DEFAULT_PARAMS, PointsModel, matchup_features
from .simulate import simulate
from .ratings import snapshot

log = logging.getLogger(__name__)

CTX_COLS = ["loc", "rest_diff", "tz_shift", "miles", "wind_eff", "cold_eff"]


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def walk_forward_features(tg_all: pd.DataFrame, tg: pd.DataFrame, params: dict, first_season: int) -> pd.DataFrame:
    """Variables del modelo para cada fila equipo-partido, con ratings previos al partido."""
    rows = []
    target = tg_all[(tg_all["season"] >= first_season) & tg_all["pts_for"].notna()]
    for (season, week), grp in target.groupby(["season", "week"]):
        snap = snapshot(tg, season * 30 + week, params)
        for r in grp.itertuples(index=False):
            ctx = {c: getattr(r, c) for c in CTX_COLS}
            ctx = {k: (0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in ctx.items()}
            f = matchup_features(snap, r.team, r.opp, ctx)
            f.update(game_id=r.game_id, season=season, week=week, team=r.team, opp=r.opp, loc=r.loc,
                     game_type=r.game_type, pts=r.pts_for, pts_against=r.pts_against)
            rows.append(f)
    return pd.DataFrame(rows)


def fit_season_models(feat: pd.DataFrame, seasons: list[int], ridge: float, min_train_seasons: int = 2):
    """Modelo walk-forward por temporada: entrena con temporadas < S."""
    models = {}
    for s in seasons:
        train = feat[feat["season"] < s]
        if train["season"].nunique() < min_train_seasons:
            continue
        models[s] = PointsModel.fit(train, train["pts"].to_numpy(float), ridge,
                                    train["game_id"].to_numpy(), train["week"].to_numpy())
    return models


def predict_games(feat: pd.DataFrame, models: dict, mc: bool, n_sims: int = N_SIMS,
                  sd_scale: float = 1.0) -> pd.DataFrame:
    """Una fila por partido con proyecciones del local (A) y visitante (B)."""
    out = []
    for s, m in models.items():
        fs = feat[feat["season"] == s].copy()
        if fs.empty:
            continue
        fs["pred"] = m.predict(fs)
        for gid, g in fs.groupby("game_id"):
            if len(g) != 2:
                continue
            g = g.sort_values("loc", ascending=False)  # local primero (o el primero en neutral)
            a, b = g.iloc[0], g.iloc[1]
            sd = m.sd_for_week(int(a["week"])) * sd_scale
            playoff = a["game_type"] != "REG"
            if mc:
                sim = simulate(a["pred"], b["pred"], a["pace"], sd, sd, m.resid_corr, n=n_sims,
                               playoff=playoff, seed=abs(hash(gid)) % (2**32))
                p_a, p_tie = sim["p_a"], sim["p_tie"]
            else:
                msd = math.sqrt(2 * sd * sd * (1 - m.resid_corr))
                p_a, p_tie = norm_cdf((a["pred"] - b["pred"]) / msd), 0.0
            out.append({
                "game_id": gid, "season": s, "week": int(a["week"]), "game_type": a["game_type"],
                "team_a": a["team"], "team_b": b["team"], "loc_a": int(a["loc"]),
                "pred_a": float(a["pred"]), "pred_b": float(b["pred"]),
                "pts_a": float(a["pts"]), "pts_b": float(b["pts"]),
                "p_a": float(p_a), "p_tie": float(p_tie),
            })
    return pd.DataFrame(out)


def metrics(pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {}
    res = np.where(pred["pts_a"] > pred["pts_b"], 1.0, np.where(pred["pts_a"] < pred["pts_b"], 0.0, 0.5))
    p = pred["p_a"] + 0.5 * pred.get("p_tie", 0)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    decided = res != 0.5
    err = np.concatenate([pred["pred_a"] - pred["pts_a"], pred["pred_b"] - pred["pts_b"]])
    return {
        "games": int(len(pred)),
        "mae_points": float(np.mean(np.abs(err))),
        "rmse_points": float(np.sqrt(np.mean(err ** 2))),
        "bias_points": float(np.mean(err)),
        "mae_margin": float(np.mean(np.abs((pred["pred_a"] - pred["pred_b"]) - (pred["pts_a"] - pred["pts_b"])))),
        "mae_total": float(np.mean(np.abs((pred["pred_a"] + pred["pred_b"]) - (pred["pts_a"] + pred["pts_b"])))),
        "brier": float(np.mean((p - res) ** 2)),
        "log_loss": float(-np.mean(res * np.log(p) + (1 - res) * np.log(1 - p))),
        "accuracy": float(np.mean((p[decided] > 0.5) == (res[decided] == 1))),
    }


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> list[dict]:
    res = np.where(pred["pts_a"] > pred["pts_b"], 1.0, np.where(pred["pts_a"] < pred["pts_b"], 0.0, 0.5))
    p = (pred["p_a"] + 0.5 * pred.get("p_tie", 0)).to_numpy()
    # Se usa la probabilidad del favorito para que cada partido aparezca una vez.
    fav_p = np.where(p >= 0.5, p, 1 - p)
    fav_res = np.where(p >= 0.5, res, 1 - res)
    edges = np.linspace(0.5, 1.0, bins // 2 + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fav_p >= lo) & (fav_p < hi if hi < 1 else fav_p <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()), "pred": float(fav_p[m].mean()),
                    "obs": float(fav_res[m].mean())})
    return out


def baselines(tg: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Modelos de referencia ingenuos para contextualizar la precisión."""
    out = {}
    hist = tg.copy()
    # 1) Promedio de puntos de la liga en temporadas previas + probabilidad 50%
    rows = []
    for s, g in pred.groupby("season"):
        prev = hist[hist["season"] < s]
        lg = prev["pts_for"].mean()
        home_wr = (prev[prev["loc"] == 1]["pts_for"].to_numpy() > prev[prev["loc"] == 1]["pts_against"].to_numpy()).mean()
        for r in g.itertuples(index=False):
            rows.append({**r._asdict(), "pred_a": lg, "pred_b": lg, "p_a": home_wr if r.loc_a == 1 else 0.5, "p_tie": 0.0})
    out["promedio_liga"] = metrics(pd.DataFrame(rows))
    # 2) Promedios de la temporada (puntos anotados y permitidos) sin ajuste por rival
    rows = []
    for r in pred.itertuples(index=False):
        prior = hist[(hist["season"] == r.season) & (hist["week"] < r.week)]
        if prior.empty:
            prior = hist[hist["season"] == r.season - 1]
        def avg(team, col):
            v = prior[prior["team"] == team][col]
            return v.mean() if len(v) else hist["pts_for"].mean()
        pa = (avg(r.team_a, "pts_for") + avg(r.team_b, "pts_against")) / 2
        pb = (avg(r.team_b, "pts_for") + avg(r.team_a, "pts_against")) / 2
        rows.append({**r._asdict(), "pred_a": pa, "pred_b": pb, "p_a": norm_cdf((pa - pb) / 13.5), "p_tie": 0.0})
    out["promedios_simples"] = metrics(pd.DataFrame(rows))
    return out


def run(val_seasons: tuple[int, ...] | None = None, test_seasons: tuple[int, ...] | None = None,
        grid: dict | None = None, save: bool = True) -> dict:
    t0 = time.time()
    _, tg_all, tg = data.load()
    first_feat = FIRST_SEASON + 1  # la primera temporada sólo aporta historia a los ratings
    val_seasons = val_seasons or (FIRST_SEASON + 3, FIRST_SEASON + 4, FIRST_SEASON + 5)
    test_seasons = test_seasons or tuple(range(max(val_seasons) + 1, CURRENT_SEASON + 1))
    grid = grid or {"half_life": [6.0, 10.0, 16.0], "lam": [2.0, 4.0, 8.0], "ridge_beta": [30.0, 300.0, 1500.0]}

    # 1) Búsqueda de hiperparámetros con temporadas de validación (sin MC: aproximación normal)
    search = []
    feat_cache: dict[tuple, pd.DataFrame] = {}
    for hl, lam in itertools.product(grid["half_life"], grid["lam"]):
        params = {**DEFAULT_PARAMS, "half_life": hl, "lam": lam}
        key = (hl, lam)
        feat_cache[key] = walk_forward_features(tg_all, tg, params, first_feat)
        for rb in grid["ridge_beta"]:
            models = fit_season_models(feat_cache[key], list(val_seasons), rb)
            m = metrics(predict_games(feat_cache[key], models, mc=False))
            search.append({"half_life": hl, "lam": lam, "ridge_beta": rb, **m})
            log.info("hl=%s lam=%s rb=%s mae=%.3f ll=%.4f", hl, lam, rb, m["mae_points"], m["log_loss"])
    search_df = pd.DataFrame(search)
    # criterio: MAE de puntos (objetivo principal) con desempate por log-loss
    search_df["score"] = search_df["mae_points"] + 2 * search_df["log_loss"]
    best = search_df.sort_values("score").iloc[0]
    params = {**DEFAULT_PARAMS, "half_life": float(best["half_life"]), "lam": float(best["lam"]),
              "ridge_beta": float(best["ridge_beta"])}
    feat = feat_cache[(params["half_life"], params["lam"])]

    # 1b) Calibración de la dispersión: factor sobre la desviación estándar
    # residual elegido por log-loss en validación (afecta sólo a probabilidades).
    val_models = fit_season_models(feat, list(val_seasons), params["ridge_beta"])
    scale_search = []
    for sc in grid.get("sd_scale", [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]):
        m = metrics(predict_games(feat, val_models, mc=False, sd_scale=sc))
        scale_search.append({"sd_scale": sc, "log_loss": m["log_loss"], "brier": m["brier"]})
    params["sd_scale"] = float(min(scale_search, key=lambda r: r["log_loss"])["sd_scale"])

    # 2) Evaluación fuera de muestra con Monte Carlo completo
    models = fit_season_models(feat, list(val_seasons) + list(test_seasons), params["ridge_beta"])
    pred = predict_games(feat, models, mc=True, sd_scale=params["sd_scale"])
    test_pred = pred[pred["season"].isin(test_seasons)]
    val_pred = pred[pred["season"].isin(val_seasons)]
    by_season = {int(s): metrics(g) for s, g in pred.groupby("season")}

    result = {
        "params": params,
        "validation_seasons": list(val_seasons),
        "test_seasons": list(test_seasons),
        "validation": metrics(val_pred),
        "test": metrics(test_pred),
        "by_season": by_season,
        "calibration_test": calibration_table(test_pred),
        "baselines_test": baselines(tg, test_pred),
        "search": search_df.drop(columns=["score"]).to_dict(orient="records"),
        "sd_scale_search": scale_search,
        "n_sims": N_SIMS,
        "runtime_s": round(time.time() - t0, 1),
        "created_at": db.utcnow(),
    }

    # 3) Modelo de producción: entrenado con todo lo jugado hasta hoy
    prod = PointsModel.fit(feat, feat["pts"].to_numpy(float), params["ridge_beta"],
                           feat["game_id"].to_numpy(), feat["week"].to_numpy())
    result["production_model"] = prod.to_dict()
    if save:
        db.kv_set("model_params", params)
        db.kv_set("production_model", prod.to_dict())
        db.kv_set("backtest", {k: v for k, v in result.items() if k != "production_model"})
        db.write_df(pred, "backtest_games")
    return result
