import numpy as np
import pandas as pd

from app.ratings import TEAMS, fit_metric
from app.simulate import simulate
from app.sources.nflverse import BETTING_COLUMNS
from app.model import FEATURES


def _synthetic_games(n_weeks=17, seed=0):
    rng = np.random.default_rng(seed)
    off = {t: rng.normal(0, 4) for t in TEAMS}
    de = {t: rng.normal(0, 4) for t in TEAMS}
    rows = []
    for w in range(1, n_weeks + 1):
        perm = rng.permutation(TEAMS)
        for i in range(0, 32, 2):
            h, a = perm[i], perm[i + 1]
            for team, opp, loc in ((h, a, 1), (a, h, -1)):
                rows.append({"season": 2025, "week": w, "team": team, "opp": opp, "loc": loc,
                             "pts_for": 22 + off[team] + de[opp] + 1.0 * loc + rng.normal(0, 3)})
    df = pd.DataFrame(rows)
    df["t"] = df["season"] * 30 + df["week"]
    df["oi"] = df["team"].map({t: i for i, t in enumerate(TEAMS)})
    df["di"] = df["opp"].map({t: i for i, t in enumerate(TEAMS)})
    return df, off, de


def test_ratings_recover_true_strengths():
    df, off, de = _synthetic_games()
    r = fit_metric(df, "pts", now_t=2025 * 30 + 30, half_life=1000, lam=0.5)
    est = np.array([r.off[t] for t in TEAMS])
    true = np.array([off[t] for t in TEAMS])
    assert np.corrcoef(est, true)[0, 1] > 0.9
    assert 0.3 < r.home < 1.7


def test_ratings_use_only_past_games():
    df, _, _ = _synthetic_games()
    cutoff = 2025 * 30 + 5
    r1 = fit_metric(df, "pts", cutoff, 10, 4)
    df2 = df.copy()
    df2.loc[df2["t"] >= cutoff, "pts_for"] = 999  # alterar el futuro no debe cambiar nada
    r2 = fit_metric(df2, "pts", cutoff, 10, 4)
    assert r1.off == r2.off and r1.mu == r2.mu


def test_simulation_probabilities_and_means():
    sim = simulate(27.0, 21.0, 11.0, 10.0, 10.0, 0.05, n=10_000, seed=1)
    assert sim["n_sims"] == 10_000
    assert abs(sim["p_a"] + sim["p_b"] + sim["p_tie"] - 1) < 1e-9
    assert sim["p_a"] > 0.6
    assert abs(sim["mean_a"] - 27) < 1.5 and abs(sim["mean_b"] - 21) < 1.5
    assert 11 < sim["margin_sd"] < 17


def test_simulation_symmetric_for_equal_teams():
    sim = simulate(23.0, 23.0, 11.0, 10.0, 10.0, 0.0, n=20_000, seed=2)
    assert abs(sim["p_a"] - sim["p_b"]) < 0.03


def test_no_betting_inputs():
    joined = " ".join(FEATURES)
    for col in BETTING_COLUMNS:
        assert col not in joined
    assert not any(k in joined for k in ("spread", "moneyline", "odds", "vegas", "total_line"))
