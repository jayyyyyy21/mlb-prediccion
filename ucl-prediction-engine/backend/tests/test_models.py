"""Tests del motor estadístico con datos sintéticos (no requieren red ni base de datos real)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.models.distributions import count_pmf, fit_rho, goal_markets, sample_counts, score_matrix
from app.models.strength import Obs, StrengthModel
from app.providers.espn import american_to_decimal, parse_event
from app.models.dataset import comp_covariates, stats_valid


def test_score_matrix_normalized_and_dc_effect():
    m0 = score_matrix(1.4, 1.1, 0.0)
    m1 = score_matrix(1.4, 1.1, -0.1)
    assert m0.sum() == pytest.approx(1.0)
    assert m1.sum() == pytest.approx(1.0)
    # rho negativo aumenta 0-0 y 1-1 (Dixon-Coles)
    assert m1[0, 0] > m0[0, 0] and m1[1, 1] > m0[1, 1]


def test_goal_markets_consistency():
    g = goal_markets(score_matrix(1.6, 0.9, -0.05))
    assert g["home_win"] + g["draw"] + g["away_win"] == pytest.approx(1.0, abs=1e-9)
    assert g["over_1_5"] >= g["over_2_5"] >= g["over_3_5"]
    assert g["home_win"] > g["away_win"]


def test_fit_rho_recovers_sign():
    rng = np.random.default_rng(0)
    lam, mu, rho = 1.3, 1.0, -0.12
    M = score_matrix(lam, mu, rho)
    idx = rng.choice(M.size, size=20000, p=M.ravel())
    hg, ag = np.divmod(idx, M.shape[1])
    est = fit_rho(np.full(len(hg), lam), np.full(len(hg), mu), hg, ag, np.ones(len(hg)))
    assert est == pytest.approx(rho, abs=0.04)


def test_strength_model_recovers_ordering():
    rng = np.random.default_rng(1)
    teams = [f"t{i}" for i in range(12)]
    att = {t: a for t, a in zip(teams, np.linspace(-0.4, 0.4, 12))}
    dfn = {t: d for t, d in zip(teams, np.linspace(0.3, -0.3, 12))}
    obs = []
    for _ in range(40):
        for i in teams:
            for j in teams:
                if i == j or rng.random() > 0.15:
                    continue
                lam = math.exp(0.1 + 0.25 + att[i] - dfn[j])
                mu = math.exp(0.1 + att[j] - dfn[i])
                obs.append(Obs(i, j, 1.0, rng.poisson(lam), 1.0))
                obs.append(Obs(j, i, 0.0, rng.poisson(mu), 1.0))
    m = StrengthModel({t: "L" for t in teams}, lam_team=1.0).fit(obs)
    est = [m.ratings(t)["attack"] for t in teams]
    assert np.corrcoef(est, list(att.values()))[0, 1] > 0.9
    assert m.home_adv[0] == pytest.approx(0.25, abs=0.08)
    e, sd = m.eta("t11", "t0", 1.0)
    assert sd > 0
    # Encogimiento: con shrink=0 la predicción ignora equipos
    m.shrink = 0.0
    assert m.mean("t11", "t0", 1.0) == pytest.approx(m.mean("t0", "t11", 1.0))


def test_negative_binomial_mean_and_overdispersion():
    rng = np.random.default_rng(2)
    x = sample_counts(rng, np.full(200000, 5.0), 0.1)
    assert x.mean() == pytest.approx(5.0, rel=0.02)
    assert x.var() == pytest.approx(5.0 + 0.1 * 25, rel=0.05)
    pmf = count_pmf(5.0, 0.1)
    assert pmf.sum() == pytest.approx(1.0)


def test_american_odds_conversion():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal("+100") == pytest.approx(2.0)
    assert american_to_decimal(None) is None


def test_stats_quality_filter():
    ok = {"possession": 55, "shots": 12}, {"possession": 45, "shots": 8}
    zeros = {"possession": 0, "shots": 0, "corners": 0}, {"possession": 0, "shots": 0, "corners": 0}
    assert stats_valid(*ok)
    assert not stats_valid(*zeros)
    assert not stats_valid({}, {})


def test_comp_covariates():
    assert comp_covariates("uefa.champions") == (1.0, 0.0)
    assert comp_covariates("uefa.champions_qual") == (0.0, 1.0)
    assert comp_covariates("eng.1") == (0.0, 0.0)


def test_parse_event_minimal():
    ev = {
        "id": "1", "date": "2026-11-03T20:00Z", "season": {"year": 2026, "slug": "league-phase"},
        "competitions": [{
            "status": {"type": {"name": "STATUS_FULL_TIME", "completed": True}},
            "venue": {"fullName": "Estadio", "address": {"city": "Madrid", "country": "Spain"}},
            "competitors": [
                {"homeAway": "home", "score": "2", "team": {"id": "10", "displayName": "A"},
                 "statistics": [{"name": "wonCorners", "displayValue": "6"}, {"name": "possessionPct", "displayValue": "58.1"}]},
                {"homeAway": "away", "score": "1", "team": {"id": "11", "displayName": "B"},
                 "statistics": [{"name": "wonCorners", "displayValue": "3"}]},
            ],
            "odds": [None],
        }],
    }
    p = parse_event(ev, "uefa.champions")
    assert p["season"] == "2026-27" and p["phase"] == "league-phase"
    assert p["home"]["score"] == 2 and p["home"]["stats"]["corners"] == 6
    assert p["odds"] is None
    assert p["kickoff"] == datetime(2026, 11, 3, 20, 0)


def test_opportunities_no_bet_without_odds():
    from app.services.opportunities import opportunities
    M = score_matrix(1.5, 1.0, -0.05)
    g = goal_markets(M)
    bands = {k: [g[k] - 0.05, g[k] + 0.05] for k in ("home_win", "draw", "away_win", "over_1_5", "over_2_5", "over_3_5", "btts")}
    pred = {
        "goals": {**g, "bands_p10_p90": bands, "home_xg_sd_log": 0.2, "away_xg_sd_log": 0.2},
        "corners": {"total_over": {"9.5": 0.5}, "home_over": {"4.5": 0.5}, "away_over": {"4.5": 0.4},
                    "home_more": 0.5, "away_more": 0.4},
        "shots_on_target": {"total_over": {"8.5": 0.5}, "home_over": {"3.5": 0.6}, "away_over": {"3.5": 0.4}},
    }
    out = opportunities(pred, ("A", "B"), None, {"level": "Media"}, None)
    assert out["recommendation"]["decision"] == "NO BET"
    assert all(m["verdict"].startswith("Sin cuota") for m in out["markets"])
    # Con cuotas muy generosas sí aparece valor si la confianza lo permite
    odds = {"home": 10.0, "draw": 4.0, "away": 1.2, "source": "test", "provider": "x"}
    out2 = opportunities(pred, ("A", "B"), odds, {"level": "Media"},
                         {"test": {"result_1x2": {"model": {"log_loss": 0.9}, "baseline": {"log_loss": 1.0}}}})
    assert out2["recommendation"]["decision"] == "BET"
    assert any(p["market"] == "Gana A" for p in out2["recommendation"]["picks"])


def test_transfermarkt_parser_and_status():
    from datetime import date
    from app.providers.transfermarkt import merge_for_match, parse_absences, status_for_match

    def row(name, pid, pos, reason, since, ret, missed, comp=None):
        img = f'<img src="x" title="{comp}" alt="{comp}" class="" /> ' if comp else ""
        return (f'<tr class="odd"><td> <table class="inline-table"> <tr> <td rowspan="2"></td> <td class="hauptlink"> '
                f'<a title="{name}" href="/x/profil/spieler/{pid}">{name}</a> </td> </tr> <tr> <td>{pos}</td> </tr> </table> '
                f'</td><td class="zentriert">25</td><td class="links hauptlink img-vat">{img}{reason}</td>'
                f'<td class="zentriert">{since}</td><td class="zentriert">{ret}</td><td class="zentriert"><a href="#">{missed}</a></td></tr>')

    page = ('<table class="items"><tbody><tr><td class="extrarow bg_blau_20 hauptlink" colspan="6">Injuries</td></tr>'
            + row("Ana", 1, "Centre-Back", "Knee injury", "01/10/2026", "20/11/2026", 5)
            + row("Bea", 2, "Left Winger", "Muscle injury", "30/10/2026", "-", 1)
            + row("Cris", 3, "Goalkeeper", "Back injury", "01/01/2026", "01/11/2026", 20)
            + '<tr><td class="extrarow bg_blau_20 hauptlink" colspan="6">Suspensions</td></tr>'
            + row("Ana", 1, "Centre-Back", "No eligibility", "03/09/2026", "28/01/2027", 8, "UEFA Champions League")
            + row("Dani", 4, "Centre-Forward", "Red card suspension", "01/11/2026", "10/11/2026", 1, "Premier League")
            + '</tbody></table><div class="keys"></div>')
    ab = parse_absences(page)
    assert [a.player for a in ab] == ["Ana", "Bea", "Cris", "Ana", "Dani"]
    assert ab[0].expected_return == date(2026, 11, 20) and ab[0].position == "Centre-Back"
    assert ab[3].competition == "UEFA Champions League"
    d = date(2026, 11, 4)
    assert status_for_match(ab[0], d) == "confirmed"
    assert status_for_match(ab[1], d) == "probable"   # lesión reciente sin fecha
    assert status_for_match(ab[2], d) == "probable"   # regreso justo antes del partido
    assert status_for_match(ab[4], d) is None         # sanción de liga: no aplica en UCL
    merged = {m["player"]: m for m in merge_for_match(ab, d)}
    assert set(merged) == {"Ana", "Bea", "Cris"} and "No eligibility" in merged["Ana"]["reason"]
