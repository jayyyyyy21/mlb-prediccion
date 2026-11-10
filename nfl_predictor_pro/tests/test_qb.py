import pandas as pd

from app.qb import QBModel


def _games():
    rows = []
    # Equipo AAA: el titular (bueno) juega las semanas 1-6; el suplente (malo) las semanas 7-8.
    for w in range(1, 9):
        qb, epa = ("starter", 0.20) if w <= 6 else ("backup", -0.20)
        rows.append({"game_id": f"g{w}", "season": 2025, "week": w, "team": "AAA", "gsis_id": qb,
                     "passer": qb, "dropbacks": 40, "qb_epa_sum": 40 * epa})
    return pd.DataFrame(rows)


def test_no_change_when_same_qb_played_everything():
    snap = QBModel(_games()).at(2025, 7)  # sólo semanas 1-6: todo lo jugó el titular
    f = snap.features("AAA", "starter")
    assert abs(f["qb_delta"]) < 1e-9
    assert f["qb_level"] <= 0.20 + 1e-9  # nunca por encima de lo observado


def test_returning_starter_raises_and_backup_lowers_projection():
    snap = QBModel(_games()).at(2025, 9)  # los ratings mezclan titular y suplente
    assert snap.features("AAA", "starter")["qb_delta"] > 0
    assert snap.features("AAA", "backup")["qb_delta"] < 0
    assert snap.features("AAA", None)["qb_delta"] == 0  # QB desconocido: sin ajuste


def test_uses_only_past_games():
    g = _games()
    before = QBModel(g).at(2025, 5).features("AAA", "starter")
    g2 = g.copy()
    g2.loc[g2["week"] >= 5, "qb_epa_sum"] = -999.0  # alterar el futuro no debe cambiar nada
    assert QBModel(g2).at(2025, 5).features("AAA", "starter") == before


def test_unknown_qb_gets_replacement_level():
    snap = QBModel(_games()).at(2025, 9)
    assert snap.level("novato") == snap.replacement
