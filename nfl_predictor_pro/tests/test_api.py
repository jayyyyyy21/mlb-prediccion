"""Pruebas de humo de la API. Requieren datos descargados (python -m app.cli init)."""
import os

import pytest

from app.config import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="sin base de datos local")
os.environ["NFLPP_AUTO_UPDATE_HOURS"] = "0"
os.environ["NFLPP_WEATHER"] = "false"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200 and r.json()["model_ready"]


def test_week_predictions(client):
    d = client.get("/api/predictions").json()
    assert d["games"]
    for g in d["games"]:
        p = g["projection"]
        assert abs(p["p_home"] + p["p_away"] + p["p_tie"] - 1) < 1e-3
        assert 3 <= p["home_points"] <= 60 and 3 <= p["away_points"] <= 60
        assert g["confidence"] in ("Alto", "Moderado", "Bajo")


def test_simulator_qb_absence_lowers_points(client):
    base = client.post("/api/simulate", json={"home": "KC", "away": "BUF", "neutral": True}).json()
    qb = next(p for p in base["key_players"]["home"] if p["position"] == "QB")
    out = client.post("/api/simulate", json={"home": "KC", "away": "BUF", "neutral": True,
                                             "out_home": [qb["gsis_id"]]}).json()
    assert out["projection"]["home_points"] <= base["projection"]["home_points"]


def test_export_csv(client):
    r = client.get("/api/export?fmt=csv")
    assert r.status_code == 200 and "prob_local" in r.text
