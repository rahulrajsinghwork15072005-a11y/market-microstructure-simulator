from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_dashboard():
    r = client.get("/")
    assert r.status_code == 200
    assert "Microstructure" in r.text


def test_scenarios():
    assert "balanced" in client.get("/api/scenarios").json()["scenarios"]


def test_demo_run():
    b = client.get("/api/demo/run?scenario=balanced").json()
    assert b["n_trades"] > 0
    assert len(b["mids"]) > 0


def test_demo_stylized_facts():
    assert "excess_kurtosis" in client.get("/api/demo/stylized-facts").json()


def test_demo_impact():
    assert "kyle_lambda" in client.get("/api/demo/impact").json()


def test_demo_book():
    assert "snapshot" in client.get("/api/demo/book").json()


def test_simulate_post():
    r = client.post("/api/simulate", json={"n_events": 500, "seed": 1})
    assert r.status_code == 200
    assert r.json()["n_trades"] >= 0


def test_unknown_scenario_404():
    assert client.get("/api/demo/run?scenario=nope").status_code == 404
