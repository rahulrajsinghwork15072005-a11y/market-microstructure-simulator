from fastapi.testclient import TestClient

from app.main import app
from app.observability import RateLimiter

client = TestClient(app)


def test_version_endpoint():
    body = client.get("/api/version").json()
    assert body["name"] == "market-microstructure-simulator"
    assert "version" in body


def test_ready_endpoint():
    assert client.get("/api/ready").json()["ready"] is True


def test_metrics_endpoint():
    client.get("/api/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_security_headers_and_request_id():
    r = client.get("/api/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert "x-request-id" in r.headers


def test_rate_limiter():
    limiter = RateLimiter(limit=2, window=60.0)
    assert limiter.allow("c") and limiter.allow("c")
    assert limiter.allow("c") is False
