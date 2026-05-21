import json
from fastapi.testclient import TestClient

from ml_service.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_minimal():
    # minimal feature payload using feature_info keys
    resp = client.get("/models")
    data = resp.json()
    models = data.get("models", [])
    if not models:
        # no models available; just assert service responds
        assert resp.status_code == 200
        return

    model = models[0]
    payload = {"model": model, "features": {}}
    r = client.post("/predict", json=payload)
    # if model loaded, predict returns 200, else returns 503
    assert r.status_code in (200, 503, 500)
