from fastapi.testclient import TestClient
from api.main import app
from src.preprocess import FEATURES

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_features_return_422():
    assert client.post("/predict", json={"Amount": 5}).status_code == 422


def test_negative_amount_returns_422():
    payload = {f: 0.0 for f in FEATURES}
    payload["Amount"] = -10
    assert client.post("/predict", json=payload).status_code == 422


def test_valid_request(monkeypatch):
    monkeypatch.setattr("api.main.predict_transaction", lambda features: {
        "fraud_probability": 0.7, "is_flagged": True, "threshold": 0.5,
        "risk_level": "high", "model_name": "test"})
    response = client.post("/predict", json={f: 0.0 for f in FEATURES})
    assert response.status_code == 200
    assert response.json()["is_flagged"] is True
