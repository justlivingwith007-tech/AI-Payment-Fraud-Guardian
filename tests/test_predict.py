import pytest
from src.preprocess import FEATURES
from src.predict import predict_transaction


class FakeModel:
    def predict_proba(self, X):
        return [[0.1, 0.9]]


def test_prediction_flags_high_risk():
    artifact = {"pipeline": FakeModel(), "threshold": 0.5, "model_name": "fake"}
    result = predict_transaction({f: 1.0 for f in FEATURES}, artifact)
    assert result["is_flagged"] is True
    assert result["fraud_probability"] == 0.9


def test_missing_feature_rejected():
    with pytest.raises(ValueError):
        predict_transaction({"Amount": 10}, {"pipeline": FakeModel(), "threshold": 0.5})
