"""Inference from a trusted, trained pipeline."""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.preprocess import FEATURES

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"


@lru_cache(maxsize=1)
def load_model(path=None):
    """Only load a model created locally; joblib/pickle is unsafe for untrusted files."""
    model_path = Path(path) if path else DEFAULT_MODEL
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run python -m src.train first.")
    return joblib.load(model_path)


def predict_transaction(features, artifact=None):
    artifact = artifact if artifact is not None else load_model()
    if set(features) != set(FEATURES):
        raise ValueError(f"Expected exactly these features: {FEATURES}")
    values = np.array([features[f] for f in FEATURES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All input values must be finite numbers")
    if features["Amount"] < 0 or features["Time"] < 0:
        raise ValueError("Amount and Time cannot be negative")
    frame = pd.DataFrame([dict(zip(FEATURES, values))], columns=FEATURES)
    probability = float(artifact["pipeline"].predict_proba(frame)[0][1])
    threshold = float(artifact["threshold"])
    return {"fraud_probability": round(probability, 6),
            "is_flagged": probability >= threshold,
            "threshold": round(threshold, 6),
            "risk_level": "high" if probability >= threshold else "low",
            "model_name": artifact["model_name"]}
