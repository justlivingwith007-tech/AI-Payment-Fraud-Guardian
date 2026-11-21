"""Train a baseline and Random Forest, select threshold on validation only."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocess import FEATURES, load_dataset, split_dataset

ROOT = Path(__file__).resolve().parents[1]


def select_threshold(y_true, probabilities):
    """Maximise validation F1, never optimise threshold on held-out test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def train(data_path=None, output=None):
    df = load_dataset(data_path) if data_path else load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    candidates = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=2,
                                                  class_weight="balanced_subsample", n_jobs=2, random_state=42)),
        ]),
    }
    results = {}
    for name, pipeline in candidates.items():
        print(f"Training {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_val)[:, 1]
        results[name] = {"validation_pr_auc": float(average_precision_score(y_val, probs)),
                         "threshold": select_threshold(y_val, probs)}
    best = max(results, key=lambda name: results[name]["validation_pr_auc"])
    # Retrain chosen model on TRAIN only: validation remains independent of fitting.
    chosen = candidates[best]
    artifact = {"pipeline": chosen, "features": FEATURES, "threshold": results[best]["threshold"],
                "model_name": best, "validation_results": results}
    target = Path(output) if output else ROOT / "models" / "fraud_model.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target, compress=3)
    from src.evaluate import evaluate_artifact
    metrics = evaluate_artifact(artifact, X_test, y_test)
    report = {"selected_model": best, "validation": results, "test": metrics,
              "split": "64% train / 16% validation / 20% held-out test; stratified random split",
              "caveat": "Random split is a learning baseline, not a temporal out-of-time evaluation."}
    (target.parent / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    train(args.data, args.output)
