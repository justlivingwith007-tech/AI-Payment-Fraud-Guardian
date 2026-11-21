"""Evaluation on a held-out test split; no threshold tuning here."""
import argparse
import json
import joblib
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from src.preprocess import load_dataset, split_dataset


def evaluate_artifact(artifact, X_test, y_test):
    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]
    predictions = (probabilities >= artifact["threshold"]).astype(int)
    return {"pr_auc": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, predictions, labels=[0, 1]).ravel().tolist(),
            "threshold": float(artifact["threshold"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    model = joblib.load(args.model)  # Load only trusted local artifacts.
    df = load_dataset(args.data) if args.data else load_dataset()
    _, _, X_test, _, _, y_test = split_dataset(df)
    print(json.dumps(evaluate_artifact(model, X_test, y_test), indent=2))
