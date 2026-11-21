"""Generate portfolio-ready evaluation charts from saved model metrics.

Usage (from project root): python -m src.visualize
No model retraining and no test-set threshold tuning.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNG files without opening a chart window.
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "metrics.json"
OUTPUT_DIR = ROOT / "reports" / "figures"


def annotate_bars(ax, bars, suffix="%"):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(f"{value:.2f}{suffix}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)


def make_charts():
    with METRICS_PATH.open(encoding="utf-8") as file:
        report = json.load(file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test = report["test"]

    # Figure 1: confusion matrix in the scikit-learn order TN, FP, FN, TP.
    tn, fp, fn, tp = test["confusion_matrix_tn_fp_fn_tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="Transactions")
    ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Fraud detection — test confusion matrix", pad=14)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{matrix[r, c]:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if matrix[r, c] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: held-out test-set classification metrics.
    labels = ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
    keys = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    values = [100 * test[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Fraud detection — held-out test performance", pad=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    annotate_bars(ax, bars)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: validation-only model comparison (not test-set scores).
    validation = report["validation"]
    models = ["Logistic Regression", "Random Forest"]
    keys = ["logistic_regression", "random_forest"]
    values = [100 * validation[k]["validation_pr_auc"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(models, values)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Validation PR-AUC (%)")
    ax.set_title("Model selection — validation PR-AUC", pad=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    annotate_bars(ax, bars)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved 3 charts to:", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    make_charts()
