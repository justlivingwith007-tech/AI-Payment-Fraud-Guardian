"""Data loading and leakage-safe preprocessing helpers."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def load_dataset(path=DEFAULT_DATA):
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    df[TARGET] = df[TARGET].astype(int)
    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset needs both legitimate and fraudulent examples")
    return df


def split_dataset(df, test_size=0.20, validation_size=0.20, random_state=42):
    """Return train, validation, test splits; validation_size is fraction of remaining train set."""
    X, y = df[FEATURES], df[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=validation_size, random_state=random_state, stratify=y_dev
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
