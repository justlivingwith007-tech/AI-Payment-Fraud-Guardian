import pandas as pd
from src.preprocess import FEATURES, load_dataset, split_dataset


def test_splits_are_disjoint():
    rows = []
    for i in range(100):
        record = {name: float(i) for name in FEATURES}
        record["Class"] = i % 2
        rows.append(record)
    df = pd.DataFrame(rows)
    a, b, c, ya, yb, yc = split_dataset(df)
    assert len(a) + len(b) + len(c) == 100
    assert set(a.index).isdisjoint(b.index)
    assert set(a.index).isdisjoint(c.index)
    assert set(b.index).isdisjoint(c.index)
    assert len(ya) + len(yb) + len(yc) == 100


def test_invalid_class_rejected(tmp_path):
    row = {name: 1 for name in FEATURES}
    df = pd.DataFrame([{**row, "Class": 2}])
    file = tmp_path / "invalid.csv"
    df.to_csv(file, index=False)
    import pytest
    with pytest.raises(ValueError):
        load_dataset(file)
