import pytest
import pandas as pd
from pathlib import Path
from csvx.core.loader import load_dataset, save_dataset, get_file_metadata, detect_separator


def test_load_and_save_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_file = tmp_path / "test.csv"
    save_dataset(df, csv_file)

    loaded = load_dataset(csv_file)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["a", "b"]
    assert loaded["a"].tolist() == [1, 2, 3]


def test_load_and_save_parquet(tmp_path):
    df = pd.DataFrame({"num": [10.5, 20.2], "label": ["A", "B"]})
    pq_file = tmp_path / "test.parquet"
    save_dataset(df, pq_file)

    loaded = load_dataset(pq_file)
    assert len(loaded) == 2
    assert "label" in loaded.columns


def test_load_and_save_json(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "val": ["foo", "bar"]})
    json_file = tmp_path / "test.json"
    save_dataset(df, json_file)

    loaded = load_dataset(json_file)
    assert len(loaded) == 2


def test_load_sampling(tmp_path):
    df = pd.DataFrame({"id": list(range(100))})
    csv_file = tmp_path / "big.csv"
    save_dataset(df, csv_file)

    sample_n = load_dataset(csv_file, sample_rows=10)
    assert len(sample_n) == 10

    sample_pct = load_dataset(csv_file, sample_pct=20)
    assert len(sample_pct) == 20


def test_file_metadata(tmp_path):
    df = pd.DataFrame({"x": [1]})
    csv_file = tmp_path / "meta.csv"
    save_dataset(df, csv_file)

    meta = get_file_metadata(csv_file)
    assert meta["file_name"] == "meta.csv"
    assert meta["size_bytes"] > 0
    assert meta["format"] == "csv"
