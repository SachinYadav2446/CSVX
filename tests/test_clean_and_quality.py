import pandas as pd
from csvx.quality.scorer import compute_quality_score
from csvx.cleaning.engine import clean_dataset
from csvx.cleaning.query import evaluate_query
from csvx.comparison.diff import compare_datasets


def test_quality_scorer():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "w", "v"],
    })
    q = compute_quality_score(df)
    assert q["overall_score"] > 80
    assert "Missing Values" in q["subscores"]


def test_clean_dataset():
    df = pd.DataFrame({
        "First Name ": [" Alice", "Bob ", "Bob ", "Charlie"],
        "Salary": [50000.0, None, None, 100000.0],
    })
    cleaned, actions = clean_dataset(
        df,
        remove_duplicates=True,
        fill_missing="median",
        normalize_columns=True,
        strip_whitespace=True,
    )

    assert "first_name" in cleaned.columns
    assert "salary" in cleaned.columns
    assert cleaned["first_name"].iloc[0] == "Alice"
    assert cleaned["salary"].isna().sum() == 0
    assert len(cleaned) == 3


def test_query_filter():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "salary": [30000, 60000, 90000, 120000],
    })
    res = evaluate_query(df, "age > 25 and salary < 100000")
    assert len(res) == 2
    assert list(res["age"]) == [30, 40]


def test_compare_datasets():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "c": [5, 6, 7]})
    comp = compare_datasets(df1, df2, "v1", "v2")

    assert comp["old_rows"] == 2
    assert comp["new_rows"] == 3
    assert comp["added_columns"] == ["c"]
    assert comp["removed_columns"] == ["b"]
