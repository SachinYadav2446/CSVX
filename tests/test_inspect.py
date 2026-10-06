import pandas as pd
from csvx.analysis.inspect import inspect_dataset
from csvx.core.types import infer_column_type, infer_dataframe_types


def test_inspect_dataset():
    df = pd.DataFrame({
        "id": [1, 2, 3, 3],
        "name": ["Alice", "Bob", "Charlie", "Charlie"],
        "age": [25, None, 30, 30],
    })
    res = inspect_dataset(df, "test.csv", file_size_bytes=100)

    assert res["rows"] == 4
    assert res["columns"] == 3
    assert res["duplicates"] == 1
    assert len(res["column_summary"]) == 3

    age_summary = next(c for c in res["column_summary"] if c["name"] == "age")
    assert age_summary["missing_count"] == 1
    assert age_summary["unique_count"] == 2


def test_infer_column_type():
    assert infer_column_type(pd.Series([1, 2, 3, 4], name="val")) == "Integer"
    assert infer_column_type(pd.Series([1.2, 3.4, 5.6], name="salary")) == "Float"
    assert infer_column_type(pd.Series(["test@example.com", "user@site.org"], name="email")) == "Email"
    assert infer_column_type(pd.Series([True, False, True], name="flag")) == "Boolean"
    assert infer_column_type(pd.Series(["A", "B", "A", "B", "A"], name="category")) == "Categorical"
