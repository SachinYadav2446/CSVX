import pandas as pd
import numpy as np
from csvx.analysis.describe import describe_column, describe_dataframe


def test_describe_column():
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    s = pd.Series(data, name="scores")
    stats = describe_column(s)

    assert stats["count"] == 10
    assert stats["mean"] == 55.0
    assert stats["median"] == 55.0
    assert stats["min"] == 10.0
    assert stats["max"] == 100.0
    assert stats["q1"] == 32.5
    assert stats["q3"] == 77.5
    assert stats["iqr"] == 45.0


def test_describe_dataframe():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["a", "b", "c", "d", "e"],
    })
    res = describe_dataframe(df)
    assert len(res) >= 1
    assert res[0]["name"] == "num"
