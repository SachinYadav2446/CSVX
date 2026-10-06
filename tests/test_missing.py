import pandas as pd
from csvx.quality.missing import analyze_missing_values
from csvx.quality.duplicates import analyze_duplicates
from csvx.analysis.outliers import detect_dataframe_outliers, detect_column_outliers
from csvx.analysis.correlation import compute_correlation_matrix, find_strong_correlations


def test_analyze_missing():
    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": ["x", "y", "z", "w"],
    })
    res = analyze_missing_values(df)
    assert res["total_missing"] == 2
    assert len(res["columns_missing"]) == 1
    assert res["columns_missing"][0]["column"] == "a"
    assert res["columns_missing"][0]["missing_count"] == 2


def test_analyze_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": ["a", "b", "b", "c"],
    })
    res = analyze_duplicates(df)
    assert res["total_rows"] == 4
    assert res["duplicate_rows"] == 1
    assert res["unique_rows"] == 3


def test_detect_outliers():
    # Normal data + 1 extreme outlier
    data = [10, 12, 11, 13, 12, 11, 10, 12, 13, 1000]
    s = pd.Series(data, name="val")
    res = detect_column_outliers(s, method="iqr")
    assert res["outlier_count"] == 1
    assert res["indices"] == [9]

    res_z = detect_column_outliers(s, method="zscore", threshold=2.0)
    assert res_z["outlier_count"] >= 1


def test_correlation():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10], # perfectly correlated
        "z": [5, 4, 3, 2, 1], # negatively correlated
    })
    corr = compute_correlation_matrix(df)
    assert abs(corr.loc["x", "y"] - 1.0) < 1e-5
    assert abs(corr.loc["x", "z"] - (-1.0)) < 1e-5

    strong = find_strong_correlations(corr, threshold=0.9)
    assert len(strong) == 3
