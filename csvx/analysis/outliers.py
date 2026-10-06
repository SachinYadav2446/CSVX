"""Outlier detection using IQR, Z-Score, and Modified Z-Score methods."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def detect_column_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect outliers for a single numeric column."""
    clean_s = series.dropna()
    total_valid = len(clean_s)

    if total_valid < 4 or not pd.api.types.is_numeric_dtype(series):
        return {
            "column": series.name,
            "method": method,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "indices": [],
        }

    values = clean_s.values.astype(float)
    method_lower = method.lower()
    indices = []
    lower_bound = None
    upper_bound = None

    if method_lower == "iqr":
        th = threshold if threshold is not None else 1.5
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        lower_bound = q1 - th * iqr
        upper_bound = q3 + th * iqr
        mask = (values < lower_bound) | (values > upper_bound)
        indices = clean_s.index[mask].tolist()

    elif method_lower in ["zscore", "z-score"]:
        th = threshold if threshold is not None else 3.0
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            mask = z_scores > th
            lower_bound = mean - th * std
            upper_bound = mean + th * std
            indices = clean_s.index[mask].tolist()

    elif method_lower in ["modified-zscore", "modified_zscore", "mad"]:
        th = threshold if threshold is not None else 3.5
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 0:
            mod_z = 0.6745 * np.abs(values - median) / mad
            mask = mod_z > th
            lower_bound = median - (th * mad / 0.6745)
            upper_bound = median + (th * mad / 0.6745)
            indices = clean_s.index[mask].tolist()

    else:
        raise ValueError(f"Unknown outlier method '{method}'. Choose from 'iqr', 'zscore', 'modified-zscore'.")

    outlier_count = len(indices)
    outlier_pct = (outlier_count / max(1, total_valid)) * 100.0

    return {
        "column": series.name,
        "method": method,
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "indices": indices,
    }


def detect_dataframe_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: Optional[float] = None,
    column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Detect outliers across all or specific numerical columns in a DataFrame."""
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        return [detect_column_outliers(df[column], method=method, threshold=threshold)]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        detect_column_outliers(df[col], method=method, threshold=threshold)
        for col in numeric_cols
    ]


def display_outliers_report(results: List[Dict[str, Any]]):
    """Render outlier analysis report with Rich."""
    print_header("OUTLIER REPORT")

    if not results:
        console.print("[yellow]No numerical columns found to analyze for outliers.[/yellow]")
        return

    table = create_table(["Column", "Method", "Outlier Count", "Outlier %", "Lower Bound", "Upper Bound"])
    total_outliers = 0

    for res in results:
        total_outliers += res["outlier_count"]
        cnt_style = "bold red" if res["outlier_count"] > 0 else "green"
        lb_str = format_number(res["lower_bound"]) if res["lower_bound"] is not None else "N/A"
        ub_str = format_number(res["upper_bound"]) if res["upper_bound"] is not None else "N/A"

        table.add_row(
            res["column"],
            res["method"].upper(),
            Text(format_number(res["outlier_count"]), style=cnt_style),
            Text(f"{res['outlier_pct']:.2f}%", style=cnt_style),
            lb_str,
            ub_str,
        )

    console.print(table)
    console.print(f"\n[bold]Total detected outlier instances:[/bold] {format_number(total_outliers)}")
