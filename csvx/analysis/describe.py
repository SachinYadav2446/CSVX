"""Descriptive statistics computation for numeric and categorical columns."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def describe_column(series: pd.Series) -> Dict[str, Any]:
    """Calculate thorough descriptive statistics for a single series."""
    clean_s = series.dropna()
    total_count = len(series)
    valid_count = len(clean_s)

    if valid_count == 0 or not pd.api.types.is_numeric_dtype(series):
        # Fallback for non-numeric or empty
        mode_val = clean_s.mode().iloc[0] if len(clean_s) > 0 else None
        return {
            "name": series.name,
            "type": "non-numeric",
            "count": valid_count,
            "missing": total_count - valid_count,
            "unique": series.nunique(dropna=True),
            "mode": mode_val,
        }

    # Numeric statistics
    values = clean_s.values.astype(float)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if valid_count > 1 else 0.0
    var_val = float(np.var(values, ddof=1)) if valid_count > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    median_val = float(np.median(values))

    q1_val = float(np.percentile(values, 25))
    q3_val = float(np.percentile(values, 75))
    iqr_val = float(q3_val - q1_val)

    skew_val = float(stats.skew(values, bias=False)) if valid_count > 2 else 0.0
    kurt_val = float(stats.kurtosis(values, bias=False)) if valid_count > 3 else 0.0

    mode_res = clean_s.mode()
    mode_val = float(mode_res.iloc[0]) if not mode_res.empty else None

    return {
        "name": series.name,
        "type": "numeric",
        "count": valid_count,
        "missing": total_count - valid_count,
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "std": std_val,
        "var": var_val,
        "min": min_val,
        "max": max_val,
        "q1": q1_val,
        "q3": q3_val,
        "iqr": iqr_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }


def describe_dataframe(df: pd.DataFrame, column: Optional[str] = None) -> List[Dict[str, Any]]:
    """Describe one or all columns in a DataFrame."""
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        return [describe_column(df[column])]

    # Describe all numeric columns first, then others
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # Fallback to all columns if no numeric cols
        return [describe_column(df[c]) for c in df.columns]

    return [describe_column(df[col]) for col in numeric_cols]


def display_describe_report(stats_list: List[Dict[str, Any]]):
    """Render the descriptive statistics report with Rich."""
    if not stats_list:
        console.print("[yellow]No columns to describe.[/yellow]")
        return

    if len(stats_list) == 1:
        stat = stats_list[0]
        print_header("NUMERICAL SUMMARY", f"Column: {stat['name']}")
        
        if stat["type"] == "non-numeric":
            table = create_table(["Metric", "Value"])
            table.add_row("Count", format_number(stat["count"]))
            table.add_row("Missing", format_number(stat["missing"]))
            table.add_row("Unique", format_number(stat["unique"]))
            table.add_row("Mode", str(stat["mode"]))
            console.print(table)
            return

        table = create_table(["Metric", "Value"])
        table.add_row("Count", format_number(stat["count"]))
        table.add_row("Mean", format_number(stat["mean"]))
        table.add_row("Median", format_number(stat["median"]))
        table.add_row("Mode", format_number(stat["mode"]))
        table.add_row("Std Dev", format_number(stat["std"]))
        table.add_row("Variance", format_number(stat["var"]))
        table.add_row("Minimum", format_number(stat["min"]))
        table.add_row("Maximum", format_number(stat["max"]))
        table.add_row("Q1 (25%)", format_number(stat["q1"]))
        table.add_row("Q3 (75%)", format_number(stat["q3"]))
        table.add_row("IQR", format_number(stat["iqr"]))
        table.add_row("Skewness", f"{stat['skewness']:.2f}")
        table.add_row("Kurtosis", f"{stat['kurtosis']:.2f}")
        console.print(table)
    else:
        print_header("NUMERICAL DESCRIPTIVE STATISTICS")
        table = create_table([
            "Column", "Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max", "IQR", "Skew"
        ])
        for stat in stats_list:
            if stat["type"] == "numeric":
                table.add_row(
                    stat["name"],
                    format_number(stat["count"]),
                    format_number(stat["mean"]),
                    format_number(stat["std"]),
                    format_number(stat["min"]),
                    format_number(stat["q1"]),
                    format_number(stat["median"]),
                    format_number(stat["q3"]),
                    format_number(stat["max"]),
                    format_number(stat["iqr"]),
                    f"{stat['skewness']:.2f}",
                )
        console.print(table)
