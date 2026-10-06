"""Categorical frequency analysis and cross-tabulation."""

from typing import Dict, Any, List, Optional
import pandas as pd
from rich.table import Table
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def analyze_category_column(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    """Analyze frequencies and distributions for a single categorical column."""
    clean_s = series.dropna()
    total_valid = len(clean_s)
    total_len = len(series)
    unique_cnt = series.nunique(dropna=True)

    if total_valid == 0:
        return {
            "column": series.name,
            "unique_count": 0,
            "total_count": total_len,
            "missing_count": total_len,
            "top_values": [],
        }

    val_counts = clean_s.value_counts()
    top_counts = val_counts.head(top_n)
    other_count = val_counts.iloc[top_n:].sum() if len(val_counts) > top_n else 0

    top_values = []
    for val, count in top_counts.items():
        pct = (count / total_valid) * 100.0
        top_values.append({
            "value": str(val),
            "count": int(count),
            "pct": pct,
        })

    if other_count > 0:
        other_pct = (other_count / total_valid) * 100.0
        top_values.append({
            "value": "Other",
            "count": int(other_count),
            "pct": other_pct,
        })

    return {
        "column": series.name,
        "unique_count": unique_cnt,
        "total_count": total_len,
        "missing_count": total_len - total_valid,
        "top_values": top_values,
    }


def analyze_categories(df: pd.DataFrame, column: Optional[str] = None, top_n: int = 10) -> List[Dict[str, Any]]:
    """Analyze all categorical columns or a specific one."""
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        return [analyze_category_column(df[column], top_n=top_n)]

    cat_cols = df.select_dtypes(include=["object", "category", "bool", "string", "str"]).columns.tolist()
    if not cat_cols:
        # Fallback to columns with <= 30 unique values
        cat_cols = [col for col in df.columns if df[col].nunique() <= 30]

    return [analyze_category_column(df[col], top_n=top_n) for col in cat_cols]


def compute_crosstab(df: pd.DataFrame, row_col: str, col_col: str, normalize: Optional[str] = None) -> pd.DataFrame:
    """Compute cross tabulation matrix between two columns."""
    if row_col not in df.columns:
        raise KeyError(f"Row column '{row_col}' not found.")
    if col_col not in df.columns:
        raise KeyError(f"Column '{col_col}' not found.")

    return pd.crosstab(df[row_col], df[col_col], normalize=normalize, margins=True)


def display_categories_report(results: List[Dict[str, Any]]):
    """Render category frequencies with Rich."""
    if not results:
        console.print("[yellow]No categorical columns found to analyze.[/yellow]")
        return

    for res in results:
        print_header("CATEGORY ANALYSIS", f"Column: {res['column']}")
        console.print(f"Unique values: [bold cyan]{format_number(res['unique_count'])}[/bold cyan]")
        console.print(f"Missing values: [bold red]{format_number(res['missing_count'])}[/bold red]\n")

        table = create_table(["Value", "Count", "Percentage", "Bar"])
        for item in res["top_values"]:
            pct = item["pct"]
            bar_len = int(pct / 4) # 25 blocks for 100%
            bar = "█" * bar_len + "░" * (25 - bar_len)
            table.add_row(
                item["value"],
                format_number(item["count"]),
                f"{pct:.1f}%",
                Text(bar, style="cyan"),
            )
        console.print(table)
        console.print()


def display_crosstab_report(ct: pd.DataFrame, row_col: str, col_col: str):
    """Render cross-tabulation table with Rich."""
    print_header("CROSS-TABULATION", f"{row_col} (rows) × {col_col} (columns)")

    headers = [row_col] + [str(c) for c in ct.columns]
    table = create_table(headers)

    for idx, row in ct.iterrows():
        row_str = [str(idx)]
        for val in row:
            if isinstance(val, float) and not val.is_integer():
                row_str.append(f"{val:.2%}" if val <= 1.0 else f"{val:,.2f}")
            else:
                row_str.append(format_number(val))
        table.add_row(*row_str)

    console.print(table)
