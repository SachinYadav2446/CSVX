"""Duplicate row detection and subset duplicate analysis."""

from typing import Dict, Any, List, Optional
import pandas as pd
from rich.table import Table
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def analyze_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """Detect full or subset duplicate rows."""
    total_rows = len(df)
    if subset:
        for col in subset:
            if col not in df.columns:
                raise KeyError(f"Subset column '{col}' not found in dataset.")
        dup_mask = df.duplicated(subset=subset, keep="first")
    else:
        dup_mask = df.duplicated(keep="first")

    duplicate_rows = int(dup_mask.sum())
    unique_rows = total_rows - duplicate_rows
    duplicate_rate = (duplicate_rows / max(1, total_rows)) * 100.0

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "unique_rows": unique_rows,
        "duplicate_rate": duplicate_rate,
        "subset": subset,
    }


def display_duplicates_report(report: Dict[str, Any]):
    """Render duplicate analysis report with Rich."""
    subtitle = f"Subset: {', '.join(report['subset'])}" if report.get("subset") else None
    print_header("DUPLICATE ANALYSIS", subtitle)

    table = create_table(["Metric", "Count", "Rate"])
    dup_style = "bold red" if report["duplicate_rows"] > 0 else "green"

    table.add_row("Total rows", format_number(report["total_rows"]), "100.0%")
    table.add_row(
        "Duplicate rows",
        Text(format_number(report["duplicate_rows"]), style=dup_style),
        Text(f"{report['duplicate_rate']:.2f}%", style=dup_style),
    )
    table.add_row(
        "Unique rows",
        format_number(report["unique_rows"]),
        f"{100.0 - report['duplicate_rate']:.2f}%",
    )

    console.print(table)
    if report["duplicate_rows"] > 0:
        console.print("\n[dim]Tip: Use `csvx clean <file> --remove-duplicates` to eliminate duplicate records.[/dim]")
