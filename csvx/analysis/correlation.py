"""Correlation matrix computation and strong relationship detection."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from rich.text import Text

from csvx.console import console, create_table, print_header


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix for all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method=method)


def find_strong_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Find pairs of columns with absolute correlation >= threshold."""
    if corr_matrix.empty:
        return []

    strong_pairs = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            val = corr_matrix.loc[col1, col2]
            if not np.isnan(val) and abs(val) >= threshold:
                strong_pairs.append({
                    "col1": col1,
                    "col2": col2,
                    "correlation": float(val),
                    "abs_correlation": float(abs(val)),
                })

    strong_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return strong_pairs


def display_correlation_report(
    corr_matrix: pd.DataFrame,
    strong_only: bool = False,
    threshold: float = 0.7,
    method: str = "pearson",
):
    """Render correlation matrix or strong correlations with Rich."""
    if corr_matrix.empty:
        console.print("[yellow]Not enough numeric columns (minimum 2 required) to compute correlation.[/yellow]")
        return

    if strong_only:
        print_header("STRONG CORRELATIONS", f"Method: {method.capitalize()} (|r| >= {threshold})")
        strong_pairs = find_strong_correlations(corr_matrix, threshold=threshold)
        if not strong_pairs:
            console.print(f"[green]No variable pairs with correlation >= {threshold}[/green]")
            return

        table = create_table(["Relationship", "Correlation", "Strength"])
        for pair in strong_pairs:
            r = pair["correlation"]
            r_str = f"{r:+.2f}"
            if abs(r) >= 0.9:
                strength = "[bold magenta]Very Strong[/bold magenta]"
            elif abs(r) >= 0.8:
                strength = "[bold red]Strong[/bold red]"
            else:
                strength = "[bold yellow]Moderate/Strong[/bold yellow]"

            table.add_row(
                f"{pair['col1']} ↔ {pair['col2']}",
                Text(r_str, style="bold cyan"),
                strength,
            )
        console.print(table)
    else:
        print_header("CORRELATION MATRIX", f"Method: {method.capitalize()}")
        cols = corr_matrix.columns.tolist()
        headers = [""] + cols
        table = create_table(headers)

        for row_col in cols:
            row_vals = [f"[bold]{row_col}[/bold]"]
            for col_col in cols:
                val = corr_matrix.loc[row_col, col_col]
                if np.isnan(val):
                    row_vals.append("N/A")
                elif row_col == col_col:
                    row_vals.append(Text(f"{val:.2f}", style="dim"))
                elif abs(val) >= 0.7:
                    row_vals.append(Text(f"{val:+.2f}", style="bold red"))
                elif abs(val) >= 0.4:
                    row_vals.append(Text(f"{val:+.2f}", style="bold yellow"))
                else:
                    row_vals.append(f"{val:+.2f}")
            table.add_row(*row_vals)

        console.print(table)
