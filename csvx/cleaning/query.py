"""Dataset filtering and query evaluation engine."""

from typing import Optional, Tuple
import pandas as pd
from rich.table import Table

from csvx.console import console, create_table, format_number, print_header, print_success


def evaluate_query(df: pd.DataFrame, query_str: str) -> pd.DataFrame:
    """Filter DataFrame using Pandas expression engine."""
    # Replace single '=' with '==' if accidentally written by user
    clean_expr = query_str.strip()
    # Normalize AND/OR to and/or
    clean_expr = clean_expr.replace(" AND ", " and ").replace(" OR ", " or ")

    try:
        filtered_df = df.query(clean_expr).reset_index(drop=True)
        return filtered_df
    except Exception as e:
        raise ValueError(f"Failed to execute query '{query_str}': {e}")


def display_query_results(df: pd.DataFrame, query_str: str, max_rows: int = 15):
    """Render query results preview in the console."""
    print_header("QUERY RESULTS", f"Condition: {query_str}")
    console.print(f"Matched: [bold green]{format_number(len(df))}[/bold green] rows\n")

    if df.empty:
        console.print("[yellow]Query matched 0 rows.[/yellow]")
        return

    preview_df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    table = create_table(cols)

    for _, row in preview_df.iterrows():
        row_vals = [format_number(val) for val in row]
        table.add_row(*row_vals)

    console.print(table)
    if len(df) > max_rows:
        console.print(f"\n[dim]Showing first {max_rows} of {len(df):,} rows.[/dim]")
