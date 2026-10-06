"""Dataset version comparison and schema diff engine."""

from typing import Dict, Any, List, Set, Tuple
import pandas as pd
from rich.table import Table
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def compare_datasets(df_old: pd.DataFrame, df_new: pd.DataFrame, name_old: str = "OLD", name_new: str = "NEW") -> Dict[str, Any]:
    """Compare two DataFrames across dimensions, missingness, duplicates, and schema."""
    old_rows, old_cols = df_old.shape
    new_rows, new_cols = df_new.shape

    old_missing_pct = (df_old.isna().sum().sum() / max(1, df_old.size)) * 100.0
    new_missing_pct = (df_new.isna().sum().sum() / max(1, df_new.size)) * 100.0

    old_dup_pct = (df_old.duplicated().sum() / max(1, old_rows)) * 100.0
    new_dup_pct = (df_new.duplicated().sum() / max(1, new_rows)) * 100.0

    old_cols_set = set(df_old.columns)
    new_cols_set = set(df_new.columns)

    added_cols = sorted(list(new_cols_set - old_cols_set))
    removed_cols = sorted(list(old_cols_set - new_cols_set))
    common_cols = sorted(list(old_cols_set & new_cols_set))

    type_changes = []
    for col in common_cols:
        old_t = str(df_old[col].dtype)
        new_t = str(df_new[col].dtype)
        if old_t != new_t:
            type_changes.append({
                "column": col,
                "old_type": old_t,
                "new_type": new_t,
            })

    return {
        "name_old": name_old,
        "name_new": name_new,
        "old_rows": old_rows,
        "new_rows": new_rows,
        "old_cols": old_cols,
        "new_cols": new_cols,
        "old_missing_pct": old_missing_pct,
        "new_missing_pct": new_missing_pct,
        "old_dup_pct": old_dup_pct,
        "new_dup_pct": new_dup_pct,
        "added_columns": added_cols,
        "removed_columns": removed_cols,
        "type_changes": type_changes,
    }


def display_comparison_report(comp: Dict[str, Any]):
    """Render comparison report in the console."""
    print_header("DATASET COMPARISON", f"{comp['name_old']} vs {comp['name_new']}")

    table = create_table(["Metric", comp["name_old"], comp["name_new"], "Difference"])

    # Row diff
    row_diff = comp["new_rows"] - comp["old_rows"]
    row_diff_str = f"{row_diff:+d}" if row_diff != 0 else "0"
    table.add_row("Rows", format_number(comp["old_rows"]), format_number(comp["new_rows"]), row_diff_str)

    # Col diff
    col_diff = comp["new_cols"] - comp["old_cols"]
    col_diff_str = f"{col_diff:+d}" if col_diff != 0 else "0"
    table.add_row("Columns", format_number(comp["old_cols"]), format_number(comp["new_cols"]), col_diff_str)

    # Missing diff
    miss_diff = comp["new_missing_pct"] - comp["old_missing_pct"]
    table.add_row("Missing %", f"{comp['old_missing_pct']:.2f}%", f"{comp['new_missing_pct']:.2f}%", f"{miss_diff:+.2f}%")

    # Duplicates diff
    dup_diff = comp["new_dup_pct"] - comp["old_dup_pct"]
    table.add_row("Duplicates %", f"{comp['old_dup_pct']:.2f}%", f"{comp['new_dup_pct']:.2f}%", f"{dup_diff:+.2f}%")

    console.print(table)

    # Added columns
    if comp["added_columns"]:
        console.print()
        print_header("NEW COLUMNS")
        for col in comp["added_columns"]:
            console.print(f"[bold green]+ {col}[/bold green]")

    # Removed columns
    if comp["removed_columns"]:
        console.print()
        print_header("REMOVED COLUMNS")
        for col in comp["removed_columns"]:
            console.print(f"[bold red]- {col}[/bold red]")

    # Data type changes
    if comp["type_changes"]:
        console.print()
        print_header("SCHEMA TYPE CHANGES")
        for tc in comp["type_changes"]:
            console.print(f"[yellow]~ {tc['column']}: {tc['old_type']} → {tc['new_type']}[/yellow]")
