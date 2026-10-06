"""Dataset inspection and structural summary analysis."""

from typing import Dict, Any, List
import pandas as pd
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from csvx.console import console, create_table, format_number, format_bytes, print_header
from csvx.core.types import infer_column_type


def inspect_dataset(df: pd.DataFrame, file_name: str, file_size_bytes: int = 0) -> Dict[str, Any]:
    """Calculate inspection statistics for a DataFrame."""
    n_rows = len(df)
    n_cols = len(df.columns)
    memory_usage = int(df.memory_usage(deep=True).sum())
    n_duplicates = int(df.duplicated().sum())

    columns_info = []
    for col in df.columns:
        s = df[col]
        missing_cnt = int(s.isna().sum())
        missing_pct = (missing_cnt / n_rows * 100.0) if n_rows > 0 else 0.0
        unique_cnt = int(s.nunique(dropna=True))
        raw_type = str(s.dtype)
        semantic_type = infer_column_type(s)

        columns_info.append({
            "name": col,
            "raw_type": raw_type,
            "semantic_type": semantic_type,
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "unique_count": unique_cnt,
        })

    return {
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "rows": n_rows,
        "columns": n_cols,
        "memory_bytes": memory_usage,
        "duplicates": n_duplicates,
        "column_summary": columns_info,
    }


def display_inspect_report(report: Dict[str, Any]):
    """Render the inspection report with Rich formatting."""
    print_header("CSVX DATASET REPORT")

    info_lines = Text()
    info_lines.append(f"File:       {report['file_name']}\n", style="bold white")
    info_lines.append(f"Rows:       {format_number(report['rows'])}\n")
    info_lines.append(f"Columns:    {format_number(report['columns'])}\n")
    info_lines.append(f"Memory:     {format_bytes(report['memory_bytes'])}\n")
    if report.get("file_size_bytes"):
        info_lines.append(f"File Size:  {format_bytes(report['file_size_bytes'])}\n")
    
    dup_pct = (report['duplicates'] / max(1, report['rows'])) * 100
    dup_style = "bold red" if report['duplicates'] > 0 else "green"
    info_lines.append("Duplicates: ", style="white")
    info_lines.append(f"{format_number(report['duplicates'])} ({dup_pct:.1f}%)\n", style=dup_style)

    console.print(Panel(info_lines, border_style="cyan", title="Overview", title_align="left"))

    console.print()
    print_header("COLUMN SUMMARY")
    
    table = create_table(["Column", "Data Type", "Inferred Type", "Missing", "Missing %", "Unique Values"])
    for col in report["column_summary"]:
        missing_style = "red" if col["missing_count"] > 0 else "green"
        table.add_row(
            col["name"],
            col["raw_type"],
            col["semantic_type"],
            Text(format_number(col["missing_count"]), style=missing_style),
            Text(f"{col['missing_pct']:.1f}%", style=missing_style),
            format_number(col["unique_count"]),
        )

    console.print(table)
