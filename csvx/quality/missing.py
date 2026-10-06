"""Missing value analysis and imputation recommendations."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from rich.table import Table
from rich.text import Text

from csvx.console import console, create_table, format_number, print_header


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values per column and suggest imputation strategies."""
    n_rows = len(df)
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    total_missing_pct = (total_missing / max(1, total_cells)) * 100.0

    columns_missing = []
    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        if missing_count > 0:
            missing_pct = (missing_count / max(1, n_rows)) * 100.0

            # Suggest strategy
            if missing_pct > 60:
                suggestion = "Drop column (>60% missing)"
            elif pd.api.types.is_numeric_dtype(s):
                clean_s = s.dropna()
                if len(clean_s) > 2:
                    skew = stats.skew(clean_s, bias=False)
                    if abs(skew) > 1.0:
                        suggestion = "Consider median imputation (skewed distribution)"
                    else:
                        suggestion = "Consider mean imputation (symmetric distribution)"
                else:
                    suggestion = "Consider median imputation"
            elif pd.api.types.is_datetime64_any_dtype(s):
                suggestion = "Consider forward/backward fill"
            else:
                suggestion = "Consider mode or 'Unknown' constant imputation"

            columns_missing.append({
                "column": col,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "suggestion": suggestion,
            })

    # Sort columns by missing count descending
    columns_missing.sort(key=lambda x: x["missing_count"], reverse=True)

    return {
        "total_missing": total_missing,
        "total_missing_pct": total_missing_pct,
        "total_rows": n_rows,
        "columns_missing": columns_missing,
    }


def display_missing_report(report: Dict[str, Any], suggest: bool = False):
    """Render missing value report with Rich."""
    print_header("MISSING VALUE REPORT")

    if not report["columns_missing"]:
        console.print("[bold green]✓ No missing values found in the dataset![/bold green]")
        return

    table = create_table(["Column", "Missing Count", "Percentage"])
    for col_info in report["columns_missing"]:
        pct_style = "bold red" if col_info["missing_pct"] > 20 else "yellow"
        table.add_row(
            col_info["column"],
            format_number(col_info["missing_count"]),
            Text(f"{col_info['missing_pct']:.2f}%", style=pct_style),
        )

    console.print(table)
    console.print(f"\n[bold]Total missing cells:[/bold] {format_number(report['total_missing'])} ({report['total_missing_pct']:.2f}% of all cells)")

    if suggest:
        console.print()
        print_header("RECOMMENDATIONS")
        for col_info in report["columns_missing"]:
            console.print(f"[bold]{col_info['column']:<18}[/bold] → {col_info['suggestion']}")
