"""Comprehensive Data Quality Scoring engine."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from csvx.console import console, create_table, format_number, print_header
from csvx.quality.missing import analyze_missing_values
from csvx.quality.duplicates import analyze_duplicates
from csvx.quality.consistency import check_dataset_consistency
from csvx.analysis.outliers import detect_dataframe_outliers


def compute_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate an overall 0-100 data quality score along with individual dimension scores:
    - Missing Values (weight: 25%)
    - Duplicates (weight: 20%)
    - Outliers (weight: 20%)
    - Data Types (weight: 15%)
    - Consistency (weight: 20%)
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    issues = []
    actions = []

    # 1. Missing Values Score
    missing_info = analyze_missing_values(df)
    missing_pct = missing_info["total_missing_pct"]
    missing_score = max(0, int(round(100.0 - (missing_pct * 3.0))))
    for col_info in missing_info["columns_missing"]:
        issues.append(f"{col_info['missing_pct']:.1f}% missing {col_info['column']} values")
        actions.append(f"Handle missing {col_info['column']} values ({col_info['suggestion']})")

    # 2. Duplicate Rows Score
    dup_info = analyze_duplicates(df)
    dup_pct = dup_info["duplicate_rate"]
    dup_score = max(0, int(round(100.0 - (dup_pct * 4.0))))
    if dup_info["duplicate_rows"] > 0:
        issues.append(f"{dup_info['duplicate_rate']:.2f}% duplicate rows ({dup_info['duplicate_rows']:,} rows)")
        actions.append("Remove duplicate records")

    # 3. Outlier Score
    outlier_list = detect_dataframe_outliers(df, method="iqr")
    total_outliers = sum(res["outlier_count"] for res in outlier_list)
    numeric_cells = max(1, len(outlier_list) * n_rows)
    outlier_cell_pct = (total_outliers / numeric_cells) * 100.0 if outlier_list else 0.0
    outlier_score = max(0, int(round(100.0 - (outlier_cell_pct * 4.0))))
    for res in outlier_list:
        if res["outlier_count"] > 0:
            issues.append(f"{res['outlier_count']:,} {res['column']} outliers")
            actions.append(f"Investigate {res['column']} outliers")

    # 4. Consistency Score & Data Types
    consistency_info = check_dataset_consistency(df)
    consistency_score = consistency_info["score"]
    for iss in consistency_info["issues"]:
        issues.append(iss)
        actions.append(f"Clean and format values in {iss.split(':')[0]}")

    # Data Types Validity Score
    type_issues_count = sum(1 for iss in consistency_info["issues"] if "Mixed" in iss or "Constant" in iss)
    data_types_score = max(0, 100 - type_issues_count * 15)

    # Weighted Overall Score
    weights = {
        "missing": 0.25,
        "duplicates": 0.20,
        "outliers": 0.20,
        "data_types": 0.15,
        "consistency": 0.20,
    }

    overall_score = int(round(
        missing_score * weights["missing"]
        + dup_score * weights["duplicates"]
        + outlier_score * weights["outliers"]
        + data_types_score * weights["data_types"]
        + consistency_score * weights["consistency"]
    ))
    overall_score = max(0, min(100, overall_score))

    # Deduplicate actions
    dedup_actions = []
    for act in actions:
        if act not in dedup_actions:
            dedup_actions.append(act)

    return {
        "overall_score": overall_score,
        "subscores": {
            "Missing Values": missing_score,
            "Duplicates": dup_score,
            "Outliers": outlier_score,
            "Data Types": data_types_score,
            "Consistency": consistency_score,
        },
        "issues": issues,
        "recommendations": dedup_actions,
    }


def display_quality_report(report: Dict[str, Any]):
    """Render the Quality Score card and diagnostics with Rich."""
    score = report["overall_score"]
    if score >= 85:
        score_color = "bold green"
        badge = "EXCELLENT / CLEAN"
    elif score >= 70:
        score_color = "bold yellow"
        badge = "MODERATE QUALITY / ATTENTION NEEDED"
    else:
        score_color = "bold red"
        badge = "POOR QUALITY / ACTION REQUIRED"

    panel_text = Text(justify="center")
    panel_text.append("CSVX DATA QUALITY\n\n", style="bold white")
    panel_text.append(f"SCORE: {score}/100\n", style=score_color)
    panel_text.append(f"[{badge}]", style="dim")

    console.print()
    console.print(Panel(panel_text, border_style=score_color, expand=False, padding=(1, 6)))
    console.print()

    # Subscores table
    table = create_table(["Quality Dimension", "Score", "Rating"])
    for dim, subscore in report["subscores"].items():
        if subscore >= 85:
            color, rating = "green", "Good"
        elif subscore >= 70:
            color, rating = "yellow", "Moderate"
        else:
            color, rating = "red", "Poor"

        table.add_row(
            dim,
            Text(f"{subscore}/100", style=f"bold {color}"),
            Text(rating, style=color),
        )
    console.print(table)

    # Issues Detected
    if report["issues"]:
        console.print()
        print_header("ISSUES DETECTED")
        for iss in report["issues"][:10]:
            console.print(f" [bold red]⚠[/bold red] {iss}")
        if len(report["issues"]) > 10:
            console.print(f" ... and {len(report['issues']) - 10} more issues.")

    # Recommended Actions
    if report["recommendations"]:
        console.print()
        print_header("RECOMMENDED ACTIONS")
        for i, act in enumerate(report["recommendations"][:8], 1):
            console.print(f" {i}. [bold cyan]{act}[/bold cyan]")
    else:
        console.print("\n[bold green]✓ No critical data quality issues identified.[/bold green]")
