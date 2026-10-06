"""Automated rule-based and statistical dataset insights generator."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from csvx.console import console, print_header
from csvx.analysis.describe import describe_dataframe
from csvx.analysis.outliers import detect_dataframe_outliers
from csvx.analysis.correlation import compute_correlation_matrix, find_strong_correlations


def generate_dataset_insights(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze DataFrame patterns and generate bulleted statistical insights and actions."""
    insights = []
    recommendations = []

    n_rows = len(df)
    n_cols = len(df.columns)

    # 1. Dimension insights
    if n_rows < 50:
        insights.append(f"Dataset is relatively small with only {n_rows} rows.")
    elif n_rows > 100000:
        insights.append(f"Large dataset containing {n_rows:,} rows and {n_cols} columns.")

    # 2. Missing value insights
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        missing_by_col = df.isna().sum()
        high_missing = missing_by_col[missing_by_col > 0]
        for col, cnt in high_missing.items():
            pct = (cnt / n_rows) * 100
            insights.append(f"{pct:.1f}% ({cnt:,}) of '{col}' values are missing.")
            if pct > 50:
                recommendations.append(f"Consider dropping column '{col}' (>50% missing values).")
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    recommendations.append(f"Consider median/mean imputation for numeric column '{col}'.")
                else:
                    recommendations.append(f"Consider mode or constant imputation for categorical column '{col}'.")

    # 3. Duplicate row insights
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        dup_pct = (n_dups / n_rows) * 100
        insights.append(f"Found {n_dups:,} duplicate rows ({dup_pct:.1f}% of total).")
        recommendations.append("Remove duplicate records using `csvx clean --remove-duplicates`.")

    # 4. Outlier insights
    outlier_res = detect_dataframe_outliers(df, method="iqr")
    for res in outlier_res:
        if res["outlier_count"] > 0:
            insights.append(f"Detected {res['outlier_count']:,} potential outliers in '{res['column']}' (IQR method).")
            recommendations.append(f"Investigate outliers in '{res['column']}' or use robust scaling / trimming.")

    # 5. Skewness insights
    stats_list = describe_dataframe(df)
    for stat in stats_list:
        if stat.get("type") == "numeric":
            skew = stat.get("skewness", 0.0)
            if skew > 1.5:
                insights.append(f"'{stat['name']}' is highly right-skewed (skewness = {skew:.2f}).")
                recommendations.append(f"Consider log or Box-Cox transformation for '{stat['name']}' before modeling.")
            elif skew < -1.5:
                insights.append(f"'{stat['name']}' is highly left-skewed (skewness = {skew:.2f}).")

    # 6. Correlation insights
    corr_matrix = compute_correlation_matrix(df)
    strong_pairs = find_strong_correlations(corr_matrix, threshold=0.75)
    for pair in strong_pairs[:5]:
        rel_type = "positive" if pair["correlation"] > 0 else "negative"
        insights.append(f"'{pair['col1']}' and '{pair['col2']}' have a strong {rel_type} correlation (r = {pair['correlation']:.2f}).")
        if abs(pair["correlation"]) > 0.9:
            recommendations.append(f"High multicollinearity between '{pair['col1']}' and '{pair['col2']}'. Consider feature selection.")

    # 7. Categorical column anomalies
    obj_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    for col in obj_cols:
        clean_s = df[col].dropna().astype(str)
        # Check casing inconsistency: e.g. "NY", "Ny", "ny"
        lower_counts = clean_s.str.lower().nunique()
        raw_counts = clean_s.nunique()
        if raw_counts > lower_counts:
            insights.append(f"Column '{col}' contains case-inconsistent categories ({raw_counts} distinct vs {lower_counts} case-insensitive).")
            recommendations.append(f"Normalize casing and trim whitespace in column '{col}'.")

    if not insights:
        insights.append("Dataset appears well-formed with no obvious structural anomalies.")
    if not recommendations:
        recommendations.append("Dataset is clean and ready for exploratory modeling and visualization.")

    return {
        "insights": insights,
        "recommendations": recommendations,
    }


def display_explain_report(data: Dict[str, List[str]]):
    """Render the explain report to console."""
    print_header("DATASET INSIGHTS")
    for i, insight in enumerate(data["insights"], 1):
        console.print(f" {i}. [bold]{insight}[/bold]")

    console.print()
    print_header("RECOMMENDATIONS")
    for rec in data["recommendations"]:
        console.print(f" → [yellow]{rec}[/yellow]")
