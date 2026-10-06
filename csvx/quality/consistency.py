"""Data consistency, mixed types, and formatting anomaly checks."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def check_column_consistency(series: pd.Series) -> Dict[str, Any]:
    """Check a single column for whitespace, casing, and mixed types."""
    clean_s = series.dropna()
    issues = []

    if len(clean_s) == 0:
        return {"column": series.name, "issues": issues, "score": 100}

    # 1. Mixed types in object column
    if series.dtype == object:
        types_present = set(type(x).__name__ for x in clean_s)
        if len(types_present) > 1:
            issues.append(f"Mixed python types detected: {', '.join(sorted(types_present))}")

        str_vals = clean_s.astype(str)

        # 2. Leading / trailing whitespace
        whitespace_count = (str_vals != str_vals.str.strip()).sum()
        if whitespace_count > 0:
            issues.append(f"{whitespace_count} values have leading/trailing whitespace")

        # 3. Case inconsistency in categorical values
        unique_raw = str_vals.nunique()
        unique_lower = str_vals.str.lower().nunique()
        if unique_raw > unique_lower:
            issues.append(f"Case inconsistencies detected ({unique_raw} variants vs {unique_lower} unique)")

    # 4. Constant or zero-variance column
    if series.nunique() == 1:
        issues.append("Constant column (only 1 unique value)")

    score = max(0, 100 - len(issues) * 25)
    return {
        "column": series.name,
        "issues": issues,
        "score": score,
    }


def check_dataset_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Check entire dataset consistency."""
    col_results = [check_column_consistency(df[col]) for col in df.columns]
    all_issues = []
    total_score = 0

    for res in col_results:
        total_score += res["score"]
        for issue in res["issues"]:
            all_issues.append(f"{res['column']}: {issue}")

    avg_score = int(round(total_score / max(1, len(df.columns))))
    return {
        "score": avg_score,
        "issues": all_issues,
        "column_details": col_results,
    }
