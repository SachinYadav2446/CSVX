"""Data cleaning engine for safe, configurable transformations."""

import re
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

from csvx.console import console, print_header, print_success, print_warning, format_number
from csvx.core.loader import save_dataset
from csvx.analysis.outliers import detect_dataframe_outliers


def normalize_column_name(name: str) -> str:
    """Convert column name to snake_case, removing punctuation."""
    name = re.sub(r"[^\w\s]", "", str(name).strip())
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
    fill_missing: Optional[str] = None,
    normalize_columns: bool = False,
    strip_whitespace: bool = True,
    drop_outliers: Optional[str] = None,
    drop_missing_col_pct: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply configured cleaning operations to a DataFrame.
    Returns (cleaned_df, list_of_actions_taken).
    """
    cleaned_df = df.copy()
    actions = []

    # 1. Normalize column names
    if normalize_columns:
        old_cols = cleaned_df.columns.tolist()
        new_cols = [normalize_column_name(c) for c in old_cols]
        cleaned_df.columns = new_cols
        if old_cols != new_cols:
            actions.append("Normalized column names to snake_case")

    # 2. Strip whitespace in string columns
    if strip_whitespace:
        str_cols = cleaned_df.select_dtypes(include=["object", "string", "str"]).columns
        stripped_count = 0
        for col in str_cols:
            s = cleaned_df[col]
            if s.dropna().map(lambda x: isinstance(x, str)).any():
                trimmed = s.map(lambda x: x.strip() if isinstance(x, str) else x)
                diffs = (s != trimmed).sum()
                if diffs > 0:
                    stripped_count += diffs
                    cleaned_df[col] = trimmed
        if stripped_count > 0:
            actions.append(f"Trimmed whitespace on {stripped_count:,} string values")

    # 3. Remove duplicate rows
    if remove_duplicates:
        init_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        dups_removed = init_rows - len(cleaned_df)
        if dups_removed > 0:
            actions.append(f"Removed {dups_removed:,} duplicate rows")

    # 4. Drop columns with excessive missing percentage
    if drop_missing_col_pct is not None and 0 < drop_missing_col_pct < 100:
        thresh = drop_missing_col_pct / 100.0
        n_rows = len(cleaned_df)
        cols_to_drop = [c for c in cleaned_df.columns if cleaned_df[c].isna().sum() / max(1, n_rows) >= thresh]
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            actions.append(f"Dropped {len(cols_to_drop)} columns with >={drop_missing_col_pct}% missing values: {', '.join(cols_to_drop)}")

    # 5. Fill or handle missing values
    if fill_missing:
        method = fill_missing.lower().strip()
        total_missing_before = int(cleaned_df.isna().sum().sum())

        if method == "drop":
            init_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna().reset_index(drop=True)
            dropped_rows = init_rows - len(cleaned_df)
            actions.append(f"Dropped {dropped_rows:,} rows with missing values")

        elif method == "median":
            filled_count = 0
            for col in cleaned_df.columns:
                if cleaned_df[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        med = cleaned_df[col].median()
                        cnt = int(cleaned_df[col].isna().sum())
                        cleaned_df[col] = cleaned_df[col].fillna(med)
                        filled_count += cnt
                    else:
                        # Fallback to mode for non-numeric
                        mode_s = cleaned_df[col].mode()
                        if not mode_s.empty:
                            cnt = int(cleaned_df[col].isna().sum())
                            cleaned_df[col] = cleaned_df[col].fillna(mode_s.iloc[0])
                            filled_count += cnt
            if filled_count > 0:
                actions.append(f"Filled {filled_count:,} missing values using median (and mode for categorical)")

        elif method == "mean":
            filled_count = 0
            for col in cleaned_df.columns:
                if cleaned_df[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        mean_val = cleaned_df[col].mean()
                        cnt = int(cleaned_df[col].isna().sum())
                        cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                        filled_count += cnt
                    else:
                        mode_s = cleaned_df[col].mode()
                        if not mode_s.empty:
                            cnt = int(cleaned_df[col].isna().sum())
                            cleaned_df[col] = cleaned_df[col].fillna(mode_s.iloc[0])
                            filled_count += cnt
            if filled_count > 0:
                actions.append(f"Filled {filled_count:,} missing values using mean (and mode for categorical)")

        elif method == "mode":
            filled_count = 0
            for col in cleaned_df.columns:
                if cleaned_df[col].isna().sum() > 0:
                    mode_s = cleaned_df[col].mode()
                    if not mode_s.empty:
                        cnt = int(cleaned_df[col].isna().sum())
                        cleaned_df[col] = cleaned_df[col].fillna(mode_s.iloc[0])
                        filled_count += cnt
            if filled_count > 0:
                actions.append(f"Filled {filled_count:,} missing values using mode")

        elif method in ["ffill", "pad"]:
            cnt = int(cleaned_df.isna().sum().sum())
            cleaned_df = cleaned_df.ffill().bfill()
            actions.append(f"Filled missing values using forward/backward fill")

        else:
            # Custom constant value
            cnt = int(cleaned_df.isna().sum().sum())
            cleaned_df = cleaned_df.fillna(fill_missing)
            actions.append(f"Filled missing values with '{fill_missing}'")

    # 6. Drop outliers
    if drop_outliers:
        outlier_res = detect_dataframe_outliers(cleaned_df, method=drop_outliers)
        outlier_indices = set()
        for res in outlier_res:
            outlier_indices.update(res["indices"])

        if outlier_indices:
            cleaned_df = cleaned_df.drop(index=list(outlier_indices)).reset_index(drop=True)
            actions.append(f"Dropped {len(outlier_indices):,} outlier rows using {drop_outliers.upper()}")

    return cleaned_df, actions
