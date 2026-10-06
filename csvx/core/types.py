"""Semantic and statistical data type detection for tabular columns."""

import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
URL_REGEX = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)


def infer_column_type(series: pd.Series, sample_size: int = 500) -> str:
    """
    Infer the high-level semantic type of a pandas Series.

    Possible types:
    - 'Integer'
    - 'Float'
    - 'Boolean'
    - 'DateTime'
    - 'Email'
    - 'URL'
    - 'UUID'
    - 'ID'
    - 'Categorical'
    - 'String'
    """
    clean_s = series.dropna()
    if len(clean_s) == 0:
        return "Empty"

    # Take sample for fast pattern matching
    sample = clean_s.iloc[:sample_size] if len(clean_s) > sample_size else clean_s

    # Check for Boolean
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if sample.dtype == object or pd.api.types.is_string_dtype(series):
        bool_candidates = {str(x).strip().lower() for x in sample.unique()}
        if bool_candidates.issubset({"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}):
            return "Boolean"

    # Check for DateTime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
    if sample.dtype == object or pd.api.types.is_string_dtype(series):
        # Try converting sample to datetime
        try:
            # Avoid single numbers or short tokens being parsed as dates
            str_sample = sample.astype(str)
            if str_sample.str.len().mean() >= 6:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(str_sample, errors="coerce")
                    if converted.notna().mean() > 0.8:
                        return "DateTime"
        except Exception:
            pass

    # Check for Numeric types
    if pd.api.types.is_integer_dtype(series):
        # Could it be an ID column?
        col_name = str(series.name).lower()
        if ("id" in col_name or "key" in col_name or col_name.endswith("_num")) and series.nunique() / len(series) > 0.8:
            return "ID"
        return "Integer"

    if pd.api.types.is_float_dtype(series):
        # Check if float values are actually whole integers
        if (clean_s % 1 == 0).all():
            col_name = str(series.name).lower()
            if ("id" in col_name or "key" in col_name) and series.nunique() / len(series) > 0.8:
                return "ID"
            return "Integer"
        return "Float"

    # Check String/Object semantic patterns
    str_sample = sample.astype(str).str.strip()

    # Email
    if str_sample.str.match(EMAIL_REGEX).mean() > 0.8:
        return "Email"

    # URL
    if str_sample.str.match(URL_REGEX).mean() > 0.8:
        return "URL"

    # UUID
    if str_sample.str.match(UUID_REGEX).mean() > 0.8:
        return "UUID"

    # ID-like column names & high uniqueness
    col_name = str(series.name).lower()
    total_len = len(series)
    unique_len = series.nunique()
    if ("id" in col_name or "uuid" in col_name or "code" in col_name or "key" in col_name) and (unique_len / total_len > 0.7):
        return "ID"

    # Categorical: low cardinality relative to dataset size
    if isinstance(series.dtype, pd.CategoricalDtype) or (unique_len <= 50 and (unique_len / max(1, total_len) < 0.2 or total_len < 100)):
        return "Categorical"

    return "String"


def infer_dataframe_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer semantic types for all columns in a DataFrame."""
    return {col: infer_column_type(df[col]) for col in df.columns}
