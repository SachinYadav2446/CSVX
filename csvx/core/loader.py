"""Unified data loader supporting CSV, TSV, Parquet, JSON, JSONL, and Excel."""

import os
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import pandas as pd


SUPPORTED_FORMATS = [".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".xlsx", ".xls"]


def detect_separator(file_path: Union[str, Path], n_bytes: int = 10240) -> str:
    """Detect CSV delimiter using csv.Sniffer or fallback."""
    import csv

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(n_bytes)
            if not sample:
                return ","
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",\t;|:")
            return dialect.delimiter
    except Exception:
        # Fallback check
        if str(file_path).endswith(".tsv"):
            return "\t"
        return ","


def load_dataset(
    file_path: Union[str, Path],
    sample_rows: Optional[int] = None,
    sample_pct: Optional[float] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a dataset into a pandas DataFrame from various file formats.

    Supported formats: CSV, TSV, Parquet, JSON, JSONL, Excel.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path, **kwargs)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", **kwargs)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True, **kwargs)
    elif suffix == ".json":
        try:
            df = pd.read_json(path, lines=False, **kwargs)
        except Exception:
            df = pd.read_json(path, lines=True, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, **kwargs)
    elif suffix == ".csv":
        # Auto-detect delimiter if not explicitly provided
        sep = kwargs.pop("sep", None) or kwargs.pop("delimiter", None)
        if sep is None:
            sep = detect_separator(path)
        df = pd.read_csv(path, sep=sep, **kwargs)
    else:
        # Default try CSV
        sep = kwargs.pop("sep", None) or detect_separator(path)
        df = pd.read_csv(path, sep=sep, **kwargs)

    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)
    elif sample_pct is not None and 0 < sample_pct < 100:
        frac = sample_pct / 100.0
        df = df.sample(frac=frac, random_state=42).reset_index(drop=True)

    return df


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file size, modification time, and existence."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = path.stat()
    return {
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "size_bytes": stat.st_size,
        "format": path.suffix.lower().lstrip("."),
    }


def save_dataset(df: pd.DataFrame, output_path: Union[str, Path], **kwargs: Any) -> Path:
    """Save a DataFrame to a given file format based on suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in [".parquet", ".pq"]:
        df.to_parquet(path, index=False, **kwargs)
    elif suffix in [".json", ".jsonl"]:
        lines = suffix == ".jsonl" or kwargs.pop("lines", False)
        orient = "records" if lines else kwargs.pop("orient", "records")
        df.to_json(path, orient=orient, lines=lines, **kwargs)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        df.to_excel(path, index=False, **kwargs)
    else:
        # Default to CSV
        df.to_csv(path, index=False, **kwargs)

    return path
