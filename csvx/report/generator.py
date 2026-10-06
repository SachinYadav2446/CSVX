"""Full standalone EDA HTML and JSON report generator."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader

from csvx.analysis.inspect import inspect_dataset
from csvx.analysis.describe import describe_dataframe
from csvx.analysis.outliers import detect_dataframe_outliers
from csvx.analysis.correlation import compute_correlation_matrix, find_strong_correlations
from csvx.analysis.categories import analyze_categories
from csvx.quality.missing import analyze_missing_values
from csvx.quality.duplicates import analyze_duplicates
from csvx.quality.scorer import compute_quality_score
from csvx.visualization.plots import generate_plot


def json_serializer(obj: Any) -> Any:
    """Helper to convert NumPy/Pandas types for JSON dump."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def generate_eda_report(
    df: pd.DataFrame,
    file_name: str,
    output_dir: Union[str, Path] = "report",
) -> Path:
    """
    Generate a complete EDA report directory containing:
    - report.html
    - summary.json
    - missing_values.png
    - correlations.png
    - distributions/<column>.png
    - outliers/<column>.png
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = out_dir / "distributions"
    dist_dir.mkdir(exist_ok=True)
    outliers_dir = out_dir / "outliers"
    outliers_dir.mkdir(exist_ok=True)

    # 1. Compute Analyses
    inspect_res = inspect_dataset(df, file_name=file_name)
    stats_res = describe_dataframe(df)
    missing_res = analyze_missing_values(df)
    duplicates_res = analyze_duplicates(df)
    quality_res = compute_quality_score(df)
    outliers_res = detect_dataframe_outliers(df)
    categories_res = analyze_categories(df)
    corr_matrix = compute_correlation_matrix(df)
    strong_corr = find_strong_correlations(corr_matrix)

    # 2. Generate Visualizations
    charts = {
        "missing": None,
        "correlation": None,
        "distributions": [],
        "outliers": [],
    }

    # Missing values plot
    if missing_res["total_missing"] > 0:
        missing_img = out_dir / "missing_values.png"
        generate_plot(df, plot_type="missing", output_path=missing_img)
        charts["missing"] = "missing_values.png"

    # Correlation heatmap
    if not corr_matrix.empty and corr_matrix.shape[1] >= 2:
        corr_img = out_dir / "correlations.png"
        generate_plot(df, plot_type="correlation", output_path=corr_img)
        charts["correlation"] = "correlations.png"

    # Distribution histograms for numeric columns (top 6)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:6]:
        clean_col = "".join(c for c in col if c.isalnum() or c in "_-")
        dist_img = dist_dir / f"{clean_col}.png"
        generate_plot(df, plot_type="histogram", output_path=dist_img, column=col)
        charts["distributions"].append({
            "title": f"Distribution: {col}",
            "path": f"distributions/{clean_col}.png",
        })

    # Outlier boxplots for numeric columns (top 6)
    for col in numeric_cols[:6]:
        clean_col = "".join(c for c in col if c.isalnum() or c in "_-")
        box_img = outliers_dir / f"{clean_col}.png"
        generate_plot(df, plot_type="box", output_path=box_img, column=col)
        charts["outliers"].append({
            "title": f"Box Plot: {col}",
            "path": f"outliers/{clean_col}.png",
        })

    # 3. Export summary.json
    summary_data = {
        "file_name": file_name,
        "timestamp": datetime.now().isoformat(),
        "inspection": inspect_res,
        "quality": quality_res,
        "missing": missing_res,
        "duplicates": duplicates_res,
        "statistics": stats_res,
        "outliers": outliers_res,
        "categories": categories_res,
        "strong_correlations": strong_corr,
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=json_serializer)

    # 4. Render HTML Report
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))

    # Add custom formatting filter
    def number_format(val):
        if val is None or (isinstance(val, float) and (val != val)):
            return "N/A"
        if isinstance(val, (int, np.integer)):
            return f"{val:,}"
        if isinstance(val, (float, np.floating)):
            if val.is_integer():
                return f"{int(val):,}"
            return f"{val:,.2f}"
        return str(val)

    env.filters["number_format"] = number_format

    template = env.get_template("template.html")
    html_content = template.render(
        file_name=file_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rows=len(df),
        columns=len(df.columns),
        inspect=inspect_res,
        quality=quality_res,
        missing=missing_res,
        duplicates=duplicates_res,
        stats=stats_res,
        outliers=outliers_res,
        charts=charts,
    )

    html_path = out_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path
