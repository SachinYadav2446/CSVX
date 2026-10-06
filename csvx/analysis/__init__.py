"""Analysis layer modules for CSVX."""

from csvx.analysis.inspect import inspect_dataset, display_inspect_report
from csvx.analysis.describe import describe_dataframe, describe_column, display_describe_report
from csvx.analysis.outliers import detect_dataframe_outliers, detect_column_outliers, display_outliers_report
from csvx.analysis.correlation import compute_correlation_matrix, find_strong_correlations, display_correlation_report
from csvx.analysis.categories import analyze_categories, compute_crosstab, display_categories_report, display_crosstab_report
from csvx.analysis.explain import generate_dataset_insights, display_explain_report

__all__ = [
    "inspect_dataset",
    "display_inspect_report",
    "describe_dataframe",
    "describe_column",
    "display_describe_report",
    "detect_dataframe_outliers",
    "detect_column_outliers",
    "display_outliers_report",
    "compute_correlation_matrix",
    "find_strong_correlations",
    "display_correlation_report",
    "analyze_categories",
    "compute_crosstab",
    "display_categories_report",
    "display_crosstab_report",
    "generate_dataset_insights",
    "display_explain_report",
]
