"""Main CLI entry point for CSVX using Typer and Rich."""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.text import Text

from csvx import __version__
from csvx.console import (
    console,
    error_console,
    print_header,
    print_error,
    print_success,
    create_table,
    format_number,
)
from csvx.core.loader import load_dataset, save_dataset, get_file_metadata
from csvx.core.types import infer_dataframe_types
from csvx.analysis.inspect import inspect_dataset, display_inspect_report
from csvx.analysis.describe import describe_dataframe, display_describe_report
from csvx.analysis.outliers import detect_dataframe_outliers, display_outliers_report
from csvx.analysis.correlation import (
    compute_correlation_matrix,
    find_strong_correlations,
    display_correlation_report,
)
from csvx.analysis.categories import (
    analyze_categories,
    compute_crosstab,
    display_categories_report,
    display_crosstab_report,
)
from csvx.analysis.explain import generate_dataset_insights, display_explain_report
from csvx.quality.missing import analyze_missing_values, display_missing_report
from csvx.quality.duplicates import analyze_duplicates, display_duplicates_report
from csvx.quality.scorer import compute_quality_score, display_quality_report
from csvx.cleaning.engine import clean_dataset
from csvx.cleaning.query import evaluate_query, display_query_results
from csvx.comparison.diff import compare_datasets, display_comparison_report
from csvx.visualization.plots import generate_plot
from csvx.report.generator import generate_eda_report
from csvx.pipeline.runner import run_pipeline

app = typer.Typer(
    name="csvx",
    help="CSVX — Fast, powerful command-line data analysis and data quality toolkit.",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]CSVX[/bold cyan] version [bold green]{__version__}[/bold green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show CSVX version and exit.", callback=version_callback, is_eager=True
    ),
):
    """CSVX — Command-Line Data Analysis & Data Quality Toolkit."""
    pass


@app.command("inspect")
def inspect_cmd(
    file_path: Path = typer.Argument(..., help="Path to tabular dataset (CSV, Parquet, JSON, TSV, etc.)"),
):
    """Inspect dataset dimensions, memory footprint, column types, and missing values."""
    try:
        meta = get_file_metadata(file_path)
        df = load_dataset(file_path)
        report = inspect_dataset(df, file_name=meta["file_name"], file_size_bytes=meta["size_bytes"])
        display_inspect_report(report)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("describe")
def describe_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    column: Optional[str] = typer.Option(None, "--column", "-c", help="Specific column to describe"),
):
    """Compute comprehensive descriptive statistics (mean, std, IQR, skewness, kurtosis, etc.)."""
    try:
        df = load_dataset(file_path)
        stats_list = describe_dataframe(df, column=column)
        display_describe_report(stats_list)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("missing")
def missing_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    suggest: bool = typer.Option(False, "--suggest", "-s", help="Provide smart imputation recommendations"),
):
    """Analyze missing value counts, percentages, and imputation strategies."""
    try:
        df = load_dataset(file_path)
        report = analyze_missing_values(df)
        display_missing_report(report, suggest=suggest)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("duplicates")
def duplicates_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    subset: Optional[str] = typer.Option(None, "--subset", help="Comma-separated columns to check for duplicates"),
):
    """Detect and quantify duplicate rows in the dataset."""
    try:
        df = load_dataset(file_path)
        subset_cols = [c.strip() for c in subset.split(",")] if subset else None
        report = analyze_duplicates(df, subset=subset_cols)
        display_duplicates_report(report)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("outliers")
def outliers_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    method: str = typer.Option("iqr", "--method", "-m", help="Method: 'iqr', 'zscore', or 'modified-zscore'"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Outlier threshold override"),
    column: Optional[str] = typer.Option(None, "--column", "-c", help="Specific column to check"),
):
    """Detect statistical outliers using IQR, Z-Score, or Modified Z-Score."""
    try:
        df = load_dataset(file_path)
        results = detect_dataframe_outliers(df, method=method, threshold=threshold, column=column)
        display_outliers_report(results)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("correlation")
def correlation_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    method: str = typer.Option("pearson", "--method", "-m", help="Correlation method: 'pearson', 'spearman', 'kendall'"),
    strong: bool = typer.Option(False, "--strong", "-s", help="Display only strongly correlated pairs"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Correlation threshold for strong pairs"),
):
    """Compute correlation matrix and highlight strong relationships."""
    try:
        df = load_dataset(file_path)
        matrix = compute_correlation_matrix(df, method=method)
        display_correlation_report(matrix, strong_only=strong, threshold=threshold, method=method)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("categories")
def categories_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    column: Optional[str] = typer.Option(None, "--column", "-c", help="Specific categorical column"),
    top: int = typer.Option(10, "--top", "-n", help="Number of top categories to display"),
):
    """Analyze frequency distributions of categorical columns."""
    try:
        df = load_dataset(file_path)
        results = analyze_categories(df, column=column, top_n=top)
        display_categories_report(results)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("crosstab")
def crosstab_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    row: str = typer.Option(..., "--row", "-r", help="Row column name"),
    column: str = typer.Option(..., "--column", "-c", help="Column column name"),
    normalize: Optional[str] = typer.Option(None, "--normalize", help="Normalize: 'index', 'columns', 'all'"),
):
    """Compute cross-tabulation between two categorical columns."""
    try:
        df = load_dataset(file_path)
        ct = compute_crosstab(df, row_col=row, col_col=column, normalize=normalize)
        display_crosstab_report(ct, row_col=row, col_col=column)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("types")
def types_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
):
    """Infer semantic column types (Integer, Float, DateTime, Email, ID, etc.)."""
    try:
        df = load_dataset(file_path)
        inferred = infer_dataframe_types(df)
        print_header("COLUMN TYPE ANALYSIS", f"File: {file_path.name}")
        table = create_table(["Column", "Inferred Semantic Type", "Pandas Dtype"])
        for col, t in inferred.items():
            table.add_row(col, Text(t, style="bold cyan"), str(df[col].dtype))
        console.print(table)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("quality")
def quality_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
):
    """Calculate overall Data Quality Score (0-100) and actionable recommendations."""
    try:
        df = load_dataset(file_path)
        report = compute_quality_score(df)
        display_quality_report(report)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("clean")
def clean_cmd(
    file_path: Path = typer.Argument(..., help="Path to input dataset"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (default: cleaned_<file>)"),
    remove_duplicates: bool = typer.Option(False, "--remove-duplicates", help="Remove duplicate rows"),
    fill_missing: Optional[str] = typer.Option(None, "--fill-missing", help="Fill missing: 'median', 'mean', 'mode', 'drop', or value"),
    normalize_columns: bool = typer.Option(False, "--normalize-columns", help="Convert column names to snake_case"),
    strip_whitespace: bool = typer.Option(True, "--strip-whitespace/--no-strip-whitespace", help="Trim leading/trailing whitespace in strings"),
    drop_outliers: Optional[str] = typer.Option(None, "--drop-outliers", help="Drop outlier rows: 'iqr' or 'zscore'"),
    drop_missing_cols_pct: Optional[float] = typer.Option(None, "--drop-missing-cols-pct", help="Drop columns with missing %% >= threshold"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite original file directly"),
):
    """Safely clean and transform a dataset."""
    try:
        df = load_dataset(file_path)
        print_header("CLEANING DATASET...", f"File: {file_path.name}")

        cleaned_df, actions = clean_dataset(
            df,
            remove_duplicates=remove_duplicates,
            fill_missing=fill_missing,
            normalize_columns=normalize_columns,
            strip_whitespace=strip_whitespace,
            drop_outliers=drop_outliers,
            drop_missing_col_pct=drop_missing_cols_pct,
        )

        for act in actions:
            print_success(act)

        if not actions:
            console.print("[yellow]No cleaning options were specified. Dataset unmodified.[/yellow]")

        if overwrite:
            out_file = file_path
        elif output:
            out_file = output
        else:
            out_file = file_path.parent / f"cleaned_{file_path.name}"

        save_dataset(cleaned_df, out_file)
        console.print(f"\nSaved cleaned dataset to: [bold green]{out_file}[/bold green]")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("query")
def query_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    query_expr: str = typer.Argument(..., help="Query filter expression, e.g. 'salary > 50000 and age < 35'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save query result to file"),
    max_rows: int = typer.Option(15, "--max-rows", "-n", help="Max preview rows to print"),
):
    """Filter dataset rows using expressions and save or preview results."""
    try:
        df = load_dataset(file_path)
        filtered = evaluate_query(df, query_expr)
        display_query_results(filtered, query_expr, max_rows=max_rows)

        if output:
            save_dataset(filtered, output)
            print_success(f"Saved {len(filtered):,} matched rows to {output}")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("compare")
def compare_cmd(
    old_file: Path = typer.Argument(..., help="Path to original / baseline dataset"),
    new_file: Path = typer.Argument(..., help="Path to modified / current dataset"),
):
    """Compare two versions of a dataset (dimensions, missingness, schema drift)."""
    try:
        df_old = load_dataset(old_file)
        df_new = load_dataset(new_file)
        comp = compare_datasets(df_old, df_new, name_old=old_file.name, name_new=new_file.name)
        display_comparison_report(comp)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("convert")
def convert_cmd(
    file_path: Path = typer.Argument(..., help="Source dataset path"),
    to: str = typer.Option(..., "--to", "-t", help="Target format: 'parquet', 'csv', 'json', 'jsonl', 'tsv'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Target destination path"),
):
    """Convert dataset between formats (CSV, Parquet, JSON, TSV)."""
    try:
        df = load_dataset(file_path)
        target_fmt = to.lower().lstrip(".")
        if output:
            out_path = output
        else:
            out_path = file_path.parent / f"{file_path.stem}.{target_fmt}"

        save_dataset(df, out_path)
        print_success(f"Converted [bold]{file_path.name}[/bold] → [bold green]{out_path}[/bold green]")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("sample")
def sample_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    rows: Optional[int] = typer.Option(None, "--rows", "-r", help="Number of random rows to sample"),
    percentage: Optional[float] = typer.Option(None, "--percentage", "-p", help="Percentage of rows to sample (1-100)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save sampled dataset to file"),
):
    """Extract a representative random sample from a large dataset."""
    try:
        df = load_dataset(file_path, sample_rows=rows, sample_pct=percentage)
        print_header("DATASET SAMPLE", f"Extracted {len(df):,} rows")
        if output:
            save_dataset(df, output)
            print_success(f"Saved sampled data to {output}")
        else:
            # Preview sample
            preview = df.head(10)
            table = create_table([str(c) for c in df.columns])
            for _, r in preview.iterrows():
                table.add_row(*[format_number(v) for v in r])
            console.print(table)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("plot")
def plot_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    plot_type: str = typer.Option(..., "--type", "-t", help="Plot type: 'histogram', 'box', 'scatter', 'correlation', 'missing'"),
    column: Optional[str] = typer.Option(None, "--column", "-c", help="Target column for univariate plots"),
    x: Optional[str] = typer.Option(None, "--x", help="X-axis column for scatter plots"),
    y: Optional[str] = typer.Option(None, "--y", help="Y-axis column for scatter plots"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output image path (default: plot_<type>.png)"),
    title: Optional[str] = typer.Option(None, "--title", help="Custom chart title"),
):
    """Generate exploratory data analysis visualizations (Matplotlib/Seaborn)."""
    try:
        df = load_dataset(file_path)
        out_p = output or Path(f"plot_{plot_type.lower()}.png")
        saved_path = generate_plot(df, plot_type=plot_type, output_path=out_p, column=column, x=x, y=y, title=title)
        print_success(f"Generated [bold]{plot_type}[/bold] plot: [bold green]{saved_path}[/bold green]")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("report")
def report_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
    output_dir: Path = typer.Option(Path("report"), "--output-dir", "-o", help="Output directory for HTML report and assets"),
):
    """Generate a comprehensive standalone interactive HTML EDA report."""
    try:
        df = load_dataset(file_path)
        print_header("GENERATING EDA REPORT...", f"Dataset: {file_path.name}")
        html_p = generate_eda_report(df, file_name=file_path.name, output_dir=output_dir)
        print_success(f"Report ready! Open in browser: [bold green]{html_p.resolve()}[/bold green]")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("pipeline")
def pipeline_cmd(
    pipeline_file: Path = typer.Argument(..., help="Path to YAML pipeline specification"),
    dataset_file: Path = typer.Argument(..., help="Path to dataset"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output path for final dataset"),
):
    """Execute a batch multi-step processing pipeline defined in YAML."""
    try:
        run_pipeline(pipeline_file, dataset_file, output_path=output)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


@app.command("explain")
def explain_cmd(
    file_path: Path = typer.Argument(..., help="Path to dataset"),
):
    """Generate rule-based automated statistical insights and recommendations."""
    try:
        df = load_dataset(file_path)
        res = generate_dataset_insights(df)
        display_explain_report(res)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
