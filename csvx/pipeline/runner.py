"""Pipeline execution engine for reproducible batch workflows."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import pandas as pd

from csvx.console import console, print_header, print_success, print_error
from csvx.core.loader import load_dataset, save_dataset
from csvx.analysis.inspect import inspect_dataset, display_inspect_report
from csvx.analysis.describe import describe_dataframe, display_describe_report
from csvx.analysis.outliers import detect_dataframe_outliers, display_outliers_report
from csvx.analysis.correlation import compute_correlation_matrix, display_correlation_report
from csvx.quality.scorer import compute_quality_score, display_quality_report
from csvx.cleaning.engine import clean_dataset
from csvx.report.generator import generate_eda_report


def run_pipeline(
    pipeline_config_path: Union[str, Path],
    dataset_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Execute a multi-step data pipeline defined in YAML/JSON."""
    cfg_path = Path(pipeline_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_config_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    steps = config.get("pipeline", [])
    if not steps:
        raise ValueError("Pipeline configuration contains no steps.")

    print_header("CSVX PIPELINE EXECUTION", f"Config: {cfg_path.name} | Dataset: {Path(dataset_path).name}")
    df = load_dataset(dataset_path)

    for i, step in enumerate(steps, 1):
        if isinstance(step, str):
            step_name = step.lower().strip()
            step_params = {}
        elif isinstance(step, dict):
            step_name = list(step.keys())[0].lower().strip()
            step_params = step[step_name] or {}
        else:
            continue

        console.print(f"\n[bold cyan]Step {i}:[/bold cyan] [bold]{step_name}[/bold]")

        if step_name == "inspect":
            report = inspect_dataset(df, file_name=Path(dataset_path).name)
            display_inspect_report(report)

        elif step_name in ["quality", "quality_score"]:
            q_res = compute_quality_score(df)
            display_quality_report(q_res)

        elif step_name == "describe":
            stat_res = describe_dataframe(df, column=step_params.get("column"))
            display_describe_report(stat_res)

        elif step_name in ["detect_outliers", "outliers"]:
            method = step_params.get("method", "iqr")
            th = step_params.get("threshold")
            out_res = detect_dataframe_outliers(df, method=method, threshold=th)
            display_outliers_report(out_res)

        elif step_name in ["correlation", "corr"]:
            method = step_params.get("method", "pearson")
            strong = step_params.get("strong", False)
            corr_mat = compute_correlation_matrix(df, method=method)
            display_correlation_report(corr_mat, strong_only=strong, method=method)

        elif step_name in ["remove_duplicates", "dedup"]:
            init_cnt = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            print_success(f"Removed {init_cnt - len(df):,} duplicate rows.")

        elif step_name in ["fill_missing", "impute"]:
            method = step_params.get("method", "median")
            df, actions = clean_dataset(df, fill_missing=method)
            for act in actions:
                print_success(act)

        elif step_name in ["normalize_columns", "normalize"]:
            df, actions = clean_dataset(df, normalize_columns=True)
            for act in actions:
                print_success(act)

        elif step_name == "clean":
            df, actions = clean_dataset(
                df,
                remove_duplicates=step_params.get("remove_duplicates", False),
                fill_missing=step_params.get("fill_missing"),
                normalize_columns=step_params.get("normalize_columns", False),
                drop_outliers=step_params.get("drop_outliers"),
            )
            for act in actions:
                print_success(act)

        elif step_name == "report":
            out_dir = step_params.get("output_dir", "report")
            html_p = generate_eda_report(df, file_name=Path(dataset_path).name, output_dir=out_dir)
            print_success(f"Generated comprehensive report at {html_p}")

        else:
            console.print(f"[yellow]Unknown step '{step_name}', skipping.[/yellow]")

    if output_path:
        save_dataset(df, output_path)
        print_success(f"Saved transformed dataset to {output_path}")

    return df
