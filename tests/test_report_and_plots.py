import pytest
import pandas as pd
from pathlib import Path
from csvx.visualization.plots import generate_plot
from csvx.report.generator import generate_eda_report
from csvx.pipeline.runner import run_pipeline


def test_generate_plots(tmp_path):
    df = pd.DataFrame({
        "salary": [50000, 60000, 75000, 90000, 150000],
        "experience": [2, 4, 6, 8, 15],
    })

    hist_p = tmp_path / "hist.png"
    generate_plot(df, "histogram", hist_p, column="salary")
    assert hist_p.exists() and hist_p.stat().st_size > 0

    box_p = tmp_path / "box.png"
    generate_plot(df, "box", box_p, column="salary")
    assert box_p.exists() and box_p.stat().st_size > 0

    scatter_p = tmp_path / "scatter.png"
    generate_plot(df, "scatter", scatter_p, x="experience", y="salary")
    assert scatter_p.exists() and scatter_p.stat().st_size > 0

    corr_p = tmp_path / "corr.png"
    generate_plot(df, "correlation", corr_p)
    assert corr_p.exists() and corr_p.stat().st_size > 0


def test_generate_report(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "age": [25, 30, None, 40],
        "salary": [50000, 60000, 70000, 200000],
        "dept": ["IT", "HR", "IT", "Sales"],
    })

    report_dir = tmp_path / "eda_report"
    html_file = generate_eda_report(df, "test_data.csv", output_dir=report_dir)

    assert html_file.exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "correlations.png").exists() or (report_dir / "report.html").exists()


def test_pipeline_execution(tmp_path):
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Bob"],
        "Age": [25, 30, 30],
        "Salary": [50000.0, None, None],
    })
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    pipeline_yaml = tmp_path / "pipe.yml"
    pipeline_yaml.write_text(
        """
pipeline:
  - inspect
  - remove_duplicates
  - fill_missing:
      method: median
  - normalize_columns
        """,
        encoding="utf-8",
    )

    out_csv = tmp_path / "out.csv"
    res_df = run_pipeline(pipeline_yaml, csv_file, output_path=out_csv)

    assert len(res_df) == 2
    assert "salary" in res_df.columns
    assert res_df["salary"].isna().sum() == 0
    assert out_csv.exists()
