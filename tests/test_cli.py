import pytest
from typer.testing import CliRunner
from csvx.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "CSVX version" in result.stdout


def test_cli_inspect():
    result = runner.invoke(app, ["inspect", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "CSVX DATASET REPORT" in result.stdout
    assert "COLUMN SUMMARY" in result.stdout


def test_cli_describe():
    result = runner.invoke(app, ["describe", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "NUMERICAL" in result.stdout


def test_cli_missing():
    result = runner.invoke(app, ["missing", "examples/sample_data.csv", "--suggest"])
    assert result.exit_code == 0
    assert "MISSING VALUE REPORT" in result.stdout
    assert "RECOMMENDATIONS" in result.stdout


def test_cli_duplicates():
    result = runner.invoke(app, ["duplicates", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "DUPLICATE ANALYSIS" in result.stdout


def test_cli_outliers():
    result = runner.invoke(app, ["outliers", "examples/sample_data.csv", "--method", "iqr"])
    assert result.exit_code == 0
    assert "OUTLIER REPORT" in result.stdout


def test_cli_correlation():
    result = runner.invoke(app, ["correlation", "examples/sample_data.csv", "--strong"])
    assert result.exit_code == 0
    assert "STRONG CORRELATIONS" in result.stdout or "CORRELATION" in result.stdout


def test_cli_quality():
    result = runner.invoke(app, ["quality", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "SCORE:" in result.stdout


def test_cli_types():
    result = runner.invoke(app, ["types", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "COLUMN TYPE ANALYSIS" in result.stdout


def test_cli_clean(tmp_path):
    out_csv = tmp_path / "cleaned.csv"
    result = runner.invoke(app, [
        "clean",
        "examples/sample_data.csv",
        "--remove-duplicates",
        "--fill-missing", "median",
        "--normalize-columns",
        "--output", str(out_csv),
    ])
    assert result.exit_code == 0
    assert out_csv.exists()


def test_cli_query(tmp_path):
    out_csv = tmp_path / "filtered.csv"
    result = runner.invoke(app, [
        "query",
        "examples/sample_data.csv",
        "salary > 60000",
        "--output", str(out_csv),
    ])
    assert result.exit_code == 0
    assert out_csv.exists()


def test_cli_convert(tmp_path):
    out_pq = tmp_path / "sample.parquet"
    result = runner.invoke(app, [
        "convert",
        "examples/sample_data.csv",
        "--to", "parquet",
        "--output", str(out_pq),
    ])
    assert result.exit_code == 0
    assert out_pq.exists()


def test_cli_sample(tmp_path):
    out_sample = tmp_path / "sampled.csv"
    result = runner.invoke(app, [
        "sample",
        "examples/sample_data.csv",
        "--rows", "5",
        "--output", str(out_sample),
    ])
    assert result.exit_code == 0
    assert out_sample.exists()


def test_cli_explain():
    result = runner.invoke(app, ["explain", "examples/sample_data.csv"])
    assert result.exit_code == 0
    assert "DATASET INSIGHTS" in result.stdout
