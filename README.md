# CSVX — Command-Line Data Analysis & Data Quality Toolkit

**CSVX** is an advanced, lightweight command-line toolkit designed for data analysts, data scientists, ML engineers, and developers who need to quickly inspect, understand, clean, analyze, validate, visualize, and transform tabular datasets directly from the terminal without opening a Jupyter Notebook or writing repetitive Pandas code.

---

## ⚡ Key Highlights

- 🚀 **Zero Notebook Friction**: Perform full exploratory data analysis (EDA) and data validation with single terminal commands.
- 📊 **Automated Quality Scoring**: Instantly compute a comprehensive 0–100 Data Quality Score with subscores for missing values, duplicates, outliers, types, and consistency.
- 🧹 **Safe Data Cleaning Engine**: Impute missing data, deduplicate, strip whitespace, drop outliers, and normalize column names with non-destructive defaults.
- 📈 **Instant Visualizations**: Generate histograms, box plots, scatter plots, missingness charts, and correlation heatmaps in high-resolution PNG format.
- 📑 **Comprehensive HTML Reports**: Generate standalone, interactive EDA reports with charts, KPIs, statistics, and recommendations.
- 🔁 **Reproducible Pipelines**: Define and execute declarative multi-step data pipelines in YAML or JSON.
- 🔄 **Format Agnostic**: Seamlessly inspect, query, and convert between **CSV, Parquet, JSON, JSONL, and TSV**.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CSVX.git
cd CSVX

# Install in editable mode
pip install -e .

# Or with dev/testing dependencies
pip install -e ".[dev]"
```

---

## 🛠️ CLI Commands & Usage

### 1. Dataset Inspection
Summarize dataset dimensions, memory footprint, column data types, missing counts, and unique values.
```bash
csvx inspect data.csv
```

### 2. Descriptive Statistics
Compute 13 univariate statistical metrics (Count, Mean, Median, Mode, Std Dev, Variance, Min, Max, Q1, Q3, IQR, Skewness, Kurtosis).
```bash
csvx describe data.csv
csvx describe data.csv --column salary
```

### 3. Missing Value Analysis & Recommendations
Quantify missing data and generate smart imputation suggestions (e.g. median for skewed numeric, mode for categorical).
```bash
csvx missing data.csv
csvx missing data.csv --suggest
```

### 4. Duplicate Detection
Detect and analyze duplicate rows across the whole dataset or specific column subsets.
```bash
csvx duplicates data.csv
csvx duplicates data.csv --subset customer_id,email
```

### 5. Outlier Detection
Detect numerical anomalies using standard statistical techniques:
```bash
csvx outliers data.csv --method iqr
csvx outliers data.csv --method zscore --threshold 3.0
csvx outliers data.csv --method modified-zscore
```

### 6. Correlation Analysis
Generate correlation matrices or isolate strong relationships:
```bash
csvx correlation data.csv
csvx correlation data.csv --strong --threshold 0.75
```

### 7. Categorical Analysis & Cross-Tabulation
Analyze frequency distributions of categorical features and create two-way contingency tables:
```bash
csvx categories data.csv
csvx categories data.csv --column department --top 5
csvx crosstab data.csv --row gender --column purchased
```

### 8. Data Quality Score
Compute an overall 0–100 Data Quality Score with diagnostic issues and prioritized recommendations:
```bash
csvx quality data.csv
```

### 9. Semantic Data Type Detection
Infer semantic types such as Email, URL, UUID, ID, DateTime, Integer, Float, and Categorical:
```bash
csvx types data.csv
```

### 10. Data Cleaning & Transformation
Apply safe, non-destructive cleaning transformations:
```bash
csvx clean data.csv \
    --remove-duplicates \
    --fill-missing median \
    --normalize-columns \
    --output cleaned_data.csv
```

### 11. Query Engine
Filter datasets using intuitive expressions directly from the terminal:
```bash
csvx query data.csv "salary > 50000 and age < 40"
csvx query data.csv "department == 'Engineering'" --output engineers.csv
```

### 12. Dataset Comparison
Compare two dataset versions for schema drift, row/column count differences, and quality changes:
```bash
csvx compare old.csv new.csv
```

### 13. Format Conversion
Convert between CSV, Parquet, JSON, and TSV:
```bash
csvx convert data.csv --to parquet
csvx convert data.parquet --to csv
```

### 14. Dataset Sampling
Extract representative random samples from large files:
```bash
csvx sample huge.csv --rows 1000 --output sample.csv
csvx sample huge.csv --percentage 5 --output sample.csv
```

### 15. Visualization Plots
Generate publication-quality Matplotlib/Seaborn charts:
```bash
csvx plot data.csv --type histogram --column salary --output salary_hist.png
csvx plot data.csv --type box --column salary --output salary_box.png
csvx plot data.csv --type scatter --x experience --y salary --output exp_vs_sal.png
csvx plot data.csv --type correlation --output corr_matrix.png
csvx plot data.csv --type missing --output missing_bars.png
```

### 16. Standalone HTML EDA Report
Generate a complete, modern HTML report with embedded styles, summary JSON, and visualizations:
```bash
csvx report data.csv --output-dir report/
```

### 17. Declarative Pipeline Execution
Define workflows in YAML and execute them reproducibly:
```bash
csvx pipeline pipeline.yml data.csv --output processed.csv
```

*Example `pipeline.yml`:*
```yaml
pipeline:
  - inspect
  - quality
  - describe
  - detect_outliers:
      method: iqr
  - correlation:
      strong: true
  - remove_duplicates
  - fill_missing:
      method: median
  - normalize_columns
  - report:
      output_dir: pipeline_report
```

### 18. Automated Dataset Insights
Generate statistical explanations and action items:
```bash
csvx explain data.csv
```

---

## 🏛️ Architecture Overview

```
CSVX/
├── cli.py                     # Typer CLI application & command dispatch
├── console.py                 # Rich terminal styling and formatting
├── core/
│   ├── loader.py              # Multi-format data loader (CSV, Parquet, JSON, TSV, Excel)
│   └── types.py               # Semantic data type inference
├── analysis/
│   ├── inspect.py             # Dimensions, memory, and structural metadata
│   ├── describe.py            # Comprehensive descriptive statistics
│   ├── correlation.py         # Correlation matrices & strong pairwise filter
│   ├── outliers.py            # IQR, Z-Score, and MAD outlier detection
│   ├── categories.py          # Category distributions & cross-tabulation
│   └── explain.py             # Rule-based natural language insights
├── quality/
│   ├── scorer.py              # Overall Quality Score (0-100) & dimension breakdown
│   ├── missing.py             # Missing value metrics & imputation advice
│   ├── duplicates.py          # Duplicate rows & subset duplicates
│   └── consistency.py         # Type consistency, case variations, whitespace
├── cleaning/
│   ├── engine.py              # Safe data transformation engine
│   └── query.py               # Expression filter engine
├── comparison/
│   └── diff.py                # Dataset schema diff & version comparison
├── visualization/
│   └── plots.py               # Matplotlib & Seaborn visualization engine
├── report/
│   ├── generator.py           # HTML and JSON report generator
│   └── template.html          # Interactive HTML template
└── pipeline/
    └── runner.py              # YAML pipeline execution engine
```

---

## 🧪 Testing

Run the automated test suite with pytest:
```bash
pytest -v
```

---

## 📄 License
MIT License.
