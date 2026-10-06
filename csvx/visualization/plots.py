"""Plot generation engine using Matplotlib and Seaborn."""

from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")


def generate_plot(
    df: pd.DataFrame,
    plot_type: str,
    output_path: Union[str, Path],
    column: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 30,
) -> Path:
    """Generate and save a visualization plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p_type = plot_type.lower().strip()

    plt.figure(figsize=(9, 6), dpi=150)

    if p_type in ["hist", "histogram"]:
        target_col = column or x
        if not target_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric column available to plot histogram.")
            target_col = numeric_cols[0]
        
        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found.")
        
        clean_s = df[target_col].dropna()
        sns.histplot(clean_s, kde=True, bins=bins, color="#2b5c8f")
        plt.title(title or f"Distribution of {target_col}", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel("Count", fontsize=12)

    elif p_type in ["box", "boxplot"]:
        target_col = column or y or x
        if not target_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric column available to plot boxplot.")
            target_col = numeric_cols[0]

        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found.")

        clean_s = df[target_col].dropna()
        sns.boxplot(x=clean_s, color="#48cae4")
        plt.title(title or f"Box Plot of {target_col}", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel(target_col, fontsize=12)

    elif p_type in ["scatter", "scatterplot"]:
        if not x or not y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Scatter plot requires both --x and --y numeric columns.")
            x = x or numeric_cols[0]
            y = y or numeric_cols[1]

        if x not in df.columns or y not in df.columns:
            raise KeyError(f"Columns '{x}' and/or '{y}' not found.")

        sns.scatterplot(data=df, x=x, y=y, alpha=0.7, color="#0077b6")
        plt.title(title or f"{x} vs {y}", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)

    elif p_type in ["correlation", "heatmap", "corr"]:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValueError("Correlation plot requires at least 2 numerical columns.")
        
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
        plt.title(title or "Correlation Matrix Heatmap", fontsize=14, fontweight="bold", pad=12)

    elif p_type in ["missing", "missing_values", "nulls"]:
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        if missing_counts.empty:
            plt.text(0.5, 0.5, "No Missing Values Found", horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title(title or "Missing Values", fontsize=14, fontweight="bold")
        else:
            missing_pct = (missing_counts / len(df)) * 100.0
            sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#e63946")
            plt.title(title or "Missing Values Percentage by Column", fontsize=14, fontweight="bold", pad=12)
            plt.xlabel("Missing (%)", fontsize=12)
            plt.ylabel("Column", fontsize=12)

    else:
        raise ValueError(f"Unsupported plot type '{plot_type}'. Choose from 'histogram', 'box', 'scatter', 'correlation', 'missing'.")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
