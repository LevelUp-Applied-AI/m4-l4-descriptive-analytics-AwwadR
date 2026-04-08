import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data_profile(df, output_dir):
    """Generate a text data profile for any DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    profile_lines = [
        "Automated EDA Data Profile",
        "=" * 60,
        "",
        "Shape",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Columns",
        ", ".join(df.columns.astype(str).tolist()),
        "",
        "Data Types",
        df.dtypes.to_string(),
        "",
        "Missing Values Summary",
        pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pct": missing_pct
        }).to_string(),
        "",
        "Descriptive Statistics",
        df.describe(include="all").to_string(),
    ]

    out_path = os.path.join(output_dir, "auto_data_profile.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(profile_lines))

    return out_path

def plot_numeric_distributions(df, output_dir, columns=None):
    """Generate histogram plots for numeric columns."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    numeric_df = df.select_dtypes(include=["number"])

    if columns is not None:
        numeric_df = numeric_df[columns]

    saved_files = []

    for col in numeric_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(numeric_df[col].dropna(), kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{col}_auto_distribution.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files

def plot_correlation_heatmap(df, output_dir, columns=None):
    """Generate a correlation heatmap for numeric columns."""
    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include=["number"])

    if columns is not None:
        numeric_df = numeric_df[columns]

    # Remove columns with zero variance
    zero_var_cols = numeric_df.columns[numeric_df.nunique() <= 1]
    numeric_df = numeric_df.drop(columns=zero_var_cols)

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Automated Correlation Heatmap")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "auto_correlation_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path

def plot_missing_data(df, output_dir):
    """Generate a missing-data heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Data Pattern")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "missing_data_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path

def summarize_outliers_iqr(df):
    """Summarize outliers in numeric columns using the IQR method."""
    numeric_df = df.select_dtypes(include=["number"])
    rows = []

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(outlier_count)
        })

    return pd.DataFrame(rows)

def generate_eda_report(df, output_dir="output", numeric_columns=None, style="whitegrid"):
    """Generate a full automated EDA report for any DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style=style)

    profile_path = generate_data_profile(df, output_dir)
    distribution_files = plot_numeric_distributions(df, output_dir, columns=numeric_columns)
    heatmap_path = plot_correlation_heatmap(df, output_dir, columns=numeric_columns)
    missing_path = plot_missing_data(df, output_dir)

    outlier_df = summarize_outliers_iqr(df)
    outlier_path = os.path.join(output_dir, "outlier_summary.csv")
    outlier_df.to_csv(outlier_path, index=False)

    return {
        "profile_path": profile_path,
        "distribution_files": distribution_files,
        "heatmap_path": heatmap_path,
        "missing_path": missing_path,
        "outlier_summary_path": outlier_path,
    }