import os
import pandas as pd
from eda_report import (
    generate_data_profile,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_missing_data,
    summarize_outliers_iqr,
    generate_eda_report,
)


def test_generate_data_profile(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })

    path = generate_data_profile(df, tmp_path)
    assert os.path.exists(path)


def test_plot_numeric_distributions(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10.0, 11.0, 12.0, 13.0]
    })

    files = plot_numeric_distributions(df, tmp_path)
    assert len(files) == 2
    for path in files:
        assert os.path.exists(path)


def test_plot_correlation_heatmap(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8]
    })

    path = plot_correlation_heatmap(df, tmp_path)
    assert os.path.exists(path)


def test_plot_missing_data(tmp_path):
    df = pd.DataFrame({
        "x": [1, None, 3],
        "y": [None, 2, 3]
    })

    path = plot_missing_data(df, tmp_path)
    assert os.path.exists(path)


def test_summarize_outliers_iqr():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 100]
    })

    result = summarize_outliers_iqr(df)
    assert "outlier_count" in result.columns
    assert len(result) == 1


def test_generate_eda_report(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, None, 6, 8],
        "label": ["a", "b", "c", "d"]
    })

    results = generate_eda_report(df, output_dir=tmp_path)

    assert os.path.exists(results["profile_path"])
    assert os.path.exists(results["heatmap_path"])
    assert os.path.exists(results["missing_path"])
    assert os.path.exists(results["outlier_summary_path"])
    assert len(results["distribution_files"]) == 2