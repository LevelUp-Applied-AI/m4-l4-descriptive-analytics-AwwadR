"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

OUTPUT_DIR = "output"
DATA_PATH = "data/student_performance.csv"

def save_text_file(path, content):
    """Save plain text content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def format_p_value(p_value):
    """Format p-values cleanly for reports."""
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.4f}"

def interpret_effect_size_cohens_d(d_value):
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d_value)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def cohens_d(group1, group2):
    """Compute Cohen's d for two independent groups."""
    group1 = np.array(group1, dtype=float)
    group2 = np.array(group2, dtype=float)

    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)

    pooled_std = np.sqrt(
        (((n1 - 1) * std1 **2) + ((n2 - 1) * std2 ** 2)) / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the cleaned dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    df = pd.read_csv(filepath)

    original_shape = df.shape
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    # Missing-data handling decisions
    handling_notes = []

    if "commute_minutes" in df.columns:
        commute_missing = int(df["commute_minutes"].isnull().sum())
        if commute_missing > 0:
            commute_median = df["commute_minutes"].median()
            df["commute_minutes"] = df["commute_minutes"].fillna(commute_median)
            handling_notes.append(
                f"- commute_minutes: filled {commute_missing} missing values with median "
                f"({commute_median:.2f}) because the column is numeric and median is more "
                f"robust for potentially skewed data."
            )
    
    if "study_hours_weekly" in df.columns:
        study_missing = int(df["study_hours_weekly"].isnull().sum())
        if study_missing > 0:
            before_drop = len(df)
            df = df.dropna(subset=["study_hours_weekly"]).copy()
            rows_dropped = before_drop - len(df)
            handling_notes.append(
                f"- study_hours_weekly: dropped {rows_dropped} rows with missing values "
                f"because the missing share is relatively small and the variable is important "
                f"for correlation and hypothesis interpretation."
            )

    if "scholarship" in df.columns:
        scholarship_missing = int(df["scholarship"].isnull().sum())
        if scholarship_missing > 0:
            handling_notes.append(
                f"- scholarship: {scholarship_missing} missing values were left as missing "
                f"for general analysis, and rows with missing scholarship values were "
                f"excluded automatically in the chi-square contingency table."
        )
    
    if not handling_notes:
        handling_notes.append("- No missing-value handling was needed.")

    profile_lines = [
        "Lab 4 Data Profile",
        "=" * 60,
        "",
        "Original Dataset Shape",
        f"Rows: {original_shape[0]}",
        f"Columns: {original_shape[1]}",
        "",
        "Final Dataset Shape After Cleaning",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Columns",
        ", ".join(df.columns.tolist()),
        "",
        "Data Types",
        df.dtypes.to_string(),
        "",
        "Missing Values Summary (Before Cleaning)",
        pd.DataFrame(
            {
                "missing_count": missing_counts,
                "missing_pct": missing_pct
            }
        ).to_string(),
        "",
        "Descriptive Statistics (Numeric Columns)",
        df.describe().to_string(),
        "",
        "Handling Decisions",
        "\n".join(handling_notes),
    ]

    save_text_file(
        os.path.join(OUTPUT_DIR, "data_profile.txt"),
        "\n".join(profile_lines)
    )

    return df
    
def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        list[str]: list of saved plot file paths
    """
    saved_files = []

    sns.set_theme(style="whitegrid")

    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct"]
    for col in numeric_cols:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax)
            skew_val = df[col].dropna().skew()

            if skew_val > 0.5:
                skew_label = "right-skewed"
            elif skew_val < -0.5:
                skew_label = "left-skewed"
            else:
                skew_label = "roughly symmetric"

            ax.set_title(f"{col} distribution appears {skew_label}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.tight_layout()

            out_path = os.path.join(OUTPUT_DIR, f"{col}_distribution.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(out_path)

    if "department" in df.columns and "gpa" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="department", y="gpa", ax=ax)
        ax.set_title("GPA varies across departments")
        ax.set_xlabel("Department")
        ax.set_ylabel("GPA")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, "gpa_by_department.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    if "scholarship" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        scholarship_counts = df["scholarship"].value_counts()
        scholarship_counts.plot(kind="bar", ax=ax)
        ax.set_title("Scholarship categories are unevenly distributed")
        ax.set_xlabel("Scholarship")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, "scholarship_counts.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def get_top_correlated_pairs(corr_matrix, top_n=2):
    """Return top correlated variable pairs excluding self-correlation."""
    pairs = []

    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                pairs.append((col1, col2, abs(corr_matrix.loc[col1, col2])))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

     Returns:
        dict: contains correlation matrix and top pairs
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # Drop zero-variance columns if any
    nunique = numeric_df.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        numeric_df = numeric_df.drop(columns=zero_var_cols)

    corr_matrix = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    ax.set_title("Correlation heatmap of numeric variables")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    top_pairs = get_top_correlated_pairs(corr_matrix, top_n=2)
    scatter_paths = []

    for col1, col2, corr_value in top_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=col1, y=col2, ax=ax, alpha=0.6)
        ax.set_title(f"{col1} vs {col2} (|r| = {corr_value:.2f})")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"scatter_{col1}_vs_{col2}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        scatter_paths.append(out_path)

    return {
        "corr_matrix": corr_matrix,
        "top_pairs": top_pairs,
        "heatmap_path": heatmap_path,
        "scatter_paths": scatter_paths,
    }


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns."""
    results = {}

    print("\n=== Hypothesis Testing Results ===")

    # Hypothesis 1: internship vs GPA
    if "has_internship" in df.columns and "gpa" in df.columns:
        internship_yes = df.loc[df["has_internship"] == "Yes", "gpa"].dropna()
        internship_no = df.loc[df["has_internship"] == "No", "gpa"].dropna()

        t_stat, p_value = stats.ttest_ind(
            internship_yes,
            internship_no,
            equal_var=False,
            nan_policy="omit"
        )
        d_value = cohens_d(internship_yes, internship_no)

        results["internship_ttest"] = {
            "test": "Independent samples t-test",
            "hypothesis": "Students with internships have a higher GPA than students without internships.",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(d_value),
            "group_means": {
                "internship_yes_mean_gpa": float(internship_yes.mean()),
                "internship_no_mean_gpa": float(internship_no.mean()),
            },
        }

        sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"

        print("\n1) Internship vs GPA")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {format_p_value(p_value)}")
        print(f"   Cohen's d: {d_value:.4f} ({interpret_effect_size_cohens_d(d_value)} effect)")
        print(f"   Interpretation: The GPA difference is {sig_text}.")

    # Hypothesis 2: scholarship vs department
    if "scholarship" in df.columns and "department" in df.columns:
        contingency_table = pd.crosstab(df["scholarship"], df["department"])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        results["scholarship_chi_square"] = {
            "test": "Chi-square test of independence",
            "hypothesis": "Scholarship status is associated with department.",
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "contingency_table": contingency_table,
        }

        sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"

        print("\n2) Scholarship vs Department")
        print(f"   chi-square statistic: {chi2:.4f}")
        print(f"   p-value: {format_p_value(p_value)}")
        print(f"   degrees of freedom: {dof}")
        print(f"   Interpretation: The association is {sig_text}.")

    # Optional extra: ANOVA for GPA across departments
    if "department" in df.columns and "gpa" in df.columns:
        dept_groups = []
        dept_names = []

        for dept in sorted(df["department"].dropna().unique()):
            gpas = df.loc[df["department"] == dept, "gpa"].dropna()
            if len(gpas) > 1:
                dept_groups.append(gpas)
                dept_names.append(dept)

        if len(dept_groups) >= 3:
            f_stat, p_value = stats.f_oneway(*dept_groups)

            results["department_anova"] = {
                "test": "One-way ANOVA",
                "hypothesis": "Average GPA differs across departments.",
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "departments_tested": dept_names,
            }

            sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"

            print("\n3) GPA across Departments (ANOVA)")
            print(f"   F-statistic: {f_stat:.4f}")
            print(f"   p-value: {format_p_value(p_value)}")
            print(f"   Interpretation: Department GPA differences are {sig_text}.")

    return results


def build_findings_markdown(df, corr_results, test_results, distribution_files):
    """Create FINDINGS.md content."""
    corr_matrix = corr_results["corr_matrix"]
    top_pairs = corr_results["top_pairs"]

    gpa_skew = df["gpa"].dropna().skew() if "gpa" in df.columns else np.nan
    study_skew = df["study_hours_weekly"].dropna().skew() if "study_hours_weekly" in df.columns else np.nan
    attendance_skew = df["attendance_pct"].dropna().skew() if "attendance_pct" in df.columns else np.nan

    def skew_label(value):
        if pd.isna(value):
            return "not available"
        if value > 0.5:
            return "right-skewed"
        if value < -0.5:
            return "left-skewed"
        return "roughly symmetric"

    internship_section = "Not available."
    if "internship_ttest" in test_results:
        t = test_results["internship_ttest"]
        internship_sig = "statistically significant" if t["p_value"] < 0.05 else "not statistically significant"
        internship_section = (
            f"An independent samples t-test compared GPA between students with and without internships. "
            f"The test returned t = {t['t_statistic']:.4f} and p {format_p_value(t['p_value'])}. "
            f"This result is {internship_sig}. "
            f"Cohen's d = {t['cohens_d']:.4f}, indicating a "
            f"{interpret_effect_size_cohens_d(t['cohens_d'])} effect size."
        )

    scholarship_section = "Not available."
    if "scholarship_chi_square" in test_results:
        c = test_results["scholarship_chi_square"]
        scholarship_sig = "statistically significant" if c["p_value"] < 0.05 else "not statistically significant"
        scholarship_section = (
            f"A chi-square test of independence examined whether scholarship status is associated with department. "
            f"The test returned chi-square = {c['chi2_statistic']:.4f}, "
            f"p = {format_p_value(c['p_value'])}, and df = {c['degrees_of_freedom']}. "
            f"This association is {scholarship_sig}."
        )

    top_pair_lines = []
    for col1, col2, value in top_pairs:
        top_pair_lines.append(f"- `{col1}` and `{col2}`: |r| = {value:.3f}")

    recommendations = [
        "1. Strengthen academic support programs that help students build consistent weekly study habits, because study time is expected to align positively with GPA.",
        "2. Review department-level GPA differences and provide targeted tutoring or advising where GPA spread or median performance looks weaker.",
        "3. If internship students show meaningfully higher GPA, the university should expand structured internship access and monitor whether that pattern remains stable over time.",
    ]

    md = f"""# FINDINGS

## 1. Dataset Overview
The cleaned dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.
Main columns include: `{ "`, `".join(df.columns.tolist()) }`.

Notable data-quality issues:
- `commute_minutes` had missing values and was filled using the median.
- `scholarship` had missing values; these were kept as missing in the general dataset and implicitly excluded from the chi-square contingency analysis.

See `output/data_profile.txt`.

## 2. Distribution Analysis
- `gpa` appears **{skew_label(gpa_skew)}**.
- `study_hours_weekly` appears **{skew_label(study_skew)}**.
- `attendance_pct` appears **{skew_label(attendance_skew)}**.
- The department box plot helps compare GPA medians, spread, and potential outliers across departments.
- Scholarship categories are not evenly distributed.

Charts:
- `output/gpa_distribution.png`
- `output/study_hours_weekly_distribution.png`
- `output/attendance_pct_distribution.png`
- `output/gpa_by_department.png`
- `output/scholarship_counts.png`

## 3. Correlation Analysis
Top correlated numeric pairs:
{chr(10).join(top_pair_lines)}

The correlation heatmap shows which numeric variables move together most strongly. These relationships may reflect meaningful academic patterns, but correlation does not imply causation.

Charts:
- `output/correlation_heatmap.png`
{chr(10).join([f"- `{path}`" for path in corr_results["scatter_paths"]])}

## 4. Hypothesis Testing

### Internship and GPA
{internship_section}

### Scholarship and Department
{scholarship_section}
"""

    if "department_anova" in test_results:
        a = test_results["department_anova"]
        anova_sig = "statistically significant" if a["p_value"] < 0.05 else "not statistically significant"
        md += f"""
### GPA Differences Across Departments (ANOVA)
A one-way ANOVA tested whether average GPA differs across departments.
The result was F = {a['f_statistic']:.4f}, p = {format_p_value(a['p_value'])}.
This result is {anova_sig}.
"""

    md += f"""

## 5. Recommendations
{chr(10).join(recommendations)}

## 6. Generated Files
- `output/data_profile.txt`
{chr(10).join([f"- `{path}`" for path in distribution_files])}
- `output/correlation_heatmap.png`
{chr(10).join([f"- `{path}`" for path in corr_results["scatter_paths"]])}
"""

    return md


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_profile(DATA_PATH)
    distribution_files = plot_distributions(df)
    corr_results = plot_correlations(df)
    test_results = run_hypothesis_tests(df)

    findings_md = build_findings_markdown(df, corr_results, test_results, distribution_files)
    save_text_file("FINDINGS.md", findings_md)

    print("\n=== Output Files Created ===")
    print("- output/data_profile.txt")
    for path in distribution_files:
        print(f"- {path}")
    print(f"- {corr_results['heatmap_path']}")
    for path in corr_results["scatter_paths"]:
        print(f"- {path}")
    print("- FINDINGS.md")


if __name__ == "__main__":
    main()
