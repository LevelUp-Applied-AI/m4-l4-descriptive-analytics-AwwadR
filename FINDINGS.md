# FINDINGS

## 1. Dataset Overview
The cleaned dataset contains **2000 rows** and **10 columns**.
Main columns include: `student_id`, `department`, `semester`, `course_load`, `study_hours_weekly`, `gpa`, `attendance_pct`, `has_internship`, `commute_minutes`, `scholarship`.

Notable data-quality issues:
- `commute_minutes` had missing values and was filled using the median.
- `scholarship` had missing values; these were kept as missing in the general dataset and implicitly excluded from the chi-square contingency analysis.

See `output/data_profile.txt`.

## 2. Distribution Analysis
- `gpa` appears **roughly symmetric**.
- `study_hours_weekly` appears **roughly symmetric**.
- `attendance_pct` appears **roughly symmetric**.
- The department box plot helps compare GPA medians, spread, and potential outliers across departments.
- Scholarship categories are not evenly distributed.
- The violin plot for GPA by department showed that the five departments have very similar distribution shapes and central ranges, with no strong visual evidence of major differences.

Charts:
- `output/gpa_distribution.png`
- `output/study_hours_weekly_distribution.png`
- `output/attendance_pct_distribution.png`
- `output/gpa_by_department.png`
- `output/gpa_by_department_violin.png`
- `output/scholarship_counts.png`

## 3. Correlation Analysis
Top correlated numeric pairs:
- `study_hours_weekly` and `gpa`: |r| = 0.639
- `gpa` and `attendance_pct`: |r| = 0.041

The correlation heatmap shows which numeric variables move together most strongly. These relationships may reflect meaningful academic patterns, but correlation does not imply causation.

Charts:
- `output/correlation_heatmap.png`
- `output/scatter_study_hours_weekly_vs_gpa.png`
- `output/scatter_gpa_vs_attendance_pct.png`

## 4. Hypothesis Testing

### Internship and GPA
An independent samples t-test compared GPA between students with and without internships. The test returned t = 14.2288 and p < 0.001. This result is statistically significant. Cohen's d = 0.6898, indicating a medium effect size.

### Scholarship and Department
A chi-square test of independence examined whether scholarship status is associated with department. The test returned chi-square = 13.9486, p = 0.3040, and df = 12. This association is not statistically significant.

### GPA Differences Across Departments (ANOVA)
A one-way ANOVA tested whether average GPA differs across departments.
The result was F = 0.6671, p = 0.6148.
This result was not statistically significant.

Since the ANOVA result was not statistically significant, no post-hoc pairwise t-tests were performed.


## 5. Recommendations
1. Strengthen academic support programs that help students build consistent weekly study habits, because study time is expected to align positively with GPA.
2. Review department-level GPA differences and provide targeted tutoring or advising where GPA spread or median performance looks weaker.
3. If internship students show meaningfully higher GPA, the university should expand structured internship access and monitor whether that pattern remains stable over time.

## 6. Generated Files
- `output/data_profile.txt`
- `output/gpa_distribution.png`
- `output/study_hours_weekly_distribution.png`
- `output/attendance_pct_distribution.png`
- `output/gpa_by_department.png`
- `output/gpa_by_department_violin.png`
- `output/scholarship_counts.png`
- `output/correlation_heatmap.png`
- `output/scatter_study_hours_weekly_vs_gpa.png`
- `output/scatter_gpa_vs_attendance_pct.png`
