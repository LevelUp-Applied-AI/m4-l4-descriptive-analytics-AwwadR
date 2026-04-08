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
- The violin plot for GPA by department showed that the five departments have very similar distribution shapes and central ranges, with no strong visual evidence of major differences.
- Scholarship categories are not evenly distributed.

Charts:
- `output/gpa_distribution.png`
- `output/study_hours_weekly_distribution.png`
- `output/attendance_pct_distribution.png`
- `output/gpa_by_department.png`
- `output/gpa_by_department_violin.png`
- `output/scholarship_counts.png`

## 3. Correlation Analysis
The strongest relationship was between `study_hours_weekly` and `gpa`, with **|r| = 0.639**, indicating a moderate positive relationship. This suggests that students who study more each week tend to achieve higher GPAs.

The next highest correlation was between `gpa` and `attendance_pct`, with **|r| = 0.041**, which is extremely weak and not practically meaningful.

The correlation heatmap shows which numeric variables move together most strongly. These relationships may reflect academic patterns, but correlation does not imply causation.

Charts:
- `output/correlation_heatmap.png`
- `output\scatter_study_hours_weekly_vs_gpa.png`
- `output\scatter_gpa_vs_attendance_pct.png`

## 4. Hypothesis Testing

### Internship and GPA
An independent samples t-test compared GPA between students with and without internships. The test returned **t = 14.2288** and **p < 0.001**. This result is statistically significant. **Cohen's d = 0.6898**, indicating a **medium** effect size.

### Scholarship and Department
A chi-square test of independence examined whether scholarship status is associated with department. The test returned **chi-square = 13.9486**, **p = 0.3040**, and **df = 12**. This association was not statistically significant.


### GPA Differences Across Departments (ANOVA)
A one-way ANOVA tested whether average GPA differs across departments. The result was **F = 0.6671** and **p 0.6148**. This result was not statistically significant.

Since the ANOVA result was not statistically significant, no post-hoc pairwise t-tests were performed.

## 5. Tier 3 Statistical Extensions

### Bootstrap Confidence Intervals for Mean GPA by Internship Status
- Internship = Yes: mean = **2.9828**, 95% CI = **[2.9506, 3.0151]**
- Internship = No: mean = **2.7006**, 95% CI = **[2.6794, 2.7222]**
These bootstrap intervals estimate the uncertainty around the mean GPA in each group using repeated resampling.

### Power Analysis
Using the observed effect size (**Cohen's d = 0.6898**), the estimated required sample size is **33.98 students per group** to detect the difference with **80% power** at **alpha = 0.05**.

### False Positive Rate Simulation
A simulation under the null hypothesis produced a false positive rate of **0.0448**, which can be compared to the nominal alpha level of **0.05**.


## 6. Recommendations
1. Strengthen academic support programs that help students build consistent weekly study habits, since study hours showed a moderate positive relationship with GPA.
2. Use department-level visual monitoring to continue tracking GPA spread and outliers, even though the current ANOVA did not show statistically significant mean differences.
3. Consider expanding structured internship opportunities, since students with internships showed significantly higher GPA with a medium effect size.

## 7. Generated Files
- `output/data_profile.txt`
- `output\gpa_distribution.png`
- `output\study_hours_weekly_distribution.png`
- `output\attendance_pct_distribution.png`
- `output\gpa_by_department.png`
- `output\gpa_by_department_violin.png`
- `output\scholarship_counts.png`
- `output/correlation_heatmap.png`
- `output\scatter_study_hours_weekly_vs_gpa.png`
- `output\scatter_gpa_vs_attendance_pct.png`
