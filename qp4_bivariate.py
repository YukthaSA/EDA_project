import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"

# 1. Load dataset
df = pd.read_csv(URL)
print("1. Loaded:", df.shape, "\n")

# 2. Two categorical variables -> contingency table + stacked bar plots
ct = pd.crosstab(df["Department"], df["Attrition"])
print("2. Contingency table (Department vs Attrition)\n", ct, "\n")

# 2a. Simple stacked bar plot (basic matplotlib/pandas  ~ Base R)
ct.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#4c72b0", "#dd8452"])
plt.title("2a. Stacked Bar (matplotlib): Department vs Attrition")
plt.xlabel("Department"); plt.ylabel("Number of Employees")
plt.xticks(rotation=15); plt.legend(title="Attrition")
plt.tight_layout(); plt.show()

# 2b. Stacked bar plot with seaborn (~ ggplot2)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Department", hue="Attrition", multiple="stack", shrink=0.8,
             palette="Set2")
plt.title("2b. Stacked Bar (seaborn): Department vs Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout(); plt.show()

# 3. Grouped / side-by-side bar plot
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="MaritalStatus", hue="Attrition", palette="Set1")
plt.title("3. Grouped Bar: MaritalStatus vs Attrition")
plt.xlabel("Marital Status"); plt.ylabel("Count"); plt.legend(title="Attrition")
plt.tight_layout(); plt.show()

# 4. Scatter plot of two numerical variables
plt.figure(figsize=(8, 6))
plt.scatter(df["Age"], df["MonthlyIncome"], alpha=0.5, color="teal", edgecolor="white")
plt.title("4. Scatter: Age vs MonthlyIncome")
plt.xlabel("Age"); plt.ylabel("Monthly Income")
plt.tight_layout(); plt.show()

# 5. Line plot (trend): average income by years at company
trend = df.groupby("YearsAtCompany")["MonthlyIncome"].mean()
plt.figure(figsize=(9, 5))
plt.plot(trend.index, trend.values, marker="o", color="crimson", linewidth=2)
plt.title("5. Line Plot: Mean MonthlyIncome by YearsAtCompany")
plt.xlabel("Years at Company"); plt.ylabel("Mean Monthly Income")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.show()

# 6. Heatmap of relationship among variables
num_cols = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
            "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
            "YearsWithCurrManager", "PercentSalaryHike"]
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"label": "Correlation"})
plt.title("6. Heatmap of Numerical Variables")
plt.tight_layout(); plt.show()

# 7. Bar chart comparing values (mean income by department)
mean_inc = df.groupby("Department")["MonthlyIncome"].mean().sort_values()
plt.figure(figsize=(8, 5))
bars = plt.bar(mean_inc.index, mean_inc.values, color=["#8da0cb", "#66c2a5", "#fc8d62"])
plt.bar_label(bars, fmt="%.0f")
plt.title("7. Bar Chart: Mean MonthlyIncome by Department")
plt.xlabel("Department"); plt.ylabel("Mean Monthly Income")
plt.tight_layout(); plt.show()

# 8. Density plot: numerical across categories
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="Age", hue="Attrition", fill=True, common_norm=False, alpha=0.4)
plt.title("8. Density Plot: Age by Attrition")
plt.tight_layout(); plt.show()

# 9. Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Department", y="MonthlyIncome", hue="Department", palette="Set3", legend=False)
plt.title("9. Box Plot: MonthlyIncome by Department")
plt.tight_layout(); plt.show()

# 10. Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="Department", y="MonthlyIncome", hue="Department", palette="Pastel1", legend=False)
plt.title("10. Violin Plot: MonthlyIncome by Department")
plt.tight_layout(); plt.show()

# 11. Violin + box in one graphic
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="JobSatisfaction", y="MonthlyIncome", hue="JobSatisfaction",
               palette="muted", inner=None, legend=False)
sns.boxplot(data=df, x="JobSatisfaction", y="MonthlyIncome", width=0.15,
            boxprops={"facecolor": "white", "zorder": 3}, showfliers=False)
plt.title("11. Violin + Box: MonthlyIncome by JobSatisfaction")
plt.tight_layout(); plt.show()

# 12. Correlation matrix + correlogram (lower triangle)
corr_matrix = df[num_cols].corr()
print("12. Correlation matrix\n", corr_matrix.round(2), "\n")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"})
plt.title("12. Correlogram of Numerical Variables")
plt.tight_layout(); plt.show()

# 13. Titles, axis labels, legends and parameters are set on every plot above.
print("13. All plots include titles, axis labels, legends and graphical parameters.")
