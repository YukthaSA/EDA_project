import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats               

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")
URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"

# 1. Load dataset
df = pd.read_csv(URL)
print("1. Loaded:", df.shape, "\n")

# 2 & 3. Summary statistics stored in one structured data frame
num_vars = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"]
rows = {}
for c in num_vars:
    s = df[c]
    rows[c] = {
        "Mean": s.mean(), "Median": s.median(), "Minimum": s.min(), "Maximum": s.max(),
        "Range": s.max() - s.min(), "Variance": s.var(), "Std Dev": s.std(),
        "IQR": s.quantile(0.75) - s.quantile(0.25),
        "Skewness": stats.skew(s),               # e1071::skewness (type 3)
        "Kurtosis": stats.kurtosis(s),           # e1071::kurtosis (type 3, excess)
    }
summary_df = pd.DataFrame(rows).T.round(3)
print("2/3. Summary statistics\n", summary_df, "\n")

# 4. Frequency + relative frequency table for a categorical variable; pie + bar
freq = df["Department"].value_counts()
rel = df["Department"].value_counts(normalize=True).round(4)
print("4. Frequency / Relative frequency (Department)\n",
      pd.DataFrame({"Frequency": freq, "Relative Frequency": rel}), "\n")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].pie(freq, labels=freq.index, autopct="%1.1f%%", startangle=90,
          colors=sns.color_palette("pastel"))
ax[0].set_title("4a. Pie Chart - Department")
sns.barplot(x=freq.index, y=freq.values, hue=freq.index, palette="pastel", ax=ax[1], legend=False)
ax[1].set_title("4b. Bar Plot - Department"); ax[1].set_ylabel("Frequency")
ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.show()

# 5. Frequency table with bin intervals; histogram + box plot
bins = pd.cut(df["MonthlyIncome"], bins=10)
print("5. Binned frequency table (MonthlyIncome)\n", bins.value_counts().sort_index(), "\n")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].hist(df["MonthlyIncome"], bins=10, color="skyblue", edgecolor="black")
ax[0].set_title("5a. Histogram - MonthlyIncome"); ax[0].set_xlabel("MonthlyIncome")
sns.boxplot(y=df["MonthlyIncome"], ax=ax[1], color="lightcoral")
ax[1].set_title("5b. Box Plot (outliers = dots)")
plt.tight_layout(); plt.show()

# 6. Contingency table + row / column percentages
ct = pd.crosstab(df["Department"], df["Attrition"])
print("6. Contingency table\n", ct, "\n")
print("   Row percentages (%)\n", (ct.div(ct.sum(axis=1), axis=0) * 100).round(2), "\n")
print("   Column percentages (%)\n", (ct.div(ct.sum(axis=0), axis=1) * 100).round(2), "\n")

# 7. Stacked and grouped bar plots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ct.plot(kind="bar", stacked=True, ax=ax[0], color=["#4c72b0", "#dd8452"])
ax[0].set_title("7a. Stacked Bar - Department vs Attrition"); ax[0].tick_params(axis="x", rotation=15)
ct.plot(kind="bar", ax=ax[1], color=["#4c72b0", "#dd8452"])
ax[1].set_title("7b. Grouped Bar - Department vs Attrition"); ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.show()

# 8. One categorical + one numerical: group statistics (aggregate())
agg = df.groupby("Department")["MonthlyIncome"].agg(["mean", "median", "std"]).round(2)
print("8. MonthlyIncome by Department\n", agg, "\n")

# 9. Side-by-side box plots + violin plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="Department", y="MonthlyIncome", hue="Department", ax=ax[0],
            palette="Set2", legend=False)
ax[0].set_title("9a. Side-by-side Box Plots")
sns.violinplot(data=df, x="Department", y="MonthlyIncome", hue="Department", ax=ax[1],
               palette="Set2", legend=False)
ax[1].set_title("9b. Violin Plot")
plt.tight_layout(); plt.show()

# 10. Pearson & Spearman correlation + scatter with regression line
pearson = df["Age"].corr(df["MonthlyIncome"], method="pearson")
spearman = df["Age"].corr(df["MonthlyIncome"], method="spearman")
print(f"10. Age vs MonthlyIncome -> Pearson = {pearson:.3f}, Spearman = {spearman:.3f}\n")
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x="Age", y="MonthlyIncome", scatter_kws={"alpha": 0.4},
            line_kws={"color": "red"})
plt.title("10. Scatter plot with Linear Regression Line")
plt.tight_layout(); plt.show()

# 11. Two categorical + one numerical
grp = df.groupby(["Gender", "Attrition"])["MonthlyIncome"].agg(["mean", "std"]).round(2)
print("11. MonthlyIncome by Gender x Attrition\n", grp, "\n")

g = sns.catplot(data=df, x="Attrition", y="MonthlyIncome", col="Gender", kind="box",
                hue="Attrition", palette="Set2", height=4, legend=False)
g.figure.suptitle("11a. Faceted Box Plot", y=1.03)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Gender", y="MonthlyIncome", hue="Attrition", errorbar="sd",
            capsize=0.1, palette="Set1")
plt.title("11b. Grouped Bar Chart with Error Bars (mean +/- SD)")
plt.tight_layout(); plt.show()

# 12. One categorical + two numerical
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Age", y="MonthlyIncome", hue="Attrition", style="Attrition",
                alpha=0.7, palette="Set1")
plt.title("12. Age vs MonthlyIncome by Attrition")
plt.tight_layout(); plt.show()
print("12. Correlation (Age vs MonthlyIncome) separately for each Attrition level")
for level, sub in df.groupby("Attrition"):
    print(f"    {level}:\n", sub[["Age", "MonthlyIncome"]].corr().round(3), "\n")

# 13. Correlation matrix + heatmap (>= 3 numerical features)
hm_vars = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
corr = df[hm_vars].corr()
print("13. Correlation matrix\n", corr.round(3), "\n")
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("13. Correlation Heatmap")
plt.tight_layout(); plt.show()

# 14. Pairs plot / scatterplot matrix
g = sns.pairplot(df[hm_vars + ["Attrition"]], hue="Attrition", diag_kind="kde",
                 plot_kws={"alpha": 0.4}, palette="Set1")
g.figure.suptitle("14. Pairs Plot", y=1.02)
plt.show()
