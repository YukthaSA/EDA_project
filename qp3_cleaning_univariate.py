import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")

URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"

# 1. Import
df = pd.read_csv(URL)

"""# demo corruption
rng = np.random.default_rng(7)
df.loc[rng.choice(df.index, 70, replace=False), "Age"] = np.nan
df.loc[rng.choice(df.index, 90, replace=False), "MonthlyIncome"] = np.nan
df.loc[rng.choice(df.index, 60, replace=False), "Department"] = np.nan
df["Sparse_Col"] = np.where(rng.random(len(df)) < 0.80, np.nan, 1.0)   # ~80 % missing
df = pd.concat([df, df.sample(12, random_state=3)], ignore_index=True)  # 12 duplicates"""


print("1. Dataset imported:", df.shape, "\n")

# 2. First and last six records
print("2. First six\n", df.head(6), "\n\n   Last six\n", df.tail(6), "\n")

# 3. Total missing values
print("3. Total missing values:", int(df.isna().sum().sum()), "\n")

# 4. Records with at least one missing value
print("4. Records with >= 1 missing value:", df.isna().any(axis=1).sum())
print(df[df.isna().any(axis=1)].head(), "\n")

# 5. Remove columns having > 70 % missing values
pct_missing = df.isna().mean() * 100
drop_cols = pct_missing[pct_missing > 70].index.tolist()
df = df.drop(columns=drop_cols)
print("5. Columns dropped (>70 % missing):", drop_cols, "\n")

# 6. Replace missing numerical values with the MEAN   (Age)
df["Age"] = df["Age"].fillna(df["Age"].mean())
print("6. Age missing after mean imputation:", df["Age"].isna().sum())

# 7. Replace missing numerical values with the MEDIAN (MonthlyIncome)
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
print("7. MonthlyIncome missing after median imputation:", df["MonthlyIncome"].isna().sum())

# 8. Replace missing categorical values with the MODE (Department)
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])
print("8. Department missing after mode imputation:", df["Department"].isna().sum(), "\n")

# 9. Identify duplicate rows
dups = df[df.duplicated(keep=False)]
print("9. Duplicate rows (all copies):", len(dups), "| extra copies:", df.duplicated().sum())

# 10. Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
print("10. Shape after dropping duplicates:", df.shape, "\n")

# 11. Columns with incorrect data types
print("11. Columns with questionable data types")
print("    object columns that should be categorical:",
      df.select_dtypes(exclude="number").columns.tolist())
ordinal_like = [c for c in df.select_dtypes("number").columns if df[c].nunique() <= 5]
print("    integer columns with <=5 levels (really categorical/ordinal):", ordinal_like, "\n")

# 12. Convert character variables into factors (category)
for c in df.select_dtypes(exclude=["number", "category"]).columns:
    df[c] = df[c].astype("category")
print("12. Dtypes after conversion\n", df.dtypes.value_counts(), "\n")

# 13. Convert text values to lower / upper case
df["Department_lower"] = df["Department"].astype(str).str.lower()
df["Gender_upper"] = df["Gender"].astype(str).str.upper()
print("13. Case conversion\n", df[["Department", "Department_lower", "Gender", "Gender_upper"]].head(), "\n")

# 14. Detect outliers using boxplots
num_for_box = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion"]
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, col in zip(axes, num_for_box):
    sns.boxplot(y=df[col], ax=ax, color="skyblue")
    ax.set_title(col)
fig.suptitle("14. Boxplots for outlier detection")
plt.tight_layout(); plt.show()

# 15. Identify outliers with the IQR method
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

print("15. IQR outliers (1.5 x IQR)")
for col in num_for_box:
    lo, hi = iqr_bounds(df[col])
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"    {col:25s} bounds=({lo:9.1f}, {hi:9.1f})  outliers={n}")
print()

# 16. Remove extreme outliers (3 x IQR) on MonthlyIncome
lo, hi = iqr_bounds(df["MonthlyIncome"], k=3)
before = len(df)
df_noext = df[(df["MonthlyIncome"] >= lo) & (df["MonthlyIncome"] <= hi)]
print(f"16. Extreme outliers removed: {before - len(df_noext)}  -> shape {df_noext.shape}\n")

# 17. Histogram + interpretation
plt.figure(figsize=(8, 5))
plt.hist(df["MonthlyIncome"], bins=30, color="steelblue", edgecolor="white")
plt.title("17. Histogram of MonthlyIncome"); plt.xlabel("MonthlyIncome"); plt.ylabel("Frequency")
plt.show()
sk = stats.skew(df["MonthlyIncome"])
print(f"17. Interpretation: skewness = {sk:.2f}; the distribution is "
      f"{'right (positively) skewed' if sk > 0.5 else 'left skewed' if sk < -0.5 else 'roughly symmetric'}"
      f" - most employees earn modest salaries while a few earn very high ones.\n")

# 18. Histogram with density curve
plt.figure(figsize=(8, 5))
sns.histplot(df["MonthlyIncome"], bins=30, stat="density", kde=True, color="steelblue")
plt.title("18. Histogram with density curve")
plt.show()

# 19. Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(y=df["MonthlyIncome"], color="lightgreen")
plt.title("19. Violin plot of MonthlyIncome")
plt.show()

# 20. Dot plot for a small numerical dataset
small = df["Age"].sample(30, random_state=1).sort_values().reset_index(drop=True)
plt.figure(figsize=(8, 5))
sns.stripplot(x=small, size=8, color="crimson", jitter=0.15)
plt.title("20. Dot plot of 30 sampled ages"); plt.xlabel("Age")
plt.show()

# 21. Frequency table of a categorical variable
freq = df["JobRole"].value_counts()
print("21. Frequency table (JobRole)\n", freq, "\n")

# 22. Proportion and percentage
prop = df["JobRole"].value_counts(normalize=True)
print("22. Proportion & percentage\n",
      pd.DataFrame({"Count": freq, "Proportion": prop.round(4), "Percent": (prop * 100).round(2)}), "\n")

# 23. Bar chart of frequencies
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Department", hue="Department", palette="Set2", legend=False)
plt.title("23. Frequency of each Department"); plt.ylabel("Count")
plt.show()

# 24. Horizontal bar chart (several categories)
plt.figure(figsize=(9, 5))
freq.sort_values().plot(kind="barh", color="teal")
plt.title("24. JobRole frequency (horizontal)"); plt.xlabel("Count")
plt.tight_layout(); plt.show()

# 25. Percentage bar chart
plt.figure(figsize=(9, 5))
(prop.sort_values() * 100).plot(kind="barh", color="orange")
plt.title("25. JobRole percentage"); plt.xlabel("Percentage (%)")
plt.tight_layout(); plt.show()

# 26. Pie chart (small number of categories)
plt.figure(figsize=(6, 6))
df["Attrition"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"], startangle=90)
plt.title("26. Attrition share"); plt.ylabel("")
plt.show()

# 27. Lollipop chart
f = freq.sort_values()
plt.figure(figsize=(9, 5))
plt.hlines(y=f.index, xmin=0, xmax=f.values, color="grey")
plt.plot(f.values, f.index, "o", color="purple", markersize=9)
plt.title("27. Lollipop chart - JobRole"); plt.xlabel("Count")
plt.tight_layout(); plt.show()

# 28. Dot chart (Cleveland dot plot)
plt.figure(figsize=(9, 5))
plt.scatter(f.values, f.index, color="darkgreen", s=60)
plt.grid(axis="y", linestyle=":")
plt.title("28. Dot chart - JobRole counts"); plt.xlabel("Count")
plt.tight_layout(); plt.show()

# 29. Density plot
plt.figure(figsize=(8, 5))
sns.kdeplot(df["Age"], fill=True, color="royalblue")
plt.title("29. Density plot of Age")
plt.show()

# 30. facet_wrap equivalent - distribution of one variable across groups
g = sns.displot(data=df, x="MonthlyIncome", col="Department", col_wrap=3,
                kde=True, bins=25, height=4, facet_kws={"sharey": False})
g.figure.suptitle("30. MonthlyIncome distribution by Department", y=1.03)
plt.show()

print("30. Interpretation per group")
for dept, sub in df.groupby("Department", observed=True)["MonthlyIncome"]:
    print(f"    {dept:25s} mean={sub.mean():8.1f} median={sub.median():8.1f} skew={stats.skew(sub):5.2f}"
          f" -> {'right-skewed' if stats.skew(sub) > 0.5 else 'fairly symmetric'}")
