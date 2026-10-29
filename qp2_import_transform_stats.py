import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"
INJECT_DEMO_DATA = False

# 1. Import the dataset
df = pd.read_csv(URL)
print("1. Dataset imported.\n")

if INJECT_DEMO_DATA:
    rng = np.random.default_rng(42)
    df = df.copy()
    df.loc[rng.choice(df.index, 60, replace=False), "Age"] = np.nan
    df.loc[rng.choice(df.index, 80, replace=False), "MonthlyIncome"] = np.nan
    df.loc[rng.choice(df.index, 50, replace=False), "Department"] = np.nan
    df.loc[rng.choice(df.index, 40, replace=False), "YearsAtCompany"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # 10 duplicates

# 2. First 10 records
print("2. First 10 records\n", df.head(10), "\n")

# 3. Last 10 records
print("3. Last 10 records\n", df.tail(10), "\n")

# 4. Rows and columns
print(f"4. Rows = {df.shape[0]}, Columns = {df.shape[1]}\n")

# 5. Column names
print("5. Column names:\n", list(df.columns), "\n")

# 6. Data types / structure
print("6. Data types")
df.info()
print()

# 7. Summary statistics
print("7. Summary statistics\n", df.describe(include="all").T.head(15), "\n")

# 8 & 10. Missing values in each column
print("8/10. Missing values per column\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# 9. Total missing values
print("9. Total missing values:", int(df.isna().sum().sum()), "\n")

# 11. Rows with at least one missing value
rows_with_na = df[df.isna().any(axis=1)]
print(f"11. Rows containing at least one missing value: {len(rows_with_na)}")
print(rows_with_na.head(), "\n")

# 12. Remove all rows containing missing values
df_dropna = df.dropna()
print(f"12. After dropping rows with NaN: {df_dropna.shape}")
print(df_dropna.head(), "\n")

# 13. Replace missing values in a numerical column with the MEAN
df_fill = df.copy()
df_fill["Age"] = df_fill["Age"].fillna(df_fill["Age"].mean())
print("13. Age NaN after mean fill:", df_fill["Age"].isna().sum())

# 14. Replace missing values in another numerical column with the MEDIAN
df_fill["MonthlyIncome"] = df_fill["MonthlyIncome"].fillna(df_fill["MonthlyIncome"].median())
print("14. MonthlyIncome NaN after median fill:", df_fill["MonthlyIncome"].isna().sum())

# 15. Replace missing values in a categorical column with the MODE
df_fill["Department"] = df_fill["Department"].fillna(df_fill["Department"].mode()[0])
print("15. Department NaN after mode fill:", df_fill["Department"].isna().sum(), "\n")

# 16. Forward fill  (equivalent of zoo::na.locf in R)
df_ffill = df.copy()
df_ffill["YearsAtCompany"] = df_ffill["YearsAtCompany"].ffill()
print("16. YearsAtCompany NaN after forward fill:", df_ffill["YearsAtCompany"].isna().sum())

# 17. Backward fill
df_bfill = df.copy()
df_bfill["YearsAtCompany"] = df_bfill["YearsAtCompany"].bfill()
print("17. YearsAtCompany NaN after backward fill:", df_bfill["YearsAtCompany"].isna().sum())

# 18. Replace remaining missing values with a constant (0 / "Unknown")
df_const = df.copy()
num_cols = df_const.select_dtypes(include="number").columns
cat_cols = df_const.select_dtypes(exclude="number").columns
df_const[num_cols] = df_const[num_cols].fillna(0)
df_const[cat_cols] = df_const[cat_cols].fillna("Unknown")
print("18. Remaining NaN after constant fill:", int(df_const.isna().sum().sum()), "\n")

# 19. Remove duplicate records
print("19. Duplicate rows found:", df_fill.duplicated().sum())
df_nodup = df_fill.drop_duplicates()
print("    Shape after removing duplicates:", df_nodup.shape, "\n")

# 20. Convert appropriate variables into factors (category dtype)
df_clean = df_nodup.copy()
factor_cols = ["Attrition", "BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "Over18", "OverTime"]
df_clean[factor_cols] = df_clean[factor_cols].astype("category")
print("20. Dtypes after converting to category:\n", df_clean[factor_cols].dtypes, "\n")

# 21. Rename column names
df_clean = df_clean.rename(columns={"MonthlyIncome": "Monthly_Income",
                                    "DistanceFromHome": "Distance_From_Home"})
print("21. Renamed columns ->", [c for c in df_clean.columns if "_" in c], "\n")

# 22. New derived column
df_clean["Annual_Income"] = df_clean["Monthly_Income"] * 12
print("22. Derived column Annual_Income\n", df_clean[["Monthly_Income", "Annual_Income"]].head(), "\n")

# 23. summary() equivalent
print("23. Summary statistics\n", df_clean.describe().T.head(12), "\n")

# 24. Min and max
print("24. Min / Max of Monthly_Income:", df_clean["Monthly_Income"].min(),
      "/", df_clean["Monthly_Income"].max())

# 25. Range
print("25. Range of Monthly_Income:",
      df_clean["Monthly_Income"].max() - df_clean["Monthly_Income"].min())

# 26. Variance
print("26. Variance of Monthly_Income:", round(df_clean["Monthly_Income"].var(), 2))

# 27. Standard deviation
print("27. Std. deviation of Monthly_Income:", round(df_clean["Monthly_Income"].std(), 2))

# 28. IQR
q1, q3 = df_clean["Monthly_Income"].quantile([0.25, 0.75])
print("28. IQR of Monthly_Income:", q3 - q1)
